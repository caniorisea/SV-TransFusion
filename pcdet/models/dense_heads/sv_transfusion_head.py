import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils

from .transfusion_head import SeparateHead_Transfusion, TransFusionHead


def voxel_indices_to_metric_xyz(indices_zyx: torch.Tensor, voxel_size, pc_range):
    """
    indices_zyx: (L, 3) int tensor in (z,y,x)
    return xyz: (L, 3) float tensor in meters (x,y,z)
    """
    zyx = indices_zyx.to(torch.float32)
    vs = torch.tensor(voxel_size, device=zyx.device, dtype=torch.float32)  # (x,y,z)
    pr = torch.tensor(pc_range[0:3], device=zyx.device, dtype=torch.float32)
    x = (zyx[:, 2] + 0.5) * vs[0] + pr[0]
    y = (zyx[:, 1] + 0.5) * vs[1] + pr[1]
    z = (zyx[:, 0] + 0.5) * vs[2] + pr[2]
    return torch.stack([x, y, z], dim=-1)


class SVQIModule(nn.Module):
    def __init__(self, d_model, svqi_cfg, voxel_size, pc_range):
        super().__init__()
        self.cfg = svqi_cfg
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.d_model = d_model

        self.radius = float(getattr(svqi_cfg, 'RADIUS', 1.5))
        self.max_neighbors = int(getattr(svqi_cfg, 'MAX_NEIGHBORS', 64))
        self.use_dist = bool(getattr(svqi_cfg, 'USE_DIST_NORM', True))
        pos_hidden = int(getattr(svqi_cfg, 'POS_MLP_HIDDEN', 128))

        in_dim = 4 if self.use_dist else 3
        self.pos_mlp = nn.Sequential(
            nn.Linear(in_dim, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, d_model),
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    @torch.no_grad()
    def _knn_in_radius(self, q_xyz, v_xyz):
        """
        q_xyz: (K,3)
        v_xyz: (L,3)
        return idx: (K,max_neighbors) long padded with -1
        """
        # dist: (K,L)
        dist = torch.cdist(q_xyz.unsqueeze(0), v_xyz.unsqueeze(0)).squeeze(0)
        mask = dist <= self.radius
        K = q_xyz.shape[0]
        idx = torch.full((K, self.max_neighbors), -1, device=q_xyz.device, dtype=torch.long)


        for k in range(K):
            sel = torch.nonzero(mask[k], as_tuple=False).squeeze(-1)
            if sel.numel() == 0:
                continue
            d = dist[k, sel]
            order = torch.argsort(d)[: self.max_neighbors]
            picked = sel[order]
            idx[k, :picked.numel()] = picked
        return idx

    def forward(self, q_feat, q_xyz, sp_feat, sp_indices_full):
        """
        q_feat: (B,K,C)
        q_xyz:  (B,K,3) meters
        sp_feat: (L,C) sparse voxel features
        sp_indices_full: (L,4) [b,z,y,x]
        return: (B,K,C)
        """
        B, K, C = q_feat.shape
        out = q_feat.new_zeros((B, K, C))

        batch_idx = sp_indices_full[:, 0].long()
        indices_zyx = sp_indices_full[:, 1:4].long()
        v_xyz_all = voxel_indices_to_metric_xyz(indices_zyx, self.voxel_size, self.pc_range)  # (L,3)

        for b in range(B):
            sel = (batch_idx == b)
            if sel.sum() == 0:
                continue
            v_feat = sp_feat[sel]     # (Lb,C)
            v_xyz = v_xyz_all[sel]    # (Lb,3)
            qf = q_feat[b]            # (K,C)
            qx = q_xyz[b]             # (K,3)

            nn_idx = self._knn_in_radius(qx, v_xyz)  # (K,M)

            # per query attention (slow but ok)
            for k in range(K):
                ids = nn_idx[k]
                valid = ids >= 0
                if valid.sum() == 0:
                    continue
                vf = v_feat[ids[valid]]  # (Nk,C)
                rel = v_xyz[ids[valid]] - qx[k:k+1]  # (Nk,3)
                if self.use_dist:
                    d = torch.norm(rel, dim=-1, keepdim=True)
                    rel_in = torch.cat([rel, d], dim=-1)
                else:
                    rel_in = rel
                pe = self.pos_mlp(rel_in)
                val = vf + pe  # (Nk,C)

                q = self.q_proj(qf[k:k+1])      # (1,C)
                kk = self.k_proj(val)           # (Nk,C)
                vv = self.v_proj(val)           # (Nk,C)
                attn = torch.softmax((q @ kk.transpose(0, 1)) / (C ** 0.5), dim=-1)  # (1,Nk)
                agg = attn @ vv  # (1,C)
                out[b, k] = self.out_proj(agg).squeeze(0)

        return out


class GatedFusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2 * c, c), nn.Sigmoid())

    def forward(self, feat_2d, feat_3d):
        """
        feat_2d/feat_3d: (B,K,C)
        """
        a = self.fc(torch.cat([feat_2d, feat_3d], dim=-1))
        return a * feat_2d + (1.0 - a) * feat_3d


def info_nce_loss(z, labels, tau=0.07):
    """
    Supervised contrastive (simple):
    z: (N,C) normalized
    labels: (N,) long, -1 for ignore
    """
    device = z.device
    valid = labels >= 0
    z = z[valid]
    y = labels[valid]
    if z.shape[0] <= 1:
        return z.new_tensor(0.0)

    z = F.normalize(z, dim=-1)
    sim = (z @ z.t()) / tau  # (N,N)
    # mask self
    eye = torch.eye(sim.shape[0], device=device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    pos = (y[:, None] == y[None, :]) & (~eye)
    # log-softmax over all except self
    logp = F.log_softmax(sim, dim=-1)
    # for anchors with at least one positive
    denom = pos.sum(dim=-1)
    keep = denom > 0
    if keep.sum() == 0:
        return z.new_tensor(0.0)
    loss = -(logp[pos].view(-1).sum() / denom[keep].sum().clamp(min=1))
    return loss


class SVTransFusionHead(TransFusionHead):
    """
    - SVQI (3D sparse voxel interaction)
    - QCD (denoising + contrastive) training-only
    """
    def __init__(self, model_cfg, input_channels, num_class, class_names,
                 grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True):
        super().__init__(model_cfg, input_channels, num_class, class_names,
                         grid_size, point_cloud_range, voxel_size, predict_boxes_when_training)

        # decoder layers count
        self.num_decoder_layers = int(getattr(model_cfg, 'NUM_DECODER_LAYERS', 1))

        self.svqi_cfg = getattr(model_cfg, 'SVQI', None)
        self.qcd_cfg = getattr(model_cfg, 'QCD', None)

        hidden_channel = self.model_cfg.HIDDEN_CHANNEL

        # SVQI
        self.enable_svqi = self.svqi_cfg is not None and bool(getattr(self.svqi_cfg, 'ENABLED', False))
        if self.enable_svqi:
            self.svqi = SVQIModule(hidden_channel, self.svqi_cfg, voxel_size, point_cloud_range)
            self.fusion = GatedFusion(hidden_channel)

        # QCD
        self.enable_qcd = self.qcd_cfg is not None and bool(getattr(self.qcd_cfg, 'ENABLED', False))
        if self.enable_qcd:
            self.dn_groups = int(getattr(self.qcd_cfg, 'NUM_GROUPS', 3))
            self.center_noise = getattr(self.qcd_cfg, 'CENTER_NOISE_SCALE', [0.2, 0.2, 0.2])
            self.size_noise = float(getattr(self.qcd_cfg, 'SIZE_NOISE_SCALE', 0.4))
            self.tau = float(getattr(self.qcd_cfg, 'CONTRASTIVE_TAU', 0.07))

        self._last_dn_loss = None
        self._last_con_loss = None

    def _build_dn_queries(self, gt_boxes, lidar_feat_flatten, bev_pos):
        """
        gt_boxes: (B, M, 10?) in batch_dict (yours: gt_boxes [...,:-1] is bbox, last is label)
        return dn_query_feat, dn_query_pos_xy, dn_labels
        """
        # gt_boxes includes label at last dim in OpenPCDet batch_dict
        B = gt_boxes.shape[0]
        device = gt_boxes.device
        dn_feats = []
        dn_pos = []
        dn_labels = []

        # bev feature map size
        # bev_pos is (B, HW, 2) already flipped to (x,y) in parent predict()
        # lidar_feat_flatten is (B, C, HW)
        _, C, HW = lidar_feat_flatten.shape

        for b in range(B):
            gtb = gt_boxes[b]
            # filter valid
            valid = (gtb[:, 3] > 0) & (gtb[:, 4] > 0)
            gtb = gtb[valid]
            if gtb.numel() == 0:
                # no dn
                dn_feats.append(lidar_feat_flatten.new_zeros((C, 0)))
                dn_pos.append(bev_pos.new_zeros((0, 2)))
                dn_labels.append(torch.empty((0,), device=device, dtype=torch.long))
                continue

            boxes = gtb[:, :-1]  # (N,9?) x,y,z,w,l,h,yaw,vx,vy
            labels = gtb[:, -1].long() - 1  # to 0-based

            N = boxes.shape[0]
            # repeat groups
            boxes_rep = boxes[None].repeat(self.dn_groups, 1, 1).view(-1, boxes.shape[-1])
            labels_rep = labels[None].repeat(self.dn_groups, 1).view(-1)

            # noise center
            wlh = boxes_rep[:, 3:6].clamp(min=1e-3)
            noise = torch.zeros_like(boxes_rep[:, 0:3])
            noise[:, 0] = (torch.rand_like(noise[:, 0]) * 2 - 1) * self.center_noise[0] * wlh[:, 0]
            noise[:, 1] = (torch.rand_like(noise[:, 1]) * 2 - 1) * self.center_noise[1] * wlh[:, 1]
            noise[:, 2] = (torch.rand_like(noise[:, 2]) * 2 - 1) * self.center_noise[2] * wlh[:, 2]
            boxes_rep[:, 0:3] = boxes_rep[:, 0:3] + noise

            # noise size scale
            scale = (torch.rand((boxes_rep.shape[0], 3), device=device) * 2 - 1) * self.size_noise + 1.0
            boxes_rep[:, 3:6] = (boxes_rep[:, 3:6] * scale).clamp(min=1e-3)

            # map dn centers to BEV feature indices
            x = boxes_rep[:, 0]
            y = boxes_rep[:, 1]
            coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
            coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

            # clamp to feature map
            x_size = self.grid_size[0] // self.feature_map_stride
            y_size = self.grid_size[1] // self.feature_map_stride
            cx = coor_x.clamp(0, x_size - 1)
            cy = coor_y.clamp(0, y_size - 1)


            idx = (cy.long() * x_size + cx.long()).clamp(0, HW - 1)  # (Nrep,)

            # gather feature as initial dn query feat
            feat = lidar_feat_flatten[b].gather(dim=-1, index=idx[None, :].expand(C, -1))  # (C,Nrep)
            pos = torch.stack([cx, cy], dim=-1)  # (Nrep,2) in feature coords (x,y)

            dn_feats.append(feat)
            dn_pos.append(pos)
            dn_labels.append(labels_rep)

        # pad to max dn count in batch
        max_dn = max([f.shape[1] for f in dn_feats])
        if max_dn == 0:
            return None, None, None

        dn_query_feat = lidar_feat_flatten.new_zeros((B, C, max_dn))
        dn_query_pos = bev_pos.new_zeros((B, max_dn, 2))
        dn_query_label = torch.full((B, max_dn), -1, device=device, dtype=torch.long)

        for b in range(B):
            n = dn_feats[b].shape[1]
            if n == 0:
                continue
            dn_query_feat[b, :, :n] = dn_feats[b]
            dn_query_pos[b, :n] = dn_pos[b]
            dn_query_label[b, :n] = dn_labels[b]

        # add category embedding
        one_hot = F.one_hot(dn_query_label.clamp(min=0), num_classes=self.num_classes).permute(0, 2, 1).float()
        # ignore (-1) rows: set to 0
        one_hot = one_hot * (dn_query_label[:, None, :] >= 0).float()
        dn_query_feat = dn_query_feat + self.class_encoding(one_hot)

        return dn_query_feat, dn_query_pos, dn_query_label

    def predict(self, inputs, batch_dict=None):
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # (B,C,HW)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)      # (B,HW,2)

        # query init heatmap
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner

        if self.dataset_name == "nuScenes":
            local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.dataset_name == "Waymo":
            local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)

        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)  # (B,num_cls,HW)

        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., : self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]

        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )  # (B,C,K)
        self.query_labels = top_proposals_class

        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        query_feat = query_feat + self.class_encoding(one_hot.float())

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )  # (B,K,2) in (x,y) grid coords
        query_pos = query_pos.flip(dims=[-1])  # to (y,x)? original code flips to xy, keep consistent
        bev_pos_flip = bev_pos.flip(dims=[-1])

        # ---------------- QCD: build dn queries (training only) ----------------
        dn_query_feat = dn_query_pos = dn_query_label = None
        attn_mask = None
        if self.training and self.enable_qcd and batch_dict is not None and ('gt_boxes' in batch_dict):
            dn_query_feat, dn_query_pos, dn_query_label = self._build_dn_queries(
                batch_dict['gt_boxes'], lidar_feat_flatten, bev_pos_flip
            )
            if dn_query_feat is not None:
                # concat dn in front: (B,C,Kdn+K)
                query_feat = torch.cat([dn_query_feat, query_feat], dim=-1)
                query_pos = torch.cat([dn_query_pos, query_pos], dim=1)
                # build simple attention mask: prevent leakage between dn and matching queries
                Kdn = dn_query_feat.shape[-1]
                Kmatch = self.num_proposals
                total = Kdn + Kmatch
                attn_mask = torch.zeros((total, total), device=inputs.device, dtype=torch.bool)
                # block matching attending to dn and dn attending to matching (symmetric)
                attn_mask[Kdn:, :Kdn] = True
                attn_mask[:Kdn, Kdn:] = True
                # (optional) block dn groups each other: not implemented fully here

        # ---------------- Decoder loop (simulate multi-layer) ----------------
        # query_feat is (B,C,K)
        for _ in range(self.num_decoder_layers):
            query_feat = self.decoder(query_feat, lidar_feat_flatten, query_pos, bev_pos_flip, attn_mask=attn_mask)

            # SVQI fusion: needs sparse voxels and query xyz (in meters)
            if self.enable_svqi and batch_dict is not None:
                sp = batch_dict.get('encoded_spconv_tensor', None)
                if sp is not None:
                    # query_pos currently in feature coords (x,y). convert to meters (x,y), z set to 0 (or use predicted height later)
                    x = query_pos[:, :, 0] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
                    y = query_pos[:, :, 1] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
                    z = torch.zeros_like(x)
                    q_xyz = torch.stack([x, y, z], dim=-1)  # (B, Ktotal, 3)

                    q_feat_bkc = query_feat.permute(0, 2, 1)  # (B,K,C)
                    svqi_out = self.svqi(q_feat_bkc, q_xyz, sp.features, sp.indices)  # (B,K,C)
                    q_fused = self.fusion(q_feat_bkc, svqi_out)  # (B,K,C)
                    query_feat = q_fused.permute(0, 2, 1)  # back to (B,C,K)

        # prediction head
        res_layer = self.prediction_head(query_feat)  # each head: (B, out, Ktotal)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)

        # store query heatmap score only for matching queries
        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )
        res_layer["dense_heatmap"] = dense_heatmap

        # record dn info for loss
        res_layer["_dn_query_label"] = dn_query_label  # (B,Kdn) or None
        res_layer["_dn_num"] = 0 if dn_query_feat is None else dn_query_feat.shape[-1]

        return res_layer

    def forward(self, batch_dict):
        feats = batch_dict['spatial_features_2d']
        res = self.predict(feats, batch_dict=batch_dict)

        if not self.training:
            bboxes = self.get_bboxes(res)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[..., :-1]
            gt_labels_3d = gt_boxes[..., -1].long() - 1
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res, batch_dict=batch_dict)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, batch_dict=None, **kwargs):
        """
        baseline loss + QCD losses
        """
        dn_num = int(pred_dicts.get('_dn_num', 0))
        if dn_num > 0:
            # slice matching part (after dn)
            pred_match = {}
            for k, v in pred_dicts.items():
                if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[-1] >= dn_num:
                    pred_match[k] = v[:, :, dn_num:]
                else:
                    pred_match[k] = v
        else:
            pred_match = pred_dicts

        loss_main, tb = super().loss(gt_bboxes_3d, gt_labels_3d, pred_match)

        self._last_dn_loss = None
        self._last_con_loss = None

        # -------- QCD losses --------
        if self.training and self.enable_qcd and dn_num > 0:
            # dn predictions slice
            pred_dn = {}
            for k, v in pred_dicts.items():
                if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[-1] >= dn_num:
                    pred_dn[k] = v[:, :, :dn_num]
            dn_labels = pred_dicts.get('_dn_query_label', None)  # (B,dn_num)
            if dn_labels is not None:
                # --- classification dn loss ---
                cls_score_dn = pred_dn["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)
                dn_lab = dn_labels.reshape(-1)
                valid = dn_lab >= 0
                if valid.sum() > 0:
                    one_hot = torch.zeros((dn_lab.shape[0], self.num_classes), device=cls_score_dn.device, dtype=cls_score_dn.dtype)
                    one_hot[valid, dn_lab[valid]] = 1.0
                    w = torch.zeros_like(dn_lab, dtype=cls_score_dn.dtype, device=cls_score_dn.device)
                    w[valid] = 1.0
                    loss_dn_cls = self.loss_cls(cls_score_dn, one_hot, w).sum() / valid.sum().clamp(min=1)
                else:
                    loss_dn_cls = cls_score_dn.new_tensor(0.0)

                self._last_dn_loss = loss_dn_cls

                emb = torch.cat([pred_dn[h] for h in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1)  # (B,code, dn)
                emb = emb.permute(0, 2, 1).reshape(-1, emb.shape[1])  # (B*dn, code)
                self._last_con_loss = info_nce_loss(emb, dn_lab, tau=self.tau)

                tb['loss_dn_cls'] = float(loss_dn_cls.detach().cpu().item())
                tb['loss_con'] = float(self._last_con_loss.detach().cpu().item())

        # combine
        loss = loss_main
        if self._last_dn_loss is not None:
            w_dn = float(getattr(self.qcd_cfg, 'LOSS_WEIGHT_DN', 1.0))
            loss = loss + w_dn * self._last_dn_loss
        if self._last_con_loss is not None:
            w_con = float(getattr(self.qcd_cfg, 'LOSS_WEIGHT_CON', 0.1))
            loss = loss + w_con * self._last_con_loss

        tb['loss_total'] = float(loss.detach().cpu().item())
        return loss, tb
