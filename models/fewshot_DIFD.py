"""
DIFD for FSS
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.encoder import Res101Encoder
import numpy as np
from models.Decoders import Decoder
import random
from boundary_loss import BoundaryLoss


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
        self.criterion = nn.NLLLoss(ignore_index=255, weight=self.my_weight)
        self.criterion_b = BoundaryLoss(theta0=3, theta=5)
        self.rate = 0.95
        self.kernel = (16, 8)
        self.stride = 8
        self.num = 100
        self.fuse = torch.nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)  # 1
        self.n_shots = len(supp_imgs[0])  # 1
        self.n_queries = len(qry_imgs)  # 1
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]    # 1
        supp_bs = supp_imgs[0][0].shape[0]      # 1
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)

        # encoder output
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]


        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]


        ##### Get threshold #######
        self.thresh_pred = tao

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            supp_fts_ = [[F.interpolate(supp_fts[0][[epi], way, shot], size=img_size, mode='bilinear', align_corners=True)
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]
            prototype = [[self.get_masked_fts(supp_fts_[way][shot], supp_mask[[epi], way, shot])
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]  # (way, shot, (1, 512))
            prototype = self.getPrototype(prototype)

            bg_prototype = [[self.get_masked_fts(supp_fts_[way][shot], 1-supp_mask[[epi], way, shot])
                             for shot in range(self.n_shots)] for way in range(self.n_ways)]  # (way, shot, (1, 512))
            bg_prototype = self.getPrototype(bg_prototype)  # (1, (1, 512))

            if supp_mask[epi][0].sum() == 0:
                region_protos = [[self.CI(supp_fts_[way][shot], supp_mask[[epi], way, shot], self.kernel, self.stride, flag=0)
                                  for shot in range(self.n_shots)] for way in range(self.n_ways)]  # (way, shot, 2, (n_pts, 512))
            else:
                region_protos = [[self.CI(supp_fts_[way][shot], supp_mask[[epi], way, shot], self.kernel, self.stride, flag=1)
                                  for shot in range(self.n_shots)] for way in range(self.n_ways)]  # (way, shot, 2, (n_pts, 512))

            bg_pts = [torch.cat([region_protos[way][0][0], bg_prototype[way]], dim=0) for way in range(self.n_ways)]
            fg_pts = [torch.cat([region_protos[way][0][1], prototype[way]], dim=0) for way in range(self.n_ways)]

            # get fg_mathched
            matched_fts = torch.stack(
                [self.IFR_FD(fg_pts[way], qry_fts[0][epi], supp_fts[0][[epi], way, 0],
                                supp_mask[[epi], way, 0]) for way in range(self.n_ways)], dim=1)
            # get fg_preds
            fg_preds = torch.cat([self.decoder1(matched_fts[:, way]) for way in range(self.n_ways)], dim=1)

            ### get bg_mathched
            bg_matched_fts = torch.stack(
                [self.IFR_FD(bg_pts[way], qry_fts[0][epi], supp_fts[0][[epi], way, 0],
                                1-supp_mask[[epi], way, 0]) for way in range(self.n_ways)], dim=1)
            ### get bg_preds
            bg_preds = torch.cat([self.decoder2(bg_matched_fts[:, way]) for way in range(self.n_ways)], dim=1)

            fg_preds = F.interpolate(fg_preds, size=img_size, mode='bilinear', align_corners=True)
            bg_preds = F.interpolate(bg_preds, size=img_size, mode='bilinear', align_corners=True)

            preds = torch.cat([bg_preds, fg_preds], dim=1)
            preds = torch.softmax(preds, dim=1)


            outputs.append(preds)
            ''' Prototype alignment loss '''
            if train:
                align_loss_epi, b_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                 [qry_fts[n][epi] for n in range(len(qry_fts))], preds, supp_mask[epi])
                align_loss += align_loss_epi
                b_loss += b_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs, b_loss / supp_bs


    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (1, 512), (1, 1)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))   # ([1, 64, 64])

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)   # align_corners=True 元素值不会超出原上下界
        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def get_masked_fts(self, fts, mask):

        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C] (1, 1, (1, 512))
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts   (n_way, (1, 512))

        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        """
            supp_fts: (2, [1, 512, 64, 64])
            qry_fts: (2, (1, 512, 64, 64))
            pred: [1, 2, 256, 256]
            fore_mask: [Way, Shot , 256, 256]
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):

                qry_fts_ = F.interpolate(qry_fts[0], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                qry_prototype = [[self.get_masked_fts(qry_fts_, pred_mask[way + 1])]]
                qry_prototype = self.getPrototype(qry_prototype)

                qry_bg_prototype = [[self.get_masked_fts(qry_fts_, 1-pred_mask[way + 1])]]
                qry_bg_prototype = self.getPrototype(qry_bg_prototype)

                if pred_mask[way + 1].sum() == 0:
                    region_qry_protos = [self.CI(qry_fts_, pred_mask[way + 1], self.kernel, self.stride, flag=0)]
                else:
                    region_qry_protos = [self.CI(qry_fts_, pred_mask[way + 1], self.kernel, self.stride, flag=1)]

                qry_bg_pts = [torch.cat([region_qry_protos[0][0], qry_bg_prototype[0]], dim=0)]
                qry_fg_pts = [torch.cat([region_qry_protos[0][1], qry_prototype[0]], dim=0)]

                # get fg_mathched
                matched_fts = torch.stack([self.IFR_FD(qry_fg_pts[0], supp_fts[0][way, [shot]], qry_fts[0],
                                                          pred_mask[way + 1])], dim=1)
                # get fg_preds
                fg_preds = torch.cat([self.decoder1(matched_fts[:, 0])], dim=1)

                ### get bg_mathched
                bg_matched_fts = torch.stack([self.IFR_FD(qry_bg_pts[0], supp_fts[0][way, [shot]], qry_fts[0],
                                                             1-pred_mask[way + 1])], dim=1)
                ### get bg_preds
                bg_preds = torch.cat([self.decoder2(bg_matched_fts[:, 0])], dim=1)

                fg_preds = F.interpolate(fg_preds, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                bg_preds = F.interpolate(bg_preds, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                pred_ups = torch.cat([bg_preds, fg_preds], dim=1)
                pred_ups = torch.softmax(pred_ups, dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways
                b_loss += self.criterion_b(pred_ups, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, b_loss

    def get_pts(self, fts, mask):
        features_trans = fts.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1
        features_trans = features_trans[indx]  # (n_fg x 512)
        if len(features_trans) > 100:
            random.seed(9)
            features_trans = features_trans[random.sample(range(len(features_trans)), 100)]
        return features_trans

    def get_way_pts(self, fg_fts):
        """
            fg_fts: lists of list of tensor
                        expect shape: Wa x Sh x [all x C]
            fg_prototypes: [(all, 512) * way]    list of tensor
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        # fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
        #                  fg_fts]  ## concat all fg_fts
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]

        return prototypes

    def get_prior(self, fts, prototype):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 32, 32) (1, 512)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        prior = torch.sigmoid(F.cosine_similarity(fts, prototype[..., None, None], dim=1))

        return prior     # ([1, 32, 32])


    def CI(self, fts, mask, kernel_size=(16, 16), stride=8, flag=1, aspect_ratio_range=(0.5, 2)):
        # SI
        b, c, height, width = fts.size()
        kernel_h, kernel_w = kernel_size
        n_fg = mask.sum()
        n_bg = height * width - n_fg
        fg_pts = []
        bg_pts = []
        fg_pts.append(torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (n_fg + 1e-5))  # 1 x C
        bg_pts.append(torch.sum(fts * (1 - mask[None, ...]), dim=(-2, -1)) / (n_bg + 1e-5))  # 1 x C
        patch_fts = F.unfold(fts, kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1, c, kernel_h, kernel_w)
        patch_mask = F.unfold(mask.unsqueeze(1), kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1, kernel_h, kernel_w).unsqueeze(2)
        bg_patch_mask = 1 - patch_mask
        random.seed(9)
        if flag == 1:
            fg_indx = torch.sum(patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h*kernel_w*self.rate, n_fg/150)
            fg_protos = torch.sum(patch_fts[:, fg_indx, :, :, :] * patch_mask[:, fg_indx, :, :, :], dim=(-2, -1)) \
                        / (patch_mask[:, fg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)   # (1, n, 512)
            if fg_protos.numel() == 0:
                fg_protos = self.get_pts(fts, mask)
            else:
                fg_protos = fg_protos.squeeze(0)
                if len(fg_protos) > 150:
                    fg_protos = fg_protos[random.sample(range(len(fg_protos)), 150)]
            fg_pts.append(fg_protos)

        bg_indx = torch.sum(bg_patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h*kernel_w, n_bg / 250)
        bg_protos = torch.sum(patch_fts[:, bg_indx, :, :, :] * bg_patch_mask[:, bg_indx, :, :, :], dim=(-2, -1)) \
                    / (bg_patch_mask[:, bg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)
        if bg_protos.numel() == 0:
            bg_protos = self.get_pts(fts, 1-mask)
        else:
            bg_protos = bg_protos.squeeze(0)
            if len(bg_protos) > 250:
                bg_protos = bg_protos[random.sample(range(len(bg_protos)), 250)]
        bg_pts.append(bg_protos)

        # RI
        min_aspect_ratio, max_aspect_ratio = aspect_ratio_range
        assert min_aspect_ratio > 0
        assert max_aspect_ratio > min_aspect_ratio
        torch.manual_seed(9)
        aspect_ratio = torch.rand(self.num).to(self.device) * (max_aspect_ratio - min_aspect_ratio) + min_aspect_ratio
        aspect_ratio0 = aspect_ratio[aspect_ratio <= 1]
        bbox_height0 = ((torch.rand(len(aspect_ratio0)).to(self.device) * 0.9375 + 0.0625) * (height * aspect_ratio0)).int()
        bbox_width0 = (bbox_height0 / aspect_ratio0).int()

        aspect_ratio1 = aspect_ratio[aspect_ratio > 1]
        bbox_width1 = ((torch.rand(len(aspect_ratio1)).to(self.device) * 0.9375 + 0.0625) * (width / aspect_ratio1)).int()
        bbox_height1 = (bbox_width1 * aspect_ratio1).int()

        bbox_height = torch.cat([bbox_height0, bbox_height1], dim=0)
        bbox_width = torch.cat([bbox_width0, bbox_width1], dim=0)

        x1 = (torch.rand(self.num).to(self.device) * (height - bbox_height)).int()
        y1 = (torch.rand(self.num).to(self.device) * (width - bbox_width)).int()
        x2 = x1 + bbox_height
        y2 = y1 + bbox_width
        for i in range(self.num):
            n1 = mask[:, x1[i]:x2[i], y1[i]:y2[i]].sum()
            n0 = (1 - mask[:, x1[i]:x2[i], y1[i]:y2[i]]).sum()
            if n1 >= min(bbox_height[i]*bbox_width[i]*self.rate, n_fg / self.num) and n1 < n_fg:
                prototype = self.get_masked_fts(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]], mask[:, x1[i]:x2[i], y1[i]:y2[i]])
                fg_pts.append(prototype)
            if n0 >= min(bbox_height[i]*bbox_width[i], n_bg / self.num) and n0 < n_bg:
                prototype = self.get_masked_fts(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]], 1 - mask[:, x1[i]:x2[i], y1[i]:y2[i]])
                bg_pts.append(prototype)


        fg_pts = torch.cat(fg_pts, dim=0)
        bg_pts = torch.cat(bg_pts, dim=0)

        return [bg_pts, fg_pts]

    def IFR_FD(self, pts, qry_fts, sup_fts, mask):
        """
        Args:
            pts: expect shape: (n+1) x C
            qry_fts: (N_q, 512, 64, 64)
            qry_fts: (1, 512, 64, 64)
            mask: (1, 256, 256)
        """
        n, c, h, w = qry_fts.shape
        prototype = pts[-1].unsqueeze(0)
        pts = pts[:-1]
        pts_ = F.normalize(pts, dim=-1)
        prototype = F.normalize(prototype, dim=-1)

        res = []
        for i in range(n):
            # IFR
            qry_ft = qry_fts[i].unsqueeze(0)
            fts_ = qry_ft.permute(0, 2, 3, 1)
            fts_ = F.normalize(fts_, dim=-1)
            one_sim = torch.matmul(fts_, prototype.transpose(0, 1)).permute(0, 3, 1, 2)
            sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(3, 0, 1, 2)  # [1, 64, 64, 100] --> [100, 1, 64, 64]
            pseudo_mask = torch.sigmoid(sim*(1+self.scaler*torch.sigmoid(self.thresh_pred[i+1])))
            qry_update = torch.sum(qry_ft * pseudo_mask, dim=(-2, -1)) / (pseudo_mask.sum(dim=(-2, -1)) + 1e-5)
            G = self.get_fuse_factor(fts_, sup_fts, mask)
            qry_update = torch.stack([pts*(1-G), qry_update*G], dim=0)
            pts = pts + self.fuse(qry_update.unsqueeze(0)).squeeze(0).squeeze(0)

            # FD
            pts_norm = F.normalize(pts, dim=-1)
            sim = torch.matmul(fts_, pts_norm.transpose(0, 1)).permute(3, 0, 1, 2)
            sim_map = torch.softmax(sim*(1+self.scaler*torch.sigmoid(self.thresh_pred[0])), dim=0)
            sim_0 = torch.sum(sim*sim_map, dim=0).unsqueeze(0)
            sim_1 = torch.sum(sim, dim=0).unsqueeze(0)
            matched_pts = pts.unsqueeze(2).unsqueeze(3) * sim_map
            matched_pts = torch.sum(matched_pts, dim=0).unsqueeze(0)  # [1, c, 64, 64]
            matched_fts = torch.cat([qry_ft, matched_pts, sim_0], dim=1)  # 1 x (2c+1) x h x w
            res.append(matched_fts)

        res = torch.cat(res, dim=0)

        return res

    def get_fuse_factor(self, qry_fts, sup_fts, mask):
        """
        Args:
            qry_fts: (1, 64, 64, 512)
            sup_fts: (1, 512, 64, 64)
            mask:   (1, 256, 256)
        """
        mask_ = F.interpolate(mask.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear').squeeze(0)
        if mask_.sum() == 0:
            G = torch.tensor(0.0).to(self.device)
        else:
            sup_fts_ = F.normalize(sup_fts.permute(0, 2, 3, 1).view(-1, 512), dim=-1)
            G = torch.matmul(sup_fts_, qry_fts.view(-1, 512).transpose(0, 1))
            G = G[mask_.view(-1) > 0]
            G = torch.sigmoid(torch.mean(G, dim=(-2, -1)))

        return G





