import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from model.mae import MaskedAutoencoderViT

from .layers import FPN, Projector, TransformerDecoder, MaskIoUProjector


class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(),
                                    cfg.word_len).float()
        # Multi-Modal FPN
        self.neck_with_text_state = cfg.neck_with_text_state
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

        # Mask IoU Projector
        self.pred_mask_iou = cfg.pred_mask_iou
        self.mask_iou_loss_type = cfg.mask_iou_loss_type
        self.mask_iou_loss_weight = cfg.mask_iou_loss_weight
        if self.pred_mask_iou:
            self.mask_iou_proj = MaskIoUProjector(cfg.word_dim, cfg.vis_dim,
                                                  cfg.vis_dim)
            if self.mask_iou_loss_type.lower() == 'mse':
                self.mask_iou_loss = nn.MSELoss()
            elif self.mask_iou_loss_type.lower() == 'bce':
                self.mask_iou_loss = nn.BCEWithLogitsLoss()
            else:
                assert False, 'Not support mask_iou_loss_type: {}'.format(
                    self.mask_iou_loss_type)

        # MoE
        self.use_moe_select_best_sent = cfg.use_moe_select_best_sent
        self.max_sent_num = cfg.max_sent_num
        self.use_moe_consistency_loss = cfg.use_moe_consistency_loss
        self.moe_consistency_loss_weight = cfg.moe_consistency_loss_weight
        if self.use_moe_select_best_sent:
            self.sent_selector = MaskIoUProjector(cfg.word_dim, cfg.vis_dim,
                                                  cfg.vis_dim)
            if self.use_moe_consistency_loss:
                self.moe_consistency_loss = nn.MSELoss()

        # MAE
        self.use_mae_gen_target_area = cfg.use_mae_gen_target_area
        self.mae_pretrain = cfg.mae_pretrain
        self.reconstruct_full_img = cfg.reconstruct_full_img
        self.mae_hard_example_mining_type = cfg.mae_hard_example_mining_type
        if self.use_mae_gen_target_area:
            self.mae = MaskedAutoencoderViT(patch_size=16,
                                            embed_dim=768,
                                            depth=12,
                                            num_heads=12,
                                            decoder_embed_dim=512,
                                            decoder_depth=8,
                                            decoder_num_heads=16,
                                            mlp_ratio=4)
            if self.mae_pretrain is not None and self.mae_pretrain != '':
                mae_state_dict = self.mae.state_dict()
                pre_state_dict = torch.load(self.mae_pretrain,
                                            map_location="cpu")
                mae_state_dict.update(pre_state_dict['model'])
                self.mae.load_state_dict(mae_state_dict)
                print('load MAE pretrain model from {} successfully.'.format(
                    self.mae_pretrain))

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        batch_size, _, img_h, img_w = img.shape
        vis = self.backbone.encode_image(img)

        if self.use_moe_select_best_sent:
            pred_all = img.new_zeros(
                (batch_size, self.max_sent_num, img_h // 4, img_w // 4))
            score_all = img.new_zeros((batch_size, self.max_sent_num))
            for i_sent in range(self.max_sent_num):
                f_word, state = self.backbone.encode_text(word[:, i_sent, :])
                # b, 512, 26, 26 (C4)
                if self.neck_with_text_state:
                    fq = self.neck(vis, state)
                else:
                    fq = self.neck(vis)
                b, c, h, w = fq.size()
                fq = self.decoder(fq, f_word, pad_mask[:, i_sent, :])
                fq = fq.reshape(b, c, h, w)
                # b, 1, 104, 104
                pred = self.proj(fq, state)
                score = self.sent_selector(fq, state)
                pred_all[:, i_sent:i_sent + 1] = pred
                score_all[:, i_sent] = score
            best_idx = torch.argmax(score_all, dim=1)  # b, 7
            best_idx_oh = F.one_hot(best_idx, num_classes=self.max_sent_num)
            pred_mask = torch.ones(
                (batch_size, self.max_sent_num, img_h // 4, img_w // 4),
                device=best_idx.device) * best_idx_oh[:, :, None, None]
            pred = torch.masked_select(pred_all, pred_mask.bool()).reshape(
                (batch_size, 1, img_h // 4, img_w // 4))
            if self.training and self.use_moe_consistency_loss:
                pred_all = pred_all.sigmoid()
                moe_consistency_loss = img.new_zeros(
                    (self.max_sent_num, self.max_sent_num))
                for c_i in range(self.max_sent_num):
                    for c_j in range(self.max_sent_num):
                        if c_i == c_j:
                            continue
                        moe_consistency_loss[c_i,
                                             c_j] = self.moe_consistency_loss(
                                                 pred_all[:, c_i],
                                                 pred_all[:, c_j])
        else:
            word, state = self.backbone.encode_text(word)
            # b, 512, 26, 26 (C4)
            if self.neck_with_text_state:
                fq = self.neck(vis, state)
            else:
                fq = self.neck(vis)
            b, c, h, w = fq.size()
            fq = self.decoder(fq, word, pad_mask)
            fq = fq.reshape(b, c, h, w)
            # b, 1, 104, 104
            pred = self.proj(fq, state)

        results = dict()
        results['pred'] = pred.detach()
        if self.pred_mask_iou:
            # b,
            mask_iou_pred = self.mask_iou_proj(fq, state)
            results['mask_iou_pred'] = mask_iou_pred

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            results['target'] = mask
            loss = F.binary_cross_entropy_with_logits(pred, mask)

            if self.pred_mask_iou:
                # threshold = 0.35, same as test
                pred_t = (pred.detach().reshape(
                    (b, -1)).sigmoid() > 0.35).to(torch.float)
                mask_t = (mask.detach().reshape((b, -1))).to(torch.float)
                mask_iou_label = (pred_t * mask_t).sum(-1) / (
                    (pred_t + mask_t) > 0).sum(-1)
                mask_iou_loss = self.mask_iou_loss(mask_iou_pred,
                                                   mask_iou_label)
                loss = loss + mask_iou_loss * self.mask_iou_loss_weight

            if self.use_moe_select_best_sent and self.use_moe_consistency_loss:
                loss = loss + moe_consistency_loss.mean(
                ) * self.moe_consistency_loss_weight

            if self.use_mae_gen_target_area:
                # (224, 224) same as original MAE
                mae_img = F.interpolate(img, (224, 224),
                                        mode='bilinear').detach()
                if not self.reconstruct_full_img:
                    mae_mask = F.interpolate(mask, (224, 224),
                                             mode='nearest').detach()
                    mae_img = mae_img * mae_mask
                if self.mae_hard_example_mining_type is not None:
                    pred_t = (pred.detach().sigmoid() > 0.35).to(torch.int32)
                    mask_t = mask.detach().to(torch.int32)
                    if self.mae_hard_example_mining_type == 'v0':
                        mae_hard_example = torch.logical_xor(pred_t, mask_t)
                    elif self.mae_hard_example_mining_type == 'v1':
                        pred_t = torch.where(mask < 0.5, 0, pred_t)
                        mae_hard_example = torch.logical_xor(pred_t, mask_t)
                    mae_hard_example = mae_hard_example.to(torch.float32)
                    mae_hard_example = F.interpolate(mae_hard_example,
                                                     (224, 224),
                                                     mode='nearest').detach()
                else:
                    mae_hard_example = None
                mae_loss, mae_pred, mae_mask = self.mae(
                    mae_img, hard_example=mae_hard_example)
                loss = loss + mae_loss

            results['loss'] = loss

        return results
