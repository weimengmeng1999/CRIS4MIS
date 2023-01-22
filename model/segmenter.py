import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model

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
        vis = self.backbone.encode_image(img)
        word, state = self.backbone.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
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
                    (b, -1)) > 0.35).to(torch.float)
                mask_t = (mask.detach().reshape((b, -1))).to(torch.float)
                mask_iou_label = (pred_t * mask_t).sum(-1) / (
                    (pred_t + mask_t) > 0).sum(-1)
                mask_iou_loss = self.mask_iou_loss(mask_iou_pred,
                                                   mask_iou_label)
                loss = loss + mask_iou_loss * self.mask_iou_loss_weight

            results['loss'] = loss

        return results
