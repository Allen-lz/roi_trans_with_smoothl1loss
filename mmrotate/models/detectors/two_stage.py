# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
import numpy as np
import cv2
import matplotlib.pylab as plt


class SalienceMapMse(torch.nn.Module):
    def __init__(self):
        super(SalienceMapMse, self).__init__()
        self.Dimensionality_Reduction = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        self.sg = torch.nn.Sigmoid()
        self.mseloss = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, x, gt):
        gt = gt.to(x.device)
        x = self.Dimensionality_Reduction(x)
        x = self.sg(x)
        x_h, x_w = x.shape[2:]
        the_gt = torch.nn.UpsamplingBilinear2d((x_h, x_w))(gt)

        att_loss = self.mseloss(input=x, target=the_gt)
        return att_loss

@ROTATED_DETECTORS.register_module()
class RotatedTwoStageDetector(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedTwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.attention_loss = SalienceMapMse()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 5).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def cxcywha2poly(self, bboxes):
        polygons = []
        for bbox in bboxes:
            bbox = np.array(bbox.detach().cpu())
            xc, yc, w, h, ag = bbox[:5]
            wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
            hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
            p1 = (xc - wx - hx, yc - wy - hy)
            p2 = (xc + wx - hx, yc + wy - hy)
            p3 = (xc + wx + hx, yc + wy + hy)
            p4 = (xc - wx + hx, yc - wy + hy)
            poly = np.array([p1, p2, p3, p4], dtype=np.int32)[np.newaxis, :, :]
            polygons.append(poly)
        polygons = np.concatenate(polygons, axis=0)
        return polygons

    def get_salience_map(self, img, gt_bboxes):
        salience_maps = []
        for i, bboxes in enumerate(gt_bboxes):
            _, img_h, img_w = img[i].shape
            polygons = self.cxcywha2poly(bboxes)
            salience_map = np.zeros([img_h, img_w], dtype=np.uint8)
            cv2.fillPoly(salience_map, polygons, 255)

            salience_maps.append(torch.FloatTensor(salience_map).unsqueeze(0).unsqueeze(0) / 255)

            # vis_img = np.array(img[i].detach().cpu().permute(1, 2, 0))
            # vis_img = np.array(img[i].detach().cpu().permute(1, 2, 0)) * np.array([58.395, 57.12, 57.375])[np.newaxis, np.newaxis, :] - np.array([123.675, 116.28, 103.53])[np.newaxis, np.newaxis, :]
            # vis_img = np.array(vis_img, dtype=np.uint8)
            # plt.subplot(121), plt.imshow(vis_img)
            # plt.subplot(122), plt.imshow(salience_map)
            # plt.show()
        salience_maps = torch.cat(salience_maps, dim=0)
        return salience_maps

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 这里有img和gt_bboxes, 我可以直接在这里绘制出来一个显著性检测的label map

        salience_maps = self.get_salience_map(img, gt_bboxes)
        x = self.extract_feat(img)
        losses = dict()
        # 对每个尺度的特征
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        att_loss = 0
        for feat_map in x:
            att_loss += self.attention_loss(feat_map, salience_maps) * 10
        att_loss = att_loss / salience_maps.shape[0]

        att_loss_dict = {"loss_backbone_att": att_loss}
        losses.update(att_loss_dict)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
