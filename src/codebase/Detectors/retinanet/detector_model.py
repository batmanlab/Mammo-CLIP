#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:04:18 2019

@author: viswanatha
"""

import math
import torch
import torch.nn as nn
from torchvision.ops import nms as NMS

from Detectors.detectors_utils import BBoxTransform, ClipBoxes
from Detectors.retinanet import losses
from Detectors.retinanet.anchors import Anchors
from Detectors.retinanet.efficient_net import EfficientNet


def nms(dets, thresh):
    """[summary]

    Args:
        dets ([tensor]): [bounding boxes]
        thresh ([float]): [threshold for nms]

    Returns:
        [tensor]: [description]
    """
    boxes = dets[:, :4]
    scores = dets[:, -1]
    return NMS(boxes, scores, thresh)


class PyramidFeatures(nn.Module):
    """[summary]"""

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        """[summary]

        Args:
            C3_size ([int]): [description]
            C4_size ([int]): [description]
            C5_size ([int]): [description]
            feature_size (int, optional): [num channels
                                            for layer in FPN]. Defaults to 256.
        """
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_x_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_x_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P3_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=2, padding=1
        )

    def forward(self, inputs):
        """[summary]

        Args:
            inputs ([list]): [features from the backbone]

        Returns:
            [list]: [FPN features]
        """
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = self.P4_x_upsampled(P4_x)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = self.P3_x_upsampled(P3_x)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    """[summary]"""

    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        """[summary]

        Args:
            num_features_in ([int]): [input channels]
            num_anchors (int, optional): [description]. Defaults to 9.
            feature_size (int, optional): [description]. Defaults to 256.
        """
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        """[summary]

        Args:
            x ([tensor]): [feature map]

        Returns:
            [tensor]: [regression outputs]
        """
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    """[summary]"""

    def __init__(
            self, num_features_in, num_anchors=9, num_classes=80, feature_size=256
    ):
        """[summary]

        Args:
            num_features_in ([int]): [input channels]
            num_anchors (int, optional): [description]. Defaults to 9.
            num_classes (int, optional): [description]. Defaults to 80.
            feature_size (int, optional): [description]. Defaults to 256.
        """
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        """[summary]

        Args:
            x ([tensor]): [feature map]

        Returns:
            [tensor]: [classification logits]
        """
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        # batch_size, width, height, channels = out1.shape
        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class RetinaNet(nn.Module):
    """[summary]"""

    def __init__(
            self, num_classes, backbone_network, fpn_sizes, focal_alpha=0.25, focal_gamma=2.0, domain_classifier=False
    ):
        """[summary]

        Args:
            num_classes ([int]): [description]
            backbone_network ([str]): [description]
            fpn_sizes ([list]): [number of channels
                                    in each backbone feature map]
        """
        self.inplanes = 64
        super(RetinaNet, self).__init__()
        # fpn_sizes = [160, 272, 448]
        # fpn_sizes = [56, 160, 448]
        # for b4
        # fpn_sizes = [160, 272, 448]

        # for b0
        # fpn_sizes = [112,192,1280]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss(focal_alpha, focal_gamma)

        self.domain_classifier = domain_classifier

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(
            -math.log((1.0 - prior) / prior)
        )

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

        self.efficientnet = backbone_network

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        """[summary]"""
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        # final_out, C3, C4, C5 = self.efficientnet(img_batch)
        _, C3, C4, C5 = self.efficientnet(img_batch)
        features = self.fpn([C3, C4, C5])

        regression = torch.cat(
            [self.regressionModel(feature) for feature in features], dim=1
        )

        classification = torch.cat(
            [self.classificationModel(feature) for feature in features], dim=1
        )

        if self.domain_classifier:
            return classification.view(classification.size(0), -1), regression.view(regression.size(0), -1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        scores = torch.max(classification, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores > 0.05)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        anchors_nms_idx = nms(
            torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5
        )

        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

        return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


def RetinaNet_efficientnet(
        num_classes, model_type, focal_alpha=0.25, focal_gamma=2.0, domain_classifier=False, clip_chk_pt=None,
        freeze_backbone="n"
):
    """[summary]

    Args:
        num_classes ([int]): [description]
        model_type ([str]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [torch model]: [description]
    """

    if model_type == "clip_b5_upmc" or model_type == "clip_b5_upmc_vindr":
        efficientnet = EfficientNet.from_pretrained(model_name="efficientnet-b5", model_type=model_type,
                                                    num_classes=num_classes, clip_chk_pt=clip_chk_pt,
                                                    freeze_backbone=freeze_backbone)
        efficientnet.source_layer_indexes = [26, 37]
        model = RetinaNet(
            num_classes, efficientnet, [176, 512, 2048], focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            domain_classifier=domain_classifier
        )
    elif model_type == "clip_b2_upmc" or model_type == "clip_b2_upmc_vindr":
        efficientnet = EfficientNet.from_pretrained(model_name="efficientnet-b2", model_type=model_type,
                                                    num_classes=num_classes, clip_chk_pt=clip_chk_pt,
                                                    freeze_backbone=freeze_backbone)
        efficientnet.source_layer_indexes = [15, 21]
        model = RetinaNet(
            num_classes, efficientnet, [120, 352, 1408], focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            domain_classifier=domain_classifier
        )
    else:
        raise ValueError(
            "Unsupported model type, must be one of b0, b1, b2, b3, b4, b5"
        )

    return model
