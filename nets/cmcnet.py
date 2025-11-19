# nets/cmcnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FeatureExtractor(nn.Module):
    """
    共享的特征提取网络 A，使用 VGG16 backbone + GAP，
    对应论文里 Fig.2 的 A 模块。
    """
    def __init__(self, pretrained=True, out_dim=512):
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        # 只用 feature 部分，不要原来的 classifier
        self.features = vgg.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)              # [B, 512, H', W']
        x = self.gap(x)                   # [B, 512, 1, 1]
        x = torch.flatten(x, 1)           # [B, 512]
        return x


class CMCNet(nn.Module):
    """
    Combined Matching and Classification Network (CMCNet)
    对应论文的 Fig.2:
      - A: 共享特征网络 (FeatureExtractor)
      - B: metric network 做 matching
      - C: 两个 single-view classification 头 (CC / MLO)
    """
    def __init__(
        self,
        num_classes: int = 2,
        feat_dim: int = 512,
        use_pretrained_backbone: bool = True,
        metric_hidden_dim: int = 512,
        use_distance_matching: bool = False,
    ):
        """
        Args:
            num_classes: 分类类别数，一般就是 mass / non-mass = 2
            feat_dim: 特征维度（VGG16 + GAP 默认是 512）
            use_pretrained_backbone: 是否用 ImageNet 预训练的 VGG16
            metric_hidden_dim: metric 网络中间层维度
            use_distance_matching:
                - False: 用 FC + softmax 输出 match logits（交叉熵）
                - True: 输出距离，用 contrastive loss
        """
        super(CMCNet, self).__init__()
        self.feature = FeatureExtractor(
            pretrained=use_pretrained_backbone,
            out_dim=feat_dim,
        )

        # CC / MLO 两个分类头（共享 backbone 后各自一个 FC）
        self.cls_cc = nn.Linear(feat_dim, num_classes)
        self.cls_mlo = nn.Linear(feat_dim, num_classes)

        self.use_distance_matching = use_distance_matching

        if not use_distance_matching:
            # Metric network (B)：拼接 f1,f2 后做 2 分类 (match / non-match)
            self.metric = nn.Sequential(
                nn.Linear(feat_dim * 2, metric_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(metric_hidden_dim, 2),  # logits
            )
        else:
            # 用特征距离做 matching，不需要 FC
            self.metric = None

    def forward(self, img_cc, img_mlo):
        """
        Args:
            img_cc: [B, 3, H, W] CC view patch
            img_mlo: [B, 3, H, W] MLO view patch
        Returns:
            match_output:
                - if use_distance_matching == False: [B, 2] logits
                - if use_distance_matching == True:  [B] distances
            cls_cc_logits: [B, num_classes]
            cls_mlo_logits: [B, num_classes]
            f_cc, f_mlo: [B, feat_dim] (方便需要时取特征)
        """
        f_cc = self.feature(img_cc)
        f_mlo = self.feature(img_mlo)

        # classification heads
        cls_cc_logits = self.cls_cc(f_cc)
        cls_mlo_logits = self.cls_mlo(f_mlo)

        if self.use_distance_matching:
            # Euclidean distance, 用于 contrastive loss
            diff = f_cc - f_mlo
            dist = torch.sqrt(torch.sum(diff * diff, dim=1) + 1e-8)
            match_output = dist
        else:
            # FC metric network + softmax (交叉熵)
            pair_feat = torch.cat([f_cc, f_mlo], dim=1)
            match_output = self.metric(pair_feat)

        return match_output, cls_cc_logits, cls_mlo_logits, f_cc, f_mlo


# 一个简单的 contrastive loss（如果你选用距离方式）
class ContrastiveLoss(nn.Module):
    """
    L = 1/2 * [ y * d^2 + (1-y) * max(margin - d, 0)^2 ]
    y = 1 表示匹配，y = 0 表示不匹配
    """
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances, labels):
        """
        Args:
            distances: [B], 由 CMCNet(use_distance_matching=True) 输出
            labels:    [B], 0 或 1
        """
        positive = labels * distances.pow(2)
        negative = (1 - labels) * torch.clamp(self.margin - distances, min=0.0).pow(2)
        loss = 0.5 * (positive + negative)
        return loss.mean()
