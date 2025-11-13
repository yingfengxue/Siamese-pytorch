import torch
import torch.nn as nn
from nets.vgg import VGG16  # 假设 VGG16 在这里

class CMCNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        """
        初始化 CMCNet (Combined Matching and Classification Network).
        
        参数:
        num_classes (int): 您的 positive 类别数 + 1 个 "背景" 类。
                           (在您的情况下, 应该是 4).
        pretrained (bool): 是否使用预训练的VGG权重。
        """
        super(CMCNet, self).__init__()
        
        # --- A: 特征网络 (Feature Network) ---
        # 加载VGG16骨干网络
        vgg = VGG16(pretrained, 3)
        
        # 我们只需要 'features' 部分 (共享权重)
        self.features = vgg.features
        
        # 根据论文，我们使用全局平均池化 (GAP) [cite: 149]
        # 这会把 [B, 512, H, W] 的特征图变成 [B, 512, 1, 1]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # VGG的最后一个 'features' 块输出 512 个通道
        feature_dim = 512
        
        # --- C: 分类头 (Classification Head) ---
        # 这是一个共享的FC层，用于两个分支的分类任务
        # 输出 4 (3个 positive 类 + 1个 "背景" 类)
        self.classifier_dropout = nn.Dropout(p=0.5)
        self.classifier_head = nn.Linear(feature_dim, num_classes)
        
        # --- B: 度量网络 (Metric Network Head) ---
        # 这个头接收 *串联* 的特征 [f_cc, f_mlo]
        # 串联后的维度是 512 + 512 = 1024
        # 输出 2 (代表 "不匹配" 或 "匹配")
        self.metric_dropout = nn.Dropout(p=0.5)
        self.metric_head = nn.Linear(feature_dim * 2, 2)

    def forward_once(self, x):
        """
        定义一个辅助函数，用于通过共享的骨干网络运行单个输入。
        """
        # x: [B, 3, H, W]
        x = self.features(x)   # -> [B, 512, H_out, W_out]
        x = self.gap(x)        # -> [B, 512, 1, 1]
        x = torch.flatten(x, 1) # -> [B, 512] (展平)
        return x

    def forward(self, x_cc, x_mlo):
        """
        模型的前向传播。
        接收两个输入 (cc_patch, mlo_patch)，
        返回三个输出 (cls_cc, cls_mlo, match)。
        """
        
        # --- A: 特征网络 (共享权重) ---
        # 通过共享的骨干网络分别传递CC和MLO
        f_cc = self.forward_once(x_cc)   # CC 特征: [B, 512]
        f_mlo = self.forward_once(x_mlo) # MLO 特征: [B, 512]

        # --- C: 分类输出 ---
        # 两个分支使用 *共享* 的分类头
        # out_cls_cc = self.classifier_head(f_cc)   # -> [B, 4]
        # out_cls_mlo = self.classifier_head(f_mlo) # -> [B, 4]
        out_cls_cc = self.classifier_head(self.classifier_dropout(f_cc))
        out_cls_mlo = self.classifier_head(self.classifier_dropout(f_mlo))
        
        # --- B: 匹配输出 ---
        # 串联两个特征向量
        f_concat = torch.cat((f_cc, f_mlo), dim=1) # -> [B, 1024]
        
        # 通过度量网络头
        #out_match = self.metric_head(f_concat)    # -> [B, 2]
        out_match = self.metric_head(self.metric_dropout(f_concat))
        
        # --- 返回三个输出 ---
        # 我们还返回原始特征，以防您想在 train.py 中使用对比损失 (Contrastive Loss)
        return out_cls_cc, out_cls_mlo, out_match, f_cc, f_mlo

# --- (旧代码，我们不再需要) ---
# del Siamese
# del get_img_output_length
