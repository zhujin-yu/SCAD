# scene_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
import logging
import os

logger = logging.getLogger('base')


class SceneClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, use_ucmerced_pretrain=False):
        super().__init__()
        self.num_classes = num_classes
        self.use_ucmerced_pretrain = use_ucmerced_pretrain

        # 使用预训练的MobileNetV3-Large
        backbone = mobilenet_v3_large(pretrained=pretrained)

        # MobileNetV3的特征提取部分
        self.features = backbone.features

        # MobileNetV3的全局池化和分类器部分
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # MobileNetV3-Large的最后一层特征维度是960
        if use_ucmerced_pretrain:
            # UCMerced预训练版本的结构
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(960, 512),
                nn.Hardswish(inplace=True),  # MobileNetV3使用Hardswish
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        else:
            # 标准结构
            self.classifier = nn.Sequential(
                nn.Linear(960, 512),
                nn.Hardswish(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

        # 场景特征向量输出
        self.feature_extractor = nn.Sequential(
            nn.Linear(960, 256),
            nn.Hardswish(inplace=True),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        # 特征提取
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        # 分类输出
        class_logits = self.classifier(features)

        # 场景特征
        scene_features = self.feature_extractor(features)

        return class_logits, scene_features


class SceneAdaptiveModule:
    def __init__(self, device='cuda', use_ucmerced_pretrain=True, ucmerced_checkpoint_path=None):
        self.device = device
        self.use_ucmerced_pretrain = use_ucmerced_pretrain

        self.classifier = SceneClassifier(
            num_classes=4,
            pretrained=True,
            use_ucmerced_pretrain=use_ucmerced_pretrain
        ).to(device)
        self.classifier.eval()

        # 场景类型映射
        self.scene_types = ['urban', 'farmland', 'mountain', 'water']

        # 场景对应的采样步数配置
        self.scene_timesteps = {
            'urban': 20,  # 城市场景：复杂细节，需要更多步数
            'farmland': 16,  # 农田场景：中等复杂度
            'mountain': 14,  # 山区场景：相对简单
            'water': 10  # 水体场景：最简单
        }

        # 场景对应的β调度参数
        self.scene_cosine_scales = {
            'urban': 1.8,     # 增强细节保留
            'farmland': 1.6,  # 中等细节
            'mountain': 1.4,  # 基础细节
            'water': 1.2      # 避免过度采样
        }

        # 场景对应的注意力权重配置
        self.scene_attention_weights = {
            'urban': {'clam_weight': 0.3, 'slam_weight': 0.7},  # 侧重空间细节
            'farmland': {'clam_weight': 0.6, 'slam_weight': 0.4},  # 侧重光谱区分
            'mountain': {'clam_weight': 0.4, 'slam_weight': 0.6},  # 平衡空间和光谱
            'water': {'clam_weight': 0.5, 'slam_weight': 0.5}  # 均等权重
        }

        # 加载UCMerced预训练权重
        if use_ucmerced_pretrain and ucmerced_checkpoint_path:
            self.load_ucmerced_pretrained(ucmerced_checkpoint_path)

        logger.info("Scene Adaptive Module with MobileNetV3-Large initialized")

    def load_ucmerced_pretrained(self, checkpoint_path):
        """加载UCMerced 21分类预训练权重并适配到4分类"""
        try:
            if not os.path.exists(checkpoint_path):
                logger.warning(f"❌ UCMerced checkpoint not found: {checkpoint_path}")
                return False

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 获取当前模型的状态字典
            current_state_dict = self.classifier.state_dict()

            # 过滤不匹配的键
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in current_state_dict:
                    if v.shape == current_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        # 形状不匹配的层（主要是分类头），进行智能初始化
                        if 'classifier' in k or 'feature_extractor' in k:
                            if 'weight' in k:
                                # 分类头权重初始化
                                nn.init.xavier_uniform_(current_state_dict[k])
                            elif 'bias' in k:
                                nn.init.zeros_(current_state_dict[k])
                        else:
                            filtered_state_dict[k] = v
                else:
                    logger.debug(f"Skipping key: {k}")

            # 加载过滤后的状态字典
            self.classifier.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"✅ Successfully loaded UCMerced MobileNetV3 pretrained weights from {checkpoint_path}")
            return True

        except Exception as e:
            logger.warning(f"❌ Failed to load UCMerced pretrained weights: {e}")
            return False

    def classify_scene(self, lr_image):
        """对LR图像进行场景分类"""
        with torch.no_grad():
            # 确保输入在[0,1]范围内
            if lr_image.max() > 1.0:
                lr_image = lr_image / 255.0

            class_logits, scene_features = self.classifier(lr_image)
            scene_probs = F.softmax(class_logits, dim=1)
            scene_preds = torch.argmax(scene_probs, dim=1)

            # 记录分类结果
            for i, pred in enumerate(scene_preds):
                scene_type = self.scene_types[pred.item()]
                confidence = scene_probs[i][pred].item()
                logger.info(f"Sample {i}: Scene={scene_type}, Confidence={confidence:.3f}")

        return scene_preds, scene_probs, scene_features

    def get_adaptive_config(self, scene_preds):
        """根据场景预测获取自适应配置"""
        batch_size = scene_preds.shape[0]
        timesteps = []
        cosine_scales = []
        attention_configs = []

        for i in range(batch_size):
            scene_idx = scene_preds[i].item()
            scene_type = self.scene_types[scene_idx]

            timesteps.append(self.scene_timesteps[scene_type])
            cosine_scales.append(self.scene_cosine_scales[scene_type])
            attention_configs.append(self.scene_attention_weights[scene_type])

        return timesteps, cosine_scales, attention_configs