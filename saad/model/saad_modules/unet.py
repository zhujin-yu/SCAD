import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from torchvision.models import vgg19
from torch.autograd import Variable
from math import exp
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # self.up = nn.Upsample(scale_factor=2, mode="bicubic")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            # Mish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        # 使用噪声自适应块替换原来的Block
        self.block1 = NoiseAdaptiveBlock(dim, dim_out, groups=norm_groups)
        self.block2 = NoiseAdaptiveBlock(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, noise_intensity=None):
        b, c, h, w = x.shape
        h = self.block1(x, noise_intensity)  # 传入噪声强度
        h = self.noise_func(h, time_emb)
        h = self.block2(h, noise_intensity)  # 传入噪声强度
        return h + self.res_conv(x)


##########################  场景感知的通道注意力和空间注意力  ############################
class SceneAwareCLAM(nn.Module):
    """场景感知的通道注意力"""

    def __init__(self, in_planes, ratio=16, pool_mode='Avg|Max'):
        super(SceneAwareCLAM, self).__init__()
        self.pool_mode = pool_mode
        if pool_mode.find('Avg') != -1:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool_mode.find('Max') != -1:
            self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # 场景特征映射器 - 修复维度匹配问题
        self.scene_mapper = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, in_planes),  # 直接映射到输出通道数
            nn.Sigmoid()
        )

    def forward(self, x, scene_features=None):
        if self.pool_mode == 'Avg':
            out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        elif self.pool_mode == 'Max':
            out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        elif self.pool_mode == 'Avg|Max':
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out

        # 场景自适应调整
        if scene_features is not None:
            # 确保场景特征有正确的形状 [batch_size, 128]
            if scene_features.dim() == 1:
                scene_features = scene_features.unsqueeze(0)

            # 映射到与输出相同的通道数
            scene_weights = self.scene_mapper(scene_features)

            # 调整形状以匹配注意力输出 [batch_size, channels, 1, 1]
            batch_size, channels = out.shape[0], out.shape[1]
            scene_weights = scene_weights.view(batch_size, channels, 1, 1)

            out = out * scene_weights

        out = self.sigmoid(out) * x
        return out


class SceneAwareSLAM(nn.Module):
    """场景感知的空间注意力"""

    def __init__(self, kernel_size=7, pool_mode='Avg|Max'):
        super(SceneAwareSLAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.pool_mode = pool_mode
        if pool_mode == 'Avg|Max':
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 场景特征映射器
        self.scene_mapper = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, scene_features=None):
        if self.pool_mode == 'Avg':
            out = torch.mean(x, dim=1, keepdim=True)
        elif self.pool_mode == 'Max':
            out, _ = torch.max(x, dim=1, keepdim=True)
        elif self.pool_mode == 'Avg|Max':
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)

        out = self.sigmoid(self.conv1(out))

        # 场景自适应调整
        if scene_features is not None:
            # 确保场景特征有正确的形状
            if scene_features.dim() == 1:
                scene_features = scene_features.unsqueeze(0)

            scene_weights = self.scene_mapper(scene_features)

            # 调整形状以匹配空间注意力输出 [batch_size, 1, 1, 1]
            batch_size = out.shape[0]
            scene_weights = scene_weights.view(batch_size, 1, 1, 1)

            out = out * scene_weights

        out = out * x
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class NoiseAdaptiveSwish(nn.Module):
    """噪声自适应的Swish激活函数"""

    def __init__(self, channel):
        super().__init__()
        # 学习噪声强度到激活斜率的映射
        self.slope = nn.Parameter(torch.ones(1, channel, 1, 1))  # 通道自适应斜率

    def forward(self, x, noise_intensity=None):
        """
        x: 输入特征 [B, C, H, W]
        noise_intensity: 噪声强度特征 [B, 1, H, W] 或 None
        """
        if noise_intensity is not None:
            # 确保 noise_intensity 与 x 的空间维度匹配
            if noise_intensity.shape[2:] != x.shape[2:]:
                # 使用插值调整 noise_intensity 的空间尺寸
                noise_intensity = F.interpolate(
                    noise_intensity,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            # 根据噪声强度调整激活斜率
            # 噪声越强，激活斜率越小（避免放大噪声）
            adaptive_slope = self.slope * (1 - noise_intensity)
            return x * torch.sigmoid(adaptive_slope * x)
        else:
            # 默认行为：标准Swish
            return x * torch.sigmoid(x)

class NoiseAdaptiveBlock(nn.Module):
    """带有噪声自适应激活的块"""
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim)
        self.activation = NoiseAdaptiveSwish(dim)
        self.dropout = nn.Dropout(dropout) if dropout != 0 else nn.Identity()
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x, noise_intensity=None):
        x = self.norm(x)
        x = self.activation(x, noise_intensity)
        x = self.dropout(x)
        x = self.conv(x)
        return x
# 噪声分离模块
class NoiseSeparationModule(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()

        # 多尺度特征提取
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        # 新增：跨尺度注意力门控
        self.scale_attention = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),  # 压缩通道
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Sigmoid()  # 生成尺度权重
        )
        # 噪声分量分离 - 调整输入通道数
        self.gaussian_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 1)
        )

        self.stripe_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 1)
        )

        self.salt_pepper_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 1)
        )


        self.fusion = nn.Conv2d(out_channels * 3, out_channels, 1)

    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.conv1(x)  # 1x1: 细节特征（适合椒盐噪声）
        feat2 = self.conv2(x)  # 3x3: 中等尺度特征
        feat3 = self.conv3(x)  # 5x5: 大尺度特征（适合条带噪声）

        # 特征拼接用于计算注意力权重
        multi_scale_feat = torch.cat([feat1, feat2, feat3], dim=1)

        # 计算跨尺度注意力权重 [B, C, 1, 1]
        scale_weights = self.scale_attention(multi_scale_feat)

        # 应用注意力权重到每个尺度特征
        weighted_feat1 = feat1 * scale_weights
        weighted_feat2 = feat2 * scale_weights
        weighted_feat3 = feat3 * scale_weights

        # 分离噪声分量 - 每个分支处理加权后的多尺度特征
        gaussian_noise = self.gaussian_branch(weighted_feat1 + weighted_feat2 + weighted_feat3)
        stripe_noise = self.stripe_branch(weighted_feat1 + weighted_feat2 + weighted_feat3)
        salt_pepper_noise = self.salt_pepper_branch(weighted_feat1 + weighted_feat2 + weighted_feat3)

        # 对条带噪声应用横向注意力
        stripe_noise = self.horizontal_attention(stripe_noise)

        # 融合所有噪声分量
        combined_noise = torch.cat([gaussian_noise, stripe_noise, salt_pepper_noise], dim=1)
        output = self.fusion(combined_noise)
        # 新增：计算综合噪声强度图 [B, 1, H, W]
        noise_intensity = (gaussian_noise.abs().mean(dim=1, keepdim=True) +
                          stripe_noise.abs().mean(dim=1, keepdim=True) +
                          salt_pepper_noise.abs().mean(dim=1, keepdim=True)) / 3

        return output, (gaussian_noise, stripe_noise, salt_pepper_noise), noise_intensity
    def horizontal_attention(self, x):
        """横向注意力机制，用于条带噪声处理"""
        batch, channels, height, width = x.shape

        # 计算横向注意力权重
        horizontal_pool = F.adaptive_avg_pool2d(x, (1, width))  # [B, C, 1, W]
        horizontal_weights = torch.sigmoid(horizontal_pool)

        # 应用注意力
        attended_x = x * horizontal_weights

        return attended_x

# 增强的条带噪声处理块
class StripeAwareBlock(nn.Module):
    def __init__(self, dim, dim_out, norm_groups=32, dropout=0):
        super().__init__()

        # 使用噪声自适应块
        self.main_block = NoiseAdaptiveBlock(dim, dim_out, norm_groups, dropout)

        # 横向条带检测
        self.horizontal_conv = nn.Conv2d(dim, dim_out, (1, 3), padding=(0, 1))
        self.vertical_conv = nn.Conv2d(dim, dim_out, (3, 1), padding=(1, 0))

        self.fusion = nn.Conv2d(dim_out * 3, dim_out, 1)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, noise_intensity=None):
        # 主路径 - 传入噪声强度
        main_feat = self.main_block(x, noise_intensity)

        # 横向特征（检测水平条带）
        horizontal_feat = self.horizontal_conv(x)

        # 纵向特征（检测垂直特征，用于对比）
        vertical_feat = self.vertical_conv(x)

        # 特征融合
        combined_feat = torch.cat([main_feat, horizontal_feat, vertical_feat], dim=1)
        output = self.fusion(combined_feat)

        return output + self.res_conv(x)



class SceneAwareResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0,
                 with_attn=False, use_stripe_aware=False, layer_type='down'):
        super().__init__()
        self.with_attn = with_attn
        self.use_stripe_aware = use_stripe_aware
        self.layer_type = layer_type  # 'down', 'mid', 'up'

        if use_stripe_aware:
            self.res_block = StripeAwareBlock(dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        else:
            self.res_block = ResnetBlock(
                dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)

        if with_attn:
            self.ca = SceneAwareCLAM(dim_out, pool_mode='Avg|Max')
            self.sa = SceneAwareSLAM(kernel_size=7, pool_mode='Avg|Max')

            # 注意力权重融合器
            self.attention_fusion = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # 输出CLAM和SLAM的权重
                nn.Softmax(dim=1)
            )

    def forward(self, x, time_emb, scene_features=None, attention_configs=None, noise_intensity=None):
        # 修改这一行，正确传递 noise_intensity
        if self.use_stripe_aware:
            x = self.res_block(x, noise_intensity)  # StripeAwareBlock 需要 noise_intensity
        else:
            x = self.res_block(x, time_emb, noise_intensity)  # ResnetBlock 需要 noise_intensity

        # ... 其余部分保持不变 ...
        if self.with_attn and scene_features is not None:
            # 处理场景特征维度
            batch_size = x.shape[0]

            # 如果scene_features是单个向量，扩展到batch中所有样本
            if scene_features.dim() == 1:
                scene_features = scene_features.unsqueeze(0).expand(batch_size, -1)
            elif scene_features.dim() == 2 and scene_features.shape[0] == 1:
                scene_features = scene_features.expand(batch_size, -1)

            # 场景自适应的注意力权重融合
            if attention_configs is not None and isinstance(attention_configs, dict):
                # 使用预定义的注意力配置
                clam_weight = attention_configs.get('clam_weight', 0.5)
                slam_weight = attention_configs.get('slam_weight', 0.5)
            else:
                # 动态学习注意力权重
                attention_weights = self.attention_fusion(scene_features)
                clam_weight = attention_weights[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                slam_weight = attention_weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # 根据层类型调整权重
            if self.layer_type == 'down':
                # 下采样层：侧重通道注意力（光谱信息）
                clam_weight = clam_weight * 1.2
                slam_weight = slam_weight * 0.8
            elif self.layer_type == 'up':
                # 上采样层：侧重空间注意力（细节恢复）
                clam_weight = clam_weight * 0.8
                slam_weight = slam_weight * 1.2

            # 应用场景感知的注意力
            ca_out = self.ca(x, scene_features) * clam_weight
            sa_out = self.sa(x, scene_features) * slam_weight

            x = x + ca_out + sa_out
        elif self.with_attn:
            # 回退到原始注意力机制
            x = self.ca(x)
            x = self.sa(x)

        return x
class AdaptiveUNet(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 4),
            attn_res=(8),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
            image_size=256,
            # 新增参数
            use_noise_separation=False,
            use_scene_adaptive=False
    ):
        super().__init__()

        self.use_noise_separation = use_noise_separation
        self.use_scene_adaptive = use_scene_adaptive

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        # 噪声分离模块
        if use_noise_separation:
            self.noise_separation = NoiseSeparationModule(
                in_channel, inner_channel
            )
            actual_in_channel = inner_channel
        else:
            actual_in_channel = in_channel

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size

        # 下采样路径
        downs = [nn.Conv2d(actual_in_channel, inner_channel, kernel_size=3, padding=1)]

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            use_stripe_aware = (ind < 2)

            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(SceneAwareResnetBlocWithAttn(
                    pre_channel, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=use_attn,
                    use_stripe_aware=use_stripe_aware,
                    layer_type='down'
                ))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2

        self.downs = nn.ModuleList(downs)

        # 中间层
        self.mid = nn.ModuleList([
            SceneAwareResnetBlocWithAttn(pre_channel, pre_channel,
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout,
                                         with_attn=True,
                                         layer_type='mid'),
            SceneAwareResnetBlocWithAttn(pre_channel, pre_channel,
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout,
                                         with_attn=False,
                                         layer_type='mid')
        ])

        # 上采样路径
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]

            for _ in range(0, res_blocks + 1):
                ups.append(SceneAwareResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=use_attn,
                    layer_type='up'
                ))
                pre_channel = channel_mult

            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, scene_features=None, attention_configs=None):
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        # 噪声分离 - 使用改进的模块
        noise_intensity = None
        if self.use_noise_separation:
            x, noise_components, noise_intensity = self.noise_separation(x)

            self.current_noise_components = noise_components

        feats = []
        for layer in self.downs:
            if isinstance(layer, SceneAwareResnetBlocWithAttn):
                x = layer(x, t, scene_features, attention_configs, noise_intensity)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, SceneAwareResnetBlocWithAttn):
                x = layer(x, t, scene_features, attention_configs, noise_intensity)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, SceneAwareResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t, scene_features, attention_configs, noise_intensity)
            else:
                x = layer(x)

        return self.final_conv(x)


# 保持向后兼容性
UNet = AdaptiveUNet