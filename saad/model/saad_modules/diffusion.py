from functools import partial
import numpy as np
from tqdm import tqdm
import os
import logging

from .unet import *
from .scene_classifier import SceneAdaptiveModule

logger = logging.getLogger('base')


# 遥感特殊噪声注入器
class RemoteSensingNoiseInjector:
    def __init__(self):
        self.stripe_patterns = {}

    def add_stripe_noise(self, x, stripe_intensity=0.3, stripe_period=20):
        """
        添加条带噪声
        x: 输入图像 [B, C, H, W]
        stripe_intensity: 条带强度 0-1
        stripe_period: 条带周期（像素）
        """
        device = x.device  # 获取输入张量的设备
        batch_size, channels, height, width = x.shape

        # 创建条带噪声模板
        stripe_noise = torch.zeros_like(x)
        for b in range(batch_size):
            # 随机选择条带起始位置
            start_pos = torch.randint(0, stripe_period, (1,), device=device).item()

            # 自适应条带强度：模拟真实传感器的不均匀响应
            base_intensity = stripe_intensity * (torch.rand(1, device=device).item() * 0.5 + 0.5)

            # 创建周期性条带
            for w_idx in range(start_pos, width, stripe_period):
                end_idx = min(w_idx + 1, width)
                stripe_width = torch.randint(1, 3, (1,), device=device).item()  # 条带宽度1-2像素
                end_idx = min(w_idx + stripe_width, width)

                if end_idx > w_idx:
                    # 条带内部强度变化
                    stripe_variation = 0.2 * torch.randn(channels, height, end_idx - w_idx, device=device)
                    current_intensity = base_intensity + stripe_variation

                    # 应用条带噪声
                    stripe_val = current_intensity * torch.randn(channels, height, end_idx - w_idx, device=device)
                    stripe_noise[b, :, :, w_idx:end_idx] = stripe_val

        return stripe_noise

    def add_salt_pepper_noise(self, x, salt_prob=0.002, pepper_prob=0.002):
        """
        添加椒盐噪声
        salt_prob: 盐噪声概率
        pepper_prob: 椒噪声概率
        """
        device = x.device
        batch_size, channels, height, width = x.shape
        noisy_x = x.clone()

        # 盐噪声（最大值）
        salt_mask = torch.rand(batch_size, 1, height, width, device=device) < salt_prob
        salt_mask = salt_mask.expand_as(x)
        noisy_x[salt_mask] = 1.0  # 假设图像范围[-1,1]

        # 椒噪声（最小值）
        pepper_mask = torch.rand(batch_size, 1, height, width, device=device) < pepper_prob
        pepper_mask = pepper_mask.expand_as(x)
        noisy_x[pepper_mask] = -1.0  # 假设图像范围[-1,1]

        return noisy_x - x  # 返回噪声分量

    def add_radiation_noise(self, x, distortion_intensity=0.2):
        """
        添加辐射畸变噪声（模拟大气散射导致的边缘模糊）
        """
        device = x.device
        batch_size, channels, height, width = x.shape

        # 创建径向模糊核模拟辐射畸变
        center_y, center_x = height // 2, width // 2
        y, x = torch.meshgrid(torch.arange(height, device=device).float(),
                             torch.arange(width, device=device).float())
        y = y.to(device) - center_y
        x = x.to(device) - center_x

        # 距离中心的距离
        distance = torch.sqrt(x ** 2 + y ** 2)
        max_distance = torch.sqrt(torch.tensor(center_x ** 2 + center_y ** 2, device=device).float())

        # 距离越远，模糊程度越大
        blur_intensity = distortion_intensity * (distance / max_distance)
        blur_intensity = blur_intensity.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        blur_intensity = blur_intensity.expand(batch_size, channels, -1, -1)

        # 生成随机模糊噪声
        radiation_noise = blur_intensity * torch.randn_like(x)

        return radiation_noise
# 噪声抑制损失
class NoiseSuppressLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def stripe_energy_loss(self, pred, target):
        """计算条带能量损失 - 通过傅里叶变换检测高频条纹"""
        device = pred.device
        batch_size, channels, height, width = pred.shape

        total_stripe_loss = 0.0
        for b in range(batch_size):
            for c in range(channels):
                # 傅里叶变换检测条带
                pred_fft = torch.fft.fft2(pred[b, c])
                target_fft = torch.fft.fft2(target[b, c])

                # 计算频谱幅度
                pred_mag = torch.abs(pred_fft)
                target_mag = torch.abs(target_fft)

                # 重点关注高频分量（条带通常在特定高频）
                # 创建高频掩码（忽略中心低频区域）
                center_y, center_x = height // 2, width // 2
                y, x = torch.meshgrid(torch.arange(height, device=device),
                                     torch.arange(width, device=device))
                y = y.to(device) - center_y
                x = x.to(device) - center_x

                # 距离频率中心的距离
                freq_distance = torch.sqrt(x.float() ** 2 + y.float() ** 2)
                high_freq_mask = freq_distance > min(center_x, center_y) * 0.3

                # 计算高频能量差异（条带能量）
                pred_high_freq = pred_mag[high_freq_mask]
                target_high_freq = target_mag[high_freq_mask]

                if len(pred_high_freq) > 0:
                    stripe_loss = F.l1_loss(pred_high_freq, target_high_freq)
                    total_stripe_loss += stripe_loss

        return total_stripe_loss / (batch_size * channels)

    def salt_pepper_loss(self, pred, target):
        """计算椒盐噪声损失 - 检测异常像素占比"""
        batch_size, channels, height, width = pred.shape

        total_sp_loss = 0.0
        for b in range(batch_size):
            # 计算像素值差异
            diff = torch.abs(pred[b] - target[b])

            # 检测异常像素（与周围差异大的像素）
            # 使用3x3卷积计算局部均值
            kernel = torch.ones(1, 1, 3, 3).to(pred.device) / 9.0
            pred_smooth = F.conv2d(pred[b:b + 1].mean(dim=1, keepdim=True), kernel, padding=1)
            target_smooth = F.conv2d(target[b:b + 1].mean(dim=1, keepdim=True), kernel, padding=1)

            pred_local_diff = torch.abs(pred[b:b + 1].mean(dim=1, keepdim=True) - pred_smooth)
            target_local_diff = torch.abs(target[b:b + 1].mean(dim=1, keepdim=True) - target_smooth)

            # 异常像素阈值
            abnormal_threshold = 0.3
            pred_abnormal_ratio = (pred_local_diff > abnormal_threshold).float().mean()
            target_abnormal_ratio = (target_local_diff > abnormal_threshold).float().mean()

            # 异常像素占比差异
            sp_loss = torch.abs(pred_abnormal_ratio - target_abnormal_ratio)
            total_sp_loss += sp_loss

        return total_sp_loss / batch_size

    def forward(self, pred, target):
        stripe_loss = self.stripe_energy_loss(pred, target)
        salt_pepper_loss = self.salt_pepper_loss(pred, target)

        # 加权组合两种噪声损失
        return 0.7 * stripe_loss + 0.3 * salt_pepper_loss


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)

    elif schedule == "linear_cosine":
        betas1 = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)

        steps = n_timestep + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + cosine_s) / (1 + cosine_s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas2 = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas2 = np.clip(betas2, a_min=0, a_max=0.999)

        betas = np.add(betas1, np.add(betas2, betas2))
        betas = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class AdaptiveGaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            image_size,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            scale=4,
            # 新增遥感噪声参数
            use_remote_sensing_noise=False,
            stripe_intensity=0.3,
            salt_pepper_prob=0.002,
            radiation_intensity=0.2,
            # 新增场景自适应参数
            use_scene_adaptive=False,
            # 新增 UCMerced 预训练参数
            use_ucmerced_pretrain=True,
            ucmerced_checkpoint_path=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.scale = scale

        # 遥感噪声参数
        self.use_remote_sensing_noise = use_remote_sensing_noise
        self.stripe_intensity = stripe_intensity
        self.salt_pepper_prob = salt_pepper_prob
        self.radiation_intensity = radiation_intensity
        self.noise_injector = RemoteSensingNoiseInjector()

        # 场景自适应参数
        self.use_scene_adaptive = use_scene_adaptive
        if use_scene_adaptive:
            self.scene_module = SceneAdaptiveModule(
                device='cuda',
                use_ucmerced_pretrain=use_ucmerced_pretrain,
                ucmerced_checkpoint_path=ucmerced_checkpoint_path
            )
            # 尝试加载预训练权重
            self.load_scene_classifier()

        # 效率统计
        self.total_steps_saved = 0
        self.sample_count = 0
        self.scene_inference_times = {
            'urban': [], 'farmland': [], 'mountain': [], 'water': []
        }
        self.current_scene_type = None
        self.current_timesteps = None

        # 存储不同场景的噪声调度
        self.scene_schedules = {}

        if schedule_opt is not None:
            pass

        logger.info(f"Scene Adaptive: {use_scene_adaptive}")
        logger.info(f"Use UCMerced Pretrain: {use_ucmerced_pretrain}")

    def load_scene_classifier(self):
        """尝试加载预训练的场景分类器"""
        # 使用你的绝对路径
        pretrained_paths = [
            r'F:\zhujinyu\FastDiffSR\FastDiffSR-2\FastDiffSR\checkpoints\ucmerced_mobilenetv3\best_model_21class_mobilenetv3.pth'
        ]

        for path in pretrained_paths:
            if os.path.exists(path):
                success = self.scene_module.load_ucmerced_pretrained(path)
                if success:
                    logger.info(f"✅ Successfully loaded scene classifier from {path}")
                    break
        else:
            logger.warning("❌ No pretrained scene classifier found. Using randomly initialized classifier.")
            logger.info("📝 MobileNetV3 will use ImageNet pretrained weights for feature extraction")
    def set_adaptive_noise_schedule(self, lr_images):
        """
        根据场景自适应设置噪声调度
        """
        if not self.use_scene_adaptive:
            return None

        # 场景分类
        scene_preds, scene_probs, scene_features = self.scene_module.classify_scene(lr_images)
        timesteps_list, cosine_scales, attention_configs = self.scene_module.get_adaptive_config(scene_preds)

        # 存储场景信息供UNet使用
        self.current_scene_features = scene_features
        self.current_attention_configs = attention_configs

        # 使用第一个样本的配置
        scene_idx = scene_preds[0].item()
        scene_type = self.scene_module.scene_types[scene_idx]
        timesteps = timesteps_list[0]
        cosine_scale = cosine_scales[0]

        # 记录当前场景和步数
        self.current_scene_type = scene_type
        self.current_timesteps = timesteps

        # 统计步数节省
        original_steps = 20  # 原固定步数
        steps_saved = original_steps - timesteps
        self.total_steps_saved += steps_saved
        self.sample_count += 1

        schedule_opt = self.create_scene_specific_schedule(scene_type, timesteps, cosine_scale)

        # 打印效率信息
        avg_steps_saved = self.total_steps_saved / self.sample_count if self.sample_count > 0 else 0
        logger.info(f"Scene: {scene_type}, Timesteps: {timesteps}, Steps Saved: {steps_saved}")
        logger.info(f"Average steps saved: {avg_steps_saved:.2f} steps per sample")

        return schedule_opt

    def create_scene_specific_schedule(self, scene_type, timesteps, cosine_scale):
        """
        为特定场景创建噪声调度
        """
        key = f"{scene_type}_T{timesteps}_C{cosine_scale}"
        if key in self.scene_schedules:
            return self.scene_schedules[key]

        # 基于场景类型调整调度参数
        if scene_type == 'urban':
            # 城市场景：更平缓的噪声衰减，保留更多细节
            linear_end = 1.5e-2
        elif scene_type == 'water':
            # 水体场景：更快的噪声衰减，避免过度采样
            linear_end = 0.8e-2
        else:
            linear_end = 1e-2

        schedule_opt = {
            'schedule': 'linear_cosine',
            'n_timestep': timesteps,
            'linear_start': 1e-6,
            'linear_end': linear_end,
            'cosine_s': 8e-3 * cosine_scale  # 根据场景调整余弦调度
        }

        self.scene_schedules[key] = schedule_opt
        return schedule_opt

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

        # 新增噪声抑制损失
        self.noise_suppress_loss = NoiseSuppressLoss().to(device)

    def set_new_noise_schedule(self, schedule_opt, device, lr_images=None):
        """
        修改后的set_new_noise_schedule，支持场景自适应
        """
        # 场景自适应：在推理时设置噪声调度
        if self.use_scene_adaptive and lr_images is not None and not self.training:
            adaptive_schedule = self.set_adaptive_noise_schedule(lr_images)
            if adaptive_schedule is not None:
                schedule_opt = adaptive_schedule
                logger.info(f"Applied adaptive schedule for scene: {self.current_scene_type}")

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        cosine_s = schedule_opt.get('cosine_s', 8e-3)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'],
            cosine_s=cosine_s)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def add_remote_sensing_noise(self, x_start, img_lr_up):
        """
        添加遥感特殊噪声到残差图像
        自适应噪声权重：根据区域特性调整不同噪声的强度
        """
        if not self.use_remote_sensing_noise:
            return torch.zeros_like(x_start)

        device = x_start.device  # 获取输入张量的设备
        batch_size = x_start.shape[0]
        total_noise = torch.zeros_like(x_start)

        for b in range(batch_size):
            noise_components = []
            weights = []

            # 自适应条带噪声检测和注入
            if torch.rand(1, device=device) < 0.7:  # 70%概率添加条带噪声
                # 模拟传感器特性：不同区域条带强度不同
                stripe_strength = self.stripe_intensity * (torch.rand(1, device=device).item() * 0.6 + 0.4)
                stripe_period = 15 + torch.randint(0, 15, (1,), device=device).item()  # 周期15-30像素

                stripe_noise = self.noise_injector.add_stripe_noise(
                    x_start[b:b + 1],
                    stripe_strength,
                    stripe_period
                )
                noise_components.append(stripe_noise)
                # 条带噪声权重：在明显条带区域权重更高
                stripe_weight = 0.4 + torch.rand(1, device=device).item() * 0.3  # 权重0.4-0.7
                weights.append(stripe_weight)

            # 自适应椒盐噪声
            if torch.rand(1, device=device) < 0.5:  # 50%概率添加椒盐噪声
                # 模拟云层遮挡导致的异常像素
                current_salt_prob = self.salt_pepper_prob * (torch.rand(1, device=device).item() * 0.8 + 0.2)
                current_pepper_prob = self.salt_pepper_prob * (torch.rand(1, device=device).item() * 0.8 + 0.2)

                salt_pepper_noise = self.noise_injector.add_salt_pepper_noise(
                    x_start[b:b + 1],
                    salt_prob=current_salt_prob,
                    pepper_prob=current_pepper_prob
                )
                noise_components.append(salt_pepper_noise)
                weights.append(0.1 + torch.rand(1, device=device).item() * 0.2)  # 权重0.1-0.3

            # 辐射畸变噪声
            if torch.rand(1, device=device) < 0.4:  # 40%概率添加辐射噪声
                radiation_strength = 0.1 + torch.rand(1, device=device).item() * 0.3
                radiation_noise = self.noise_injector.add_radiation_noise(
                    x_start[b:b + 1],
                    distortion_intensity=radiation_strength
                )
                noise_components.append(radiation_noise)
                weights.append(0.1 + torch.rand(1, device=device).item() * 0.2)  # 权重0.1-0.3

            # 组合噪声（自适应权重融合）
            if noise_components:
                if sum(weights) > 0:
                    # 归一化权重
                    weights = [w / sum(weights) for w in weights]
                    combined_noise = sum(w * n for w, n in zip(weights, noise_components))
                    total_noise[b] = combined_noise[0]
                else:
                    # 默认均匀权重
                    combined_noise = sum(noise_components) / len(noise_components)
                    total_noise[b] = combined_noise[0]

        return total_noise

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None,
                        scene_features=None, attention_configs=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)

        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(
                    torch.cat([condition_x, x], dim=1), noise_level,
                    scene_features=scene_features,
                    attention_configs=attention_configs))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(
                    x, noise_level,
                    scene_features=scene_features,
                    attention_configs=attention_configs))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None,
                 scene_features=None, attention_configs=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x,
            scene_features=scene_features, attention_configs=attention_configs)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    # 在 p_sample_loop 方法中，确保正确传递场景特征
    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))

        # 场景自适应：在推理时设置噪声调度
        if self.use_scene_adaptive and not self.training:
            lr_images = x_in  # 假设x_in是LR图像
            self.set_new_noise_schedule(None, device, lr_images)

        if not self.conditional:
            shape = x_in.shape
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x

            # 传递场景特征给UNet
            scene_features = None
            attention_configs = None
            if self.use_scene_adaptive and hasattr(self, 'current_scene_features'):
                scene_features = self.current_scene_features
                attention_configs = self.current_attention_configs

                # 确保配置是字典形式
                if isinstance(attention_configs, list) and len(attention_configs) > 0:
                    attention_configs = attention_configs[0] if isinstance(attention_configs[0], dict) else {}

                # 确保场景特征有正确的形状
                if scene_features is not None:
                    # 如果只有单个样本的场景特征，扩展到batch size
                    batch_size = x.shape[0]
                    if scene_features.shape[0] == 1 and batch_size > 1:
                        scene_features = scene_features.expand(batch_size, -1)

            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x,
                                    scene_features=scene_features,
                                    attention_configs=attention_configs)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)

            img = self.res2img(img, x_in)
            for j in range(len(ret_img)):
                ret_img[j] = self.res2img(ret_img[j], x_in)

        if continous:
            return ret_img
        else:
            return img

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    # 在 diffusion.py 的 p_losses 方法中，可以利用噪声分离信息
    def p_losses(self, x_in, noise=None):
        x_hr = x_in['HR']
        img_lr_up = x_in['SR']
        x_start = self.img2res(x_hr, img_lr_up)
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        # 添加遥感特殊噪声
        remote_sensing_noise = self.add_remote_sensing_noise(x_start, img_lr_up)
        combined_noise = noise if noise is not None else torch.randn_like(x_start)
        combined_noise = combined_noise + remote_sensing_noise

        # 生成带噪声的图像
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
            noise=combined_noise
        )

        # 模型预测噪声
        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([img_lr_up, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        # 如果使用了噪声分离，可以获取噪声分量信息用于增强训练
        if hasattr(self.denoise_fn, 'current_noise_components'):
            noise_components = self.denoise_fn.current_noise_components
            # 可以在这里添加基于噪声分量的额外损失项
            # 例如：对不同类型噪声的分离质量进行监督

        # 计算基础损失
        base_loss = self.loss_func(combined_noise, x_recon)

        # 噪声抑制损失
        if self.use_remote_sensing_noise:
            with torch.no_grad():
                sr_img = self.res2img(x_recon, img_lr_up)
                noise_suppress_loss = self.noise_suppress_loss(sr_img, x_hr)
            total_loss = base_loss + 0.1 * noise_suppress_loss
        else:
            total_loss = base_loss

        return total_loss
    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    def res2img(self, img_, img_lr_up, clip_input=None):
        if clip_input is None:
            clip_input = True
        if clip_input:
            img_ = img_.clamp(-1, 1)
        img_ = img_ / 2.0 + img_lr_up
        return img_

    def img2res(self, x, img_lr_up, clip_input=None):
        if clip_input is None:
            clip_input = True
        x = (x - img_lr_up) * 2.0
        if clip_input:
            x = x.clamp(-1, 1)
        return x


# 保持向后兼容性
GaussianDiffusion = AdaptiveGaussianDiffusion