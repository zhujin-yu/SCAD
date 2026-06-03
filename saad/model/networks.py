import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################

# Generator
def define_G(opt):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'tesr':
        from .tesr_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'gdp':
        from .gdp_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'fastdiffsr':
        from .fastdiffsr_modules import diffusion, unet

    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups'] = 32

    # 遥感噪声参数
    use_remote_sensing_noise = model_opt.get('use_remote_sensing_noise', False)
    use_noise_separation = model_opt.get('use_noise_separation', False)

    # 场景自适应参数
    use_scene_adaptive = model_opt.get('use_scene_adaptive', False)

    stripe_intensity = model_opt.get('stripe_intensity', 0.3)
    salt_pepper_prob = model_opt.get('salt_pepper_prob', 0.002)
    radiation_intensity = model_opt.get('radiation_intensity', 0.2)

    # 使用自适应UNet
    model = unet.AdaptiveUNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size'],
        # 新增参数
        use_noise_separation=use_noise_separation,
        use_scene_adaptive=use_scene_adaptive
    )

    # 使用自适应扩散模型
    netG = diffusion.AdaptiveGaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
        scale=int(256 / int(opt['datasets']['train']['l_resolution'])),
        # 新增遥感噪声参数
        use_remote_sensing_noise=use_remote_sensing_noise,
        stripe_intensity=stripe_intensity,
        salt_pepper_prob=salt_pepper_prob,
        radiation_intensity=radiation_intensity,
        # 新增场景自适应参数
        use_scene_adaptive=use_scene_adaptive
    )

    if opt['phase'] == 'train':
        init_weights(netG, init_type='orthogonal')

    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)

    return netG
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))



def define_G(opt):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'tesr':
        from .tesr_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'gdp':
        from .gdp_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'fastdiffsr':
        from .fastdiffsr_modules import diffusion, unet

    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups'] = 32

    # 遥感噪声参数
    use_remote_sensing_noise = model_opt.get('use_remote_sensing_noise', False)
    use_noise_separation = model_opt.get('use_noise_separation', False)

    # 场景自适应参数 - 新增 UCMerced 预训练参数
    use_scene_adaptive = model_opt.get('use_scene_adaptive', False)
    use_ucmerced_pretrain = model_opt.get('use_ucmerced_pretrain', True)  # 默认启用
    ucmerced_checkpoint_path = model_opt.get('ucmerced_checkpoint_path', None)  # 预训练权重路径

    stripe_intensity = model_opt.get('stripe_intensity', 0.3)
    salt_pepper_prob = model_opt.get('salt_pepper_prob', 0.002)
    radiation_intensity = model_opt.get('radiation_intensity', 0.2)

    # 使用自适应UNet
    model = unet.AdaptiveUNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size'],
        # 新增参数
        use_noise_separation=use_noise_separation,
        use_scene_adaptive=use_scene_adaptive
    )

    # 使用自适应扩散模型 - 传递 UCMerced 预训练参数
    netG = diffusion.AdaptiveGaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
        scale=int(256 / int(opt['datasets']['train']['l_resolution'])),
        # 新增遥感噪声参数
        use_remote_sensing_noise=use_remote_sensing_noise,
        stripe_intensity=stripe_intensity,
        salt_pepper_prob=salt_pepper_prob,
        radiation_intensity=radiation_intensity,
        # 新增场景自适应参数
        use_scene_adaptive=use_scene_adaptive,
        # 新增 UCMerced 预训练参数
        use_ucmerced_pretrain=use_ucmerced_pretrain,
        ucmerced_checkpoint_path=ucmerced_checkpoint_path
    )

    if opt['phase'] == 'train':
        init_weights(netG, init_type='orthogonal')

    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)

    return netG