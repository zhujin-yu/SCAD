import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
from skimage.measure import compare_mse
#import PerceptualSimilarity
#from core import PerceptualSimilarity
import matplotlib.pyplot as plt
from skimage import io, data
import torchvision.transforms as transforms
import torch
import lpips


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    #cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)

    io.imsave(img_path, img)



def plot_img(result_imgs, mses, psnrs, ssims, ergas, lpips, save_path):
    """
    用Matplotlib绘制图像对比图，包含指标标注
    result_imgs: [HR, LR, INF, SR] 4张图像（numpy数组，HWC格式）
    mses/psnrs/ssims/ergas/lpips: 对应图像的指标（HR/LR为None）
    save_path: 图像保存路径
    """
    # 设置画布，4张图横向排列
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['HR (Ground Truth)', 'LR (Input)', 'INF (Baseline)', 'SR (Output)']
    metrics = ['MSE: {:.4e}', 'PSNR: {:.2f}', 'SSIM: {:.4f}', 'ERGAS: {:.4f}', 'LPIPS: {:.4f}']

    for i, (ax, img, title) in enumerate(zip(axes, result_imgs, titles)):
        ax.imshow(img.astype(np.uint8))  # 显示图像（需转为uint8格式）
        ax.set_title(title, fontsize=12)
        ax.axis('off')  # 隐藏坐标轴

        # 标注指标（仅INF和SR有指标，i=2和3）
        if i >= 2:
            metric_text = '\n'.join([
                metrics[0].format(mses[i]) if mses[i] is not None else '',
                metrics[1].format(psnrs[i]) if psnrs[i] is not None else '',
                metrics[2].format(ssims[i]) if ssims[i] is not None else '',
                metrics[3].format(ergas[i]) if ergas[i] is not None else '',
                metrics[4].format(lpips[i]) if lpips[i] is not None else ''
            ])
            ax.text(10, img.shape[0] - 10, metric_text,
                    color='white', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5))  # 黑色半透明背景

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 保存图像
    plt.close()  # 关闭画布，释放内存
def calculate_mse(img1, img2):
    mse = compare_mse(img1, img2)
    return mse

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_ergas(img1, img2, scale=4):
    channel = img1.shape[2]
    mse = compare_mse(img1, img2)
    mean2 = np.mean(img1, dtype=np.float64)**2
    ergas = 100.0*np.sqrt(mse/mean2/channel)/scale
    return ergas

def calculate_lpips(img1, img2, use_gpu=True):
    # model = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu)
    # d = model.forward(img1, img2, normalize=True)
    # return d.detach().item()

    transf = transforms.ToTensor()
    test1 = transf(img1).to(torch.float32)
    test_HR = transf(img2).to(torch.float32)
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    lpips_metrc = loss_fn_alex(test1, test_HR)
    return float(lpips_metrc)






