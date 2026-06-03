import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import time
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_fastdiffsr_infer_x4.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('-log_efficiency', action='store_true', help='记录效率统计信息')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0
    test_times = []

    # === 新增：效率统计 ===
    scene_stats = {
        'urban': {'count': 0, 'time': 0, 'steps': 0},
        'farmland': {'count': 0, 'time': 0, 'steps': 0},
        'mountain': {'count': 0, 'time': 0, 'steps': 0},
        'water': {'count': 0, 'time': 0, 'steps': 0}
    }
    total_inference_time = 0
    total_samples = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)

        # === 新增：记录推理时间 ===
        torch.cuda.synchronize()
        tic = time.time()

        diffusion.test(continous=True)

        torch.cuda.synchronize()
        toc = time.time()
        inference_time = toc - tic
        test_times.append(inference_time)

        total_inference_time += inference_time
        total_samples += 1

        # === 在这里添加效率统计 ===
        if hasattr(diffusion.netG, 'current_scene_type'):
            scene_type = diffusion.netG.current_scene_type
            scene_stats[scene_type]['count'] += 1
            scene_stats[scene_type]['time'] += inference_time

            if hasattr(diffusion.netG, 'current_timesteps'):
                actual_steps = diffusion.netG.current_timesteps
                scene_stats[scene_type]['steps'] += actual_steps
                logger.info(f"Sample {idx}: Scene={scene_type}, Steps={actual_steps}, Time={inference_time:.3f}s")
        # === 效率统计结束 ===
        visuals = diffusion.get_current_visuals()

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

        avg_test_time = np.mean(test_times)
        logger.info('Average test time: %0.6e' % (avg_test_time))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    # === 新增：效率统计报告 ===
    if args.log_efficiency or opt.get('log_efficiency', False):
        logger.info("\n" + "=" * 60)
        logger.info("                 推理效率统计报告")
        logger.info("=" * 60)

        # 在推理循环结束后，添加效率统计报告
        if args.log_efficiency or opt.get('log_efficiency', False):
            logger.info("\n" + "=" * 60)
            logger.info("                 推理效率统计报告")
            logger.info("=" * 60)

            for scene_type in scene_stats:
                if scene_stats[scene_type]['count'] > 0:
                    count = scene_stats[scene_type]['count']
                    avg_time = scene_stats[scene_type]['time'] / count
                    avg_steps = scene_stats[scene_type]['steps'] / count

                    # 计算节省
                    fixed_steps = 20
                    steps_saved = fixed_steps - avg_steps
                    time_saved_per_sample = steps_saved * (avg_time / avg_steps)
                    efficiency_gain = (steps_saved / fixed_steps) * 100

                    logger.info(f" {scene_type.upper()}场景:")
                    logger.info(f"   样本数: {count}")
                    logger.info(f"   平均步数: {avg_steps:.1f} 步")
                    logger.info(f"   平均时间: {avg_time:.3f} 秒")
                    logger.info(f"   步数节省: {steps_saved:.1f} 步 ({efficiency_gain:.1f}%)")
                    logger.info(f"   时间节省/样本: {time_saved_per_sample:.3f} 秒")

        if total_samples > 0:
            avg_inference_time = total_inference_time / total_samples
            total_avg_steps = sum(stats['steps'] for stats in scene_stats.values()) / total_samples

            overall_steps_saved = 20 - total_avg_steps
            overall_efficiency_gain = (overall_steps_saved / 20) * 100
            overall_time_saving = overall_steps_saved * (avg_inference_time / total_avg_steps)

            logger.info(f" 总体统计:")
            logger.info(f"   总样本数: {total_samples}")
            logger.info(f"   平均推理时间: {avg_inference_time:.3f} 秒")
            logger.info(f"   平均采样步数: {total_avg_steps:.1f} 步")
            logger.info(f"   总体效率提升: {overall_efficiency_gain:.1f}%")
            logger.info(f"   总时间节省: {overall_time_saving * total_samples:.2f} 秒")

            # 吞吐量计算
            throughput = total_samples / total_inference_time
            fixed_throughput = total_samples / (total_inference_time * (20 / total_avg_steps))
            throughput_improvement = (throughput - fixed_throughput) / fixed_throughput * 100

            logger.info(f"   吞吐量: {throughput:.2f} 图像/秒")
            logger.info(f"   吞吐量提升: {throughput_improvement:.1f}%")

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)