import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from core import metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_fastdiffsr_train_64_256.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-log_efficiency', action='store_true', help='记录效率统计信息')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    global scale
    if opt['datasets']['train']['l_resolution'] == 64:
        scale = 4
    elif opt['datasets']['train']['l_resolution'] == 32:
        scale = 8

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
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # 数据集加载
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            train_total_samples = len(train_set)
            batch_size = dataset_opt['batch_size']
            iter_per_epoch = len(train_loader)
            logger.info(f'训练集加载完成: {train_total_samples} 样本, {iter_per_epoch} 批次/epoch')

        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)



        # 计算总轮数
        n_iter = opt['train']['n_iter']

        # === 确保所有必要变量都被定义 ===
        if 'iter_per_epoch' not in locals() or 'iter_per_epoch' not in globals():
            if args.phase == 'val':
                # 验证阶段：使用验证集信息
                if 'val_loader' in locals():
                    iter_per_epoch = len(val_loader)
                    train_total_samples = len(val_set) if 'val_set' in locals() else 0
                    batch_size = dataset_opt.get('batch_size', 1)
                else:
                    iter_per_epoch = 1000
                    train_total_samples = 0
                    batch_size = 1
                    logger.warning("验证阶段变量未正确定义，使用默认值")
            else:
                # 训练阶段：使用训练集信息
                if 'train_loader' in locals():
                    iter_per_epoch = len(train_loader)
                else:
                    iter_per_epoch = opt['train'].get('iter_per_epoch', 1000)
                    logger.warning(f"iter_per_epoch 未正确定义，使用默认值: {iter_per_epoch}")

        # 确保 train_total_samples 和 batch_size 被定义
        if 'train_total_samples' not in locals() and 'train_total_samples' not in globals():
            train_total_samples = 0
        if 'batch_size' not in locals() and 'batch_size' not in globals():
            batch_size = 1

        total_epochs = (n_iter + iter_per_epoch - 1) // iter_per_epoch
        logger.info(f"=== Training Total Info ===")
        logger.info(f"Train set total samples: {train_total_samples}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Iterations per epoch: {iter_per_epoch}")
        logger.info(f"Total iterations to train: {n_iter}")
        logger.info(f"Total training epochs: {total_epochs}")
        logger.info(f"===========================")

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))
        remaining_iter = n_iter - current_step
        remaining_epochs = (remaining_iter + iter_per_epoch - 1) // iter_per_epoch
        logger.info(f"Resumed Training Info:")
        logger.info(f"Remaining iterations: {remaining_iter}")
        logger.info(f"Remaining epochs: {remaining_epochs}")

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        total_iterations = n_iter - current_step
        train_start_time = time.time()

        while current_step < n_iter:
            current_epoch += 1
            epoch_pbar = tqdm(enumerate(train_loader),
                              desc=f"Epoch {current_epoch:3d}/{total_epochs} (Total Epochs)",
                              total=len(train_loader),
                              unit="batch")

            for _, train_data in epoch_pbar:
                current_step += 1
                if current_step > n_iter:
                    epoch_pbar.close()
                    break

                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # 学习率调度
                if diffusion.scheduler is not None:
                    diffusion.scheduler.step()

                # 在验证阶段后记录学习率
                if current_step % opt['train']['val_freq'] == 0:
                    current_lr = diffusion.optG.param_groups[0]['lr']
                    logger.info(f'Current learning rate: {current_lr:.7e}')

                    if wandb_logger:
                        wandb_logger.log_metrics({'learning_rate': current_lr})

                # 新增学习率调整逻辑
                if current_step % 100000 == 0 and current_step != 0:
                    old_lr = diffusion.optG.param_groups[0]['lr']
                    new_lr = old_lr / 2
                    diffusion.optG.param_groups[0]['lr'] = new_lr
                    logger.info(f"学习率调整：从 {old_lr:.7e} 衰减为 {new_lr:.7e}")

                # 计算进度和时间信息
                completed_iter = current_step - diffusion.begin_step
                remaining_iter = total_iterations - completed_iter
                elapsed_time = time.time() - train_start_time

                if completed_iter > 0:
                    iter_per_sec = completed_iter / elapsed_time
                    remaining_sec = int(remaining_iter / iter_per_sec)
                    remaining_time = str(timedelta(seconds=remaining_sec))
                    end_time = datetime.now() + timedelta(seconds=remaining_sec)
                    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    remaining_time = "Calculating..."
                    end_time_str = "Calculating..."

                logs = diffusion.get_current_log()
                loss_val = logs.get('loss', 0.0)

                epoch_pbar.set_description(
                    f"Epoch {current_epoch:3d}/{total_epochs} | Step {current_step:8,d}/{n_iter} | Loss: {loss_val:.4e} | "
                    f"Remain: {remaining_time} | End at: {end_time_str}"
                )

                # log
                if current_step % opt['train']['print_freq'] == 0:
                    message = '<epoch:{:3d}/{:3d}, iter:{:8,d}/{:8,d}> '.format(
                        current_epoch, total_epochs, current_step, n_iter)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # 验证
                if current_step % opt['train']['val_freq'] == 0:
                    bic_mse = 0.0
                    bic_psnr = 0.0
                    bic_ssim = 0.0
                    bic_ergas = 0.0
                    bic_lpips = 0.0
                    avg_mse = 0.0
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_ergas = 0.0
                    avg_lpips = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    # === 新增：效率统计 ===
                    scene_stats = {
                        'urban': {'count': 0, 'time': 0, 'steps': 0, 'psnr': [], 'ssim': []},
                        'farmland': {'count': 0, 'time': 0, 'steps': 0, 'psnr': [], 'ssim': []},
                        'mountain': {'count': 0, 'time': 0, 'steps': 0, 'psnr': [], 'ssim': []},
                        'water': {'count': 0, 'time': 0, 'steps': 0, 'psnr': [], 'ssim': []}
                    }
                    total_val_time = 0
                    total_val_samples = 0

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)

                        # === 新增：记录推理时间 ===
                        torch.cuda.synchronize()
                        start_time = time.time()

                        diffusion.test(continous=False)

                        torch.cuda.synchronize()
                        end_time = time.time()
                        inference_time = end_time - start_time

                        total_val_time += inference_time
                        total_val_samples += 1

                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])
                        hr_img = Metrics.tensor2img(visuals['HR'])
                        lr_img = Metrics.tensor2img(visuals['LR'])
                        fake_img = Metrics.tensor2img(visuals['INF'])

                        # 计算评估指标
                        bmse = compare_mse(fake_img, hr_img)
                        bpsnr = compare_psnr(fake_img, hr_img)
                        bssim = compare_ssim(fake_img, hr_img, multichannel=True)
                        bergas = Metrics.calculate_ergas(fake_img, hr_img, scale=scale)
                        blpips = Metrics.calculate_lpips(fake_img, hr_img)
                        smse = compare_mse(sr_img, hr_img)
                        spsnr = compare_psnr(sr_img, hr_img)
                        sssim = compare_ssim(sr_img, hr_img, multichannel=True)
                        sergas = Metrics.calculate_ergas(sr_img, hr_img, scale=scale)
                        slpips = Metrics.calculate_lpips(sr_img, hr_img)

                        # === 在这里添加效率统计（在变量定义后）===
                        if hasattr(diffusion.netG, 'current_scene_type'):
                            scene_type = diffusion.netG.current_scene_type
                            scene_stats[scene_type]['count'] += 1
                            scene_stats[scene_type]['time'] += inference_time

                            if hasattr(diffusion.netG, 'current_timesteps'):
                                actual_steps = diffusion.netG.current_timesteps
                                scene_stats[scene_type]['steps'] += actual_steps
                                # 记录质量指标（现在 spsnr 和 sssim 已经定义）
                                scene_stats[scene_type]['psnr'].append(spsnr)
                                scene_stats[scene_type]['ssim'].append(sssim)
                                logger.info(
                                    f"Sample {idx}: Scene={scene_type}, Steps={actual_steps}, Time={inference_time:.3f}s, PSNR={spsnr:.2f}, SSIM={sssim:.4f}")

                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.tif'.format(result_path, current_step, idx))
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        # 绘制评估图像
                        result_imgs = [hr_img, lr_img, fake_img, sr_img]
                        mses = [None, None, bmse, smse]
                        psnrs = [None, None, bpsnr, spsnr]
                        ssims = [None, None, bssim, sssim]
                        ergas = [None, None, bergas, sergas]
                        lpips = [None, None, blpips, slpips]
                        Metrics.plot_img(
                            result_imgs, mses, psnrs, ssims, ergas, lpips,
                            '{}/{}_{}_plot.png'.format(result_path, current_step, idx))

                        # 累积指标
                        bic_mse += bmse
                        bic_psnr += bpsnr
                        bic_ssim += bssim
                        bic_ergas += bergas
                        bic_lpips += blpips

                        avg_mse += smse
                        avg_psnr += spsnr
                        avg_ssim += sssim
                        avg_ergas += sergas
                        avg_lpips += slpips

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    # 计算平均指标
                    bic_mse /= idx
                    bic_psnr /= idx
                    bic_ssim /= idx
                    bic_ergas /= idx
                    bic_lpips /= idx

                    avg_mse /= idx
                    avg_psnr /= idx
                    avg_ssim /= idx
                    avg_ergas /= idx
                    avg_lpips /= idx

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                    # 修复日志打印语句的语法错误
                    logger_val = logging.getLogger('val')
                    if current_step == 9152:
                        logger_val.info(
                            '<epoch:{:3d}/{:3d}, iter:{:8,d}/{:8,d}, lr:{:.7e}> bic_mse: {:.5e}, bic_psnr: {:.5e}, bic_ssim：{:.5e}, bic_ergas: {:.5e}, bic_lpips: {:.5e}'.format(
                                current_epoch, total_epochs, current_step, n_iter, diffusion.optG.param_groups[0]['lr'],
                                bic_mse, bic_psnr,
                                bic_ssim, bic_ergas, bic_lpips))
                    # 修正此处的格式字符串，确保参数数量与占位符一致
                    logger_val.info(
                        '<epoch:{:3d}/{:3d}, iter:{:8,d}/{:8,d}, lr:{:.7e}> sr_mse: {:.5e}, sr_psnr: {:.5e}, sr_ssim：{:.5e}, sr_ergas: {:.5e}, sr_lpips：{:.5e}'.format(
                            current_epoch, total_epochs, current_step, n_iter, diffusion.optG.param_groups[0]['lr'],
                            avg_mse, avg_psnr, avg_ssim, avg_ergas, avg_lpips))

                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                    # === 新增：效率统计报告 ===
                    if args.log_efficiency or opt.get('log_efficiency', False):
                        logger.info("\n" + "=" * 60)
                        logger.info("                 场景自适应效率统计")
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

                                # 质量指标
                                avg_psnr = np.mean(scene_stats[scene_type]['psnr']) if scene_stats[scene_type][
                                    'psnr'] else 0
                                avg_ssim = np.mean(scene_stats[scene_type]['ssim']) if scene_stats[scene_type][
                                    'ssim'] else 0

                                logger.info(f" {scene_type.upper()}场景:")
                                logger.info(f"   样本数: {count}")
                                logger.info(f"   平均步数: {avg_steps:.1f} 步 (节省 {steps_saved:.1f}步)")
                                logger.info(f"   平均时间: {avg_time:.3f} 秒 (节省 {time_saved_per_sample:.3f}秒/样本)")
                                logger.info(f"   效率提升: {efficiency_gain:.1f}%")
                                if avg_psnr > 0:
                                    logger.info(f"   平均PSNR: {avg_psnr:.2f}, 平均SSIM: {avg_ssim:.4f}")

                        if total_val_samples > 0:
                            avg_val_time = total_val_time / total_val_samples
                            total_avg_steps = sum(stats['steps'] for stats in scene_stats.values()) / total_val_samples

                            overall_steps_saved = 20 - total_avg_steps
                            overall_efficiency_gain = (overall_steps_saved / 20) * 100
                            overall_time_saving = overall_steps_saved * (avg_val_time / total_avg_steps)

                            logger.info(f" 总体统计 (基于 {total_val_samples} 个验证样本):")
                            logger.info(f"   平均采样步数: {total_avg_steps:.1f} 步")
                            logger.info(f"   平均推理时间: {avg_val_time:.3f} 秒/样本")
                            logger.info(f"   总体效率提升: {overall_efficiency_gain:.1f}%")
                            logger.info(f"   总验证时间节省: {overall_time_saving * total_val_samples:.2f} 秒")

                            # 吞吐量计算
                            throughput = total_val_samples / total_val_time
                            fixed_throughput = total_val_samples / (total_val_time * (20 / total_avg_steps))
                            throughput_improvement = (throughput - fixed_throughput) / fixed_throughput * 100

                            logger.info(f"   吞吐量: {throughput:.2f} 图像/秒")
                            logger.info(f"   吞吐量提升: {throughput_improvement:.1f}%")

                            # 记录到TensorBoard
                            if tb_logger:
                                tb_logger.add_scalar('efficiency/avg_steps', total_avg_steps, current_step)
                                tb_logger.add_scalar('efficiency/time_saving_per_sample', overall_time_saving,
                                                     current_step)
                                tb_logger.add_scalar('efficiency/overall_gain', overall_efficiency_gain, current_step)
                                tb_logger.add_scalar('efficiency/throughput', throughput, current_step)

                # 保存模型
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info(
                        f'Saving models and training states. (Epoch: {current_epoch}/{total_epochs}, Iter: {current_step}/{n_iter})')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            epoch_pbar.close()

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        total_time = str(timedelta(seconds=int(time.time() - train_start_time)))
        logger.info(f'End of training. Total training epochs: {total_epochs}, Total training time: {total_time}')
    else:
        logger.info('Begin Model Evaluation.')
        bic_mse = 0.0
        bic_psnr = 0.0
        bic_ssim = 0.0
        bic_ergas = 0.0
        bic_lpips = 0.0

        avg_mse = 0.0
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_ergas = 0.0
        avg_lpips = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)

        # === 新增：效率统计 ===
        scene_stats = {
            'urban': {'count': 0, 'time': 0, 'steps': 0, 'psnr': [], 'ssim': []},
            'farmland': {'count': 0, 'time': 0, 'steps': 0, 'psnr': [], 'ssim': []},
            'mountain': {'count': 0, 'time': 0, 'steps': 0, 'psnr': [], 'ssim': []},
            'water': {'count': 0, 'time': 0, 'steps': 0, 'psnr': [], 'ssim': []}
        }
        total_eval_time = 0
        total_eval_samples = 0

        eval_pbar = tqdm(enumerate(val_loader), desc="Evaluating", total=len(val_loader), unit="batch")
        for _, val_data in eval_pbar:
            idx += 1
            diffusion.feed_data(val_data)

            torch.cuda.synchronize()
            start = time.time()
            diffusion.test(continous=True)
            torch.cuda.synchronize()
            end = time.time()
            inference_time = end - start
            eval_pbar.set_postfix(inference_time=f"{inference_time:.2f}s")

            total_eval_time += inference_time
            total_eval_samples += 1

            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])
            lr_img = Metrics.tensor2img(visuals['LR'])
            fake_img = Metrics.tensor2img(visuals['INF'])

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                sr_img = visuals['SR']
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]),
                        '{}/{}_{}_sr_{}.tif'.format(result_path, current_step, idx, iter))
            else:
                sr_img = Metrics.tensor2img(visuals['SR'])
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.tif'.format(result_path, current_step, idx))

            # 计算评估指标
            bmse = compare_mse(fake_img, hr_img)
            bpsnr = compare_psnr(fake_img, hr_img)
            bssim = compare_ssim(fake_img, hr_img, multichannel=True)
            bergas = Metrics.calculate_ergas(fake_img, hr_img, scale=scale)
            blpips = Metrics.calculate_lpips(fake_img, hr_img)
            smse = compare_mse(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            spsnr = compare_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            sssim = compare_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img, multichannel=True)
            sergas = Metrics.calculate_ergas(Metrics.tensor2img(visuals['SR'][-1]), hr_img, scale=scale)
            slpips = Metrics.calculate_lpips(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            # === 在这里添加效率统计（在变量定义后）===
            if hasattr(diffusion.netG, 'current_scene_type'):
                scene_type = diffusion.netG.current_scene_type
                scene_stats[scene_type]['count'] += 1
                scene_stats[scene_type]['time'] += inference_time

                if hasattr(diffusion.netG, 'current_timesteps'):
                    actual_steps = diffusion.netG.current_timesteps
                    scene_stats[scene_type]['steps'] += actual_steps
                    logger.info(f"Sample {idx}: Scene={scene_type}, Steps={actual_steps}, Time={inference_time:.3f}s")

            # === 新增：记录质量指标 ===
            if hasattr(diffusion.netG, 'current_scene_type'):
                scene_type = diffusion.netG.current_scene_type
                scene_stats[scene_type]['psnr'].append(spsnr)
                scene_stats[scene_type]['ssim'].append(sssim)

            # 绘制评估结果
            result_imgs = [hr_img, lr_img, fake_img, Metrics.tensor2img(visuals['SR'][-1])]
            mses = [None, None, bmse, smse]
            psnrs = [None, None, bpsnr, spsnr]
            ssims = [None, None, bssim, sssim]
            ergas = [None, None, bergas, sergas]
            lpips = [None, None, blpips, slpips]
            Metrics.plot_img(
                result_imgs, mses, psnrs, ssims, ergas, lpips,
                '{}/{}_{}_plot.png'.format(result_path, current_step, idx))

            # 累积指标
            bic_mse += bmse
            bic_psnr += bpsnr
            bic_ssim += bssim
            bic_ergas += bergas
            bic_lpips += blpips
            avg_mse += smse
            avg_psnr += spsnr
            avg_ssim += sssim
            avg_ergas += sergas
            avg_lpips += slpips

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, spsnr, sssim)

        eval_pbar.close()

        # 计算平均指标
        bic_mse /= idx
        bic_psnr /= idx
        bic_ssim /= idx
        bic_ergas /= idx
        bic_lpips /= idx

        avg_mse /= idx
        avg_psnr /= idx
        avg_ssim /= idx
        avg_ergas /= idx
        avg_lpips /= idx

        # 记录评估日志
        logger_val = logging.getLogger('val')
        logger_val.info(
            '<epoch:{:3d}, iter:{:8,d}> bic_mse: {:.5e}, bic_psnr: {:.5e}, bic_ssim：{:.5e}, bic_ergas: {:.5e}, bic_lpips: {:.5e}'.format(
                current_epoch, current_step, bic_mse, bic_psnr, bic_ssim, bic_ergas, bic_lpips))
        logger_val.info(
            '<epoch:{:3d}, iter:{:8,d}> sr_mse: {:.5e}, sr_psnr: {:.5e}, sr_ssim：{:.5e}, sr_ergas: {:.5e}, sr_lpips：{:.5e}'.format(
                current_epoch, current_step, avg_mse, avg_psnr, avg_ssim, avg_ergas, avg_lpips))

        # === 新增：评估效率统计报告 ===
        if args.log_efficiency or opt.get('log_efficiency', False):
            logger.info("\n" + "=" * 60)
            logger.info("                 评估效率统计报告")
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

                    # 质量指标
                    avg_psnr = np.mean(scene_stats[scene_type]['psnr']) if scene_stats[scene_type]['psnr'] else 0
                    avg_ssim = np.mean(scene_stats[scene_type]['ssim']) if scene_stats[scene_type]['ssim'] else 0

                    logger.info(f" {scene_type.upper()}场景:")
                    logger.info(f"   样本数: {count}")
                    logger.info(f"   平均步数: {avg_steps:.1f} 步 (节省 {steps_saved:.1f}步)")
                    logger.info(f"   平均时间: {avg_time:.3f} 秒 (节省 {time_saved_per_sample:.3f}秒/样本)")
                    logger.info(f"   效率提升: {efficiency_gain:.1f}%")
                    if avg_psnr > 0:
                        logger.info(f"   平均PSNR: {avg_psnr:.2f}, 平均SSIM: {avg_ssim:.4f}")

            if total_eval_samples > 0:
                avg_eval_time = total_eval_time / total_eval_samples
                total_avg_steps = sum(stats['steps'] for stats in scene_stats.values()) / total_eval_samples

                overall_steps_saved = 20 - total_avg_steps
                overall_efficiency_gain = (overall_steps_saved / 20) * 100
                overall_time_saving = overall_steps_saved * (avg_eval_time / total_avg_steps)

                logger.info(f" 总体统计 (基于 {total_eval_samples} 个评估样本):")
                logger.info(f"   平均采样步数: {total_avg_steps:.1f} 步")
                logger.info(f"   平均推理时间: {avg_eval_time:.3f} 秒/样本")
                logger.info(f"   总体效率提升: {overall_efficiency_gain:.1f}%")
                logger.info(f"   总评估时间节省: {overall_time_saving * total_eval_samples:.2f} 秒")

                # 吞吐量计算
                throughput = total_eval_samples / total_eval_time
                fixed_throughput = total_eval_samples / (total_eval_time * (20 / total_avg_steps))
                throughput_improvement = (throughput - fixed_throughput) / fixed_throughput * 100

                logger.info(f"   吞吐量: {throughput:.2f} 图像/秒")
                logger.info(f"   吞吐量提升: {throughput_improvement:.1f}%")

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
