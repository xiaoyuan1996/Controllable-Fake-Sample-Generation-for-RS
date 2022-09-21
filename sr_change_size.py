import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
#from tensorboardX import SummaryWriter
import os
import numpy as np
import copy

from fid_eval import calculate_fid_given_paths


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.cache = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / (.0001 + self.count)

        self.cache.append(self.val)
        if len(self.cache) >= 20: self.cache = self.cache[1:]
        self.avg = np.mean(self.cache)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/multiple_256.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-infer', '-i', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')


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
    #tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    change_sizes = opt["change_sizes"]


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

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    logger.info("change rate:" + "".join(["{}:{} ".format(k,v) for k,v in change_sizes.items()]))

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    # ave
    #ave_loss = AverageMeter()

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        if current_step == 0:
            change_size_idx = 0
        else:
            change_size_idx = 0
            try:
                while current_step >= int(float(list(change_sizes.keys())[change_size_idx]) * n_iter) and change_size_idx < len(list(change_sizes.keys())):
                    change_size_idx += 1
            except:
                pass
            change_size_idx -= 1

        while current_step < n_iter:

            # reset train_loader
            if current_step >= int(float(list(change_sizes.keys())[change_size_idx]) * n_iter) and change_size_idx < len(list(change_sizes.keys())):

                # print("current step: {}".format(current_step))
                logger.info('reset train_loader')
                resize_resolu =  change_sizes[list(change_sizes.keys())[change_size_idx]]
                train_dataset_opt = copy.deepcopy(opt['datasets']['train'])
                # print("src: {},{},{}".format(train_dataset_opt["l_resolution"], train_dataset_opt["r_resolution"], train_dataset_opt["batch_size"]))

                train_dataset_opt["l_resolution"], train_dataset_opt["r_resolution"] = resize_resolu, resize_resolu

                # train_dataset_opt["batch_size"] = int(train_dataset_opt["batch_size"] * (change_sizes[list(change_sizes.keys())[-1]] / resize_resolu))

                logger.info('reset train_loader: l_resolution:{}, r_resolution:{}, batch_size:{}'.format(train_dataset_opt["l_resolution"], train_dataset_opt["r_resolution"], train_dataset_opt["batch_size"]))

                train_set = Data.create_dataset(train_dataset_opt, 'train')
                train_loader = Data.create_dataloader(train_set, train_dataset_opt, 'train')

                logger.info('reset train_loader finished .')
                change_size_idx += 1

            # current_step += 1
            # continue

            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        ave_loss.update(v)
                        message += '{:s}: {:.4e} ({:.4e})'.format(k, v, ave_loss.avg)
                        #tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_is = 0.0
                    avg_brisuqe = 0.0
                    fid = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
                        # cv2.imwrite('{}/{}_{}_hr1.png'.format(result_path, current_step, idx), hr_img)
                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        # tb_logger.add_image(
                        # 'Iter_{}'.format(current_step),
                        # np.transpose(np.concatenate(
                        # (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                        # idx)
                        # avg_is += Metrics.calculate_IS(sr_img)
                        path1 = '{}/{}_{}_hr.png'.format(result_path, current_step, idx)
                        path2 = '{}/{}_{}_sr.png'.format(result_path, current_step, idx)
                        fid += calculate_fid_given_paths(path1, path2)
                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_is = 1.0
                    avg_fid = fid / idx
                    avg_brisuqe = Metrics.eval_brisque(result_path, "sr.")
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info(
                        '# Validation # FID: {:.4e} IS: {:.4e} Brisque:{:.4e} '.format(avg_fid, avg_is, avg_brisuqe))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> FID: {:.4e} IS: {:.4e} brisque:{:.4e}'.format(
                        current_epoch, current_step, avg_fid, avg_is, avg_brisuqe))
                    # tensorboard logger
                    # tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_is': avg_is,
                            'validation/avg_fid': avg_fid,
                            'validation/val_step': val_step
                        })
                        val_step += 1

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
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
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
