import torch
import data as Data
import model_train as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from core.metrics import calculate_IS
from fid_eval import calculate_fid_given_dataset
from brisque_eval import eval_brisque
#from tensorboardX import SummaryWriter
import os
import random
import numpy as np
seed = 6666
print('Random seed :{}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/infer_256.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-infer', '-i', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('-steps', '--steps', type=int, default=20)
    parser.add_argument('-eta', '--eta', type=float, default=0.0)
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    #val_path = "/data/diffusion_data/infer"

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    # Logger.setup_logger('val', val_path, 'infer_val_new', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    #tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

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
    steps = args.steps
    eta = args.eta
    print(steps,eta)
    result_path = '{}'.format(opt['path']['results'])
    fake_path = os.path.join(result_path,'lr_save')
    sr_path = os.path.join(result_path,'sr_save')
    hr_path = os.path.join(result_path,'hr_save')
    print(result_path)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(sr_path, exist_ok=True)
    os.makedirs(hr_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        #diffusion.test(continous=True)
        diffusion.test(continous=True,condition_ddim = True,steps = steps,eta = eta)
        visuals = diffusion.get_current_visuals(need_LR=False)

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
            hr_process = Metrics.tensor2img(visuals['new_HR'])
            Metrics.save_img(
                sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(sr_path, current_step, idx))
            # sr_process_1999 = os.path.join(result_path, '1')
            # sr_process_3 = os.path.join(result_path, '2')
            # sr_process_2 = os.path.join(result_path, '3')
            # os.makedirs(sr_process_1999, exist_ok=True)
            # os.makedirs(sr_process_3, exist_ok=True)
            # os.makedirs(sr_process_2, exist_ok=True)
            # Metrics.save_img(
            #     Metrics.tensor2img(visuals['SR'][2]), '{}/{}_{}_sr.png'.format(sr_process_1999, current_step, idx))
            # Metrics.save_img(
            #     Metrics.tensor2img(visuals['new_HR'][1]), '{}/{}_{}_hr5.png'.format(sr_process_1999, current_step, idx))
            # Metrics.save_img(
            #     Metrics.tensor2img(visuals['SR'][-3]), '{}/{}_{}_sr.png'.format(sr_process_3, current_step, idx))
            # Metrics.save_img(
            #     Metrics.tensor2img(visuals['new_HR'][2]), '{}/{}_{}_hr10.png'.format(sr_process_3, current_step, idx))
            # Metrics.save_img(
            #     Metrics.tensor2img(visuals['SR'][-2]), '{}/{}_{}_sr.png'.format(sr_process_2, current_step, idx))
            # Metrics.save_img(
            #     Metrics.tensor2img(visuals['new_HR'][3]), '{}/{}_{}_hr15.png'.format(sr_process_2, current_step, idx))

        Metrics.save_img(
            hr_img, '{}/{}_{}_hr.png'.format(hr_path, current_step, idx))
        Metrics.save_img(
            fake_img, '{}/{}_{}_inf.png'.format(fake_path, current_step, idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)
    status = 'sr.'
    brisque = eval_brisque(sr_path,status)
    IS = calculate_IS(sr_path)
    paths = [hr_path,sr_path]
    Fid = calculate_fid_given_dataset(paths)
    print("infer: steps = ",steps,",eta = ",eta,";IS: ",IS,",brisque: ",brisque,",Fid: ",Fid)
    logger_val = logging.getLogger('val')  # validation logger
    logger_val.info('<steps:{:4d}, eta:{:.2e}> FID: {:.4e} IS: {:.4e} brisque:{:.4e}'.format(
        steps, eta, Fid, IS, brisque))


    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
