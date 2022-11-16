import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import light_model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
leader_opt = {
"phase": "train",
"distributed": False,
"gpu_ids": [
        2
    ],
"model": {
        "which_model_G": "sr3",
        "finetune_norm": False,
        "unet": {
            "in_channel": 6,
            "out_channel": 3,
            "inner_channel": 64,
            "channel_multiplier": [
                1,
                2,
                4,
                8,
                8
            ],
            "attn_res": [
                16
            ],
            "res_blocks": 2,
            "dropout": 0.2
        },
        "beta_schedule": {
            "train": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2
            }
        },
        "diffusion": {
            "image_size": 256,
            "channels": 3,
            "loss_type": "l2",
            "is_leader": True,
            "conditional": True
        }
    }
}
leader_path = "/data/diffusion_data/save_data/large_scale/new_gen.pth"

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.net_leader = self.set_device(networks.define_G(leader_opt))
        self.set_leader_schedule()
        # gen
        logger.info(
            'load leader model in [{:s}] ...'.format(leader_path))
        self.schedule_phase = None
        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            # self.lossD_optimizer = torch.optim.Adam(list(netD.parameters()), lr=0.0001)
            self.log_dict = OrderedDict()
        network = self.net_leader
        if isinstance(self.net_leader, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, leader_path)
        network.load_state_dict(torch.load(
            leader_path), strict=False)
        self.net_leader.eval()
        self.load_network()
        self.print_network()


    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        x_leader,time,noise = self.net_leader(self.data)
        l_pix = self.netG(self.data,x_leader,time,noise)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        #print("1")
        #print(torch.min(self.data['HR'][0]))
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    # def optimize_parameters(self):
    #     self.optG.zero_grad()
    #     self.lossD_optimizer.zero_grad()
    #     loss,x_noisy,next_x = self.netG(self.data)
    #     # 判别器对于真实图片产生的损失
    #     real_output = netD(x_noisy)  # 判别器输入真实的图片，real_output对真实图片的预测结果
    #     fake_output = netD(next_x.detach())  # 判别器输入生成的图片，fake_output对生成图片的预测;detach会截断梯度，梯度就不会再传递到gen模型中了
    #
    #     g_real_loss = F.binary_cross_entropy(real_output, torch.zeros_like(real_output).float())
    #     g_fake_loss = F.binary_cross_entropy(fake_output, torch.ones_like(fake_output).float())
    #
    #     g_loss = loss + 0.5 * (g_fake_loss.to(loss.device) + g_real_loss.to(loss.device))
    #
    #     d_real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output).float())
    #     d_fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output).float())
    #     d_loss = d_real_loss + d_fake_loss
    #     # 判别器在生成图像上产生的损失
    #     d_loss.backward(retain_graph=True)
    #     # need to average in multi-gpu
    #     b, c, h, w = self.data['HR'].shape
    #     #print("1")
    #     #print(torch.min(self.data['HR'][0]))
    #     l_pix = g_loss.sum()/int(b*c*h*w)
    #     l_pix.backward(retain_graph=True)
    #     # 判别器优化
    #     self.lossD_optimizer.step()
    #     self.optG.step()
    #
    #     # set log
    #     self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False,condition_ddim = False,steps = 2000,eta = 1):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'],self.data['HR'], continous,condition_ddim,steps,eta)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'],self.data['HR'], continous,condition_ddim,steps,eta)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)
    def set_leader_schedule(self):
        schedule_opt = leader_opt['model']['beta_schedule']['train']
        if isinstance(self.net_leader, nn.DataParallel):
            self.netG.module.set_new_noise_schedule(
                schedule_opt, self.device)
        else:
            self.net_leader.set_new_noise_schedule(schedule_opt, self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict
    def get_clock_visuals(self, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            #out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            # network.load_state_dict(torch.load(
            #     gen_path), strict=(not self.opt['model']['finetune_norm']))
            network.load_state_dict(torch.load(
                gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                #从文件中加载一个用torch.save()保存的对象
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
