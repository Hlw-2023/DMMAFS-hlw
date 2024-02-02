# -*- coding: utf-8 -*-
import torch.nn as nn
import network
from ssim_un_official import *
from focalfrequencyloss import FocalFrequencyLoss
import torch
import torch.optim as optim
import torchvision
import os
import torch.nn.functional as F
from contiguous_params import ContiguousParams
import numpy as np
from utils import gradient,gradient_sobel,gradient_Isotropic



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.fusion = network.TransEX()
        self.BCE = nn.BCELoss(reduction='sum')
        if args.contiguousparams == True:
            print("ContiguousParams---")
            parametersF = ContiguousParams(self.fusion.parameters())
            self.optimizer_G = optim.Adam(parametersF.contiguous(), lr=args.lr)
        else:
            self.optimizer_G = optim.Adam(self.fusion.parameters(), lr=args.lr)


        self.loss = torch.zeros(1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.5,
                                                                    patience=2,
                                                                    verbose=True, threshold=0.0001,
                                                                    threshold_mode='rel',
                                                                    cooldown=0, min_lr=0, eps=1e-10)
        self.min_loss = 10000
        self.mean_loss = 0
        self.args = args
        if args.multiGPU:
            self.mulgpus()
        self.load()
        # [pixel  ssim  gradient score]
        self.train_mode = self.args.train_mode
        self.count_batch = 0

    def load(self, ):
        start_epoch = 0
        if self.args.load_pt:
            print("=========LOAD WEIGHTS=========")
            print(self.args.weights_path)
            checkpoint = torch.load(self.args.weights_path)
            start_epoch = checkpoint['epoch'] + 1
            try:
                if self.args.multiGPU:
                    print("load G")
                    self.fusion.load_state_dict(checkpoint['weight'])
                else:
                    print("load G single")
                    # 单卡模型读取多卡模型
                    state_dict = checkpoint['weight']
                    # create new OrderedDict that does not contain `module.`
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace('module.', '')  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.fusion.load_state_dict(new_state_dict)
            except:
                model = self.fusion
                print("weights not same ,try to load part of them")
                model_dict = model.state_dict()
                pretrained = torch.load(self.args.weights_path)['weight']

                pretrained_dict = {k: v for k, v in model_dict.items() if k in pretrained}
                left_dict = {k for k, v in model_dict.items() if k not in pretrained}
                print(left_dict)

                model_dict.update(pretrained_dict)

                model.load_state_dict(model_dict)
                print(len(model_dict), len(pretrained_dict))

            print("start_epoch:", start_epoch)
            print("=========END LOAD WEIGHTS=========")
        print("========START EPOCH: %d=========" % start_epoch)
        self.start_epoch = start_epoch

    def forward(self):
        if self.train_mode == 'fine_tune':
            self.pred = self.fusion(self.seq, self.cmap)



    def backward(self):
        if self.train_mode == 'fine_tune':

            pred = self.pred
            label = self.label

          

            loss = self.BCE(pred, label)



            self.optimizer_G.zero_grad()
            loss.backward()
            self.loss = loss

    def mulgpus(self):
        self.fusion = nn.DataParallel(self.fusion.cuda(), device_ids=self.args.GPUs, output_device=self.args.GPUs[0])


    def setdata_for_fine(self, seq, cmap,label):
        seq = seq.to(self.args.device)
        cmap = cmap.to(self.args.device)
        label = label.to(self.args.device)
        self.seq = seq
        self.cmap = cmap
        self.label = label

    def step(self):
        self.forward()
        self.backward()
        self.optimizer_G.step()
        self.count_batch += 1
        if self.train_mode=='fine_tune':
            self.print = 'Loss:ALL[%.3lf]mean[%.3f] ' % \
                         (self.loss.item(),
                          self.mean_loss/self.count_batch
                          )
        self.mean_loss += self.loss.item()




    def saveimg(self, epoch, num=0):

        if self.train_mode == 'pretrain':
            img1 = self.bright_re[0].cpu()
            img2 = self.fourier_re[0].cpu()
            img3 = self.shuffling_re[0].cpu()
            img = torchvision.utils.make_grid([img1, img2, img3], nrow=3)
            torchvision.utils.save_image(img,
                                         fp=(os.path.join('output/pretrain/result_%d_%d.jpg' % (epoch, num))))
        if self.train_mode == 'fine_tune':
            img1 = self.vi_img[0].cpu()
            img2 = self.ir_img[0].cpu()
            img3 = self.img_re[0].cpu()

            img = torchvision.utils.make_grid([img1, img2, img3], nrow=3)

            torchvision.utils.save_image(img,
                                         fp=(os.path.join(
                                             'output/fine_tune/result_%d_%d.jpg' % (epoch, num))))



    def save_BP(self, epoch):  #不需要动

        self.mean_loss = self.mean_loss/self.count_batch
        if self.min_loss > self.mean_loss:
            if self.train_mode == 'pretrain':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_fusion.pt'))
                # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_D.pt'))
                print('[%d] - Best model is saved -' % (epoch))
                print('mean loss :{%.5f} min:{%.5f}'%(self.mean_loss,self.min_loss))
            if self.train_mode == 'fine_tune':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/h_bp/best_fusion.pt'))
                # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_D.pt'))
                print('[%d] - Best model is saved -' % (epoch))
                print('mean loss :{%.5f} min:{%.5f}'%(self.mean_loss,self.min_loss))
            self.min_loss = self.mean_loss
        if epoch % 1 == 0:
            if self.train_mode == 'pretrain':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/epoch' + str(epoch) + '_fusion.pt'))
            if self.train_mode == 'fine_tune':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/h_bp/epoch' + str(epoch) + '_fusion.pt'))
        self.mean_loss = 0
        self.count_batch = 0


    def save_MF(self, epoch):

        self.mean_loss = self.mean_loss/self.count_batch
        if self.min_loss > self.mean_loss:
            if self.train_mode == 'pretrain':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_fusion.pt'))

                print('[%d] - Best model is saved -' % (epoch))
                print('mean loss :{%.5f} min:{%.5f}'%(self.mean_loss,self.min_loss))
            if self.train_mode == 'fine_tune':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/h_mf/best_fusion.pt'))

                print('[%d] - Best model is saved -' % (epoch))
                print('mean loss :{%.5f} min:{%.5f}'%(self.mean_loss,self.min_loss))
            self.min_loss = self.mean_loss
        if epoch % 1 == 0:
            if self.train_mode == 'pretrain':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/epoch' + str(epoch) + '_fusion.pt'))
            if self.train_mode == 'fine_tune':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/h_mf/epoch' + str(epoch) + '_fusion.pt'))
        self.mean_loss = 0
        self.count_batch = 0

    def save_CC(self, epoch):

        self.mean_loss = self.mean_loss / self.count_batch
        if self.min_loss > self.mean_loss:
            if self.train_mode == 'pretrain':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/best_fusion.pt'))
                # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_D.pt'))
                print('[%d] - Best model is saved -' % (epoch))
                print('mean loss :{%.5f} min:{%.5f}' % (self.mean_loss, self.min_loss))
            if self.train_mode == 'fine_tune':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/h_cc/best_fusion.pt'))
                # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_D.pt'))
                print('[%d] - Best model is saved -' % (epoch))
                print('mean loss :{%.5f} min:{%.5f}' % (self.mean_loss, self.min_loss))
            self.min_loss = self.mean_loss
        if epoch % 1 == 0:
            if self.train_mode == 'pretrain':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/epoch' + str(epoch) + '_fusion.pt'))
            if self.train_mode == 'fine_tune':
                torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                           os.path.join('weights/h_cc/epoch' + str(epoch) + '_fusion.pt'))
        self.mean_loss = 0
        self.count_batch = 0



