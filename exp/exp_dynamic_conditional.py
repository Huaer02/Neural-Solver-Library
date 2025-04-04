import os
import torch
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss
import matplotlib.pyplot as plt
from utils.visual import visual
import numpy as np


class Exp_Dynamic_Conditional(Exp_Basic):
    def __init__(self, args):
        super(Exp_Dynamic_Conditional, self).__init__(args)

    def vali(self):
        myloss = L2Loss(size_average=False)
        test_l2_full = 0
        self.model.eval()
        with torch.no_grad():
            for x, time, fx, yy in self.test_loader:
                x, time, fx, yy = x.cuda(), time.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    input_T = time[:, t:t + 1].reshape(x.shape[0], 1)
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx, T=input_T)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                if self.args.normalize:
                    pred = self.dataset.y_normalizer.decode(pred)
                test_l2_full += myloss(pred.reshape(x.shape[0], -1), yy.reshape(x.shape[0], -1)).item()
        test_loss_full = test_l2_full / (self.args.ntest)
        return test_loss_full

    def train(self):
        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else: 
            raise ValueError('Optimizer only AdamW or Adam')
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.args.pct_start)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        myloss = L2Loss(size_average=False)

        for ep in range(self.args.epochs):
            self.model.train()
            train_l2_step = 0

            for pos, time, fx, yy in self.train_loader:
                x, time, fx, yy = pos.cuda(), time.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    y = yy[..., self.args.out_dim * t:self.args.out_dim * (t + 1)]
                    input_T = time[:, t:t + 1].reshape(x.shape[0], 1)
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx, T=input_T)
                    loss = myloss(im.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
                    train_l2_step += loss.item()
                    optimizer.zero_grad()
                    loss.backward()

                    if self.args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()
            if self.args.scheduler == 'CosineAnnealingLR' or self.args.scheduler == 'StepLR':
                scheduler.step()

            train_loss_step = train_l2_step / (self.args.ntrain * float(self.args.T_out))
            print("Epoch {} Train loss step : {:.5f} ".format(ep, train_loss_step))

            test_loss_full = self.vali()
            print("Epoch {} Test loss full : {:.5f}".format(ep, test_loss_full))

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save models')
                torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('final save models')
        torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

    def test(self):
        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        rel_err = 0.0
        id = 0
        myloss = L2Loss(size_average=False)
        with torch.no_grad():
            for x, time, fx, yy in self.test_loader:
                id += 1
                x, time, fx, yy = x.cuda(), time.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                for t in range(self.args.T_out):
                    input_T = time[:, t:t + 1].reshape(x.shape[0], 1)  # B,step
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx, T=input_T)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                if self.args.normalize:
                    pred = self.dataset.y_normalizer.decode(pred)
                rel_err += myloss(pred.reshape(x.shape[0], -1), yy.reshape(x.shape[0], -1)).item()

                if id < self.args.vis_num:
                    print('visual: ', id)
                    visual(yy[:, :, -4:-2], torch.sqrt(yy[:, :, -1:] ** 2 + yy[:, :, -2:-1] ** 2),
                           torch.sqrt(pred[:, :, -1:] ** 2 + pred[:, :, -2:-1] ** 2), self.args, id)

        rel_err /= self.args.ntest
        print("rel_err:{}".format(rel_err))
