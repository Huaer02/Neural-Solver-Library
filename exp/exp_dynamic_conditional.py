import os
import torch
import time as time_module
from datetime import datetime
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss, MultiMetricLoss
import matplotlib.pyplot as plt
from utils.visual import visual
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Exp_Dynamic_Conditional(Exp_Basic):
    def __init__(self, args):
        super(Exp_Dynamic_Conditional, self).__init__(args)

        data_loss_type = getattr(args, 'data_loss_type', 'l2')
        
        is_multitask = hasattr(args, 'use_multitask') and args.use_multitask
        
        self.metric_calculator = MultiMetricLoss(
            loss_type=data_loss_type,
            is_multitask=is_multitask,
            size_average=False, 
            args=args,
            use_dwa=getattr(args, 'use_dwa', True) if is_multitask else False
        )

    def vali(self):
        vali_start_time = time_module.time()
        
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, time_data, fx, yy in tqdm(self.test_loader, unit="batchs", leave=False):
                x, time_data, fx, yy = x.cuda(), time_data.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    input_T = time_data[:, t:t + 1].reshape(x.shape[0], 1)
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx, T=input_T)
                    
                    if hasattr(self.args, 'use_multitask') and self.args.use_multitask:
                        _, final_pred, _, _ = im
                        im = final_pred
                        
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                        
                if self.args.normalize:
                    if hasattr(self.dataset, "output_normalizer"):
                        pred = self.dataset.output_normalizer.decode(pred)
                    elif hasattr(self.dataset, "y_normalizer"):
                        pred = self.dataset.y_normalizer.decode(pred)

                all_preds.append(pred.reshape(x.shape[0], -1).cpu())
                all_targets.append(yy.reshape(x.shape[0], -1).cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        test_metrics_full = self.metric_calculator.compute_data_metrics(all_preds, all_targets)
        test_l2_full = self.metric_calculator.compute_data_loss_only(all_preds, all_targets) / len(self.test_loader.dataset)

        vali_end_time = time_module.time()
        vali_time = vali_end_time - vali_start_time

        return test_l2_full, test_metrics_full, vali_time

    def train(self):
        logger.info("=" * 80)
        logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        train_start_time = time_module.time()

        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else: 
            raise ValueError('Optimizer only AdamW or Adam')
            
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.args.lr, 
                epochs=self.args.epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=self.args.pct_start
            )
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        for ep in tqdm(range(self.args.epochs), desc="Training Epochs", unit="epoch", leave=True):
            epoch_start_time = time_module.time()
            
            self.model.train()
            train_l2_step = 0
            train_l2_full = 0
            train_metrics_sum = {"l2": 0, "mae": 0, "mape": 0, "rmse": 0}
            total_samples = 0

            for pos, time_data, fx, yy in tqdm(self.train_loader, unit="batchs", leave=False):
                loss = 0
                x, time_data, fx, yy = pos.cuda(), time_data.cuda(), fx.cuda(), yy.cuda()

                for t in range(self.args.T_out):
                    y = yy[..., self.args.out_dim * t:self.args.out_dim * (t + 1)]
                    input_T = time_data[:, t:t + 1].reshape(x.shape[0], 1)
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx, T=input_T)
                    
                    if hasattr(self.args, 'use_multitask') and self.args.use_multitask:
                        # im -> (res_out, final_pred, mi_loss, club_loss)
                        res_out, final_pred, mi_loss, club_loss = im
                        res_loss = torch.mean(torch.abs(res_out))
                        im = final_pred

                        step_loss, loss_dict, step_metrics = self.metric_calculator(
                            im.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1),
                            res_loss, mi_loss, club_loss, return_all_metrics=True
                        )
                    else:
                        step_loss, step_metrics = self.metric_calculator(
                            im.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1), 
                            return_all_metrics=True
                        )

                    loss += step_loss

                    for key in train_metrics_sum.keys():
                        train_metrics_sum[key] += step_metrics[key].item() * x.shape[0]

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                train_l2_step += loss.item()
                train_l2_full += self.metric_calculator.compute_data_loss_only(
                    pred.reshape(x.shape[0], -1), yy.reshape(x.shape[0], -1)
                ).item()

                total_samples += x.shape[0]

                optimizer.zero_grad()
                loss.backward()

                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()
                    
            logger.info(f"Epoch {ep} Learning rate: {optimizer.param_groups[0]['lr']}")
            if self.args.scheduler == 'CosineAnnealingLR' or self.args.scheduler == 'StepLR':
                scheduler.step()

            train_loss_step = train_l2_step / (total_samples * float(self.args.T_out))
            train_loss_full = train_l2_full / total_samples

            train_metrics_avg = {
                key: value / (total_samples * float(self.args.T_out)) for key, value in train_metrics_sum.items()
            }
            train_metrics_avg['l2'] = train_metrics_avg['l2'] / (total_samples / float(self.args.T_out))

            epoch_train_time = time_module.time() - epoch_start_time

            test_loss_full, test_metrics_full, vali_time = self.vali()

            epoch_total_time = time_module.time() - epoch_start_time
            
            logger.info("-" * 80)
            logger.info(
                "Epoch {} Train loss step: {:.5e} ({:.8f}) Train loss full: {:.5e} ({:.8f})".format(
                    ep, train_loss_step, train_loss_step, train_loss_full, train_loss_full
                )
            )
            logger.info(
                "         Train L2: {:.5e} ({:.8f})".format(train_metrics_avg['l2'], train_metrics_avg['l2'])
            )
            logger.info(
                "         Train MAE: {:.5e} ({:.8f})".format(train_metrics_avg["mae"], train_metrics_avg["mae"])
            )
            logger.info(
                "         Train MAPE: {:.5e} ({:.8f})".format(train_metrics_avg["mape"], train_metrics_avg["mape"])
            )
            logger.info(
                "         Train RMSE: {:.5e} ({:.8f})".format(train_metrics_avg["rmse"], train_metrics_avg["rmse"])
            )

            logger.info("Epoch {} Test L2 full: {:.5e} ({:.8f})".format(ep, test_loss_full, test_loss_full))
            logger.info(
                "         Test MAE full: {:.5e} ({:.8f})".format(test_metrics_full["mae"], test_metrics_full["mae"])
            )
            logger.info(
                "         Test MAPE full: {:.5e} ({:.8f})".format(test_metrics_full["mape"], test_metrics_full["mape"])
            )
            logger.info(
                "         Test RMSE full: {:.5e} ({:.8f})".format(test_metrics_full["rmse"], test_metrics_full["rmse"])
            )
            logger.info(
                "         Train Time: {:.2f}s | Vali Time: {:.2f}s | Total Time: {:.2f}s".format(
                    epoch_train_time, vali_time, epoch_total_time
                )
            )
            logger.info("-" * 80)

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                logger.info('save models')
                torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

        total_train_time = time_module.time() - train_start_time

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        logger.info('final save models')
        torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

        logger.info("=" * 80)
        logger.info(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total training time: {total_train_time:.2f}s ({total_train_time/3600:.2f}h)")
        logger.info(f"Average time per epoch: {total_train_time/self.args.epochs:.2f}s")
        logger.info("=" * 80)

    def test(self):
        logger.info("=" * 80)
        logger.info(f"Testing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        test_start_time = time_module.time()

        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        inference_start_time = time_module.time()
        all_preds = []
        all_targets = []
        id = 0

        with torch.no_grad():
            for x, time_data, fx, yy in tqdm(self.test_loader, unit="batchs", leave=False):
                id += 1
                x, time_data, fx, yy = x.cuda(), time_data.cuda(), fx.cuda(), yy.cuda()
                
                for t in range(self.args.T_out):
                    input_T = time_data[:, t:t + 1].reshape(x.shape[0], 1)
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx, T=input_T)
                    
                    if hasattr(self.args, 'use_multitask') and self.args.use_multitask:
                        _, final_pred, _, _ = im
                        im = final_pred
                        
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                        
                if self.args.normalize:
                    if hasattr(self.dataset, "output_normalizer"):
                        pred = self.dataset.output_normalizer.decode(pred)
                    elif hasattr(self.dataset, "y_normalizer"):
                        pred = self.dataset.y_normalizer.decode(pred)

                all_preds.append(pred.reshape(x.shape[0], -1).cpu())
                all_targets.append(yy.reshape(x.shape[0], -1).cpu())

                # 可视化前几个样本
                if id <= self.args.vis_num:
                    logger.info('visual: {}'.format(id))
                    visual(yy[:, :, -4:-2], torch.sqrt(yy[:, :, -1:] ** 2 + yy[:, :, -2:-1] ** 2),
                           torch.sqrt(pred[:, :, -1:] ** 2 + pred[:, :, -2:-1] ** 2), self.args, id)

        inference_time = time_module.time() - inference_start_time

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        rel_err = self.metric_calculator.compute_data_loss_only(all_preds, all_targets) / len(self.test_loader.dataset)
        test_metrics = self.metric_calculator.compute_data_metrics(all_preds, all_targets)

        total_test_time = time_module.time() - test_start_time

        logger.info("=" * 80)
        logger.info("Final Test Results:")
        logger.info("L2 Relative Error: {:.5e} ({:.8f})".format(rel_err, rel_err))
        logger.info("MAE: {:.5e} ({:.8f})".format(test_metrics["mae"], test_metrics["mae"]))
        logger.info("MAPE: {:.5e} ({:.8f})".format(test_metrics["mape"], test_metrics["mape"]))
        logger.info("RMSE: {:.5e} ({:.8f})".format(test_metrics["rmse"], test_metrics["rmse"]))
        logger.info("-" * 40)
        logger.info("Time Statistics:")
        logger.info(f"Total test time: {total_test_time:.2f}s")
        logger.info(f"Pure inference time: {inference_time:.2f}s")
        logger.info(f"Average time per sample: {inference_time/len(self.test_loader.dataset):.4f}s")
        logger.info(f"Samples per second: {len(self.test_loader.dataset)/inference_time:.2f}")
        logger.info(f"Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)