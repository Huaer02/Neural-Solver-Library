import os
import torch
import time
from datetime import datetime
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss, DerivLoss, MultiMetricLoss
import matplotlib.pyplot as plt
from utils.visual import visual
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Exp_Steady(Exp_Basic):
    def __init__(self, args):
        super(Exp_Steady, self).__init__(args)
        
        # 初始化多指标损失计算器
        data_loss_type = getattr(args, 'data_loss_type', 'l2')
        is_multitask = hasattr(args, 'use_multitask') and args.use_multitask
        
        self.metric_calculator = MultiMetricLoss(
            loss_type=data_loss_type,
            is_multitask=is_multitask,
            size_average=False, 
            args=args,
            use_dwa=getattr(args, 'use_dwa', True) if is_multitask else False
        )
        
        # 保留原有的损失函数用于导数损失
        self.l2_loss = L2Loss(size_average=False)
        if self.args.derivloss:
            self.deriv_loss = DerivLoss(size_average=False, shapelist=self.args.shapelist)

    def vali(self):
        vali_start_time = time.time()
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for pos, fx, y in tqdm(self.test_loader, unit="batchs", leave=False):
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x, fx)
                
                if hasattr(self.args, 'use_multitask') and self.args.use_multitask:
                    _, final_pred, _, _ = out
                    out = final_pred
                
                if self.args.normalize:
                    if hasattr(self.dataset, "output_normalizer"):
                        out = self.dataset.output_normalizer.decode(out)
                    elif hasattr(self.dataset, "y_normalizer"):
                        out = self.dataset.y_normalizer.decode(out)

                all_preds.append(out.reshape(x.shape[0], -1).cpu())
                all_targets.append(y.reshape(x.shape[0], -1).cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算各种指标
        test_metrics = self.metric_calculator.compute_data_metrics(all_preds, all_targets)
        rel_err = self.metric_calculator.compute_data_loss_only(all_preds, all_targets) / len(self.test_loader.dataset)
        
        vali_end_time = time.time()
        vali_time = vali_end_time - vali_start_time
        
        return rel_err, test_metrics, vali_time

    def train(self):
        logger.info("=" * 80)
        logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        train_start_time = time.time()
        
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

        for ep in tqdm(range(self.args.epochs), desc="Training Epochs", unit="epoch", leave=True):
            epoch_start_time = time.time()
            
            self.model.train()
            train_loss = 0
            train_metrics_sum = {"l2": 0, "mae": 0, "mape": 0, "rmse": 0}
            total_samples = 0

            for pos, fx, y in tqdm(self.train_loader, unit="batchs", leave=False):
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x, fx)
                
                # 处理多任务输出
                if hasattr(self.args, 'use_multitask') and self.args.use_multitask:
                    res_out, final_pred, mi_loss, club_loss = out
                    res_loss = torch.mean(torch.abs(res_out))
                    out = final_pred
                
                if self.args.normalize:
                    if hasattr(self.dataset, "output_normalizer"):
                        out = self.dataset.output_normalizer.decode(out)
                        y = self.dataset.output_normalizer.decode(y)
                    elif hasattr(self.dataset, "y_normalizer"):
                        out = self.dataset.y_normalizer.decode(out)
                        y = self.dataset.y_normalizer.decode(y)

                # 计算主要损失
                if hasattr(self.args, 'use_multitask') and self.args.use_multitask:
                    loss, loss_dict, step_metrics = self.metric_calculator(
                        out.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1),
                        res_loss, mi_loss, club_loss, return_all_metrics=True
                    )
                else:
                    loss, step_metrics = self.metric_calculator(
                        out.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1), 
                        return_all_metrics=True
                    )
                
                # 添加导数损失（如果启用）
                if self.args.derivloss:
                    deriv_loss_val = self.deriv_loss(out, y)
                    loss = loss + 0.1 * deriv_loss_val

                train_loss += loss.item()
                
                # 累积指标
                for key in train_metrics_sum.keys():
                    train_metrics_sum[key] += step_metrics[key].item() * x.shape[0]
                
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

            train_loss = train_loss / total_samples
            train_metrics_avg = {
                key: value / total_samples for key, value in train_metrics_sum.items()
            }
            
            epoch_train_time = time.time() - epoch_start_time
            
            rel_err, test_metrics, vali_time = self.vali()
            
            epoch_total_time = time.time() - epoch_start_time
            
            logger.info("-" * 80)
            logger.info("Epoch {} Train loss: {:.5e} ({:.8f})".format(ep, train_loss, train_loss))
            logger.info("         Train L2: {:.5e} ({:.8f})".format(train_metrics_avg['l2'], train_metrics_avg['l2']))
            logger.info("         Train MAE: {:.5e} ({:.8f})".format(train_metrics_avg["mae"], train_metrics_avg["mae"]))
            logger.info("         Train MAPE: {:.5e} ({:.8f})".format(train_metrics_avg["mape"], train_metrics_avg["mape"]))
            logger.info("         Train RMSE: {:.5e} ({:.8f})".format(train_metrics_avg["rmse"], train_metrics_avg["rmse"]))
            
            logger.info("Epoch {} Test L2: {:.5e} ({:.8f})".format(ep, rel_err, rel_err))
            logger.info("         Test MAE: {:.5e} ({:.8f})".format(test_metrics["mae"], test_metrics["mae"]))
            logger.info("         Test MAPE: {:.5e} ({:.8f})".format(test_metrics["mape"], test_metrics["mape"]))
            logger.info("         Test RMSE: {:.5e} ({:.8f})".format(test_metrics["rmse"], test_metrics["rmse"]))
            logger.info("         Train Time: {:.2f}s | Vali Time: {:.2f}s | Total Time: {:.2f}s".format(
                epoch_train_time, vali_time, epoch_total_time))
            logger.info("-" * 80)

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                logger.info('save models')
                torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

        total_train_time = time.time() - train_start_time
        
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
        
        test_start_time = time.time()
        
        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        inference_start_time = time.time()
        all_preds = []
        all_targets = []
        id = 0
        
        with torch.no_grad():
            for pos, fx, y in tqdm(self.test_loader, unit="batchs", leave=False):
                id += 1
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x, fx)
                
                if hasattr(self.args, 'use_multitask') and self.args.use_multitask:
                    _, final_pred, _, _ = out
                    out = final_pred
                
                if self.args.normalize:
                    if hasattr(self.dataset, "output_normalizer"):
                        out = self.dataset.output_normalizer.decode(out)
                    elif hasattr(self.dataset, "y_normalizer"):
                        out = self.dataset.y_normalizer.decode(out)
                
                all_preds.append(out.reshape(x.shape[0], -1).cpu())
                all_targets.append(y.reshape(x.shape[0], -1).cpu())
                
                # 可视化前几个样本
                if id <= self.args.vis_num:
                    logger.info('visual: {}'.format(id))
                    visual(x, y, out, self.args, id)

        inference_time = time.time() - inference_start_time
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算最终测试指标
        rel_err = self.metric_calculator.compute_data_loss_only(all_preds, all_targets) / len(self.test_loader.dataset)
        test_metrics = self.metric_calculator.compute_data_metrics(all_preds, all_targets)
        
        total_test_time = time.time() - test_start_time
        
        # 输出最终结果
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