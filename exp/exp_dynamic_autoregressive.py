import os
import torch
import time
from datetime import datetime
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss, MultiMetricLoss
from utils.model_saver import ModelSaver
import matplotlib.pyplot as plt
from utils.visual import visual
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Exp_Dynamic_Autoregressive(Exp_Basic):
    def __init__(self, args):
        super(Exp_Dynamic_Autoregressive, self).__init__(args)

        data_loss_type = getattr(args, "data_loss_type", "l2")
        self.use_multitask = getattr(args, "use_multitask", False)

        self.metric_calculator = MultiMetricLoss(
            loss_type=data_loss_type,
            is_multitask=self.use_multitask,
            size_average=False,
            args=args,
            use_dwa=getattr(args, "use_dwa", True) if self.use_multitask else False,
        )
        
        # 初始化ModelSaver，监控测试集L2损失
        self.model_saver = ModelSaver(
            save_dir="./checkpoints",
            save_name=self.args.save_name,
            monitor_metric="test_l2",
            mode="min",
            patience=float('inf'),  # 不使用早停机制
            verbose=True
        )

    def vali(self, use_best_model=False):
        """
        验证方法
        Args:
            use_best_model: 是否使用最佳模型进行验证
        """
        if use_best_model:
            try:
                # 保存当前模型状态
                current_state = self.model.state_dict().copy()
                # 加载最佳模型
                self.model_saver.load_best_model(self.model, map_location="cuda")
                logger.info("使用最佳模型进行验证")
            except FileNotFoundError:
                logger.warning("未找到最佳模型，使用当前模型进行验证")
                use_best_model = False
        
        vali_start_time = time.time()
        test_l2_full = 0
        test_metrics_full = {"mae": 0, "mape": 0, "rmse": 0}

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, fx, yy in tqdm(self.test_loader, unit="batchs", leave=False):
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx)
                    if self.use_multitask:
                        _, final_pred, _, _ = im
                        im = final_pred
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    fx = torch.cat((fx[..., self.args.out_dim :], im), dim=-1)

                if self.args.normalize:
                    pred = self.dataset.y_normalizer.decode(pred)

                all_preds.append(pred.reshape(x.shape[0], -1).cpu())
                all_targets.append(yy.reshape(x.shape[0], -1).cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        test_metrics_full = self.metric_calculator.compute_data_metrics(all_preds, all_targets)
        test_l2_full = self.metric_calculator.compute_data_loss_only(all_preds, all_targets) / len(
            self.test_loader.dataset
        )

        vali_end_time = time.time()
        vali_time = vali_end_time - vali_start_time

        # 如果使用了最佳模型进行验证，恢复原来的模型状态
        if use_best_model:
            self.model.load_state_dict(current_state)

        return test_l2_full, test_metrics_full, vali_time

    def train(self):
        logger.info("=" * 80)
        logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        train_start_time = time.time()

        if self.args.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError("Optimizer only AdamW or Adam")

        if self.args.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.lr,
                epochs=self.args.epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=self.args.pct_start,
            )
        elif self.args.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        for ep in tqdm(range(self.args.epochs), desc="Training Epochs", unit="epoch", leave=True):
            epoch_start_time = time.time()

            self.model.train()
            train_l2_step = 0
            train_l2_full = 0
            train_metrics_sum = {"l2": 0, "mae": 0, "mape": 0, "rmse": 0}
            total_samples = 0
            for pos, fx, yy in tqdm(self.train_loader, unit="batchs", leave=False):
                loss = 0
                x, fx, yy = pos.cuda(), fx.cuda(), yy.cuda()

                for t in range(self.args.T_out):
                    y = yy[..., self.args.out_dim * t : self.args.out_dim * (t + 1)]
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx)

                    if self.use_multitask:
                        # im -> (res_out, final_pred, orthogonal_loss, residual_mi_loss)
                        res_out, final_pred, orthogonal_loss, residual_mi_loss = im
                        res_loss = torch.mean(torch.abs(res_out))
                        im = final_pred

                        step_loss, _, step_metrics = self.metric_calculator(
                            im, y, res_loss, orthogonal_loss, residual_mi_loss
                        )
                    else:
                        step_loss, step_metrics = self.metric_calculator(im, y)

                    loss += step_loss

                    for key in train_metrics_sum.keys():
                        train_metrics_sum[key] += step_metrics[key].item() * x.shape[0]

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    if self.args.teacher_forcing:
                        fx = torch.cat((fx[..., self.args.out_dim :], y), dim=-1)
                    else:
                        fx = torch.cat((fx[..., self.args.out_dim :], im), dim=-1)

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

                if self.args.scheduler == "OneCycleLR":
                    scheduler.step()
            logger.info(f"Epoch {ep} Learning rate: {optimizer.param_groups[0]['lr']}")
            if self.args.scheduler == "CosineAnnealingLR" or self.args.scheduler == "StepLR":
                scheduler.step()

            train_loss_step = train_l2_step / (total_samples * float(self.args.T_out))
            train_loss_full = train_l2_full / total_samples

            train_metrics_avg = {
                key: value / (total_samples * float(self.args.T_out)) for key, value in train_metrics_sum.items()
            }
            train_metrics_avg["l2"] /= total_samples / float(self.args.T_out)

            epoch_train_time = time.time() - epoch_start_time

            test_loss_full, test_metrics_full, vali_time = self.vali()

            epoch_total_time = time.time() - epoch_start_time
            logger.info("-" * 80)
            logger.info(
                "Epoch {} Train loss step: {:.5e} ({:.8f}) Train loss full: {:.5e} ({:.8f})".format(
                    ep, train_loss_step, train_loss_step, train_loss_full, train_loss_full
                )
            )
            logger.info("         Train L2: {:.5e} ({:.8f})".format(train_metrics_avg["l2"], train_metrics_avg["l2"]))
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

            # 使用ModelSaver管理模型保存
            metrics_dict = {
                "test_l2": test_loss_full,
                "test_mae": test_metrics_full["mae"],
                "test_mape": test_metrics_full["mape"],
                "test_rmse": test_metrics_full["rmse"],
                "train_l2": train_metrics_avg["l2"],
                "train_mae": train_metrics_avg["mae"],
                "train_mape": train_metrics_avg["mape"],
                "train_rmse": train_metrics_avg["rmse"]
            }
            
            # 更新最佳模型
            is_best = self.model_saver.update(self.model, metrics_dict, ep)

        total_train_time = time.time() - train_start_time

        logger.info("=" * 80)
        logger.info(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total training time: {total_train_time:.2f}s ({total_train_time/3600:.2f}h)")
        logger.info(f"Average time per epoch: {total_train_time/self.args.epochs:.2f}s")
        
        # 打印最佳模型信息
        best_summary = self.model_saver.get_summary()
        logger.info("-" * 40)
        logger.info("Best Model Summary:")
        logger.info(f"Best {best_summary['monitor_metric']}: {best_summary['best_score']:.6f}")
        logger.info(f"Best epoch: {best_summary['best_epoch']}")
        logger.info(f"Best model path: {best_summary['best_model_path']}")
        logger.info("=" * 80)

    def test(self):
        logger.info("=" * 80)
        logger.info(f"Testing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        test_start_time = time.time()

        # 加载最佳模型进行测试
        try:
            self.model_saver.load_best_model(self.model, map_location="cuda")
            logger.info("使用最佳模型进行测试")
        except FileNotFoundError:
            # 如果没有最佳模型，尝试加载传统命名的模型
            traditional_path = "./checkpoints/" + self.args.save_name + ".pt"
            if os.path.exists(traditional_path):
                self.model.load_state_dict(torch.load(traditional_path))
                logger.info(f"使用传统保存的模型进行测试: {traditional_path}")
            else:
                logger.error("未找到可用的模型文件进行测试")
                return
        
        self.model.eval()
        if not os.path.exists("./results/" + self.args.save_name + "/"):
            os.makedirs("./results/" + self.args.save_name + "/")

        rel_err = 0.0
        test_metrics = {"mae": 0, "mape": 0, "rmse": 0}
        id = 0

        inference_start_time = time.time()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, fx, yy in tqdm(self.test_loader, unit="batchs", leave=False):
                id += 1
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx)
                    if self.use_multitask:
                        _, final_pred, _, _ = im
                        im = final_pred
                    fx = torch.cat((fx[..., self.args.out_dim :], im), dim=-1)

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

        inference_time = time.time() - inference_start_time

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        rel_err = self.metric_calculator.compute_data_loss_only(all_preds, all_targets) / len(self.test_loader.dataset)

        test_metrics = self.metric_calculator.compute_data_metrics(all_preds, all_targets)

        total_test_time = time.time() - test_start_time

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
