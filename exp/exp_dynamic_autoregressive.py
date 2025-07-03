import os
import torch
import time
from datetime import datetime
from exp.exp_basic import Exp_Basic, count_parameters_in_logger
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss, MultiMetricLoss
import matplotlib.pyplot as plt
from utils.visual import visual
import numpy as np

import logging


class Exp_Dynamic_Autoregressive(Exp_Basic):
    def __init__(self, args):
        super(Exp_Dynamic_Autoregressive, self).__init__(args)

        self.metric_calculator = MultiMetricLoss(loss_type="l2", size_average=False)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(f"./log/{args.save_name}.log")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        self.logger = logger

        self.logger.info(self.args)
        self.logger.info(self.model)
        count_parameters_in_logger(self.model, self.logger)

    def vali(self):
        vali_start_time = time.time()

        myloss = L2Loss(size_average=False)
        test_l2_full = 0
        test_metrics_full = {"mae": 0, "mape": 0, "rmse": 0}

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, fx, yy in self.test_loader:
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx)
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

        test_l2_full = myloss(all_preds, all_targets).item() / len(self.test_loader.dataset)

        _, test_metrics_full = self.metric_calculator(all_preds, all_targets, return_all_metrics=True)

        vali_end_time = time.time()
        vali_time = vali_end_time - vali_start_time

        return test_l2_full, test_metrics_full, vali_time

    def train(self):
        self.logger.info("=" * 80)
        self.logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

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

        myloss = L2Loss(size_average=False)

        for ep in range(self.args.epochs):
            epoch_start_time = time.time()

            self.model.train()
            train_l2_step = 0
            train_l2_full = 0
            train_metrics_sum = {"mae": 0, "mape": 0, "rmse": 0}
            total_samples = 0

            for pos, fx, yy in self.train_loader:
                loss = 0
                x, fx, yy = pos.cuda(), fx.cuda(), yy.cuda()

                for t in range(self.args.T_out):
                    y = yy[..., self.args.out_dim * t : self.args.out_dim * (t + 1)]
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx)

                    step_loss = myloss(im.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
                    loss += step_loss

                    _, step_metrics = self.metric_calculator(
                        im.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1), return_all_metrics=True
                    )
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
                train_l2_full += myloss(pred.reshape(x.shape[0], -1), yy.reshape(x.shape[0], -1)).item()

                total_samples += x.shape[0]

                optimizer.zero_grad()
                loss.backward()

                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                if self.args.scheduler == "OneCycleLR":
                    scheduler.step()

            if self.args.scheduler == "CosineAnnealingLR" or self.args.scheduler == "StepLR":
                scheduler.step()

            train_loss_step = train_l2_step / (total_samples * float(self.args.T_out))
            train_loss_full = train_l2_full / total_samples

            # 需要在时间维度上也平均
            train_metrics_avg = {
                key: value / (total_samples * float(self.args.T_out)) for key, value in train_metrics_sum.items()
            }

            epoch_train_time = time.time() - epoch_start_time

            test_loss_full, test_metrics_full, vali_time = self.vali()

            epoch_total_time = time.time() - epoch_start_time

            self.logger.info(
                "Epoch {} Train L2 step: {:.5e} ({:.8f}) Train L2 full: {:.5e} ({:.8f})".format(
                    ep, train_loss_step, train_loss_step, train_loss_full, train_loss_full
                )
            )
            self.logger.info(
                "         Train MAE: {:.5e} ({:.8f})".format(train_metrics_avg["mae"], train_metrics_avg["mae"])
            )
            self.logger.info(
                "         Train MAPE: {:.5e} ({:.8f})".format(train_metrics_avg["mape"], train_metrics_avg["mape"])
            )
            self.logger.info(
                "         Train RMSE: {:.5e} ({:.8f})".format(train_metrics_avg["rmse"], train_metrics_avg["rmse"])
            )

            self.logger.info("Epoch {} Test L2 full: {:.5e} ({:.8f})".format(ep, test_loss_full, test_loss_full))
            self.logger.info(
                "         Test MAE full: {:.5e} ({:.8f})".format(test_metrics_full["mae"], test_metrics_full["mae"])
            )
            self.logger.info(
                "         Test MAPE full: {:.5e} ({:.8f})".format(test_metrics_full["mape"], test_metrics_full["mape"])
            )
            self.logger.info(
                "         Test RMSE full: {:.5e} ({:.8f})".format(test_metrics_full["rmse"], test_metrics_full["rmse"])
            )
            self.logger.info(
                "         Train Time: {:.2f}s | Vali Time: {:.2f}s | Total Time: {:.2f}s".format(
                    epoch_train_time, vali_time, epoch_total_time
                )
            )
            self.logger.info("-" * 80)

            if ep % 100 == 0:
                if not os.path.exists("./checkpoints"):
                    os.makedirs("./checkpoints")
                self.logger.info("save models")
                torch.save(self.model.state_dict(), os.path.join("./checkpoints", self.args.save_name + ".pt"))

        total_train_time = time.time() - train_start_time

        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        self.logger.info("final save models")
        torch.save(self.model.state_dict(), os.path.join("./checkpoints", self.args.save_name + ".pt"))

        self.logger.info("=" * 80)
        self.logger.info(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total training time: {total_train_time:.2f}s ({total_train_time/3600:.2f}h)")
        self.logger.info(f"Average time per epoch: {total_train_time/self.args.epochs:.2f}s")
        self.logger.info("=" * 80)

    def test(self):
        self.logger.info("=" * 80)
        self.logger.info(f"Testing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

        test_start_time = time.time()

        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists("./results/" + self.args.save_name + "/"):
            os.makedirs("./results/" + self.args.save_name + "/")

        rel_err = 0.0
        test_metrics = {"mae": 0, "mape": 0, "rmse": 0}
        id = 0
        myloss = L2Loss(size_average=False)

        inference_start_time = time.time()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, fx, yy in self.test_loader:
                id += 1
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx)
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

        rel_err = myloss(all_preds, all_targets).item()
        rel_err /= len(self.test_loader.dataset)

        _, test_metrics = self.metric_calculator(all_preds, all_targets, return_all_metrics=True)

        total_test_time = time.time() - test_start_time

        self.logger.info("=" * 80)
        self.logger.info("Final Test Results:")
        self.logger.info("L2 Relative Error: {:.5e} ({:.8f})".format(rel_err, rel_err))
        self.logger.info("MAE: {:.5e} ({:.8f})".format(test_metrics["mae"], test_metrics["mae"]))
        self.logger.info("MAPE: {:.5e} ({:.8f})".format(test_metrics["mape"], test_metrics["mape"]))
        self.logger.info("RMSE: {:.5e} ({:.8f})".format(test_metrics["rmse"], test_metrics["rmse"]))
        self.logger.info("-" * 40)
        self.logger.info("Time Statistics:")
        self.logger.info(f"Total test time: {total_test_time:.2f}s")
        self.logger.info(f"Pure inference time: {inference_time:.2f}s")
        self.logger.info(f"Average time per sample: {inference_time/len(self.test_loader.dataset):.4f}s")
        self.logger.info(f"Samples per second: {len(self.test_loader.dataset)/inference_time:.2f}")
        self.logger.info(f"Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
