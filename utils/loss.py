import torch
import torch.nn.functional as F
from einops import rearrange
import logging

logger = logging.getLogger(__name__)


class L2Loss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(L2Loss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class DerivLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, shapelist=None):
        super(DerivLoss, self).__init__()

        assert d > 0 and p > 0
        self.shapelist = shapelist
        self.de_x = L2Loss(d=d, p=p, size_average=size_average, reduction=reduction)
        self.de_y = L2Loss(d=d, p=p, size_average=size_average, reduction=reduction)

    def central_diff(self, x, h1, h2, s1, s2):
        # assuming PBC
        # x: (batch, n, feats), h is the step size, assuming n = h*w
        x = rearrange(x, "b (h w) c -> b h w c", h=s1, w=s2)
        x = F.pad(x, (0, 0, 1, 1, 1, 1), mode="constant", value=0.0)  # [b c t h+2 w+2]
        grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h1)  # f(x+h) - f(x-h) / 2h
        grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h2)  # f(x+h) - f(x-h) / 2h

        return grad_x, grad_y

    def __call__(self, out, y):
        out = rearrange(out, "b (h w) c -> b c h w", h=self.shapelist[0], w=self.shapelist[1])
        out = out[..., 1:-1, 1:-1].contiguous()
        out = F.pad(out, (1, 1, 1, 1), "constant", 0)
        out = rearrange(out, "b c h w -> b (h w) c")
        gt_grad_x, gt_grad_y = self.central_diff(
            y, 1.0 / float(self.shapelist[0]), 1.0 / float(self.shapelist[1]), self.shapelist[0], self.shapelist[1]
        )
        pred_grad_x, pred_grad_y = self.central_diff(
            out, 1.0 / float(self.shapelist[0]), 1.0 / float(self.shapelist[1]), self.shapelist[0], self.shapelist[1]
        )
        deriv_loss = self.de_x(pred_grad_x, gt_grad_x) + self.de_y(pred_grad_y, gt_grad_y)
        return deriv_loss


def mae_loss(y_pred, y_true):
    loss = torch.abs(y_pred - y_true)
    return loss.mean()


def mape_loss(y_pred, y_true):
    # Avoid division by zero
    y_true_safe = torch.where(torch.abs(y_true) < 1e-6, 1e-6, y_true)
    loss = torch.abs((y_pred - y_true) / y_true_safe)
    return loss.mean()


def rmse_loss(y_pred, y_true):
    loss = torch.pow(y_pred - y_true, 2)
    return torch.sqrt(loss.mean())


class DynamicWeightAveraging:
    """动态权重平均 (DWA) 实现"""

    def __init__(self, initial_weights, temperature=2.0, alpha=None):
        """
        Args:
            initial_weights: 初始权重列表 [data_weight, res_weight, orthogonal_weight]
            temperature: 温度参数
            alpha: 平滑系数 (0 < alpha < 1) 越接近 1 越平滑
        """
        self.temperature = temperature
        self.prev_losses = None
        self.alpha = alpha
        if isinstance(initial_weights, list):
            self.current_weights = torch.tensor(initial_weights, dtype=torch.float32).cuda()
        else:
            self.current_weights = initial_weights.clone().cuda()

    def update_weights(self, current_losses):
        """
        根据当前损失更新权重
        Args:
            current_losses: 当前各任务的损失值 (tensor or list)
        Returns:
            updated_weights: 更新后的权重 (tensor)
        """
        if isinstance(current_losses, list):
            current_losses = torch.tensor(current_losses, dtype=torch.float32)

        if self.prev_losses is None:
            # 第一次调用，返回初始权重
            self.prev_losses = current_losses.clone().detach()
            return self.current_weights
        else:
            loss_ratios = abs(current_losses) / (abs(self.prev_losses) + 1e-8)

            num_tasks = len(current_losses)
            dwa_weights = F.softmax(loss_ratios / self.temperature, dim=0) * num_tasks
            if self.alpha is None:
                self.current_weights = dwa_weights
            else:
                self.current_weights = self.alpha * self.current_weights.detach() + (1 - self.alpha) * dwa_weights
            self.prev_losses = current_losses.clone().detach()

            return self.current_weights


class MultiMetricLoss(object):
    """
    统一的损失和指标管理类
    支持单任务和多任务损失计算
    """

    def __init__(
        self,
        loss_type="l2",
        d=2,
        p=2,
        size_average=True,
        reduction=True,
        shapelist=None,
        args=None,
        use_dwa=False,
        is_multitask=False,
    ):
        """
        Args:
            loss_type: 数据损失函数类型 ('l2', 'mae', 'mape', 'rmse', 'deriv')
            is_multitask: 是否为多任务模式
            use_dwa: 是否使用动态权重平均
            args: 参数对象，包含多任务损失权重
        """
        self.loss_type = loss_type.lower()
        self.is_multitask = is_multitask
        self.size_average = size_average
        self.use_dwa = use_dwa
        self.args = args

        self.l2_loss = L2Loss(d=d, p=p, size_average=size_average, reduction=reduction)
        if shapelist is not None:
            self.deriv_loss = DerivLoss(d=d, p=p, size_average=size_average, reduction=reduction, shapelist=shapelist)
        else:
            self.deriv_loss = None

        # 损失函数映射
        self.loss_functions = {
            "l2": self.l2_loss,
            "mae": mae_loss,
            "mape": mape_loss,
            "rmse": rmse_loss,
            "deriv": self.deriv_loss,
        }

        # 验证损失类型
        if self.loss_type not in self.loss_functions:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        if self.loss_type == "deriv" and self.deriv_loss is None:
            raise ValueError("DerivLoss requires shapelist parameter")

        # 如果是多任务模式，初始化多任务相关参数
        if self.is_multitask:
            self._init_multitask_weights()

    def _init_multitask_weights(self):
        """初始化多任务权重和状态"""
        # [data_loss_weight, res_loss_weight, orthogonal_loss_weight, residual_mi_loss_weight]
        # [data_loss_active, res_loss_active, orthogonal_loss_active, residual_mi_loss_active]
        loss_weights = getattr(self.args, "loss_weights", [1.0, 1.0, 1.0, 1.0])
        loss_active = getattr(self.args, "loss_active", [True, True, True, False])

        if len(loss_weights) != 4:
            logger.warning(f"loss_weights length is {len(loss_weights)}, expected 4. Using defaults.")
            loss_weights = [1.0, 1.0, 1.0, 1.0]

        if len(loss_active) != 4:
            logger.warning(f"loss_active length is {len(loss_active)}, expected 4. Using defaults.")
            loss_active = [True, True, True, False]

        self.data_weight = loss_weights[0]
        self.res_weight = loss_weights[1]
        self.orthogonal_weight = loss_weights[2]
        self.residual_mi_weight = loss_weights[3]

        self.data_loss_active = loss_active[0]
        self.res_loss_active = loss_active[1]
        self.orthogonal_loss_active = loss_active[2]
        self.residual_mi_loss_active = loss_active[3]

        # 向后兼容性支持
        if hasattr(self.args, "mi_weight"):
            self.orthogonal_weight = getattr(self.args, "mi_weight", 1.0)
        if hasattr(self.args, "mi_loss_active"):
            self.orthogonal_loss_active = getattr(self.args, "mi_loss_active", True)

        self.dwa_alpha = getattr(self.args, "dwa_alpha", None)
        self.dwa_temperature = getattr(self.args, "dwa_temperature", 2.0)

        self.active_tasks = [
            self.data_loss_active,
            self.res_loss_active,
            self.orthogonal_loss_active,
            self.residual_mi_loss_active,
        ]
        self.num_active_tasks = sum(self.active_tasks)

        logger.info(f"Active tasks: {self.active_tasks}, Number of active tasks: {self.num_active_tasks}")
        logger.info(
            f"Initial loss weights: data={self.data_weight}, res={self.res_weight}, "
            f"orthogonal={self.orthogonal_weight}, residual_mi={self.residual_mi_weight}"
        )

        if self.use_dwa and self.num_active_tasks > 1:
            initial_weights = [self.data_weight, self.res_weight, self.orthogonal_weight, self.residual_mi_weight]
            self.dwa = DynamicWeightAveraging(
                initial_weights=initial_weights, temperature=self.dwa_temperature, alpha=self.dwa_alpha
            )
            logger.info(f"DWA initialized with weights: {initial_weights}")

    def compute_data_loss(self, y_pred, y_true):
        """
        根据指定的损失类型计算数据损失
        Args:
            y_pred: 预测值
            y_true: 真实值
        Returns:
            计算得到的损失值
        """
        loss_fn = self.loss_functions[self.loss_type]
        return loss_fn(y_pred, y_true)

    def compute_data_metrics(self, y_pred, y_true):
        metrics = {}

        # L2 relative loss
        metrics["l2"] = self.l2_loss(y_pred, y_true)

        # MAE
        metrics["mae"] = mae_loss(y_pred, y_true)

        # MAPE
        metrics["mape"] = mape_loss(y_pred, y_true)

        # RMSE
        metrics["rmse"] = rmse_loss(y_pred, y_true)

        # Derivative loss (if available)
        if self.deriv_loss is not None:
            metrics["deriv"] = self.deriv_loss(y_pred, y_true)

        return metrics

    def compute_multitask_loss(self, im, y_true, res_loss, orthogonal_loss, residual_mi_loss):
        """
        计算多任务损失
        Args:
            im: 模型输出
            y_true: 真实值
            res_loss: 残差损失
            orthogonal_loss: 正交损失
            residual_mi_loss: 残差流互信息损失
        Returns:
            total_loss, loss_dict, metrics_dict
        """
        data_loss = self.compute_data_loss(im, y_true)

        all_losses = [data_loss, res_loss, orthogonal_loss, residual_mi_loss]
        loss_names = ["data", "res", "orthogonal", "residual_mi"]

        active_losses = []
        active_names = []
        active_indices = []

        for i, (loss, active, name) in enumerate(zip(all_losses, self.active_tasks, loss_names)):
            if active:
                active_losses.append(loss)
                active_names.append(name)
                active_indices.append(i)

        if len(active_losses) == 0:
            raise ValueError("At least one loss must be active for multitask training")

        if self.use_dwa and len(active_losses) > 1:
            updated_weights = self.dwa.update_weights(torch.stack(active_losses))
            updated_weights = updated_weights.to(data_loss.device)

            weight_attrs = ["data_weight", "res_weight", "orthogonal_weight", "residual_mi_weight"]
            for i, (attr_name, new_weight) in enumerate(zip(weight_attrs, updated_weights)):
                setattr(self, attr_name, new_weight.item())

            final_weights = [updated_weights[i] for i in active_indices]

            logger.debug(
                f"DWA updated weights: data={self.data_weight:.4f}, res={self.res_weight:.4f}, "
                f"orthogonal={self.orthogonal_weight:.4f}, residual_mi={self.residual_mi_weight:.4f}"
            )
        else:
            all_weights = [self.data_weight, self.res_weight, self.orthogonal_weight, self.residual_mi_weight]
            final_weights = [torch.tensor(all_weights[i]).to(data_loss.device) for i in active_indices]

        weighted_losses = [w * loss for w, loss in zip(final_weights, active_losses)]
        total_loss = sum(weighted_losses)

        loss_dict = {
            "total": total_loss,
            "data": data_loss,
            "res": res_loss,
            "orthogonal": orthogonal_loss,
            "residual_mi": residual_mi_loss,
            "data_loss_type": self.loss_type,
            "active_losses": {name: loss for name, loss in zip(active_names, active_losses)},
            "weights": {
                "data": self.data_weight if self.data_loss_active else 0.0,
                "res": self.res_weight if self.res_loss_active else 0.0,
                "orthogonal": self.orthogonal_weight if self.orthogonal_loss_active else 0.0,
                "residual_mi": self.residual_mi_weight if self.residual_mi_loss_active else 0.0,
            },
            "active_status": {
                "data": self.data_loss_active,
                "res": self.res_loss_active,
                "orthogonal": self.orthogonal_loss_active,
                "residual_mi": self.residual_mi_loss_active,
            },
        }

        logger.debug(f"Loss computation details:")
        logger.debug(f"  Total Loss: {total_loss.item():.6f}")
        logger.debug(
            f"  Individual losses - data: {data_loss.item():.6f}, res: {res_loss.item():.6f}, "
            f"orthogonal: {orthogonal_loss.item():.6f}, residual_mi: {residual_mi_loss.item():.6f}"
        )
        logger.debug(
            f"  Current weights - data: {self.data_weight:.4f}, res: {self.res_weight:.4f}, "
            f"orthogonal: {self.orthogonal_weight:.4f}, residual_mi: {self.residual_mi_weight:.4f}"
        )

        metrics_dict = self.compute_data_metrics(im, y_true)

        return total_loss, loss_dict, metrics_dict

    def compute_data_loss_only(self, y_pred, y_true):
        """
        只计算数据损失，用于训练过程中的监控
        使用当前配置的损失类型
        """
        return self.compute_data_loss(y_pred, y_true)

    def __call__(self, *args, return_all_metrics=True, **kwargs):
        """
        统一的调用接口

        单任务模式: loss_fn(y_pred, y_true, return_all_metrics=True)
        多任务模式: loss_fn(im, y_true, res_loss, orthogonal_loss, residual_mi_loss, return_all_metrics=True)
        """
        if self.is_multitask:
            if len(args) != 5:
                raise ValueError(
                    "Multitask mode requires 5 arguments: (im, y_true, res_loss, orthogonal_loss, residual_mi_loss)"
                )

            im, y_true, res_loss, orthogonal_loss, residual_mi_loss = args
            total_loss, loss_dict, metrics_dict = self.compute_multitask_loss(
                im, y_true, res_loss, orthogonal_loss, residual_mi_loss
            )

            if return_all_metrics:
                return total_loss, loss_dict, metrics_dict
            else:
                return total_loss

        else:
            if len(args) != 2:
                raise ValueError("Single task mode requires 2 arguments: (y_pred, y_true)")

            y_pred, y_true = args

            main_loss = self.compute_data_loss(y_pred, y_true)

            if return_all_metrics:
                all_metrics = self.compute_data_metrics(y_pred, y_true)
                return main_loss, all_metrics
            else:
                return main_loss


def create_loss_function(loss_type="l2", is_multitask=False, **kwargs):
    """
    创建损失函数的便捷函数
    Args:
        loss_type: 数据损失函数类型
        is_multitask: 是否为多任务模式
        **kwargs: 其他参数
    """
    return MultiMetricLoss(loss_type=loss_type, is_multitask=is_multitask, **kwargs)
