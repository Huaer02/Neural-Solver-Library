import torch
import torch.nn.functional as F
from einops import rearrange


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

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

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
        x = rearrange(x, 'b (h w) c -> b h w c', h=s1, w=s2)
        x = F.pad(x,
                  (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
        grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h1)  # f(x+h) - f(x-h) / 2h
        grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h2)  # f(x+h) - f(x-h) / 2h

        return grad_x, grad_y

    def __call__(self, out, y):
        out = rearrange(out, 'b (h w) c -> b c h w', h=self.shapelist[0], w=self.shapelist[1])
        out = out[..., 1:-1, 1:-1].contiguous()
        out = F.pad(out, (1, 1, 1, 1), "constant", 0)
        out = rearrange(out, 'b c h w -> b (h w) c')
        gt_grad_x, gt_grad_y = self.central_diff(y, 1.0 / float(self.shapelist[0]),
                                                 1.0 / float(self.shapelist[1]), self.shapelist[0], self.shapelist[1])
        pred_grad_x, pred_grad_y = self.central_diff(out, 1.0 / float(self.shapelist[0]),
                                                     1.0 / float(self.shapelist[1]), self.shapelist[0],
                                                     self.shapelist[1])
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


class MultiMetricLoss(object):
    """
    综合损失和指标管理类
    支持多种损失函数和指标计算，可以选择任意一个作为主损失函数
    """
    def __init__(self, loss_type='l2', d=2, p=2, size_average=True, reduction=True, shapelist=None):
        """
        Args:
            loss_type: 主损失函数类型 ('l2', 'mae', 'mape', 'rmse', 'deriv')
            其他参数用于L2Loss和DerivLoss的初始化
        """
        self.loss_type = loss_type.lower()
        
        # 初始化各种损失函数
        self.l2_loss = L2Loss(d=d, p=p, size_average=size_average, reduction=reduction)
        if shapelist is not None:
            self.deriv_loss = DerivLoss(d=d, p=p, size_average=size_average, reduction=reduction, shapelist=shapelist)
        else:
            self.deriv_loss = None
            
        # 损失函数映射
        self.loss_functions = {
            'l2': self.l2_loss,
            'mae': mae_loss,
            'mape': mape_loss,
            'rmse': rmse_loss,
            'deriv': self.deriv_loss
        }
        
        # 验证损失函数类型
        if self.loss_type not in self.loss_functions:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        if self.loss_type == 'deriv' and self.deriv_loss is None:
            raise ValueError("DerivLoss requires shapelist parameter")
    
    def compute_all_metrics(self, y_pred, y_true):
        """
        计算所有指标
        Returns:
            dict: 包含所有指标的字典
        """
        metrics = {}
        
        # L2 relative loss
        metrics['l2_rel'] = self.l2_loss(y_pred, y_true)
        
        # MAE
        metrics['mae'] = mae_loss(y_pred, y_true)
        
        # MAPE
        metrics['mape'] = mape_loss(y_pred, y_true)
        
        # RMSE
        metrics['rmse'] = rmse_loss(y_pred, y_true)
        
        # Derivative loss (if available)
        if self.deriv_loss is not None:
            metrics['deriv'] = self.deriv_loss(y_pred, y_true)
        
        return metrics
    
    def __call__(self, y_pred, y_true, return_all_metrics=True):
        """
        计算主损失函数和所有指标
        Args:
            y_pred: 预测值
            y_true: 真实值
            return_all_metrics: 是否返回所有指标
        Returns:
            如果return_all_metrics=True，返回(主损失, 所有指标字典)
            否则只返回主损失
        """
        # 计算主损失
        main_loss = self.loss_functions[self.loss_type](y_pred, y_true)
        
        if return_all_metrics:
            # 计算所有指标
            all_metrics = self.compute_all_metrics(y_pred, y_true)
            return main_loss, all_metrics
        else:
            return main_loss


# 便捷函数，用于创建不同类型的损失函数
def create_loss_function(loss_type='l2', **kwargs):
    """
    创建损失函数的便捷函数
    Args:
        loss_type: 损失函数类型
        **kwargs: 其他参数
    """
    return MultiMetricLoss(loss_type=loss_type, **kwargs)