import torch
import torch.nn as nn
from .mi_club import CLUBMean, CLUBSample


class MultiBranchMIMinimizer(nn.Module):
    """
    多分支互信息最小化模块
    用于最小化多个分支信号之间的互信息
    """

    def __init__(self, signal_dim, num_branches, hidden_size=None, estimator_type="CLUBMean", lambda_mi=0.01):
        """
        Args:
            signal_dim: 每个分支信号的维度
            num_branches: 分支数量
            hidden_size: CLUB估计器的隐藏层大小
            estimator_type: 'CLUBMean' 或 'CLUBSample'
            lambda_mi: MI损失的缩放因子，用于控制数量级，默认0.01
        """
        super().__init__()
        self.num_branches = num_branches
        self.signal_dim = signal_dim
        self.lambda_mi = lambda_mi

        # 为每对分支创建一个CLUB估计器
        self.estimators = nn.ModuleDict()

        if estimator_type == "CLUBMean":
            EstimatorClass = CLUBMean
        elif estimator_type == "CLUBSample":
            EstimatorClass = CLUBSample
        else:
            raise ValueError("estimator_type must be 'CLUBMean' or 'CLUBSample'")

        # 为每对分支创建互信息估计器
        for i in range(num_branches):
            for j in range(i + 1, num_branches):
                key = f"branch_{i}_{j}"
                if estimator_type == "CLUBMean":
                    self.estimators[key] = EstimatorClass(signal_dim, signal_dim, hidden_size)
                else:  # CLUBSample
                    if hidden_size is None:
                        hidden_size = signal_dim * 2
                    self.estimators[key] = EstimatorClass(signal_dim, signal_dim, hidden_size)

    def forward(self, branch_signals):
        """
        Args:
            branch_signals: List of tensors, 每个tensor形状为[batch_size, signal_dim]

        Returns:
            total_mi: 所有分支对之间的总互信息
            mi_dict: 每对分支之间的互信息字典
        """
        if len(branch_signals) != self.num_branches:
            raise ValueError(f"Expected {self.num_branches} branch signals, got {len(branch_signals)}")

        total_mi = 0.0
        mi_dict = {}

        for i in range(self.num_branches):
            for j in range(i + 1, self.num_branches):
                key = f"branch_{i}_{j}"
                mi_estimate = self.estimators[key](branch_signals[i], branch_signals[j])
                # 直接在估计器输出时应用缩放，避免数量级过大
                scaled_mi_estimate = mi_estimate * self.lambda_mi
                mi_dict[key] = scaled_mi_estimate
                total_mi += scaled_mi_estimate

        return total_mi, mi_dict

    def learning_loss(self, branch_signals):
        """
        计算用于训练CLUB估计器的损失
        """
        total_loss = 0.0

        for i in range(self.num_branches):
            for j in range(i + 1, self.num_branches):
                key = f"branch_{i}_{j}"
                loss = self.estimators[key].learning_loss(branch_signals[i], branch_signals[j])
                total_loss += loss

        return total_loss


class AdaptiveMIMinimizer(nn.Module):
    """
    自适应互信息最小化模块
    可以动态调整不同分支对之间的权重
    """

    def __init__(self, signal_dim, num_branches, hidden_size=None, estimator_type="CLUBMean", lambda_mi=0.01):
        super().__init__()
        self.mi_minimizer = MultiBranchMIMinimizer(signal_dim, num_branches, hidden_size, estimator_type, lambda_mi)

        # 为每对分支学习一个权重
        num_pairs = num_branches * (num_branches - 1) // 2
        self.pair_weights = nn.Parameter(torch.ones(num_pairs))

    def forward(self, branch_signals):
        total_mi, mi_dict = self.mi_minimizer(branch_signals)

        # 应用学习到的权重
        weighted_mi = 0.0
        weight_idx = 0

        for i in range(self.mi_minimizer.num_branches):
            for j in range(i + 1, self.mi_minimizer.num_branches):
                key = f"branch_{i}_{j}"
                weight = torch.softmax(self.pair_weights, dim=0)[weight_idx]
                weighted_mi += weight * mi_dict[key]
                weight_idx += 1

        return weighted_mi, mi_dict

    def learning_loss(self, branch_signals):
        return self.mi_minimizer.learning_loss(branch_signals)


class ResidualFlowMIMinimizer(nn.Module):
    """
    残差流互信息最小化模块
    用于最小化相邻残差流之间的互信息，即X0和X1, X1和X2, X2和X3等
    """

    def __init__(self, signal_dim, num_blocks, hidden_size=None, estimator_type="CLUBMean", lambda_mi=0.01):
        """
        Args:
            signal_dim: 每个残差流信号的维度
            num_blocks: 块数量（残差流数量为num_blocks+1）
            hidden_size: CLUB估计器的隐藏层大小
            estimator_type: 'CLUBMean' 或 'CLUBSample'
            lambda_mi: MI损失的缩放因子，用于控制数量级，默认0.01
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.num_flows = num_blocks + 1  # X0, X1, X2, ..., X_num_blocks
        self.signal_dim = signal_dim
        self.lambda_mi = lambda_mi

        if estimator_type == "CLUBMean":
            EstimatorClass = CLUBMean
        elif estimator_type == "CLUBSample":
            EstimatorClass = CLUBSample
        else:
            raise ValueError("estimator_type must be 'CLUBMean' or 'CLUBSample'")

        # 为每对相邻残差流创建互信息估计器
        # X0-X1, X1-X2, X2-X3, ..., X_{n-1}-X_n
        self.estimators = nn.ModuleList()
        for i in range(self.num_flows - 1):
            if estimator_type == "CLUBMean":
                estimator = EstimatorClass(signal_dim, signal_dim, hidden_size)
            else:  # CLUBSample
                if hidden_size is None:
                    hidden_size = signal_dim * 2
                estimator = EstimatorClass(signal_dim, signal_dim, hidden_size)
            self.estimators.append(estimator)

    def forward(self, residual_flows):
        """
        Args:
            residual_flows: List of tensors, 每个tensor形状为[batch_size, signal_dim]
                          包含X0, X1, X2, ..., X_num_blocks

        Returns:
            total_mi: 所有相邻残差流对之间的总互信息
            mi_list: 每对相邻残差流之间的互信息列表
        """
        if len(residual_flows) != self.num_flows:
            raise ValueError(f"Expected {self.num_flows} residual flows, got {len(residual_flows)}")

        total_mi = 0.0
        mi_list = []

        # 计算相邻残差流之间的互信息
        for i in range(self.num_flows - 1):
            mi_estimate = self.estimators[i](residual_flows[i], residual_flows[i + 1])
            # 直接在估计器输出时应用缩放，避免数量级过大
            scaled_mi_estimate = mi_estimate * self.lambda_mi
            mi_list.append(scaled_mi_estimate)
            total_mi += scaled_mi_estimate

        return total_mi, mi_list

    def learning_loss(self, residual_flows):
        """
        计算用于训练CLUB估计器的损失
        """
        if len(residual_flows) != self.num_flows:
            raise ValueError(f"Expected {self.num_flows} residual flows, got {len(residual_flows)}")

        total_loss = 0.0

        for i in range(self.num_flows - 1):
            loss = self.estimators[i].learning_loss(residual_flows[i], residual_flows[i + 1])
            total_loss += loss

        return total_loss


def create_mi_loss(branch_signals, mi_minimizer, lambda_mi=1.0):
    """
    创建互信息最小化损失

    Args:
        branch_signals: 分支信号列表
        mi_minimizer: MI最小化器
        lambda_mi: 互信息损失的权重

    Returns:
        mi_loss: 互信息损失（用于最小化）
        club_loss: CLUB训练损失
    """
    # 计算互信息估计
    total_mi, _ = mi_minimizer(branch_signals)

    # 计算CLUB训练损失
    club_loss = mi_minimizer.learning_loss(branch_signals)

    # 互信息损失（我们想最小化互信息，所以直接使用估计值）
    mi_loss = lambda_mi * total_mi

    return mi_loss, club_loss
