import torch
import torch.nn.functional as F


def optimal_orthogonal_loss(signal_outputs, method="gram_matrix", batch_independent=True, channel_independent=False):
    """
    数学上严格的正交损失实现

    Args:
        signal_outputs: List of K tensors, each with shape [B, N1, N2, ..., C]
                       第一个维度是batch size B，最后一个维度是特征维度C，中间是空间维度
        method: 'gram_matrix', 'frobenius', 'canonical_correlation', 'cosine_similarity', 'mi'
        batch_independent: 是否每个batch独立计算（推荐True）
        channel_independent: 是否每个channel独立计算（根据应用场景选择）

    数学原理说明：
    对于基函数正交性，我们有K个分支，每个分支产生一个基函数集
    - 每个基函数可以看作是在空间域上定义的C维向量场
    - 正交性应该在函数空间中定义，即不同分支的基函数应该内积为0

    Reshape策略：
    1. batch_independent=True: 每个样本的基函数集合独立
    2. channel_independent=True: 每个通道独立计算，形状[K, spatial_size] per channel
    3. channel_independent=False: 通道间耦合，形状[K, spatial_size*C]

    支持的方法：
    - 'gram_matrix': Gram矩阵方法，G = XX^T 应接近单位矩阵
    - 'frobenius': 直接计算交叉内积的Frobenius范数
    - 'canonical_correlation': 基于协方差结构的典型相关分析
    - 'cosine_similarity': 基于余弦相似度的正交性（方向正交，尺度不敏感）
    - 'mi': 互信息方法（需要外部MI估计器，这里仅作占位符）
    """
    if len(signal_outputs) < 2:
        return torch.tensor(0.0, device=signal_outputs[0].device)

    K = len(signal_outputs)  # 分支数

    # 自动推断形状信息：[B, N1, N2, ..., C]
    first_signal = signal_outputs[0]
    shape = first_signal.shape
    B = shape[0]  # batch size
    C = shape[-1]  # channel/feature dimension
    spatial_dims = shape[1:-1]  # 中间的空间维度 (N1, N2, ...)
    spatial_size = int(torch.prod(torch.tensor(spatial_dims)))  # 总空间点数
    device = first_signal.device

    if not batch_independent:
        # 跨batch计算（通常不推荐，除非batch间有特殊关系）
        if channel_independent:
            return _compute_channel_independent_loss_generic(signal_outputs, method, K, B, spatial_size, C, device)
        else:
            return _compute_coupled_loss_generic(signal_outputs, method, K, B, spatial_size, C, device)
    else:
        # 每个batch独立计算（推荐）
        total_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            if channel_independent:
                # 每个通道独立计算正交性
                batch_loss = torch.tensor(0.0, device=device)
                for c in range(C):
                    # 构建当前batch当前channel的基函数矩阵 [K, spatial_size]
                    basis_matrix = torch.stack(
                        [signal_outputs[k][b, ..., c].reshape(spatial_size) for k in range(K)], dim=0
                    )  # [K, spatial_size]

                    channel_loss = _compute_orthogonal_loss_single(basis_matrix, method)
                    batch_loss += channel_loss

                total_loss += batch_loss / C  # 通道平均
            else:
                # 通道耦合，构建 [K, spatial_size*C] 矩阵
                basis_matrix = torch.stack(
                    [signal_outputs[k][b].reshape(spatial_size * C) for k in range(K)], dim=0
                )  # [K, spatial_size*C]

                batch_loss = _compute_orthogonal_loss_single(basis_matrix, method)
                total_loss += batch_loss

        return total_loss / B


def _compute_orthogonal_loss_single(basis_matrix, method):
    """
    计算单个基函数矩阵的正交损失
    Args:
        basis_matrix: [K, D] 其中K是分支数，D是基函数维度
        method: 损失计算方法
    """
    K, D = basis_matrix.shape
    device = basis_matrix.device

    if method == "gram_matrix":
        # Gram矩阵方法：G = X X^T，理想情况下应该是单位矩阵
        # 先L2归一化每个基函数
        basis_normalized = F.normalize(basis_matrix, dim=-1)  # [K, D]
        gram_matrix = torch.mm(basis_normalized, basis_normalized.T)  # [K, K]
        target = torch.eye(K, device=device)
        return F.mse_loss(gram_matrix, target)

    elif method == "frobenius":
        # Frobenius方法：直接计算交叉内积的平方和
        total_loss = torch.tensor(0.0, device=device)
        for i in range(K):
            for j in range(i + 1, K):
                # 计算第i和第j个基函数的内积
                inner_product = torch.dot(basis_matrix[i], basis_matrix[j])
                total_loss += inner_product**2

        # 归一化
        num_pairs = K * (K - 1) // 2
        return total_loss / num_pairs if num_pairs > 0 else total_loss

    elif method == "canonical_correlation":
        # 典型相关分析风格：考虑协方差结构
        # 中心化
        basis_centered = basis_matrix - basis_matrix.mean(dim=-1, keepdim=True)

        # 计算协方差矩阵
        cov_matrix = torch.mm(basis_centered, basis_centered.T) / (D - 1)

        # 对角元素（自协方差）
        diag_elements = torch.diagonal(cov_matrix)

        # 非对角元素（交叉协方差）应该为0
        off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=device)
        off_diag_elements = cov_matrix[off_diag_mask]

        # 归一化的交叉相关系数
        std_i = torch.sqrt(diag_elements).unsqueeze(1)  # [K, 1]
        std_j = torch.sqrt(diag_elements).unsqueeze(0)  # [1, K]
        correlation_matrix = cov_matrix / (std_i * std_j + 1e-8)

        # 非对角元素应该为0
        off_diag_corr = correlation_matrix[off_diag_mask]
        return torch.mean(off_diag_corr**2)

    elif method == "cosine_similarity":
        # 余弦相似度方法：计算所有分支对之间的余弦相似度
        # 数学公式：cos(θ) = (u·v)/(||u||||v||)，正交时 cos(θ) = 0

        # L2归一化所有基函数向量
        basis_normalized = F.normalize(basis_matrix, dim=-1, eps=1e-8)  # [K, D]

        # 方法1：通过Gram矩阵计算所有余弦相似度
        cosine_similarity_matrix = torch.mm(basis_normalized, basis_normalized.T)  # [K, K]

        # 提取非对角元素（即不同分支间的余弦相似度）
        off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=device)
        off_diag_cosines = cosine_similarity_matrix[off_diag_mask]

        # 正交性损失：余弦相似度的平方和（希望为0）
        cosine_loss = torch.mean(off_diag_cosines**2)

        return cosine_loss

        # 方法2：逐对计算（注释掉，但保留作为参考）
        # total_loss = torch.tensor(0.0, device=device)
        # for i in range(K):
        #     for j in range(i + 1, K):
        #         cos_sim = F.cosine_similarity(
        #             basis_matrix[i].unsqueeze(0),
        #             basis_matrix[j].unsqueeze(0),
        #             dim=1
        #         )
        #         total_loss += cos_sim ** 2
        #
        # num_pairs = K * (K - 1) // 2
        # return total_loss / num_pairs if num_pairs > 0 else total_loss

    elif method == "mi":
        # MI方法在这里不实现，应该在模型中使用MI估计器处理
        # 这里只是一个占位符，返回0损失
        raise NotImplementedError("MI method should be handled in the model with MI estimator, not in this function")

    else:
        raise ValueError(
            f"Unknown method: {method}. Supported methods: 'gram_matrix', 'frobenius', 'canonical_correlation', 'cosine_similarity'"
        )


def _compute_channel_independent_loss_generic(signal_outputs, method, K, B, spatial_size, C, device):
    """跨batch，每通道独立的损失计算 - 泛化版本"""
    total_loss = torch.tensor(0.0, device=device)

    for c in range(C):
        # 构建所有batch所有分支在第c个通道的矩阵 [K, B*spatial_size]
        basis_matrix = torch.stack(
            [signal_outputs[k][..., c].reshape(B * spatial_size) for k in range(K)], dim=0
        )  # [K, B*spatial_size]

        channel_loss = _compute_orthogonal_loss_single(basis_matrix, method)
        total_loss += channel_loss

    return total_loss / C


def _compute_coupled_loss_generic(signal_outputs, method, K, B, spatial_size, C, device):
    """跨batch，通道耦合的损失计算 - 泛化版本"""
    # 构建所有batch所有分支的完整矩阵 [K, B*spatial_size*C]
    basis_matrix = torch.stack(
        [signal_outputs[k].reshape(B * spatial_size * C) for k in range(K)], dim=0
    )  # [K, B*spatial_size*C]

    return _compute_orthogonal_loss_single(basis_matrix, method)
