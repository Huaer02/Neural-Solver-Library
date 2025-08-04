import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.OrthoSolver_layers import OrthoSolverBlockList
from layers.mi_minimizer import MultiBranchMIMinimizer
from layers.Basic import MLP
from layers.Embedding import timestep_embedding, unified_pos_embedding


class Model(nn.Module):
    """
    OrthoSolver Model
    Decomposition FNO with Mutual Information Minimization

    This model uses CLUB (Contrastive Learning Upper Bound) to estimate and minimize
    mutual information between different branch signals. The key design principle is:

    1. CLUB estimators are trained independently using their own optimizer
    2. Only MI estimates (not CLUB training losses) participate in main model training
    3. This separation prevents interference between CLUB training and main model training

    CLUB hyperparameters:
    - club_lr: Learning rate for CLUB optimizer (default: 0.1)
    - club_train_steps: Number of training steps for CLUB per forward pass (default: 5)
    - club_sample_ratio: Ratio of samples used for CLUB training (default: 0.1)
    - lambda_mi: Weight of MI loss in main model training (default: 0.1)
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'OrthoSolver'
        self.args = args
        
        # Basic configuration
        self.space_dim_args = getattr(args, 'space_dim', 2)
        self.space_dim = len(self.args.shapelist)
        if self.space_dim not in [1, 2, 3]:
            raise ValueError(f"Unsupported space_dim: {self.space_dim}. Must be 1, 2, or 3.")
        
        self.space_resolution = getattr(args, 'shapelist', [64, 64] if self.space_dim == 2 else [64])
        self.num_nodes = np.prod(self.space_resolution)
        
        # Model parameters
        self.modes = getattr(args, 'modes', 12)
        self.width = getattr(args, 'n_hidden', 64)
        self.n_layers = getattr(args, 'n_layers', 4)
        self.num_blocks = getattr(args, 'num_blocks', 4)
        
        # Input/Output dimensions
        self.fun_dim = getattr(args, 'fun_dim', 0)
        self.out_dim = getattr(args, 'out_dim', 1)
        
        # Get the appropriate block class based on space_dim
        self.DecomBlock = OrthoSolverBlockList[self.space_dim]
        if self.DecomBlock is None:
            raise ValueError(f"No OrthoSolver block implementation for {self.space_dim}D")
        
        # Preprocessing
        if getattr(args, 'unified_pos', False) and getattr(args, 'geotype', 'structured') != 'unstructured':
            self.pos = unified_pos_embedding(args.shapelist, args.ref)
            self.preprocess = MLP(
                args.fun_dim + args.ref ** len(args.shapelist), 
                args.n_hidden * 2,
                args.n_hidden, 
                n_layers=0, 
                res=False, 
                act=getattr(args, 'act', 'relu')
            )
        else:
            self.preprocess = MLP(
                args.fun_dim + self.space_dim_args, 
                args.n_hidden * 2, 
                args.n_hidden,
                n_layers=0, 
                res=False, 
                act=getattr(args, 'act', 'relu')
            )
        
        # Time embedding
        if getattr(args, 'time_input', False):
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden), 
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden)
            )
        
        # Multi-task options
        self.use_multitask = getattr(args, 'use_multitask', True)
        self.use_weight_fusion = getattr(args, 'use_weight_fusion', True)
        self.use_residual = getattr(args, 'use_residual', True)
        self.use_mi_loss = getattr(args, 'use_mi_loss', True)
        
        # OrthoSolver Blocks - using the dimension-specific block class
        self.fno_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            block = self.DecomBlock(
                modes=self.modes,
                width=self.width,
                input_dim=self.width + self.space_dim_args,  # include grid coordinates
                output_dim=self.out_dim,
                n_layers=self.n_layers,
                mode=getattr(args, 'fft_mode', 'full'),
                use_fork=True,
                dropout=getattr(args, 'dropout', 0.1),
                share_weight=getattr(args, 'share_weight', False),
                share_fork=getattr(args, 'share_fork', False),
                factor=getattr(args, 'factor', 2),
                ff_weight_norm=getattr(args, 'ff_weight_norm', False),
                n_ff_layers=getattr(args, 'n_ff_layers', 2),
                gain=getattr(args, 'gain', 1),
                layer_norm=getattr(args, 'layer_norm', False),
                coefficient_dim=getattr(args, 'coefficient_dim', 1),
                coefficient_only=getattr(args, 'coefficient_only', True),
            )
            self.fno_blocks.append(block)
        
        # MI minimizer for orthogonalization using basic_out signals
        if self.use_mi_loss:
            self.mi_hidden_size = getattr(args, 'mi_hidden_size', self.width)
            self.mi_estimator_type = getattr(args, 'mi_estimator_type', 'CLUBSample')
            self.lambda_mi = getattr(args, 'lambda_mi', 0.1)
            self.club_lr = getattr(args, 'club_lr', 0.1)
            self.club_train_steps = getattr(args, 'club_train_steps', 5)
            self.club_sample_ratio = getattr(args, 'club_sample_ratio', 0.1)

            self.mi_minimizer = MultiBranchMIMinimizer(
                signal_dim=self.width,  # basic_out has dimension [..., width]
                num_branches=self.num_blocks,
                hidden_size=self.mi_hidden_size,
                estimator_type=self.mi_estimator_type,
            )
            self.club_optimizer = torch.optim.Adam(self.mi_minimizer.parameters(), lr=self.club_lr)

        # Weight fusion
        if self.use_weight_fusion:
            self.weights = nn.ParameterList()
            for i in range(self.num_blocks):
                weight_shape = (1, *self.space_resolution, 1, self.out_dim)
                weight = nn.Parameter(torch.zeros(size=weight_shape), requires_grad=True)
                self.weights.append(weight)
            
            self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        if self.use_weight_fusion:
            for weight in self.weights:
                nn.init.uniform_(weight.data, a=-0.5, b=0.5)
    
    
    def _train_club_estimators(self, branch_signals):
        """单独训练CLUB估计器 - 参考STEVE的实现"""
        if not hasattr(self, 'club_optimizer'):
            return

        # 将信号reshape为 [total_samples, feature_dim] 形式
        # branch_signals: list of [batch_size, *space_resolution, feature_dim]
        reshaped_signals = []
        for signal in branch_signals:
            # Reshape to [batch_size * num_spatial_points, feature_dim]
            batch_size = signal.shape[0]
            num_spatial = np.prod(signal.shape[1:-1])  # 计算空间点总数
            feature_dim = signal.shape[-1]

            reshaped = signal.reshape(batch_size * num_spatial, feature_dim)
            reshaped_signals.append(reshaped)

        # 随机采样以提高训练效率
        if len(reshaped_signals) > 0:
            total_samples = reshaped_signals[0].shape[0]
            if self.club_sample_ratio < 1.0:
                num_samples = max(1, int(total_samples * self.club_sample_ratio))
                random_indices = np.random.choice(total_samples, num_samples, replace=False)
                sampled_signals = [signal[random_indices].detach() for signal in reshaped_signals]
            else:
                sampled_signals = [signal.detach() for signal in reshaped_signals]

            # 多轮训练CLUB估计器
            self.mi_minimizer.train()
            for _ in range(self.club_train_steps):
                self.club_optimizer.zero_grad()
                club_loss = self.mi_minimizer.learning_loss(sampled_signals)
                club_loss.backward()
                self.club_optimizer.step()

            # 恢复评估模式
            self.mi_minimizer.eval()

        
    def forward(self, x, fx=None, T=None, **kwargs):
        """
        Forward pass
        Args:
            x: Position/coordinate tensor [batch, N, space_dim]
            fx: Function values [batch, N, fun_dim]
            T: Time step (optional)
        Returns:
            For single task: prediction tensor
            For multi-task: (res_out, prediction, mi_loss)
        """
        batch_size, N, _ = x.shape
        
        # Prepare input
        if getattr(self.args, 'unified_pos', False):
            x_pos = self.pos.repeat(x.shape[0], 1, 1)
        else:
            x_pos = x
        
        if fx is not None:
            input_features = torch.cat((x_pos, fx), -1)
        else:
            input_features = x_pos
        
        # Preprocess
        processed_input = self.preprocess(input_features)
        
        # Time embedding
        if T is not None and hasattr(self, 'time_fc'):
            time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            time_emb = self.time_fc(time_emb)
            processed_input = processed_input + time_emb
        
        processed_input = processed_input.reshape(batch_size, *self.space_resolution, -1)
        grid = x.reshape(batch_size, *self.space_resolution, -1)
        all_pred = torch.zeros(batch_size, *self.space_resolution, 1, self.out_dim, device=x.device)
        
        cur_x = processed_input
        all_basic_outputs = []  # Store basic_out for MI computation
        
        # Weight fusion preparation
        if self.use_weight_fusion:
            sum_exp_weight = sum(torch.exp(weight) for weight in self.weights)
        
        # Process through blocks
        for i, block in enumerate(self.fno_blocks):
            # Forward through OrthoSolver block
            # SerialOrthoSolverBlock outputs: forecast, res_out, basic_out
            forecast_out, block_res_out, basic_out = block(cur_x, grid)
            
            # Collect basic_out for MI computation (orthogonalization)
            if self.use_mi_loss:
                all_basic_outputs.append(basic_out)
            
            # Weight fusion for forecast output
            if self.use_weight_fusion:
                w = torch.exp(self.weights[i]) / sum_exp_weight
                weighted_pred = forecast_out * w
                all_pred += weighted_pred
            else:
                all_pred += forecast_out / self.num_blocks
            
            # Update current state for next block using residual connection
            # cur_x = cur_x - res_out (as specified by user)
            if self.use_residual:
                cur_x = cur_x - block_res_out
        
        # Final residual output is the final cur_x (after all residual subtractions)
        final_res_out = cur_x if self.use_residual else processed_input

        # Compute MI losses using basic_out for orthogonalization
        mi_loss = torch.tensor(0.0, device=x.device)
        
        if self.use_mi_loss and len(all_basic_outputs) >= 2:
            # 单独训练CLUB估计器
            if self.training:
                self._train_club_estimators(all_basic_outputs)
        
            # 计算MI估计值用于主模型损失
            # 将basic_out信号reshape为CLUB可接受的格式
            reshaped_signals = []
            for signal in all_basic_outputs:
                batch_size = signal.shape[0]
                num_spatial = np.prod(signal.shape[1:-1])
                feature_dim = signal.shape[-1]
                reshaped = signal.reshape(batch_size * num_spatial, feature_dim)
                reshaped_signals.append(reshaped)

            mi_estimate, _ = self.mi_minimizer(reshaped_signals)
            mi_loss = self.lambda_mi * mi_estimate

        # Reshape outputs for compatibility
        final_pred = all_pred.reshape(batch_size, N, -1)
        final_res_out = final_res_out.reshape(batch_size, N, -1)
        
        # Return based on task type
        if self.use_multitask:
            return final_res_out, final_pred, mi_loss
        else:
            return final_pred