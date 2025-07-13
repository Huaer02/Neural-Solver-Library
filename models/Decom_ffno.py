import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Decom_Layers import DecomBlockList
from layers.mi_minimizer import MultiBranchMIMinimizer, create_mi_loss
from layers.Basic import MLP, WNLinear
from layers.Embedding import timestep_embedding, unified_pos_embedding


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Decom-FFNO'
        self.args = args
        
        # Basic configuration
        self.space_dim = getattr(args, 'space_dim', 2)
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
        self.DecomBlock = DecomBlockList[self.space_dim]
        if self.DecomBlock is None:
            raise ValueError(f"No block implementation for {self.space_dim}D")
        
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
                args.fun_dim + args.space_dim, 
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
        
        # FNO Blocks - using the dimension-specific block class
        self.fno_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            block = self.DecomBlock(
                modes=self.modes,
                width=self.width,
                input_dim=self.width + self.space_dim,  # include grid coordinates
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
            )
            self.fno_blocks.append(block)
        
        # Residual output layers
        if self.use_residual:
            self.res_out_fc = WNLinear(
                self.width, 
                self.fun_dim if self.fun_dim > 0 else self.out_dim,
                wnorm=getattr(args, 'ff_weight_norm', False)
            )
            self.layer_norm_res = nn.LayerNorm(self.fun_dim if self.fun_dim > 0 else self.out_dim)
        
        # MI minimizer for multi-task learning
        if self.use_mi_loss:
            self.signal_feature_method = getattr(args, 'signal_feature_method', 'spatial_statistics')
            self.mi_hidden_size = getattr(args, 'mi_hidden_size', self.width)
            self.mi_estimator_type = getattr(args, 'mi_estimator_type', 'CLUBMean')
            self.lambda_mi = getattr(args, 'lambda_mi', 0.1)

            signal_dim = self._get_signal_feature_dim()
            self.mi_minimizer = MultiBranchMIMinimizer(
                signal_dim=signal_dim,
                num_branches=self.num_blocks,
                hidden_size=self.mi_hidden_size,
                estimator_type=self.mi_estimator_type,
            )
        
        # Weight fusion
        if self.use_weight_fusion:
            self.weights = nn.ParameterList()
            for i in range(self.num_blocks):
                # Create weight shape based on space dimension
                if self.space_dim == 1:
                    weight_shape = (1, self.space_resolution[0], 1, self.out_dim)
                elif self.space_dim == 2:
                    weight_shape = (1, *self.space_resolution, 1, self.out_dim)
                else:  # 3D
                    weight_shape = (1, *self.space_resolution, 1, self.out_dim)
                
                weight = nn.Parameter(torch.zeros(size=weight_shape), requires_grad=True)
                self.weights.append(weight)
            
            self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        if self.use_weight_fusion:
            for weight in self.weights:
                nn.init.uniform_(weight.data, a=-0.5, b=0.5)
    
    def _get_signal_feature_dim(self):
        """Calculate signal feature dimension based on extraction method"""
        if self.signal_feature_method == "spatial_pooling":
            # global_avg + global_max + region features
            if self.space_dim == 1:
                return self.width * (2 + 2)  # 2 global + 2 regions
            elif self.space_dim == 2:
                return self.width * (2 + 4)  # 2 global + 4 regions
            else:  # 3D
                return self.width * (2 + 8)  # 2 global + 8 regions
        elif self.signal_feature_method == "spatial_statistics":
            # mean + std + max + gradients
            if self.space_dim == 1:
                return self.width * (3 + 1)  # 3 stats + 1 gradient
            elif self.space_dim == 2:
                return self.width * (3 + 2)  # 3 stats + 2 gradients
            else:  # 3D
                return self.width * (3 + 3)  # 3 stats + 3 gradients
        else:
            return self.width
    
    def _extract_signal_features(self, signal_tensor, method="spatial_pooling"):
        """Extract features from signal tensor for MI calculation"""
        batch_size = signal_tensor.shape[0]
        
        # Determine spatial dimensions
        spatial_dims = tuple(range(1, len(signal_tensor.shape) - 1))
        
        if method == "spatial_pooling":
            # Global average and max pooling
            global_avg = signal_tensor.mean(dim=spatial_dims)
            global_max = signal_tensor
            for dim in spatial_dims:
                global_max = global_max.max(dim=dim)[0]
            
            # Region pooling based on space dimension
            region_features = []
            
            if self.space_dim == 1:
                # Split into 2 regions
                space_res = signal_tensor.shape[1]
                mid = space_res // 2
                regions = [
                    signal_tensor[:, :mid, :],
                    signal_tensor[:, mid:, :]
                ]
            elif self.space_dim == 2:
                # Split into 4 regions
                h, w = signal_tensor.shape[1], signal_tensor.shape[2]
                h_mid, w_mid = h // 2, w // 2
                regions = [
                    signal_tensor[:, :h_mid, :w_mid, :],
                    signal_tensor[:, :h_mid, w_mid:, :],
                    signal_tensor[:, h_mid:, :w_mid, :],
                    signal_tensor[:, h_mid:, w_mid:, :]
                ]
            else:  # 3D
                # Split into 8 regions
                d, h, w = signal_tensor.shape[1], signal_tensor.shape[2], signal_tensor.shape[3]
                d_mid, h_mid, w_mid = d // 2, h // 2, w // 2
                regions = [
                    signal_tensor[:, :d_mid, :h_mid, :w_mid, :],
                    signal_tensor[:, :d_mid, :h_mid, w_mid:, :],
                    signal_tensor[:, :d_mid, h_mid:, :w_mid, :],
                    signal_tensor[:, :d_mid, h_mid:, w_mid:, :],
                    signal_tensor[:, d_mid:, :h_mid, :w_mid, :],
                    signal_tensor[:, d_mid:, :h_mid, w_mid:, :],
                    signal_tensor[:, d_mid:, h_mid:, :w_mid, :],
                    signal_tensor[:, d_mid:, h_mid:, w_mid:, :],
                ]
            
            for region in regions:
                region_dims = tuple(range(1, len(region.shape) - 1))
                region_features.append(region.mean(dim=region_dims))
            
            features = torch.cat([global_avg, global_max] + region_features, dim=1)
            return features
        
        elif method == "spatial_statistics":
            # Statistical features
            global_mean = signal_tensor.mean(dim=spatial_dims)
            global_std = signal_tensor.std(dim=spatial_dims)
            global_max = signal_tensor
            for dim in spatial_dims:
                global_max = global_max.max(dim=dim)[0]
            
            # Gradient features based on dimension
            grad_features = []
            if self.space_dim == 1:
                grad_x = torch.diff(signal_tensor, dim=1).abs().mean(dim=1)
                grad_features.append(grad_x)
            elif self.space_dim == 2:
                grad_x = torch.diff(signal_tensor, dim=1).abs().mean(dim=(1, 2))
                grad_y = torch.diff(signal_tensor, dim=2).abs().mean(dim=(1, 2))
                grad_features.extend([grad_x, grad_y])
            else:  # 3D
                grad_x = torch.diff(signal_tensor, dim=1).abs().mean(dim=(1, 2, 3))
                grad_y = torch.diff(signal_tensor, dim=2).abs().mean(dim=(1, 2, 3))
                grad_z = torch.diff(signal_tensor, dim=3).abs().mean(dim=(1, 2, 3))
                grad_features.extend([grad_x, grad_y, grad_z])
            
            features = torch.cat([global_mean, global_std, global_max] + grad_features, dim=1)
            return features
        
        else:
            # Default: simple average pooling
            return signal_tensor.mean(dim=spatial_dims)
    
    def forward(self, x, fx=None, T=None, **kwargs):
        """
        Forward pass
        Args:
            x: Position/coordinate tensor [batch, N, space_dim]
            fx: Function values [batch, N, fun_dim]
            T: Time step (optional)
        Returns:
            For single task: prediction tensor
            For multi-task: (res_out, prediction, mi_loss, club_loss)
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
        
        # Reshape for processing based on dimension
        if self.space_dim == 1:
            processed_input = processed_input.reshape(batch_size, self.space_resolution[0], -1)
            grid = x.reshape(batch_size, self.space_resolution[0], -1)
            all_pred = torch.zeros(batch_size, self.space_resolution[0], 1, self.out_dim, device=x.device)
        elif self.space_dim == 2:
            processed_input = processed_input.reshape(batch_size, *self.space_resolution, -1)
            grid = x.reshape(batch_size, *self.space_resolution, -1)
            all_pred = torch.zeros(batch_size, *self.space_resolution, 1, self.out_dim, device=x.device)
        else:  # 3D
            processed_input = processed_input.reshape(batch_size, *self.space_resolution, -1)
            grid = x.reshape(batch_size, *self.space_resolution, -1)
            all_pred = torch.zeros(batch_size, *self.space_resolution, 1, self.out_dim, device=x.device)
        
        cur_x = processed_input
        all_block_signal_outputs = []
        
        # Weight fusion preparation
        if self.use_weight_fusion:
            sum_exp_weight = sum(torch.exp(weight) for weight in self.weights)
        
        # Process through blocks
        for i, block in enumerate(self.fno_blocks):
            # Forward through FNO block
            pred_out, fno_res_out, signal_out = block(cur_x, grid)
            
            # Extract signal features for MI computation
            if self.use_mi_loss:
                signal_features = self._extract_signal_features(
                    signal_out, method=self.signal_feature_method
                )
                all_block_signal_outputs.append(signal_features)
            
            # Handle residual output
            if self.use_residual and i == 0:  # Only compute residual from first block
                res_out = self.res_out_fc(fno_res_out)
                res_out = self.layer_norm_res(res_out)
                # Reshape back to original input shape
                res_out = res_out.reshape(batch_size, N, -1)
            
            # Weight fusion
            if self.use_weight_fusion:
                w = torch.exp(self.weights[i]) / sum_exp_weight
                weighted_pred = pred_out * w
                all_pred += weighted_pred
            else:
                all_pred += pred_out / self.num_blocks
            
            # Update current state for next block
            if self.use_residual:
                cur_x = cur_x - fno_res_out
        
        # Compute MI losses
        mi_loss = torch.tensor(0.0, device=x.device)
        club_loss = torch.tensor(0.0, device=x.device)
        
        if self.use_mi_loss and len(all_block_signal_outputs) >= 2:
            mi_loss, club_loss = create_mi_loss(
                all_block_signal_outputs, 
                self.mi_minimizer, 
                lambda_mi=self.lambda_mi
            )
        
        # Reshape output for compatibility
        final_pred = all_pred.reshape(batch_size, N, -1)
        
        # Return based on task type
        if self.use_multitask:
            if self.use_residual:
                return res_out, final_pred, mi_loss, club_loss
            else:
                dummy_res = torch.zeros_like(final_pred)
                return dummy_res, final_pred, mi_loss, club_loss
        else:
            return final_pred