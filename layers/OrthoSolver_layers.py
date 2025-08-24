import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.Basic import WNLinear, WNFeedForward, MLP, PreNorm
from layers.Decom_Layers import DecomConvList


class SerialOrthoSolverBlock2D(nn.Module):
    """2D OrthoSolverBlock2D"""

    def __init__(
        self,
        modes,
        width,
        input_dim=12,
        output_dim=5,
        dropout=0.0,
        in_dropout=0.0,
        n_layers=4,
        share_weight: bool = False,
        share_fork=False,
        factor=2,
        ff_weight_norm=False,
        n_ff_layers=2,
        gain=1,
        layer_norm=False,
        use_fork=False,
        mode="full",
        coefficient_dim=1,
        coefficient_only=True,
    ):
        # input: X_i: [b, n1, n2,..., in_dim] -> [b, n1, n2,..., c]
        # Basic Operator: X_i: [b, n1, n2,..., c] -> B_i: [b, n1, n2,..., c]
        # Coefficient Operator: X_i: [b, n1, n2,..., c] -> C_i: [b, n1, n2,..., 1]
        # Solution Operator:
        ##  Coefficient only: C_i: [b, n1, n2,..., 1] -> [b, n1, n2,..., 1]
        ##  Coefficient + Basic: B_i: [b, n1, n2,..., c] + C_i: [b, n1, n2,..., 1] -> \hat{C_i} :[b, n1, n2,..., 1]

        # forecast: B_i * \hat{C_i}: [b, n1, n2,..., c] * [b, n1, n2,..., 1] -> MLP -> [b, n1, n2,..., out_dim]
        # res_out: B_i * C_i: [b, n1, n2,..., c] * [b, n1, n2,..., 1] -> [b, n1, n2,..., c]
        # basic_out: B_i: [b, n1, n2,..., c]
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coefficient_dim = coefficient_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork
        self.coefficient_only = coefficient_only
        DecomSpectralConv2d = DecomConvList[2]

        # Shared components
        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        # Shared Fourier weights
        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(4):  # 2D needs 4 weights
                weight = torch.FloatTensor(width, width, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param, gain=gain)
                self.fourier_weight.append(param)

        # Basic Operators: X_i -> B_i
        self.basic_operators = nn.ModuleList([])
        for _ in range(n_layers):
            self.basic_operators.append(
                DecomSpectralConv2d(
                    in_dim=width,
                    out_dim=width,
                    n_modes=modes,
                    forecast_ff=None,
                    backcast_ff=None,
                    fourier_weight=self.fourier_weight,
                    factor=factor,
                    ff_weight_norm=ff_weight_norm,
                    n_ff_layers=n_ff_layers,
                    layer_norm=layer_norm,
                    use_fork=False,
                    dropout=dropout,
                    mode=mode,
                )
            )

        # Coefficient Operators: X_i -> C_i
        self.coefficient_operators = nn.ModuleList([])
        for _ in range(n_layers):
            self.coefficient_operators.append(
                PreNorm(
                    dim=width,
                    fn=MLP(
                        n_input=width,
                        n_hidden=width * factor,
                        n_layers=n_ff_layers,
                        n_output=self.coefficient_dim,
                        act="relu",
                    ),
                )
            )

        # Solution Operators: C_i -> \hat{C_i} or [B_i, C_i] -> \hat{C_i}
        self.solution_operators = nn.ModuleList([])
        for _ in range(n_layers):
            solution_input_dim = coefficient_dim if coefficient_only else (width + coefficient_dim)
            self.solution_operators.append(
                DecomSpectralConv2d(
                    in_dim=solution_input_dim,
                    out_dim=coefficient_dim,
                    n_modes=modes,
                    forecast_ff=None,
                    backcast_ff=None,
                    fourier_weight=None,
                    factor=factor,
                    ff_weight_norm=ff_weight_norm,
                    n_ff_layers=n_ff_layers,
                    layer_norm=layer_norm,
                    use_fork=use_fork,
                    dropout=dropout,
                    mode=mode,
                )
            )

        # Final output layer for forecast: B_i * \hat{C_i} -> forecast
        # self.out = nn.Sequential(
        #     WNLinear(self.width, self.width * factor, wnorm=ff_weight_norm),
        #     nn.ReLU(),
        #     WNLinear(self.width * factor, self.output_dim, wnorm=ff_weight_norm),
        # )
        self.out = PreNorm(
            dim=width,
            fn=nn.Sequential(
                WNLinear(self.width, self.width * factor, wnorm=ff_weight_norm),
                nn.ReLU(),
                WNLinear(self.width * factor, self.output_dim, wnorm=ff_weight_norm),
            ),
        )

    def forward(self, x, grid=None, **kwargs):
        if grid is not None:
            x = torch.cat((x, grid), dim=-1)

        # Input projection: [b, n1, n2, in_dim] -> [b, n1, n2, c]
        x = self.in_proj(x)
        x = self.drop(x)

        forecast = 0
        forecast_list = []

        for i in range(self.n_layers):
            basic_operator = self.basic_operators[i]
            coefficient_operator = self.coefficient_operators[i]
            solution_operator = self.solution_operators[i]

            # Basic Operator: X_i -> B_i [b, n1, n2, c]
            B_i, _ = basic_operator(x)  # Only use backcast output

            # Coefficient Operator: X_i -> C_i [b, n1, n2, 1]
            C_i = coefficient_operator(x)

            # Solution Operator: C_i -> \hat{C_i} or [B_i, C_i] -> \hat{C_i}
            if self.coefficient_only:
                # Only use coefficient
                C_hat_i, _ = solution_operator(C_i)
            else:
                # Concatenate basic and coefficient
                combined_input = torch.cat([B_i, C_i], dim=-1)
                C_hat_i, _ = solution_operator(combined_input)

            if self.use_fork:
                # forecast: B_i * \hat{C_i} -> MLP -> [b, n1, n2, out_dim]
                forecast_input = B_i * C_hat_i  # Element-wise multiplication
                f_out = self.out(forecast_input)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            # Update x for next layer using residual connection
            x = x + B_i

        if not self.use_fork:
            # If not using fork, output from the final state
            forecast = self.out(x)

        # res_out: B_i * C_i -> [b, n1, n2, c]
        res_out = B_i * C_i  # Element-wise multiplication, output has shape [b, n1, n2, c]

        # basic_out: B_i -> [b, n1, n2, c]
        basic_out = B_i

        # Add dimension for compatibility
        forecast = torch.unsqueeze(forecast, dim=-2)

        return forecast, res_out, basic_out


class ParalleOrthoSolverBlock2D(nn.Module):
    """2D OrthoSolverBlock2D"""

    def __init__(
        self,
        modes,
        width,
        input_dim=12,
        output_dim=5,
        dropout=0.0,
        in_dropout=0.0,
        n_layers=4,
        share_weight: bool = False,
        share_fork=False,
        factor=2,
        ff_weight_norm=False,
        n_ff_layers=2,
        gain=1,
        layer_norm=False,
        use_fork=False,
        mode="full",
        coefficient_dim=1,
        coefficient_only=True,
    ):
        # input: X_i: [b, n1, n2,..., in_dim] -> [b, n1, n2,..., c]
        # Basic Operator: X_i: [b, n1, n2,..., c] -> B_i: [b, n1, n2,..., c]
        # Coefficient Operator: X_i: [b, n1, n2,..., c] -> C_i: [b, n1, n2,..., 1]
        # Solution Operator:
        ##  Coefficient only: C_i: [b, n1, n2,..., 1] -> [b, n1, n2,..., 1]
        ##  Coefficient + Basic: B_i: [b, n1, n2,..., c] + C_i: [b, n1, n2,..., 1] -> \hat{C_i} :[b, n1, n2,..., 1]

        # forecast: B_i * \hat{C_i}: [b, n1, n2,..., c] * [b, n1, n2,..., 1] -> MLP -> [b, n1, n2,..., out_dim]
        # res_out: B_i * C_i: [b, n1, n2,..., c] * [b, n1, n2,..., 1] -> [b, n1, n2,..., c]
        # basic_out: B_i: [b, n1, n2,..., c]
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coefficient_dim = coefficient_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork
        self.coefficient_only = coefficient_only
        DecomSpectralConv2d = DecomConvList[2]

        # Shared components
        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        # Shared Fourier weights
        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(4):  # 2D needs 4 weights
                weight = torch.FloatTensor(width, width, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param, gain=gain)
                self.fourier_weight.append(param)

        # Basic Operators: X_i -> B_i
        self.basic_operators = nn.ModuleList([])
        for _ in range(n_layers):
            self.basic_operators.append(
                DecomSpectralConv2d(
                    in_dim=width,
                    out_dim=width,
                    n_modes=modes,
                    forecast_ff=None,
                    backcast_ff=None,
                    fourier_weight=self.fourier_weight,
                    factor=factor,
                    ff_weight_norm=ff_weight_norm,
                    n_ff_layers=n_ff_layers,
                    layer_norm=layer_norm,
                    use_fork=False,  # Basic operator doesn't use fork
                    dropout=dropout,
                    mode=mode,
                )
            )

        # Coefficient Operators: X_i -> C_i
        self.coefficient_operators = nn.ModuleList([])
        for _ in range(n_layers):
            self.coefficient_operators.append(
                MLP(
                    n_input=width,
                    n_hidden=width * factor,
                    n_layers=n_ff_layers,
                    n_output=width,  # Keep same dimension as input for residual connection
                    act="relu",
                )
            )

        # Final coefficient projection: width -> coefficient_dim
        self.final_coeff_proj = WNLinear(width, self.coefficient_dim, wnorm=ff_weight_norm)

        # Solution Operators: C_i -> \hat{C_i} or [B_i, C_i] -> \hat{C_i}
        self.solution_operators = nn.ModuleList([])
        for _ in range(n_layers):
            solution_input_dim = coefficient_dim if coefficient_only else (width + coefficient_dim)
            self.solution_operators.append(
                DecomSpectralConv2d(
                    in_dim=solution_input_dim,
                    out_dim=coefficient_dim,
                    n_modes=modes,
                    forecast_ff=None,
                    backcast_ff=None,
                    fourier_weight=None,  # Solution operator has its own weights
                    factor=factor,
                    ff_weight_norm=ff_weight_norm,
                    n_ff_layers=n_ff_layers,
                    layer_norm=layer_norm,
                    use_fork=False,
                    dropout=dropout,
                    mode=mode,
                )
            )

        # Final output layer for forecast: B_i * \hat{C_i} -> forecast
        self.out = nn.Sequential(
            WNLinear(self.width, self.width * factor, wnorm=ff_weight_norm),
            nn.ReLU(),
            WNLinear(self.width * factor, self.output_dim, wnorm=ff_weight_norm),
        )

    def forward(self, x, grid=None, **kwargs):
        if grid is not None:
            x = torch.cat((x, grid), dim=-1)

        # Input projection: [b, n1, n2, in_dim] -> [b, n1, n2, c]
        x = self.in_proj(x)
        x = self.drop(x)

        # Initialize two separate residual chains
        x_basic = x.clone()  # For basic operators
        x_coeff = x.clone()  # For coefficient operators

        forecast = 0
        forecast_list = []

        # Process through layers with separate residual chains
        for i in range(self.n_layers):
            basic_operator = self.basic_operators[i]
            coefficient_operator = self.coefficient_operators[i]

            # Basic Operator Chain: X_basic_i -> B_i with residual connection
            B_i, _ = basic_operator(x_basic)  # Only use backcast output
            x_basic = x_basic + B_i  # Residual connection for basic chain

            # Coefficient Operator Chain: X_coeff_i -> C_i with residual connection
            C_i = coefficient_operator(x_coeff)
            x_coeff = x_coeff + C_i  # Residual connection for coefficient chain

            if self.use_fork:
                # Intermediate forecast using current B_i and C_i
                # This is optional - you might want to remove this if only final result matters
                pass

        # After all layers, get final B_i and C_i
        final_B_i = x_basic
        final_C_i_raw = x_coeff  # This is still in width dimension

        # Project final coefficient to coefficient_dim
        final_C_i = self.final_coeff_proj(final_C_i_raw)  # [b, n1, n2, coefficient_dim]

        # Now apply solution operator to final results
        solution_operator = self.solution_operators[-1]  # Use the last solution operator

        # Solution Operator: final_C_i -> \hat{C_i} or [final_B_i, final_C_i] -> \hat{C_i}
        if self.coefficient_only:
            # Only use final coefficient
            C_hat_i, _ = solution_operator(final_C_i)
        else:
            # Concatenate final basic and coefficient
            combined_input = torch.cat([final_B_i, final_C_i], dim=-1)
            C_hat_i, _ = solution_operator(combined_input)

        # Final outputs using the processed results
        # forecast: final_B_i * \hat{C_i} -> MLP -> [b, n1, n2, out_dim]
        forecast_input = final_B_i * C_hat_i  # Element-wise multiplication
        forecast = self.out(forecast_input)

        # res_out: final_B_i * final_C_i -> [b, n1, n2, c]
        res_out = final_B_i * final_C_i  # Element-wise multiplication

        # basic_out: final_B_i -> [b, n1, n2, c]
        basic_out = final_B_i

        # Add dimension for compatibility
        forecast = torch.unsqueeze(forecast, dim=-2)

        return forecast, res_out, basic_out


class SerialOrthoSolverBlock1D(nn.Module):
    """1D OrthoSolverBlock1D"""

    def __init__(
        self,
        modes,
        width,
        input_dim=12,
        output_dim=5,
        dropout=0.0,
        in_dropout=0.0,
        n_layers=4,
        share_weight: bool = False,
        share_fork=False,
        factor=2,
        ff_weight_norm=False,
        n_ff_layers=2,
        gain=1,
        layer_norm=False,
        use_fork=False,
        mode="full",
        coefficient_dim=1,
        coefficient_only=True,
    ):
        # input: X_i: [b, n1, in_dim] -> [b, n1, c]
        # Basic Operator: X_i: [b, n1, c] -> B_i: [b, n1, c]
        # Coefficient Operator: X_i: [b, n1, c] -> C_i: [b, n1, 1]
        # Solution Operator:
        ##  Coefficient only: C_i: [b, n1, 1] -> [b, n1, 1]
        ##  Coefficient + Basic: B_i: [b, n1, c] + C_i: [b, n1, 1] -> \hat{C_i} :[b, n1, 1]

        # forecast: B_i * \hat{C_i}: [b, n1, c] * [b, n1, 1] -> MLP -> [b, n1, out_dim]
        # res_out: B_i * C_i: [b, n1, c] * [b, n1, 1] -> [b, n1, c]
        # basic_out: B_i: [b, n1, c]
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coefficient_dim = coefficient_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork
        self.coefficient_only = coefficient_only
        DecomSpectralConv1d = DecomConvList[1]

        # Shared components
        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        # Basic Operators: X_i -> B_i
        self.basic_operators = nn.ModuleList([])
        for _ in range(n_layers):
            self.basic_operators.append(
                DecomSpectralConv1d(
                    in_channels=width,
                    out_channels=width,
                    modes1=modes,
                )
            )

        # Coefficient Operators: X_i -> C_i
        self.coefficient_operators = nn.ModuleList([])
        for _ in range(n_layers):
            # self.coefficient_operators.append(
            #     MLP(
            #         n_input=width,
            #         n_hidden=width * factor,
            #         n_layers=n_ff_layers,
            #         n_output=self.coefficient_dim,
            #         act='relu'
            #     )
            # )
            self.coefficient_operators.append(
                PreNorm(
                    dim=width,
                    fn=MLP(
                        n_input=width,
                        n_hidden=width * factor,
                        n_layers=n_ff_layers,
                        n_output=self.coefficient_dim,
                        act="relu",
                    ),
                )
            )

        # Solution Operators: C_i -> \hat{C_i} or [B_i, C_i] -> \hat{C_i}
        self.solution_operators = nn.ModuleList([])
        for _ in range(n_layers):
            solution_input_dim = coefficient_dim if coefficient_only else (width + coefficient_dim)
            self.solution_operators.append(
                DecomSpectralConv1d(
                    in_channels=solution_input_dim,
                    out_channels=coefficient_dim,
                    modes1=modes,
                )
            )

        # Final output layer for forecast: B_i * \hat{C_i} -> forecast
        self.out = nn.Sequential(
            WNLinear(self.width, self.width * factor, wnorm=ff_weight_norm),
            nn.ReLU(),
            WNLinear(self.width * factor, self.output_dim, wnorm=ff_weight_norm),
        )
        self.out = PreNorm(dim=width, fn=self.out)

    def forward(self, x, grid=None, **kwargs):
        if grid is not None:
            x = torch.cat((x, grid), dim=-1)

        # Input projection: [b, n1, in_dim] -> [b, n1, c]
        x = self.in_proj(x)
        x = self.drop(x)

        forecast = 0
        forecast_list = []

        for i in range(self.n_layers):
            basic_operator = self.basic_operators[i]
            coefficient_operator = self.coefficient_operators[i]
            solution_operator = self.solution_operators[i]

            # Basic Operator: X_i -> B_i [b, n1, c]
            # For 1D, we need to transpose for convolution: [b, n1, c] -> [b, c, n1]
            x_transposed = x.transpose(-1, -2)
            B_i_transposed = basic_operator(x_transposed)
            B_i = B_i_transposed.transpose(-1, -2)  # Back to [b, n1, c]

            # Coefficient Operator: X_i -> C_i [b, n1, 1]
            C_i = coefficient_operator(x)

            # Solution Operator: C_i -> \hat{C_i} or [B_i, C_i] -> \hat{C_i}
            if self.coefficient_only:
                # Only use coefficient
                C_i_transposed = C_i.transpose(-1, -2)
                C_hat_i_transposed = solution_operator(C_i_transposed)
                C_hat_i = C_hat_i_transposed.transpose(-1, -2)
            else:
                # Concatenate basic and coefficient
                combined_input = torch.cat([B_i, C_i], dim=-1)
                combined_transposed = combined_input.transpose(-1, -2)
                C_hat_i_transposed = solution_operator(combined_transposed)
                C_hat_i = C_hat_i_transposed.transpose(-1, -2)

            if self.use_fork:
                # forecast: B_i * \hat{C_i} -> MLP -> [b, n1, out_dim]
                forecast_input = B_i * C_hat_i  # Element-wise multiplication
                f_out = self.out(forecast_input)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            # Update x for next layer using residual connection
            x = x + B_i

        if not self.use_fork:
            # If not using fork, output from the final state
            forecast = self.out(x)

        # res_out: B_i * C_i -> [b, n1, c]
        res_out = B_i * C_i  # Element-wise multiplication, output has shape [b, n1, c]

        # basic_out: B_i -> [b, n1, c]
        basic_out = B_i

        # Add dimension for compatibility
        forecast = torch.unsqueeze(forecast, dim=-2)

        return forecast, res_out, basic_out


class SerialOrthoSolverBlock3D(nn.Module):
    """3D OrthoSolverBlock3D"""

    def __init__(
        self,
        modes,
        width,
        input_dim=12,
        output_dim=5,
        dropout=0.0,
        in_dropout=0.0,
        n_layers=4,
        share_weight: bool = False,
        share_fork=False,
        factor=2,
        ff_weight_norm=False,
        n_ff_layers=2,
        gain=1,
        layer_norm=False,
        use_fork=False,
        mode="full",
        coefficient_dim=1,
        coefficient_only=True,
    ):
        # input: X_i: [b, n1, n2, n3, in_dim] -> [b, n1, n2, n3, c]
        # Basic Operator: X_i: [b, n1, n2, n3, c] -> B_i: [b, n1, n2, n3, c]
        # Coefficient Operator: X_i: [b, n1, n2, n3, c] -> C_i: [b, n1, n2, n3, 1]
        # Solution Operator:
        ##  Coefficient only: C_i: [b, n1, n2, n3, 1] -> [b, n1, n2, n3, 1]
        ##  Coefficient + Basic: B_i: [b, n1, n2, n3, c] + C_i: [b, n1, n2, n3, 1] -> \hat{C_i} :[b, n1, n2, n3, 1]

        # forecast: B_i * \hat{C_i}: [b, n1, n2, n3, c] * [b, n1, n2, n3, 1] -> MLP -> [b, n1, n2, n3, out_dim]
        # res_out: B_i * C_i: [b, n1, n2, n3, c] * [b, n1, n2, n3, 1] -> [b, n1, n2, n3, c]
        # basic_out: B_i: [b, n1, n2, n3, c]
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coefficient_dim = coefficient_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork
        self.coefficient_only = coefficient_only
        DecomSpectralConv3d = DecomConvList[3]

        # Shared components
        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        # Basic Operators: X_i -> B_i
        self.basic_operators = nn.ModuleList([])
        for _ in range(n_layers):
            self.basic_operators.append(
                DecomSpectralConv3d(
                    in_channels=width,
                    out_channels=width,
                    modes1=modes,
                    modes2=modes,
                    modes3=modes,
                )
            )

        # Coefficient Operators: X_i -> C_i
        self.coefficient_operators = nn.ModuleList([])
        for _ in range(n_layers):
            # self.coefficient_operators.append(
            #     MLP(
            #         n_input=width,
            #         n_hidden=width * factor,
            #         n_layers=n_ff_layers,
            #         n_output=self.coefficient_dim,
            #         act='relu'
            #     )
            # )
            self.coefficient_operators.append(
                PreNorm(
                    dim=width,
                    fn=MLP(
                        n_input=width,
                        n_hidden=width * factor,
                        n_layers=n_ff_layers,
                        n_output=self.coefficient_dim,
                        act="relu",
                    ),
                )
            )

        # Solution Operators: C_i -> \hat{C_i} or [B_i, C_i] -> \hat{C_i}
        self.solution_operators = nn.ModuleList([])
        for _ in range(n_layers):
            solution_input_dim = coefficient_dim if coefficient_only else (width + coefficient_dim)
            self.solution_operators.append(
                DecomSpectralConv3d(
                    in_channels=solution_input_dim,
                    out_channels=coefficient_dim,
                    modes1=modes,
                    modes2=modes,
                    modes3=modes,
                )
            )

        # Final output layer for forecast: B_i * \hat{C_i} -> forecast
        self.out = nn.Sequential(
            WNLinear(self.width, self.width * factor, wnorm=ff_weight_norm),
            nn.ReLU(),
            WNLinear(self.width * factor, self.output_dim, wnorm=ff_weight_norm),
        )
        self.out = PreNorm(dim=width, fn=self.out)

    def forward(self, x, grid=None, **kwargs):
        if grid is not None:
            x = torch.cat((x, grid), dim=-1)

        # Input projection: [b, n1, n2, n3, in_dim] -> [b, n1, n2, n3, c]
        x = self.in_proj(x)
        x = self.drop(x)

        forecast = 0
        forecast_list = []

        for i in range(self.n_layers):
            basic_operator = self.basic_operators[i]
            coefficient_operator = self.coefficient_operators[i]
            solution_operator = self.solution_operators[i]

            # Basic Operator: X_i -> B_i [b, n1, n2, n3, c]
            # For 3D, we need to transpose for convolution: [b, n1, n2, n3, c] -> [b, c, n1, n2, n3]
            x_transposed = x.permute(0, -1, 1, 2, 3)
            B_i_transposed = basic_operator(x_transposed)
            B_i = B_i_transposed.permute(0, 2, 3, 4, 1)  # Back to [b, n1, n2, n3, c]

            # Coefficient Operator: X_i -> C_i [b, n1, n2, n3, 1]
            C_i = coefficient_operator(x)

            # Solution Operator: C_i -> \hat{C_i} or [B_i, C_i] -> \hat{C_i}
            if self.coefficient_only:
                # Only use coefficient
                C_i_transposed = C_i.permute(0, -1, 1, 2, 3)
                C_hat_i_transposed = solution_operator(C_i_transposed)
                C_hat_i = C_hat_i_transposed.permute(0, 2, 3, 4, 1)
            else:
                # Concatenate basic and coefficient
                combined_input = torch.cat([B_i, C_i], dim=-1)
                combined_transposed = combined_input.permute(0, -1, 1, 2, 3)
                C_hat_i_transposed = solution_operator(combined_transposed)
                C_hat_i = C_hat_i_transposed.permute(0, 2, 3, 4, 1)

            if self.use_fork:
                # forecast: B_i * \hat{C_i} -> MLP -> [b, n1, n2, n3, out_dim]
                forecast_input = B_i * C_hat_i  # Element-wise multiplication
                f_out = self.out(forecast_input)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            # Update x for next layer using residual connection
            x = x + B_i

        if not self.use_fork:
            # If not using fork, output from the final state
            forecast = self.out(x)

        # res_out: B_i * C_i -> [b, n1, n2, n3, c]
        res_out = B_i * C_i  # Element-wise multiplication, output has shape [b, n1, n2, n3, c]

        # basic_out: B_i -> [b, n1, n2, n3, c]
        basic_out = B_i

        # Add dimension for compatibility
        forecast = torch.unsqueeze(forecast, dim=-2)

        return forecast, res_out, basic_out


# OrthoSolver Block list for different dimensions
OrthoSolverBlockList = [None, SerialOrthoSolverBlock1D, SerialOrthoSolverBlock2D, SerialOrthoSolverBlock3D]
