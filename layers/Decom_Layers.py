import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.Basic import WNLinear, WNFeedForward


class DecomSpectralConv1d(nn.Module):
    """1D Spectral Convolution for Decomposition FNO"""
    def __init__(self, in_channels, out_channels, modes1):
        super(DecomSpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, : self.modes1] = self.compl_mul1d(x_ft[:, :, : self.modes1], self.weights1)

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class DecomSpectralConv2d(nn.Module):
    """2D Spectral Convolution for Decomposition FNO"""
    def __init__(
        self,
        in_dim,
        out_dim,
        n_modes,
        forecast_ff,
        backcast_ff,
        fourier_weight,
        factor,
        ff_weight_norm,
        n_ff_layers,
        layer_norm,
        use_fork,
        dropout,
        mode,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.mode = mode
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(4):
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = WNFeedForward(out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = WNFeedForward(out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        if self.mode != "no-fourier":
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, "b m n i -> b i m n")
        B, I, M, N = x.shape

        # Y dimension
        x_fty = torch.fft.rfft(x, dim=-1, norm="ortho")
        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)

        if self.mode == "full":
            out_ft[:, :, :, : self.n_modes] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, : self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]),
            )
            out_ft[:, :, :, -self.n_modes :] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, -self.n_modes :],
                torch.view_as_complex(self.fourier_weight[2]),
            )
        elif self.mode == "low-pass":
            out_ft[:, :, :, : self.n_modes] = x_fty[:, :, :, : self.n_modes]

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm="ortho")

        # X dimension
        x_ftx = torch.fft.rfft(x, dim=-2, norm="ortho")
        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)

        if self.mode == "full":
            out_ft[:, :, : self.n_modes, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, : self.n_modes, :],
                torch.view_as_complex(self.fourier_weight[1]),
            )
            out_ft[:, :, -self.n_modes :, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, -self.n_modes :, :],
                torch.view_as_complex(self.fourier_weight[3]),
            )
        elif self.mode == "low-pass":
            out_ft[:, :, : self.n_modes, :] = x_ftx[:, :, : self.n_modes, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm="ortho")

        # Combining dimensions
        x = xx + xy
        x = rearrange(x, "b i m n -> b m n i")
        return x


class DecomSpectralConv3d(nn.Module):
    """3D Spectral Convolution for Decomposition FNO"""
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(DecomSpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNOFactorizedBlock1D(nn.Module):
    """1D Factorized FNO Block"""
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
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork
        
        # Shared components
        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        # 1D Spectral layers
        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(
                DecomSpectralConv1d(
                    in_channels=width,
                    out_channels=width,
                    modes1=modes,
                )
            )

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, self.output_dim, wnorm=ff_weight_norm),
        )
        self.res_out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, self.width, wnorm=ff_weight_norm),
        )

    def forward(self, x, grid=None, **kwargs):
        if grid is not None:
            x = torch.cat((x, grid), dim=-1)

        forecast = 0
        x = self.in_proj(x)
        x = self.drop(x)
        # [B, N, C] -> [B, C, N]
        # 1D processing
        x = x.transpose(-1, -2)  # [batch, width, seq_len]
        forecast_list = []

        for i in range(self.n_layers):
            x_fourier = self.spectral_layers[i](x)
            x_fourier = x_fourier.transpose(-1, -2)  # [batch, seq_len, width]

            if self.backcast_ff:
                b = self.backcast_ff(x_fourier)
            else:
                b = x_fourier

            if self.use_fork:
                if self.forecast_ff:
                    f = self.forecast_ff(x_fourier)
                else:
                    f = x_fourier

                f_out = self.out(f)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            x = x + b.transpose(-1, -2)

        x = x.transpose(-1, -2)

        if not self.use_fork:
            forecast = self.out(x)
            res_out = self.res_out(x)
            signal_out = x
        else:
            res_out = self.res_out(x)
            signal_out = x

        # Add dimension for compatibility
        forecast = torch.unsqueeze(forecast, dim=-2)
        
        return forecast, res_out, signal_out


class FNOFactorizedBlock2D(nn.Module):
    """2D Factorized FNO Block"""
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
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork
        
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

        # 2D Spectral layers
        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(
                DecomSpectralConv2d(
                    in_dim=width,
                    out_dim=width,
                    n_modes=modes,
                    forecast_ff=self.forecast_ff,
                    backcast_ff=self.backcast_ff,
                    fourier_weight=self.fourier_weight,
                    factor=factor,
                    ff_weight_norm=ff_weight_norm,
                    n_ff_layers=n_ff_layers,
                    layer_norm=layer_norm,
                    use_fork=use_fork,
                    dropout=dropout,
                    mode=mode,
                )
            )

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, self.output_dim, wnorm=ff_weight_norm),
        )
        self.res_out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, self.width, wnorm=ff_weight_norm),
        )

    def forward(self, x, grid=None, **kwargs):
        if grid is not None:
            x = torch.cat((x, grid), dim=-1)

        forecast = 0
        x = self.in_proj(x)
        x = self.drop(x)
        
        # 2D processing
        forecast_list = []
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, f = layer(x)

            if self.use_fork:
                f_out = self.out(f)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            x = x + b

        if not self.use_fork:
            forecast = self.out(x)
            res_out = self.res_out(x)
            signal_out = x
        else:
            res_out = self.res_out(x)
            signal_out = x
        forecast = torch.unsqueeze(forecast, dim=-2)
        return forecast, res_out, signal_out


class FNOFactorizedBlock3D(nn.Module):
    """3D Factorized FNO Block"""
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
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork
        
        # Shared components
        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = WNFeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        # 3D Spectral layers
        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(
                DecomSpectralConv3d(
                    in_channels=width,
                    out_channels=width,
                    modes1=modes,
                    modes2=modes,
                    modes3=modes,
                )
            )

        # Additional processing layers for 3D
        self.spectral_processing = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_processing.append(
                nn.Sequential(
                    WNLinear(width, width, wnorm=ff_weight_norm),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, self.output_dim, wnorm=ff_weight_norm),
        )
        self.res_out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, self.width, wnorm=ff_weight_norm),
        )

    def forward(self, x, grid=None, **kwargs):
        if grid is not None:
            x = torch.cat((x, grid), dim=-1)

        forecast = 0
        x = self.in_proj(x)
        x = self.drop(x)
        
        # 3D processing
        x = x.transpose(-1, -2)  # [batch, width, d1, d2, d3]
        forecast_list = []

        for i in range(self.n_layers):
            x_fourier = self.spectral_layers[i](x)
            x_fourier = x_fourier.transpose(-1, -2)  # [batch, d1, d2, d3, width]
            
            # Additional processing for 3D
            x_processed = self.spectral_processing[i](x_fourier)

            if self.backcast_ff:
                b = self.backcast_ff(x_processed)
            else:
                b = x_processed

            if self.use_fork:
                if self.forecast_ff:
                    f = self.forecast_ff(x_processed)
                else:
                    f = x_processed

                f_out = self.out(f)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            x = x + b.transpose(-1, -2)

        x = x.transpose(-1, -2)

        if not self.use_fork:
            forecast = self.out(x)
            res_out = self.res_out(x)
            signal_out = x
        else:
            res_out = self.res_out(x)
            signal_out = x
        forecast = torch.unsqueeze(forecast, dim=-2)
        return forecast, res_out, signal_out


# Block list for different dimensions
DecomBlockList = [None, FNOFactorizedBlock1D, FNOFactorizedBlock2D, FNOFactorizedBlock3D]

# Convolution list for different dimensions  
DecomConvList = [None, DecomSpectralConv1d, DecomSpectralConv2d, DecomSpectralConv3d]

