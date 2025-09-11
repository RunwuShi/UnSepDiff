import os
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F


class downsample(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                padding,
                pool_kernel):
        super().__init__()
        self.layer_norm0 = nn.LayerNorm(in_channels)
        self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.downsample_mean = nn.AvgPool1d(kernel_size=pool_kernel)
        # self.downsample_max = nn.MaxPool1d(kernel_size=pool_kernel)

        self.activation = nn.GELU()


    def forward(self, x):
        h = self.layer_norm0(x.permute(0, 2, 1)).permute(0, 2, 1) # [batch, F, T]
        h = self.conv0(h)
        h = self.activation(h)

        h = self.layer_norm1(h.permute(0, 2, 1)).permute(0, 2, 1) # [batch, F, T]
        h = self.conv1(h)
        h = self.activation(h)

        h = self.downsample_mean(h)

        resi = self.conv2(x)
        resi = self.downsample_mean(resi)

        out = h + resi

        return out

class SingleAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(SingleAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.W_q = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_k = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_v = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_o = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        B, T, _ = x.size()

        x = self.norm(x)
        
        Q = torch.matmul(x, self.W_q)  # [B, T, hidden_dim]
        K = torch.matmul(x, self.W_k)  # [B, T, hidden_dim]
        V = torch.matmul(x, self.W_v)  # [B, T, hidden_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=Q.dtype, device=Q.device))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        out = torch.matmul(out, self.W_o)
        
        out = x + out
        
        return out


class stftpreprocess(nn.Module):
    def __init__(self,                 
                n_fft: int,
                hop_length: int,
                win_length: int, 
                press,
                **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(self.win_length)
        window = window / window.sum()   
        window = window.pow(0.5)         
        self.register_buffer('stft_window', window)
        
        self.press = press
        
    def forward(self, x, spec_type = "complex"):
        """
        :param x: [B, wave length]
        :return: [B, F, T] complex
        """
        x = x / (x.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        pad = (self.n_fft - self.hop_length) // 2
        x = x.unsqueeze(1)  # [B, 1, wave_length]
        x = F.pad(x, (pad, pad), mode='reflect')
        x = x.squeeze(1)  # [B, wave_length + 2*pad]
        
        # stft
        spec_ori = torch.stft(x, 
                            n_fft = self.n_fft, 
                            hop_length = self.hop_length, 
                            win_length = self.win_length, 
                            window = self.stft_window, 
                            return_complex=True)
        
        # compress complex
        if spec_type == "complex":
            if self.press == "log":
                spec = torch.log(torch.clamp(spec_ori.abs(), min=1e-5)) * torch.exp(1j * spec_ori.angle())  # [B, F, T], complex
            else:
                spec = torch.pow(spec_ori.abs(), self.press) * torch.exp(1j * spec_ori.angle())  # [B, F, T], complex
        elif spec_type == "amplitude":
            if self.press == "log":
                spec = torch.log(torch.clamp(spec_ori.abs(), min=1e-5))   # [B, F, T], complex
            else:
                spec = torch.pow(spec_ori.abs(), self.press)  # [B, F, T], complex
        
        return spec



class mix_src_encoder(nn.Module):
    def __init__(self, 
                h_channels,
                config,
                **kwargs):
        super().__init__()

        # preprocess
        self.preprocess = stftpreprocess(**config['FFT'], press = 0.5)

        # fft
        n_fft = config['FFT']['n_fft']
        self.n_freqs = int(n_fft // 2 + 1) # 257

        # conv2d
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

        # conv1d
        h_channels = h_channels
        self.conv2 = nn.Conv1d(in_channels=self.n_freqs, out_channels=h_channels, kernel_size=3, stride=1,padding=1)

        # down sample
        conv_kernels = [3, 3, 3, 3, 5]
        paddings = [1, 1, 1, 1, 2] 
        pool_kernels = [1, 1, 1, 2, 2]

        self.layers = nn.ModuleList(
            [downsample(in_channels = h_channels, 
                        out_channels = h_channels, 
                        kernel_size=k, 
                        padding=p,
                        pool_kernel=pk)
            for k, p, pk in zip(conv_kernels, paddings, pool_kernels)])

        self.activation = nn.GELU()
        # atten out 
        self.proj = nn.Linear(h_channels, 1) # nn.Linear(62, 64)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # preprocess
        spec = self.preprocess(x) # [batch, 2 (real, img), F, T]s
        specs = torch.stack([spec.real, spec.imag], dim = 1)

        # conv2d
        h = self.conv1(specs).squeeze(1) # [batch, F, T]
        h = self.activation(h)
        
        # conv1d
        h = self.conv2(h) # [batch, h channel, T], [64, 126]
        h = self.activation(h)

        # layers, [64, 126] -> [64, 63]-> [64, 32] -> [64, 15]
        for conv in self.layers:
            h = conv(h)

        h = self.activation(h)
        h = h.transpose(-1,-2)
        h = self.proj(h) 
        h = self.sigmoid(h).squeeze(-1)
        out = torch.mean(h, dim=-1) # [B]
        
        return out



