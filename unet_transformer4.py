import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock1D(nn.Module):
    def __init__(self, dim, heads=8, inner_dim=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, depth, model_dim, heads, inner_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock1D(model_dim, heads, inner_dim) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerUNet(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=96, max_hidden_dim=1024,
                 encoder_depth=6, decoder_depth=6, bottleneck_depth=24,
                 kernel_size=5, stride=2, attention_heads=16,
                 model_dim=512, inner_dim= 2048,
                 residual=True):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.align_convs = nn.ModuleList()
        self.skip_dims = []
        self.residual = residual

        # Encoder
        in_channels = input_channels
        for i in range(encoder_depth):
            out_channels = min(hidden_dim * (2 ** i), max_hidden_dim)
            self.encoder.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
            self.skip_dims.append(out_channels)
            in_channels = out_channels

        # Bottleneck: Conv1D -> Transformer -> Conv1D
        self.bottleneck_proj = nn.Linear(in_channels, model_dim)
        self.pos_encoding = PositionalEncoding1D(model_dim)
        self.transformer = TransformerStack(bottleneck_depth, model_dim, attention_heads, inner_dim)
        self.bottleneck_reproj = nn.Linear(model_dim, in_channels)

        # Decoder
        for i in range(decoder_depth):
            out_channels = self.skip_dims[-(i + 1)]
            self.decoder.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, output_padding=1))
            self.align_convs.append(nn.Conv1d(out_channels, out_channels, 1))
            in_channels = out_channels

        self.final_conv = nn.Conv1d(in_channels, input_channels, kernel_size=1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        original_input = x
        skips = []

        # Encoder
        for layer in self.encoder:
            x = F.gelu(layer(x))
            skips.append(x)

        # Bottleneck
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        x = self.bottleneck_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.bottleneck_reproj(x)
        x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

        # Decoder with skip connections
        for i, (layer, align) in enumerate(zip(self.decoder, self.align_convs)):
            x = F.gelu(layer(x))
            skip = skips[-(i + 1)]
            if skip.shape[-1] != x.shape[-1]:
                skip = F.interpolate(skip, size=x.shape[-1], mode='nearest')
            x = x + align(skip)

        out = self.final_conv(x)
        return out + original_input if self.residual else out

    def get_loss(self, model_outputs, ground_truth):
        return self.criterion(model_outputs, ground_truth)
