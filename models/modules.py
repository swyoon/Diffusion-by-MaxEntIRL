import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.parametrizations as P
from torch.nn import utils


def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'swish':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')

class ResBlockV2(nn.Module):
    def __init__(self, in_channel, out_channel, n_class=None, downsample=False, use_spectral_norm=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel,
                               out_channel,
                               3,
                               padding=1,
                               bias=False if n_class is not None else True)

        self.conv2 = nn.Conv2d(out_channel,
                               out_channel,
                               3,
                               padding=1,
                               bias=False if n_class is not None else True)

        if use_spectral_norm:
            self.conv1 = P.spectral_norm(self.conv1)
            self.conv2 = P.spectral_norm(self.conv2)

        self.class_embed = None

        if n_class is not None:
            class_embed = nn.Embedding(n_class, out_channel * 2 * 2)
            class_embed.weight.data[:, : out_channel * 2] = 1
            class_embed.weight.data[:, out_channel * 2 :] = 0

            self.class_embed = class_embed

        self.skip = None

        if in_channel != out_channel or downsample:
            if use_spectral_norm:
                self.skip = nn.Sequential(
                    P.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, bias=False))
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 1, bias=False)
                )

        self.downsample = downsample

    def forward(self, input, class_id=None):
        out = input

        out = self.conv1(out)

        if self.class_embed is not None:
            embed = self.class_embed(class_id).view(input.shape[0], -1, 1, 1)
            weight1, weight2, bias1, bias2 = embed.chunk(4, 1)
            out = weight1 * out + bias1

        out = F.leaky_relu(out, negative_slope=0.2)

        out = self.conv2(out)

        if self.class_embed is not None:
            out = weight2 * out + bias2

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = F.leaky_relu(out, negative_slope=0.2)

        return out


class IGEBMEncoderV2(nn.Module):
    """Neural Network used in IGEBM
    replace spectral norm implementation
    add learn_out_scaling"""
    def __init__(self, in_chan=3, out_chan=1, n_class=None, use_spectral_norm=False, keepdim=True,
                 out_activation='linear', avg_pool_dim=1, learn_out_scale=False, nh=128):
        super().__init__()
        self.keepdim = keepdim
        self.use_spectral_norm = use_spectral_norm
        self.avg_pool_dim = avg_pool_dim

        if use_spectral_norm:
            self.conv1 = P.spectral_norm(nn.Conv2d(in_chan, nh, 3, padding=1))
        else:
            self.conv1 = nn.Conv2d(in_chan, nh, 3, padding=1)

        self.blocks = nn.ModuleList(
            [
                ResBlockV2(nh, nh, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                ResBlockV2(nh, nh, n_class, use_spectral_norm=use_spectral_norm),
                ResBlockV2(nh, nh*2, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                ResBlockV2(nh*2, nh*2, n_class, use_spectral_norm=use_spectral_norm),
                ResBlockV2(nh*2, nh*2, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                ResBlockV2(nh*2, nh*2, n_class, use_spectral_norm=use_spectral_norm),
            ]
        )

        if keepdim:
            self.linear = nn.Conv2d(nh*2, out_chan, 1, 1, 0)
        else:
            self.linear = nn.Linear(nh*2, out_chan)

        self.out_activation = get_activation(out_activation)
        self.pre_activation = None
        self.learn_out_scale = learn_out_scale
        if learn_out_scale:
            self.out_scale = nn.Linear(1, 1, bias=True)

    def forward(self, input, y=None):
        out = self.conv1(input)

        out = F.leaky_relu(out, negative_slope=0.2)

        for block in self.blocks:
            out = block(out, y)

        out = F.relu(out)
        if self.keepdim:
            out = F.adaptive_avg_pool2d(out, (self.avg_pool_dim, self.avg_pool_dim))
        else:
            out = out.view(out.shape[0], out.shape[1], -1).sum(2)

        out = self.linear(out)
        if self.learn_out_scale:
            out = self.out_scale(out)
        self.pre_activation = out
        if self.out_activation is not None:
            out = self.out_activation(out)

        return out

    def load_pretrained(self, ckpt):
        """load conv1 and blocks from ckpt"""
        conv1_dict = {}
        blocks_dict = {}
        for k, v in ckpt['state_dict'].items():
            # remove 'net.' in k
            k.strip('net.')

            k_ = k[4:]
            if k_.startswith('conv1'):
                conv1_dict[k_[len('conv1.'):]] = v
            elif k_.startswith('blocks'):
                blocks_dict[k_[len('blocks.'):]] = v

        self.conv1.load_state_dict(conv1_dict)
        self.blocks.load_state_dict(blocks_dict)


def process_single_t(x, t):
    """make single integer t into a vector of an appropriate size"""
    if isinstance(t, int) or len(t.shape) == 0 or len(t) == 1:
        t = torch.ones([x.shape[0]], dtype=torch.long, device=x.device) * t
    return t