import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pydensecrf.densecrf as dcrf
import numpy as np
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

from .dinats import NATransformerLayer as NAattention
from functools import partial
from .sgformer import SGBlock
# from .dinats import NATransformerLayer as NAattention
from .utils import LayerNorm, GRN
from .van import LKABlock
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from .ConvNext import ConvBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False).to(device)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False).to(device) # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1).to(device)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False).to(device),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False).to(device),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class aspp_attention(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(aspp_attention, self).__init__()

        self.se = Squeeze_Excite_Block(out_dims)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)).to(device),
            nn.Conv2d(in_dims, out_dims, 1, 1).to(device),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True),
            # Squeeze_Excite_Block(out_dims)
        )

        self.aspp_block0 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 1, stride=1).to(device),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True),
            # Squeeze_Excite_Block(out_dims)
        )

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ).to(device),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True),
            # Squeeze_Excite_Block(out_dims)
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ).to(device),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True),
            # Squeeze_Excite_Block(out_dims)
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ).to(device),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True),
            # Squeeze_Excite_Block(out_dims)
        )

        self.se = Squeeze_Excite_Block(5*out_dims)

        self.output = nn.Conv2d(5 * out_dims, out_dims, 1)

    def forward(self, x):
        size = x.shape[2:]
        x_ = self.pool(x).to(device)
        x_ = F.interpolate(x_, size=size, mode='bilinear', align_corners=False)
        x = x.to(device)
        # x0 = self.cbam(self.aspp_block0(x))
        # x1 = self.cbam(self.aspp_block1(x))
        # x2 = self.cbam(self.aspp_block2(x))
        # x3 = self.cbam(self.aspp_block3(x))
        x0 = self.aspp_block0(x)
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x_, x0, x1, x2, x3], dim=1)
        out = self.se(out)

        return self.output(out)


#
class Conv2Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features             #in:192,out:192   in:384,out:384
        hidden_features = hidden_features or in_features       #hidden:768=192*4   hidden:1536=384*4
        self.fc1 = nn.Linear(in_features, hidden_features)     #in:192,hidden:768   in:768,hidden:3072
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)    #hidden:768,out:192   hidden:3072  out:768
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape   #(32,56,56,96)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)   #(32,8,7,8,7,96)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]num_windows=(H//Mh)*(W//Mw)即窗口的个数
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)   #(2048,7,7,96)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim   #96 192  384 768
        self.window_size = window_size  # Wh, Ww（7，7）
        self.num_heads = num_heads      #num_heads=3  6  12 24
        head_dim = dim // num_heads  #96//3=32  192//6=32 每个head对应的dimention
        self.scale = qk_scale or head_dim ** -0.5     #根号d分之一

        # define a parameter table of relative position bias 每个head的relative_position_bias_table不同
        self.relative_position_bias_table = nn.Parameter(   #relative_position_bias_table是一个169行3列
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])   #coords_h：tensor([0,1,2,3,4,5,6]
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww（2，7，7）
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Mh*Mw, Mh*Mw
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Mh*Mw, Mh*Mw, 2
        #将二元索引变为一元索引的过程
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0(49,49,2)即所有图像块相对于x轴的位置信息
        #所有行+(M-1)
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1   #行标*(2M-1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww（49，49），相对位置索引
        self.register_buffer("relative_position_index", relative_position_index)  #将它放入模型缓存区,不变

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   #一个全连接层，一次性得到qkvin:96,out=96*3=288    in:768,out=768*3=2304  in_features=384, out_features=1152
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)    #多个head拼接，然后通过Wo映射 in:96, out:96   in:768, out:768
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)   #relative_position_bias_table:(169,12)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape  #(2048,49,96)
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    #(3,2048,3,49,32)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)切片

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        #针对每个heads操作
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))#最后两个维度相乘

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)  #unsqueeze方法加入batch_size这个维度

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # attn.view: [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn) #不同区域softmax变为0
        else:
            attn = self.softmax(attn) #对每一行进行sofmax处理

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]最后两个维度信息拼接在一起
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)  #融合多个head，线性映射
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,kernel_size=7,dilation=1,):
        super().__init__()
        self.dim = dim   #96 192 384
        self.input_resolution = input_resolution    # （56，56）（28，28）(14，14）（7，7）
        self.num_heads = num_heads    #3 6 12 24
        self.window_size = window_size  #7
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim) #transformer的encoder的第一个layer_norm
        self.attn = WindowAttention( #W-MSA or SW-MSA
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  #768  3072  1536
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:  #shift_size=3=window_size//2
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution   #(28,28）
            #拥有和feature map 一样的通道排列顺序[1, H, W, 1]，方便后续的window partition
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1 初始化
            #h、w各3个切片
            h_slices = (slice(0, -self.window_size),  #[0,-7]
                        slice(-self.window_size, -self.shift_size),  #[-7,-3]
                        slice(-self.shift_size, None)) #[-3,]
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0    #cnt=9个区域
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh,Mw,1] 将img_mask划分成一个一个窗口
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) #[nW, Mh*Mw]-1推理，第二个维度为Mh*Mw
            ## [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]=[nW, Mh*Mw, Mh*Mw]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            #同一区域为0， 不同区域为-100
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        # self.natattn = NATransformerLayer(
        #     dim,
        #     kernel_size=kernel_size,
        #     dilation=dilation,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )

    def forward(self, x):
        H, W = self.input_resolution    #(56,56)输入特征层的高宽
        B, L, C = x.shape     #(32,3136,96)
        assert L == H * W, "input feature has wrong size"

        shortcut = x   #(32,3136,96)
        x = self.norm1(x)
        x = x.view(B, H, W, C)   #(32,56,56,96)

        # cyclic shift
        if self.shift_size > 0: #sw-msa在H，W维度上，从上向下移shift_size，从左向右移shift_size
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:   #w-msa
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C划分窗口
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)#[nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C拼回feature map

        # reverse cyclic shift
        if self.shift_size > 0:#sw-msa在H，W维度上，从下向上移shift_size，从右向左移shift_size
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # x = self.natattn(x)
        x = x.view(B, H * W, C)


        # FFN
        x = shortcut + self.drop_path(x)  #encoder中的残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x.view(B,H,W,C)
        # x = self.natattn(x)
        # x = x.view(B, H * W, C)


        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  #全连接层，输入4dim
        self.norm = norm_layer(4 * dim)  #在Linear之前使用

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution #input_resolution存入了高宽
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # [B,H/2,W/2,C]蓝色 h,w从0开始，间隔为2取样
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C  绿色 高度为1，宽度为0，间隔为2取样
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C  黄色
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C  绿色
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C -1表示在最后一个维度进行拼接
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)  #通过layernorm对channel进行处理
        x = self.reduction(x)    # [B,H/2*W/2,2*C]

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  #（7，7）
        self.dim = dim       #768
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()    #in:768,0ut=1536
        self.norm = norm_layer(dim // dim_scale)
        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(dim//2),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        # x = self.up(x)
        # x = x.view(B, -1, C//2)
        x = x.view(B,-1,C//4)
        x = self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)    #[B,H*W,C]
        x= self.norm(x)

        return x

class SharedData:
    def __init__(self):
        self.global_list = []  # 实例变量

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,kernel_size=7,
                 drop_path_rate1=0.3,dilations=None,shared_data=None):

        super().__init__()
        self.dim = dim  # 96  192 384 768
        self.input_resolution = input_resolution      #(img_size//patch_size)224//4=56（56，56） （28，28）  (14,14)  (7,7)
        self.depth = depth    #2 2 6
        self.use_checkpoint = use_checkpoint
        self.H=input_resolution[0]
        self.W=input_resolution[1]
        self.shared_data = shared_data
        # self.re = []


        # build blocks
        self.blocks = nn.ModuleList([  #存储当前这个stage中所有swin transformer block
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,   #num_heads=3,6,12,24
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,   #drop_path
                                 norm_layer=norm_layer)
            for i in range(depth)])    #depth =2时，上面循环两次


        dp_rates1 = [x.item() for x in torch.linspace(0, drop_path_rate1, 2)]
        self.na = nn.ModuleList([
            NAattention(
                dim=dim,
                depth=2,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=1 if dilations is None else dilations[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dp_rates1[i]
                if isinstance(dp_rates1, list)
                else dp_rates1,
                norm_layer=norm_layer,
            )
            for i in range(2)
        ])
        # self.fusion = AdaptiveWeightedFusion(self.dim)

        # patch merging layer
        if downsample is not None: #实例化
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # for blk in self.blocks:   #传递每个block,得到输出
        #     if self.use_checkpoint:
        #         x = checkpoint.checkpoint(blk, x)
        #     else:
        #         x = blk(x)   #x=（32，3136，96）
        #
        # B, _, _ = x.shape
        # # c = x
        # # self.re.append(c)
        #
        # a = x.view(B, self.dim, self.H, self.W)
        # b = x.view(B, self.H, self.W, self.dim)
        #
        # for natblk in self.na:
        #     b = natblk(b)
        # b = b.view(B, self.dim, self.H, self.W)
        # x = a + b
        #
        # x = x.view(B, self.H * self.W, self.dim)
        for blk,natblk in zip(self.blocks,self.na):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
                B, _, _ = x.shape
                a = x.view(B, self.dim, self.H, self.W)
                b = x.view(B, self.H, self.W, self.dim)
                b = natblk(b)
                b = b.view(B, self.dim, self.H, self.W)
                x = a + b
                x = x.view(B, self.H * self.W, self.dim)
                # B1, _, _ = x.shape
                # c = x.view(B1, self.H, self.W, self.dim)#
                # x = blk(x)
                # B, _, _ = x.shape
                # a = x.view(B, self.dim, self.H, self.W)
                # b = x.view(B, self.H, self.W, self.dim)
                # b = natblk(b+c)#
                # b = b.view(B, self.dim, self.H, self.W)
                # x = a + b
                # x = x.view(B, self.H * self.W, self.dim)

        if self.downsample is not None: #不为none,就下采样，即patch merging层
            x = self.downsample(x)


        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,kernel_size=7,
                 dilations=None,drop_path_rate1=0.3):

        super().__init__()
        self.dim = dim     #768 2：384 3:96
        self.input_resolution = input_resolution    #2：（14，14）
        self.depth = depth    #2：6
        self.use_checkpoint = use_checkpoint
        self.H = input_resolution[0]
        self.W = input_resolution[1]



        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,   #dim:384 input_resolution:(14,14)
                                 num_heads=num_heads, window_size=window_size,   #num_heads:12
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])  #i:0

         # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate1, 2)]
        self.na = nn.ModuleList([
            NAattention(
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=1 if dilations is None else dilations[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dp_rates[i]
                if isinstance(dp_rates, list)
                else dp_rates,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        # # self.fusion = AdaptiveWeightedFusion(self.dim)



        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # # B, _, _ = x.shape
        # # a = x.view(B, self.dim, self.H, self.W)
        # #
        # # b = x.view(B, self.H, self.W, self.dim)
        # #
        # # for natblk in self.na:
        # #     b = natblk(b)
        # # b = b.view(B, self.dim, self.H, self.W)
        # # x = a+b
        # # x = x.view(B, self.H * self.W, self.dim)
        for blk,natblk in zip(self.blocks,self.na):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
                B, _, _ = x.shape
                a = x.view(B, self.dim, self.H, self.W)

                b = x.view(B, self.H, self.W, self.dim)
                b = natblk(b)
                b = b.view(B, self.dim, self.H, self.W)
                x = a + b
                x = x.view(B, self.H * self.W, self.dim)

        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SGBasiclayer(nn.Module):
    def __init__(self, input_resolution, embed_dims_s, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 num_heads_s=1, mlp_ratios_s=1, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,drop_path=0.,
                 depths_s=1, sr_ratios=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths_s
        # self.num_stages = num_stages
        self.input_resolution = input_resolution
        # self.num_patches = img_size//4

        self.block = nn.ModuleList([SGBlock(dim=embed_dims_s, mask=True if (i%2==1) else False, num_heads=num_heads_s,
           mlp_ratio=mlp_ratios_s, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
             sr_ratio=sr_ratios, linear=linear)
             for i in range(depths_s)])
        self.norm = norm_layer(embed_dims_s)


    def forward(self, x):
        B,_, _= x.shape
        mask=None
        H, W = self.input_resolution
        for blk in self.block:
            x, mask = blk(x, H, W, mask)
        # x = self.norm(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = x.mean(dim=1)

        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)            #（224，224）
        patch_size = to_2tuple(patch_size)        #（4，4)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]    #[56，56]
        self.img_size = img_size                  #（224，224）
        self.patch_size = patch_size              #（4，4)
        self.patches_resolution = patches_resolution    #[56,56]
        self.num_patches = patches_resolution[0] * patches_resolution[1]  #56*56=3136

        self.in_chans = in_chans    #3
        self.embed_dim = embed_dim   #96

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)     #INPUT：3，OUTPUT：96
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape#(32,3,224,224)
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #flatten：[B,C,H,W]->[B,C,HW]
        #transpose:[B,C,HW]->[B,HW,C]
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C(32,3136,96)

        if self.norm is not None:
            x = self.norm(x)# B Ph*Pw C(32,3136,96)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


# # 定义自适应加权融合模块
# class AdaptiveWeightedFusion(nn.Module):
#     def __init__(self, in_channels):
#         super(AdaptiveWeightedFusion, self).__init__()
#
#         # 全连接层
#         self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1).cuda()
#
#         # 激活函数
#         self.activation = nn.Sigmoid()
#
#     def forward(self, feat1, feat2):
#         # 全局平均池化
#         pooled_feat1 = torch.mean(feat1, dim=(2, 3), keepdim=True)
#         pooled_feat2 = torch.mean(feat2, dim=(2, 3), keepdim=True)
#
#         # 全连接层
#         weighted_feat1 = self.fc(pooled_feat1)
#         weighted_feat2 = self.fc(pooled_feat2)
#
#         # 激活函数
#         weighted_feat1 = self.activation(weighted_feat1)
#         weighted_feat2 = self.activation(weighted_feat2)
#
#         # 广播乘法
#         fused_feat = feat1 * weighted_feat1 + feat2 * weighted_feat2
#
#         return fused_feat



class SwinUnetnat_lcab_three(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,drop=0.0,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", depths1=[2, 2, 2],dims1=[96, 192, 384, 768],
                 depths2=[2,2,2], in_channels=[96, 192, 384],dims3=[96, 192, 384],dims4=[96, 192, 384] ,mlp_ratios=[8,8,4],
                 embed_dims=[96, 192, 384], num_heads_s=[6, 12, 24], mlp_ratios_s=[4, 4, 4], norm_layer_s=partial(nn.LayerNorm, eps=1e-6),
                 linear=False, depths_s=[4, 4, 4], sr_ratios=[4, 2, 1], dilations1 =[[1, 16], [1, 8], [1, 4], [1, 2]],dimse=[192,384,768],
                 shared_data=None,**kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        self.num_classes = num_classes #实例化时传入类别数字为1
        self.num_layers = len(depths)  #4,depths=[2,2,6,2]
        self.embed_dim = embed_dim     #96
        self.ape = ape                 #false
        self.patch_norm = patch_norm   #True
        #stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))#768=96*2**3
        self.num_features_up = int(embed_dim * 2)   #192=96*2
        self.mlp_ratio = mlp_ratio    #4.0
        self.final_upsample = final_upsample        #{str}expand_first
        self.channels = in_channels
        self.dims3 = dims3
        self.dims4 = dims4
        self.embed_dims = embed_dims
        self.dimse = dimse

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches      #56*56=3136
        patches_resolution = self.patch_embed.patches_resolution  #List[56,56]
        self.patches_resolution = patches_resolution    #List[56,56]
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 56 * 56, 96))  # fixed sin-cos embedding
        #

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))   #(1,3136,96)
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth 针对每个swin transformerblock的drop_path_rate的设置
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule  2+2+6+2=12


        # build encoder and bottleneck layers

        self.layers = nn.ModuleList()      #用于存储构成神经网络的一系列层
        for i_layer in range(self.num_layers):   #i_layer:0-3,遍历生成每个stage
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),  #patches_resolution[0]:56
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer], #堆叠多少次swin transformer block
                               num_heads=num_heads[i_layer],    #[3，6，12，24]
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  #针对每个swin transformerblock
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None, #构建前三个stage有PatchMerging
                               use_checkpoint=use_checkpoint,
                               dilations=dilations1[i_layer]
                               )
            self.layers.append(layer)  #使用append（）方法将层添加到列表中，并且可通过迭代中的层调用整个网络的'forward（）'

        dpr3 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_s))]

        self.SGlayers = nn.ModuleList()  # 用于存储构成神经网络的一系列层
        for i in range(self.num_layers-1):  # i_layer:0-3,遍历生成每个stage
            sglayer = SGBasiclayer(embed_dims_s=dimse[i],
                                   input_resolution=(
                                       patches_resolution[0] // (2 ** (i+1)),  # patches_resolution[0]:56
                                       patches_resolution[1] // (2 ** (i+1))),
                                 num_heads_s=num_heads_s[i], mlp_ratios_s=mlp_ratios_s[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path=dpr3[sum(depths_s[:i]):sum(depths_s[:i + 1])],
                                 norm_layer=norm_layer_s,sr_ratios=sr_ratios[i], linear=linear)
            self.SGlayers.append(sglayer)

        self.asppse = aspp_attention(768,768)
        self.asppse1 = aspp_attention(96, 96)
        # for i in range(3):
        #     asppse = aspp_attention(self.embed_dims[i], self.embed_dims[i])
        #     self.asppse.append(asppse)


        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths1))]
        cur = 0
        for i in range(3):
            stage0 = nn.Sequential(
                *[Conv2Block(dim=dims1[i], drop_path=dp_rates[cur + j]) for j in range(depths1[i])]
            )
            self.stages.append(stage0)
            cur += depths1[i]

        # self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths1))]
        # cur = 0
        # for i in range(3):
        #     stage0 = nn.Sequential(
        #         *[ConvBlock(dim=dims1[i], drop_path=dp_rates[cur + j],
        #                      layer_scale_init_value=layer_scale_init_value) for j in range(depths1[i])]
        #     )
        #     self.stages.append(stage0)
        #     cur += depths1[i]

        self.stages1 = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates1 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths2))]
        cur1 = 0
        for i in range(3):
            stage1 = nn.Sequential(
                *[LKABlock(dim=dims1[i], drop_path=dp_rates1[cur1 + j],mlp_ratio=mlp_ratios[i]) for j in range(depths2[i])],

            )
            self.stages1.append(stage1)
            cur1+= depths2[i]
        self.se = []
        for i in range(3):
            se = Squeeze_Excite_Block(dims1[i])
            self.se.append(se)

        self.cb = []
        for i in range(3):
            cbam = CBAM(dims1[i])
            self.cb.append(cbam)



        dpr0 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))]

        # build decoder layers
        self.layers_up = nn.ModuleList()    #上采样层
        self.concat_back_dim = nn.ModuleList()  #维度拼接层
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),  #in:16dim, out:8dim
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),#56//8
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths_decoder[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr0[sum(depths_decoder[:(self.num_layers-1-i_layer)]):sum(depths_decoder[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                dilations=dilations1[i_layer])

            self.layers_up.append(layer_up)
            # self.aspp2 = ASPP(8 * embed_dim, 8 * embed_dim)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)  #layernorm层
        self.norm_up= norm_layer(self.embed_dim)   #self.embed_dim为标准化的输入张量的维度，norm_layer沿该维度对输入张量进行归一化

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights) #权重初始化

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features(self,x):
        #x：[B,HW,C]
        x = self.patch_embed(x)  #下采样4倍
        if self.ape:   #if true,add absolute position embedding
            x = x + self.absolute_pos_embed# Tensor(32,3136,96)
        x = self.pos_drop(x)   #随机丢失一部分输入
        x_downsample1 = []
        x_downsample = []

        x_x = []
        # for layer in self.layers:
        #     x_downsample.append(x)
        #     x = layer(x)
        for i,layer in enumerate(self.layers):

            B, _, _ = x.shape
            C = int(self.embed_dim * 2 ** i)
            H, W = (self.patches_resolution[0] // (2 ** i),
                        self.patches_resolution[1] // (2 ** i))


            if i<3:
                x_downsample1.append(x)
                # x_downsample.append(x)
                x = layer(x)
                x_x.append(x)
                y1 = x_x[i]
                # y1 = self.SGlayers[i](y1)
                y1 = y1.reshape(B, H // 2, W // 2, 2*C).permute(0, 3, 1, 2)
                # y1 = self.se[i](y1)
                deconv = nn.ConvTranspose2d(2 * C, C, kernel_size=3, stride=2, padding=1, output_padding=1).cuda()
                y1 = deconv(y1)
                y1 = y1.permute(0, 2, 3, 1).reshape(B, H*W, C)
                y1 = x_downsample1[i] + y1
                y2 = y1.permute(0, 2, 1).reshape(B, C, H, W)
                y2 = self.stages[i](y2)
                y2 = self.se[i](y2)
                # y2 = self.cb[i](y2)
                y2 = y2.permute(0, 2, 3, 1).reshape(B, H*W, C)
                # y2 = y2 + y1
                y = self.stages1[i](y1)
                y = y.permute(0, 2, 1).reshape(B, C, H, W)
                y = self.se[i](y)
                # y = self.cb[i](y)
                y = y.permute(0, 2, 3, 1).reshape(B, H * W, C)
                y3 = y1*y2
                y4 = y*y1
                y5 = y3+y4

        #     #     # y3 = torch.cat((y,y2),dim=2)
        #     #     # y3 = y3.permute(0,2,1)
        #     #     # conv = nn.Conv1d(2 * C, C, kernel_size=1).cuda()
        #     #     # y3 = conv(y3)
        #     #     # y3 = y3.permute(0, 2, 1)
        #     #     # # y = y3.permute(0, 2, 3, 1).reshape(B, H*W, C)
        #     #     # y4 = y3+x_downsample1[i]
        #     #     # # y4 = y2+y1+y
                x_downsample.append(y5)
            if i==3:
                x = layer(x)

        x = self.norm(x)
        return x, x_downsample

    #Decoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        x_up = []
        for inx, layer_up in enumerate(self.layers_up):  #enumerate函数将列表中的元素layer_up和它们的下标inx一起返回
            B, _, _ = x.shape
            C = self.embed_dim * 2 ** (self.num_layers - 1 - inx)
            H, W = (self.patches_resolution[0] // (2 ** (self.num_layers-1-inx)),self.patches_resolution[1] // (2 ** (self.num_layers-1-inx)))

            if inx == 0:
                # x_up.append(x)
                x = layer_up(x)    #上采样操作


            else:
                # x_up.append(x)
                # y1 = self.SGlayers[3-inx](x_up[inx-1])
                # deconv = nn.ConvTranspose2d(2 * C, C, kernel_size=3, stride=2, padding=1, output_padding=1).cuda()
                # y1 = y1.view(B,2*C,H//2,W//2)
                # # y1 = self.se[3-inx](y1)
                # y1 = deconv(y1)
                # y1 = y1.view(B, H * W, C)
                # x_downsample[3 - inx] = torch.cat([y1, x_downsample[3 - inx]], -1)
                # x_downsample[3 - inx] = self.concat_back_dim[inx](x_downsample[3 - inx])

                x = torch.cat([x, x_downsample[3 - inx]], -1)  #x_downsample[3-inx]与对应下采样层进行拼接
                x = self.concat_back_dim[inx](x)  #进行维度转换
                x = layer_up(x)
        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            # x = x.view(B, C, H, W)
            # a = x
            # # c = self.asppattention1(y4)
            # b = self.asppattention1(x)
            # x = self.fusion(a, b)
            # x = x.view(B, H*W, C)
            x = self.up(x)
            # x = self.asppse1(x)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            # x = self.asppse1(x)
            # x = self.asppattention1(x)
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x) #从这里开始跳到forward_features函数中去
        x = self.forward_up_features(x,x_downsample)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops