import math
import torch

from torch import nn
from functools import partial


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, fused_attn=True,
                 use_out_norm=True, is_causal=False, is_block_causal=False, block_size=1,
                 **kwargs):
        super().__init__()
        if isinstance(img_size, int):
            self.img_size = [img_size, img_size]
        else:
            self.img_size = img_size if len(img_size) == 2 else [img_size[0], img_size[0]]
        self.num_features = self.embed_dim = self.d_model = embed_dim
        self.is_causal = is_causal
        self.is_block_causal = is_block_causal
        self.block_size = block_size
        if is_block_causal:
            self.crop_size = int(patch_size * self.block_size)
        else:
            self.crop_size = patch_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.max_seq_len = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.block_mask = create_block_mask(self.max_seq_len, block_size) if self.is_block_causal else None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=is_causal,
                is_block_causal=is_block_causal, block_mask=self.block_mask, block_size=block_size,
                fused_attn=fused_attn)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if use_out_norm else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_target=False, perm=None, block_perm=None, **kwargs):
        target = self.patchify(x) if return_target else None
        x = self.patch_embed(x)
        x = self.add_cls_and_pos(x)

        if perm is not None:
            assert block_perm is not None, 'perm was given but not corresponding block_perm!'
            target = target[:, block_perm]
            x = x[:, perm]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if return_target:
            return x, target

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1]
        N = pos_embed.shape[1]

        if npatch == N:
            return pos_embed

        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return pos_embed

    def forward_return_n_last_blocks(self, x, n=1, return_patch_avgpool=False, return_layer_avgpool=False, norm_output=False):
        x = self.patch_embed(x)
        x = self.add_cls_and_pos(x)

        output = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # return last `n` blocks
            if len(self.blocks) - i <= n:
                if norm_output:
                    output.append(self.norm(x))
                else:
                    output.append(x)

        if return_layer_avgpool:
            output = torch.stack(output, dim=-1).mean(dim=-1)  # [bsize, seq, dim]
        else:  # concat all layers
            output = torch.cat(output, dim=-1)  # [bsize, seq, dim * n]

        if return_patch_avgpool:  # average pooling of sequence dimension
            output = torch.mean(output, dim=1)  # [bsize, dim] or [bsize, dim * n]

        return output

    def add_cls_and_pos(self, x):
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed
        x = self.pos_drop(x)
        return x

    def patchify(self, x):
        N, C = x.size()[:2]
        c_h, c_w = self.crop_size, self.crop_size
        crops = x.unfold(2, c_h, c_h).unfold(3, c_w, c_w)
        crops = crops.permute(0, 2, 3, 1, 4, 5).contiguous().view(N, -1, C, c_h, c_w)
        return crops


class ViTPredictor(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, fused_attn=True,
                 use_out_norm=True, is_causal=False, is_block_causal=False, block_size=1, max_seq_len=256, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = self.d_model = embed_dim
        self.is_causal = is_causal
        self.is_block_causal = is_block_causal
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.block_mask = create_block_mask(max_seq_len, block_size) if self.is_block_causal else None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=is_causal,
                is_block_causal=is_block_causal, block_mask=self.block_mask, block_size=block_size,
                fused_attn=fused_attn)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if use_out_norm else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, perm=None, **kwargs):
        # positional embeddings
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)

        # if block permutation is done in the encoder, permute decoder positional embeddings as well
        if perm is not None:
            pos_embed = pos_embed[:, perm]

        x = x + pos_embed
        x = self.pos_drop(x)

        # layers
        for blk in self.blocks:
            x = blk(x)

        # output norm
        x = self.norm(x)
        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1]
        N = pos_embed.shape[1]

        if npatch == N:
            return pos_embed

        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return pos_embed


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 is_causal=False, is_block_causal=False, block_mask=None, fused_attn=False, block_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.is_block_causal = is_block_causal
        self.block_mask = block_mask
        self.block_size = block_size
        self.fused_attn = fused_attn
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.fused_attn:
            if self.is_block_causal and self.block_mask is not None:
                if N == self.block_mask.shape[0]:
                    attn_mask = torch.logical_not(self.block_mask.to(x.device))
                else:
                    attn_mask = torch.logical_not(create_block_mask(N, self.block_size, device=x.device))
            else:
                attn_mask = None

            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask,
                is_causal=self.is_causal,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if self.is_block_causal and self.block_mask is not None:
                if N == self.block_mask.shape[0]:
                    attn.masked_fill_(self.block_mask.to(x.device), float('-inf'))
                else:
                    block_mask = create_block_mask(N, self.block_size, device=x.device)
                    attn.masked_fill_(block_mask, float('-inf'))
            elif self.is_causal:
                mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
                attn.masked_fill_(mask, float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_causal=False,
                 is_block_causal=False, block_mask=None, block_size=None, fused_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, is_causal=is_causal,
                              is_block_causal=is_block_causal, block_mask=block_mask, block_size=block_size,
                              fused_attn=fused_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def create_block_mask(num_patches, block_size, device=None):
    # assign 'block index' to each patch
    patch_grid = int(math.sqrt(num_patches))
    block_idx = torch.zeros((patch_grid, patch_grid), dtype=torch.long, device=device)
    for i in range(patch_grid):
        for j in range(patch_grid):
            block_idx[i, j] = (i // block_size) * (patch_grid // block_size) + (j // block_size)
    block_idx = block_idx.flatten()  # [num_patches]

    # causal block mask
    block_mask = torch.zeros((num_patches, num_patches), dtype=torch.bool, device=device)

    for i in range(0, num_patches):
        patch_block_idx = block_idx[i]
        block_mask[i, :] = torch.where(block_idx > patch_block_idx, True, False)
    return block_mask


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_predictor(**kwargs):
    model = ViTPredictor(qkv_bias=True, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}