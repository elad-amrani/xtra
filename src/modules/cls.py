import torch
import lightning.pytorch as pl

from typing import Optional, Any, Tuple
from torch import nn
from torchmetrics import Accuracy

from config import Config
from dataset import train_val_dataloaders
from modules import vit
from utils import config_optimizers, load_pretrained_weights, print_util_and_mem


class Classifier(pl.LightningModule):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))

        # encoder
        self.enc = vit.__dict__[args.enc_arch](**args.enc)

        if args.cls_mode == 'linear':
            self.disable_grad(self.enc)

        # classifier
        self.embed_dim_multiplier = 1 if args.cls_avgpool_layers else args.cls_n_last_blocks
        self.embed_dim = self.enc.embed_dim * self.embed_dim_multiplier

        if args.attentive_pooling:
            self.classifier = AttentionPoolingClassifier(dim=self.embed_dim,
                                                         out_features=args.cls_num_classes,
                                                         out_bn=args.cls_out_bn,
                                                         **args.attentive_pooling)
        else:
            if args.cls_out_bn:
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(self.embed_dim, affine=False, eps=1e-6),
                    nn.Linear(self.embed_dim, args.cls_num_classes)
                )
            else:
                self.classifier = nn.Linear(self.embed_dim, args.cls_num_classes)

        label_smoothing = args.label_smoothing if args.label_smoothing is not None else 0.0
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=args.cls_num_classes, top_k=1)
        self.val_acc = Accuracy(task="multiclass", num_classes=args.cls_num_classes, top_k=1)

        # print architecture
        print(self)

        # load pretrained
        self.load_pretrained_weights()

        # dataloaders
        self.train_loader, self.val_loaders, self.loader_name = train_val_dataloaders(args)

    @torch.no_grad()
    def forward(self):
        pass

    def _step(self, batch):
        images, targets = batch

        # encoder
        out = self.forward_features(images)

        # classifier
        out = self.classifier(out)

        # loss
        loss = self.criterion(out, targets)

        # accuracy
        prefix = 'train' if self.training else 'val'
        getattr(self, '{}_acc'.format(prefix))(out, targets)

        # logger
        self.log_dict({'{}_loss'.format(prefix): loss,
                       '{}_acc'.format(prefix): getattr(self, '{}_acc'.format(prefix))},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                      batch_size=getattr(self.args, '{}_batch_size'.format(prefix)))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        if self.args.trainer_fast_dev_run:
            print_util_and_mem()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loaders

    def configure_optimizers(self):
        return config_optimizers(model=self, args=self.args)

    def load_pretrained_weights(self):
        load_pretrained_weights(self)

    @staticmethod
    def disable_grad(module):
        for p in module.parameters():
            p.requires_grad = False

    def forward_features(self, images):
        out = self.enc.forward_return_n_last_blocks(images,
                                                    n=self.args.cls_n_last_blocks,
                                                    return_patch_avgpool=self.args.cls_avgpool_patchtokens,
                                                    return_layer_avgpool=self.args.cls_avgpool_layers,
                                                    norm_output=self.args.cls_norm_enc_output)
        return out


# taken from https://github.com/apple/ml-aim/blob/main/aim/torch/layers.py
class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 1,
        out_bn: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim))
        torch.nn.init.normal_(self.cls_token, std=.02)

        if out_bn:
            self.linear = nn.Sequential(
                nn.BatchNorm1d(dim, affine=False, eps=1e-6),
                nn.Linear(dim, out_features)
            )
        else:
            self.linear = nn.Linear(dim, out_features)

        self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)

        self.num_queries = num_queries

    def forward(self, x: torch.Tensor, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)

        out = self.linear(x_cls)
        return out
