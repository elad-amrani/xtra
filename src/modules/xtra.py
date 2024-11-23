import torch
import math
import lightning.pytorch as pl

from torch.nn.functional import mse_loss, l1_loss
from torch.nn import ModuleList

from config import Config
from utils import \
    config_optimizers, \
    print_util_and_mem, \
    load_pretrained_weights, \
    create_random_permutations
from modules.mlp import MLPHead
from dataset import train_val_dataloaders
from modules import vit
from modules.vit import vit_predictor


class XTRA(pl.LightningModule):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))

        # visual encoder
        self.enc = vit.__dict__[args.enc_arch](**args.enc)

        # projection
        self.proj = MLPHead(in_dim=self.enc.d_model, **args.proj)

        # predictor
        self.embed_dim = args.proj_out_dim if args.proj_nlayers > 0 else self.enc.d_model
        self.predictor = vit_predictor(embed_dim=self.embed_dim, max_seq_len=self.enc.max_seq_len, **args.predictor)

        # AIM projection
        num_tokens = args.aim_num_tokens or 1
        mlp_in_dim = self.embed_dim * args.enc_block_size ** 2 if args.enc_is_block_causal else self.embed_dim
        self.aim_proj = ModuleList([MLPHead(in_dim=mlp_in_dim, out_dim=int(3 * self.enc.crop_size ** 2),
                                            **args.aim_proj) for _ in range(num_tokens)])

        # load pretrained if exists
        self.load_pretrained_weights()

        # print architecture
        print(self)

        # dataloaders
        self.train_loader, self.val_loaders, self.loader_name = train_val_dataloaders(args)

        # random permutations
        if args.aim_pseudo_random:
            num_crops = int((self.enc.img_size[0] / self.enc.crop_size) ** 2)
            num_tokens = self.enc.max_seq_len
            self.perms, self.block_perms = create_random_permutations(num_crops, num_tokens, args.seed)

    @torch.no_grad()
    def forward(self, images, aim_idx=0, return_target=False):
        # encoder
        if return_target:
            embeds, target = self.enc(images, return_target=True)
        else:
            embeds, target = self.enc(images), None

        # projection
        embeds = self.proj(embeds)

        # predictor
        embeds = self.predictor(embeds)

        # concat embeddings of same block
        if self.args.enc_is_block_causal and self.args.enc_block_size > 1:
            embeds = self.concat_block_embeds(embeds)  # [bsize, num_blocks, embed_dim * block_size **2]

        # AIM projection
        preds = self.aim_proj[aim_idx](embeds)

        if return_target:
            return preds, target

        return preds

    def _step(self, batch):
        images, _ = batch

        # sample pseudo-random permutation if enabled
        perm, block_perm = self.choose_perm()

        # encoder
        embeds, target = self.enc(images, return_target=True, perm=perm, block_perm=block_perm)

        # projection
        embeds = self.proj(embeds)

        # when using 16-mixed, predictor may yield NaN --> convert to full precision to avoid NaN
        original_dtype = embeds.dtype
        enable_float32 = self.trainer.precision == '16-mixed'
        with torch.cuda.amp.autocast(enabled=enable_float32, dtype=torch.float32):
            # predictor
            embeds = self.predictor(embeds, perm=perm)  # [bsize, num_patches, embed_dim]
        embeds = embeds.to(original_dtype)

        # concat embeddings of same block
        if self.args.enc_is_block_causal and self.args.enc_block_size > 1:
            embeds = self.concat_block_embeds(embeds)  # [bsize, num_blocks, embed_dim * block_size **2]

        # loss
        loss = self._loss(embeds=embeds, target=target)

        return loss

    def _loss(self, embeds, target, eps=1.0e-6):
        target = target.flatten(2)  # bsize, num_crops, 3 * crop_size**2

        if self.args.aim_norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + eps) ** .5

        loss = 0.0
        for idx, proj_i in enumerate(self.aim_proj):
            pred = proj_i(embeds)

            if self.args.aim_l1_loss:
                loss += l1_loss(pred[:, :-1-idx], target[:, 1+idx:])
            else:
                loss += mse_loss(pred[:, :-1-idx], target[:, 1+idx:])
        loss /= len(self.aim_proj)

        name = 'tloss_aim' if self.training else 'vloss_aim'
        self.log(name, loss, on_step=self.training, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=self.args.train_batch_size)

        return loss

    def decropify_sample(self, x, h, w):
        """

        :param x: of shape [h*w, crop_size, crop_size, 3]
        :param h: int. number of vertical crops
        :param w: int. number of horizontal crops
        :return: original image of shape [3, h * crop_size, w * crop_size]
        """
        x = x.reshape(h, w, self.enc.crop_size, self.enc.crop_size, 3)
        x = torch.einsum('hwpqc->chpwq', x)
        x = x.reshape(3, h * self.enc.crop_size, w * self.enc.crop_size)
        return x

    def concat_block_embeds(self, x):
        bs, num_patches, dim = x.shape
        # convert to grid
        x = x.transpose(1, 2)                      # [bs, dim, num_patches]
        grid_size = int(math.sqrt(num_patches))
        x = x.view(bs, dim, grid_size, grid_size)  # [bs, dim, grid_size, grid_size]
        # 'patchify' to block_size x block_size
        c_h, c_w = self.args.enc_block_size, self.args.enc_block_size
        patches = x.unfold(2, c_h, c_h).unfold(3, c_w, c_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(bs, -1, dim, c_h, c_w)
        # flatten to a single vector per block
        x = patches.flatten(2)                     # [bs, num_blocks, dim * block_size ** 2)
        return x

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        if loss is not None:
            self.log('tloss', loss, on_step=True, on_epoch=True, prog_bar=True,
                     logger=True, sync_dist=True, batch_size=self.args.train_batch_size)
        if self.args.trainer_fast_dev_run:
            print_util_and_mem()
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch)
        if loss is not None:
            self.log('vloss_{}'.format(self.loader_name[dataloader_idx]), loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True, batch_size=self.args.val_batch_size,
                     add_dataloader_idx=False)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loaders

    def configure_optimizers(self):
        return config_optimizers(model=self, args=self.args)

    def load_pretrained_weights(self):
        load_pretrained_weights(self)

    def choose_perm(self):
        if not self.args.aim_pseudo_random:
            return None, None

        if not self.training:  # use same permutation for validation
            return self.perms[0], self.block_perms[0]

        rand_idx = torch.randint(0, self.perms.shape[0], (1,)).item()
        return self.perms[rand_idx], self.block_perms[rand_idx]
