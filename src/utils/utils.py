import os
import subprocess
import torch
import numpy as np
import urllib
import math
import random

from PIL import Image
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.optimizers.lars import LARS
from torch import nn
from torch.utils.data import WeightedRandomSampler, Sampler, Dataset
from copy import deepcopy

Image.MAX_IMAGE_PIXELS = None  # disable number of pixels limit


def use_ddp(args):
    if args.use_lsf_ccc or args.use_lsf_bv:
        num_nodes, gpu_rank = use_lsf()
    elif args.use_slurm:
        num_nodes, gpu_rank = use_slurm()
    else:
        num_nodes, gpu_rank = 1, 0
    return num_nodes, gpu_rank


def use_slurm():
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')

    gpu_rank = int(os.environ["SLURM_PROCID"])
    num_nodes = int(os.environ["SLURM_NNODES"])
    return num_nodes, gpu_rank


def use_lsf():
    print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(' ')  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    HOST_LIST = LSB_MCPU_HOSTS[::2]  # Strips the cores per node items in the list
    HOST_LIST = [node for node in HOST_LIST if node]  # remove empty strings if exist
    LSB_JOBID = os.environ["LSB_JOBID"]  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    os.environ["MASTER_ADDR"] = HOST_LIST[0]  # Sets the MasterNode to thefirst node on the list of hosts
    os.environ["MASTER_PORT"] = "5" + LSB_JOBID[-5:-1]
    os.environ["NODE_RANK"] = str(HOST_LIST.index(os.environ["HOSTNAME"].split('.')[0]))  # Uses the list index for node rank, master node rank must be 0
    os.environ["NCCL_IB_CUDA_SUPPORT"] = '1'  # Force use of infiniband
    print(os.environ["HOSTNAME"] + " MASTER_ADDR: " + os.environ["MASTER_ADDR"])
    print(os.environ["HOSTNAME"] + " MASTER_PORT: " + os.environ["MASTER_PORT"])
    print(os.environ["HOSTNAME"] + " NODE_RANK " + os.environ["NODE_RANK"])
    print(os.environ["HOSTNAME"] + " NCCL_SOCKET_IFNAME: " + os.environ["NCCL_SOCKET_IFNAME"])
    print(os.environ["HOSTNAME"] + " NCCL_IB_CUDA_SUPPORT: " + os.environ["NCCL_IB_CUDA_SUPPORT"])
    print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    print(os.environ["HOSTNAME"] + " LSB_MCPU_HOSTS: " + os.environ["LSB_MCPU_HOSTS"])
    print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    print(os.environ["HOSTNAME"] + " HOST_LIST: ")
    print(HOST_LIST)
    print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    print(os.environ["HOSTNAME"] + " HOSTNAME: " + os.environ["HOSTNAME"])
    print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    # --------------------------------------------------------------------------------------------------------------------------------------------
    num_nodes = len(HOST_LIST)
    ngpus_per_node = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    node_rank = int(os.environ["NODE_RANK"])
    gpu_rank = node_rank * ngpus_per_node + local_rank
    return num_nodes, gpu_rank


def get_scheduler(optimizer, total_steps, args):
    if args.scheduler_cosine:
        args.scheduler_cosine_max_epochs = total_steps
        if args.scheduler_cosine_warmup_pct:
            args.scheduler_cosine_warmup_epochs = int(args.scheduler_cosine_warmup_pct * total_steps)
            del args.scheduler_cosine_warmup_pct
        else:
            args.scheduler_cosine_warmup_epochs = \
                int(args.scheduler_cosine_warmup_epochs / args.trainer_max_epochs * total_steps)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **args.scheduler_cosine)
    elif args.scheduler_OneCycleLR is not None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=total_steps, **args.scheduler_OneCycleLR)
    else:  # if args.scheduler_CyclicLR is not None:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **args.scheduler_CyclicLR)
        # force scale_mode even with predefined mode
        scheduler.scale_mode = args.scheduler_CyclicLR_scale_mode
    scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
    return scheduler


class SinCosPosEmbed(nn.Module):
    def __init__(self, cls_token: bool = False):
        super().__init__()
        self.cls_token = cls_token

    def forward(self, h: int, w: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 2 == 0, embed_dim

        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, h, w])

        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        pos_embed = torch.concatenate([emb_h, emb_w], dim=1)  # (H*W, D)
        if self.cls_token:
            pos_embed = torch.concatenate(
                [torch.zeros([1, embed_dim]), pos_embed], dim=0
            )
        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(
        embed_dim: int, pos: torch.Tensor
    ) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2).float()
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb

    @staticmethod
    def interpolate_pos_encoding(new_grid, original_grid, pos_embed):
        new_h, new_w = new_grid
        orig_h, orig_w = original_grid
        if new_h == orig_h and new_w == orig_w:
            return pos_embed
        dim = pos_embed.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, orig_h, orig_w, dim).permute(0, 3, 1, 2),
            scale_factor=(new_h / orig_h, new_w / orig_w),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed


def imagenet_subset_samples(dataset, label_subset):
    # extract subset of training images
    subset_file = urllib.request.urlopen(
        "https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/" +
        str(label_subset) + "percent.txt")
    labeled_imgs = [li.decode("utf-8").split('\n')[0] for li in subset_file]
    # update dataset
    traindir = dataset.root
    dataset.samples = [(os.path.join(traindir, li.split('_')[0], li), dataset.class_to_idx[li.split('_')[0]])
                       for li in labeled_imgs]

    return dataset


def sampler_with_ratio(datasets, sampling_ratios, args):
    if len(datasets) == 1:
        return None

    assert args.trainer_limit_train_batches is not None, 'limit_train_batches must be set when sampler is ratio-based!'
    num_gpus = args.trainer_num_nodes * torch.cuda.device_count()
    num_epoch_samples = args.trainer_limit_train_batches * args.train_batch_size * num_gpus

    num_samples = sum([len(d) for d in datasets])
    weights = np.zeros(num_samples)
    start_idx = 0
    for idx, ratio in enumerate(sampling_ratios):
        d_len = len(datasets[idx])
        end_idx = start_idx + d_len
        weights[start_idx: end_idx] = ratio / d_len
        start_idx = end_idx

    sampler = CustomWeightedRandomSampler(weights, num_epoch_samples)
    return sampler


def split_dataset(dataset, split, seed):
    # get indices of train and val
    indices = list(range(len(dataset)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_len = int(len(dataset) * split)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    # create two subsets with deepcopy (to allow different transforms)
    train_set_i = torch.utils.data.Subset(dataset, indices=train_indices)
    val_set_i = torch.utils.data.Subset(deepcopy(dataset), indices=val_indices)

    train_set_i.dataset.train = True
    val_set_i.dataset.train = False

    return train_set_i, val_set_i


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


def print_ds_summary(name, num_train, num_val):
    print('\n', '*' * 30, ' {} '.format(name), '*' * 30)
    print('Number of training samples: {}'.format(num_train))
    print('Number of validation samples: {}\n'.format(num_val))


def print_config(args):
    args_dict = vars(args)
    print('*' * 30, 'Configuration', '*' * 30)
    for key in sorted(args_dict.keys()):
        print('{}: {}'.format(key, args_dict[key]))
    print('*' * 100)


def param_groups(model, lr, weight_decay, llm_lr_factor=None, enc_lr_factor=None, **kwargs):
    def is_bias_or_norm(name, param):
        return ('bias' in name) or (len(param.shape) == 1)

    def is_llm(name):
        return name.startswith('llm')

    def is_enc(name):
        return name.startswith('enc')

    params = []

    if llm_lr_factor is not None:
        params += [
            {
                "params": [p for n, p in model.named_parameters()
                           if not is_bias_or_norm(n, p) and
                           is_llm(n) and
                           p.requires_grad],
                "weight_decay": weight_decay,
                'lr': lr * llm_lr_factor
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if is_bias_or_norm(n, p) and
                           is_llm(n) and
                           p.requires_grad],
                "weight_decay": 0.0,
                'lr': lr * llm_lr_factor
            }
        ]

    if enc_lr_factor is not None:
        params += [
            {
                "params": [p for n, p in model.named_parameters()
                           if not is_bias_or_norm(n, p) and
                           is_enc(n) and
                           p.requires_grad],
                "weight_decay": weight_decay,
                'lr': lr * enc_lr_factor
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if is_bias_or_norm(n, p) and
                           is_enc(n) and
                           p.requires_grad],
                "weight_decay": 0.0,
                'lr': lr * enc_lr_factor
            }
        ]

    params += [
        {
            "params": [p for n, p in model.named_parameters()
                       if (not is_bias_or_norm(n, p)) and
                       (not is_llm(n) or llm_lr_factor is None) and
                       (not is_enc(n) or enc_lr_factor is None) and
                       p.requires_grad],
            "weight_decay": weight_decay,
            'lr': lr
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if is_bias_or_norm(n, p) and
                       (not is_llm(n) or llm_lr_factor is None) and
                       (not is_enc(n) or enc_lr_factor is None) and
                       p.requires_grad],
            "weight_decay": 0.0,
            'lr': lr
        },
    ]

    return params


def config_optimizers(model, args):
    if args.optimizer_sgd:
        optimizer = torch.optim.SGD(param_groups(model=model, **args.optimizer, **args.optimizer_sgd),
                                    **args.optimizer_sgd)
    elif args.optimizer_lars:
        optimizer = LARS(param_groups(model=model, **args.optimizer, **args.optimizer_lars),
                         **args.optimizer_lars)
    elif args.optimizer_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(param_groups(model=model, **args.optimizer, **args.optimizer_adam),
                                       **args.optimizer_adam)
    elif args.optimizer_deepspeed:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(param_groups(model=model, **args.optimizer, **args.optimizer_deepspeed),
                                     **args.optimizer_deepspeed)
    else:
        optimizer = torch.optim.AdamW(param_groups(model=model, **args.optimizer, **args.optimizer_adam),
                                      **args.optimizer_adam)

    if args.scheduler is not None:
        scheduler = get_scheduler(optimizer, total_steps=model.trainer.estimated_stepping_batches, args=args)
        return [optimizer], scheduler

    return optimizer


def print_util_and_mem():
    print("torch.cuda.utilization: {}%".format(torch.cuda.utilization()))
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
    return


def safe_getitem(func):
    def wrapper(self, index):
        try:
            return func(self, index)
        except Exception as e:
            print(f"Error retrieving item at index {index}: {e}")
            new_index = random.randint(0, len(self) - 1)
            return self.__getitem__(new_index)
    return wrapper


def load_pretrained_weights(model):
    if model.args.pretrained is not None:
        print("=> loading checkpoint '{}'".format(model.args.pretrained))
        checkpoint = torch.load(model.args.pretrained, map_location=model.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'encoder' in checkpoint:  # StoP
            state_dict = checkpoint['encoder']
            # rename prefix
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    # remove prefix
                    state_dict['enc' + k[len("module"):]] = state_dict[k]
                    del state_dict[k]
        else:
            state_dict = checkpoint  # checkpoint with weights only

        if model.args.rm_pt_param is not None:
            for k in list(state_dict.keys()):
                rm_k = any([k.startswith(x) for x in model.args.rm_pt_param])
                if rm_k:
                    del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print('*' * 50 + '\n', msg, '\n' + '*' * 50)
        print("=> loaded pre-trained model '{}'".format(model.args.pretrained))


# create random pseudo-random permutations without common prefixes
def create_random_permutations(num_crops, num_tokens, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def has_common_prefix(row, matrix, length):
        # Check if any existing row has the same prefix of given length
        for existing_row in matrix:
            if torch.equal(existing_row[:length], row[:length]):
                return True
        return False

    grid_size = int(math.sqrt(num_tokens))
    block_size = int(math.sqrt(num_tokens / num_crops))
    # set number of permutations as number of blocks to avoid two different permutations having the same prefix
    num_perms = int((grid_size / block_size) ** 2)
    block_perm = torch.zeros((num_perms, num_crops), dtype=torch.long)
    token_perm = torch.zeros((num_perms, num_tokens), dtype=torch.long)

    for i in range(num_perms):
        while True:
            # get crop permutation
            new_row = torch.randperm(num_crops)
            # Check prefixes of all lengths for all existing rows
            if all(not has_common_prefix(new_row, block_perm[:i], length) for length in range(1, num_crops)):
                block_perm[i] = new_row
                break

        # get corresponding token permutation
        token_perm_i = torch.arange(num_tokens).view(grid_size, grid_size)
        token_perm_i = token_perm_i.unfold(0, block_size, block_size).unfold(1, block_size, block_size)
        orig_size = token_perm_i.shape
        token_perm_i = token_perm_i.flatten(0, 1)

        # permute
        token_perm_i = token_perm_i[block_perm[i]]

        # revert to a sequence
        token_perm_i = token_perm_i.view(*orig_size)
        token_perm_i = torch.einsum('hwpq->hpwq', token_perm_i)
        token_perm_i = token_perm_i.reshape(num_tokens)
        token_perm[i] = token_perm_i
    return token_perm, block_perm
