import argparse
import lightning.pytorch as pl
import os

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy, DeepSpeedStrategy
from lightning.pytorch.plugins import DeepSpeedPrecision
from lightning.pytorch.callbacks import LearningRateMonitor
from torch import nn

from config import Config
from modules.get_model import get_model
from utils import use_ddp, print_config


def main():
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--save_path', type=str, default=None,
                        help='save path for checkpoints')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='path to pretrained models')
    parser.add_argument('--dist_url', type=str, default=None,
                        help='url used to set up distributed training')
    parser.add_argument('--eval', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--test', action='store_true', help='evaluate model on test set')
    parser.add_argument('--clearml_project_name', type=str, default=None,
                        help='name of ClearML project')
    parser.add_argument('--clearml_task_name', type=str, default=None,
                        help='name of ClearML task')
    parser.add_argument('--use_lsf_ccc', action='store_true', help='use LSF cluster on CCC')
    parser.add_argument('--use_lsf_bv', action='store_true', help='use LSF cluster on BlueVela')
    parser.add_argument('--use_slurm', action='store_true', help='use SLURM cluster')

    i_args = parser.parse_args()
    args = Config(i_args.config)
    if i_args.save_path is not None: args.save_path = i_args.save_path
    if i_args.pretrained is not None: args.pretrained = i_args.pretrained
    if i_args.clearml_project_name is not None: args.clearml_project_name = i_args.clearml_project_name
    if i_args.clearml_task_name is not None: args.clearml_task_name = i_args.clearml_task_name
    args.eval = args.eval or i_args.eval
    args.test = args.test or i_args.test
    args.use_lsf_ccc = args.use_lsf_ccc or i_args.use_lsf_ccc
    args.use_lsf_bv = args.use_lsf_bv or i_args.use_lsf_bv
    args.use_slurm = args.use_slurm or i_args.use_slurm
    args.trainer_default_root_dir = args.save_path

    # DDP
    args.trainer_num_nodes, global_rank = use_ddp(args)

    # create output directory
    os.makedirs(args.save_path, exist_ok=True)

    # print configuration
    print_config(args)

    # strategy
    if args.ddp_strategy:
        strategy = DDPStrategy(**args.ddp_strategy)
    elif args.fsdp_strategy:
        from functools import partial
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        policy = partial(size_based_auto_wrap_policy, min_num_params=10000)
        if args.fsdp_strategy_activation_checkpointing is not None:
            args.fsdp_strategy_activation_checkpointing = [eval(x) for x in args.fsdp_strategy_activation_checkpointing]
        strategy = FSDPStrategy(auto_wrap_policy=policy, **args.fsdp_strategy)
    elif args.deepspeed_strategy:
        precision_plugin = DeepSpeedPrecision(args.trainer_precision) if args.trainer_precision else None
        strategy = DeepSpeedStrategy(**args.deepspeed_strategy, precision_plugin=precision_plugin)
    elif args.trainer_strategy:
        strategy = args.trainer_strategy
        del args.trainer_strategy
    else:
        strategy = 'auto'

    # initialize ClearML
    if args.clearml_project_name and args.clearml_task_name and global_rank == 0:
        from clearml import Task
        Task.init(**args.clearml)

    # seed
    pl.seed_everything(args.seed)

    # model
    model = get_model(args)

    # model checkpoint monitors
    model_cp = [ModelCheckpoint(**args.checkpoint, dirpath=args.save_path, save_last=idx == 0,
                                monitor='vloss_{}'.format(name_i),
                                filename='{epoch}-{step}-{vloss_' + name_i + ':.3f}')
                for idx, name_i in enumerate(model.loader_name)]
    periodic_cp = ModelCheckpoint(**args.periodic_checkpoint, dirpath=args.save_path, filename='{epoch}-{step}')

    # callbacks
    args.trainer_callbacks = [*model_cp, periodic_cp, LearningRateMonitor(logging_interval='step')]

    # loggers
    args.trainer_logger = [TensorBoardLogger(save_dir=args.save_path, version=0),
                           CSVLogger(save_dir=args.save_path, version=0)]

    # trainer
    trainer = pl.Trainer(**args.trainer, strategy=strategy)

    if args.eval:
        trainer.validate(model, ckpt_path='last')
        return

    trainer.fit(model, ckpt_path='last')


if __name__ == '__main__':
    main()
