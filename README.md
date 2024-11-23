# XTRA: Scalable _and_ Sample-Efficient Auto-Regressive Image Models
Official PyTorch (Lightning) implementation and pretrained models of the paper _Scalable _and_ Sample-Efficient Auto-Regressive Image Models_.

### Install
    conda env create -f environment.yml

### Code Structure

```
.
├── configs                   # '.yaml' configs
│   ├── ablations             #   ablation configs
│   ├── eval                  #   evaluation configs
│   ├── pretrain              #   pre-training configs
├── scripts                   # scheduler scripts
│   ├── lsf                   #   LSF scheduler scripts
│   ├── slurm                 #   Slurm scheduler scripts
├── src                       # source files
│   ├── modules               #   moodule implemtations
│   ├── utils                 #   shared utilities
│   ├── config.py             #   config class
│   ├── dataset.py            #   datasets and data loaders
│   ├── train.py              #   training script
```

### Training
Slurm scheduler training scripts:

ViT-B/16:

    sbatch -J xtra_b_in1k ./scripts/slurm/train.sh ./config/pretrain/xtra_b_in1k_pt.yaml

ViT-H/14:

    sbatch -J xtra_h_in21k ./scripts/slurm/train.sh ./config/pretrain/xtra_h_in21k_pt.yaml