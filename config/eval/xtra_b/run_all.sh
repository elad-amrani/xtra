# attentive probing
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_att/lr_1e-4.yaml xtra_b_800ep_in1k_att xtra_b_800ep-eval-lr_1e-4-100ep-in1k_att
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_att/lr_3e-4.yaml xtra_b_800ep_in1k_att xtra_b_800ep-eval-lr_3e-4-100ep-in1k_att
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_att/lr_5e-4.yaml xtra_b_800ep_in1k_att xtra_b_800ep-eval-lr_5e-4-100ep-in1k_att
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_att/lr_1e-3.yaml xtra_b_800ep_in1k_att xtra_b_800ep-eval-lr_1e-3-100ep-in1k_att
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_att/lr_1.5e-3.yaml xtra_b_800ep_in1k_att xtra_b_800ep-eval-lr_1.5e-3-100ep-in1k_att
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_att/lr_2e-3.yaml xtra_b_800ep_in1k_att xtra_b_800ep-eval-lr_2e-3-100ep-in1k_att
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_att/lr_4e-3.yaml xtra_b_800ep_in1k_att xtra_b_800ep-eval-lr_4e-3-100ep-in1k_att

# linear probing
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_lin/lr_1e-4.yaml xtra_b_800ep_in1k_lin xtra_b_800ep-eval-lr_1e-4-100ep-in1k_lin
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_lin/lr_3e-4.yaml xtra_b_800ep_in1k_lin xtra_b_800ep-eval-lr_3e-4-100ep-in1k_lin
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_lin/lr_5e-4.yaml xtra_b_800ep_in1k_lin xtra_b_800ep-eval-lr_5e-4-100ep-in1k_lin
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_lin/lr_1e-3.yaml xtra_b_800ep_in1k_lin xtra_b_800ep-eval-lr_1e-3-100ep-in1k_lin
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_lin/lr_1.5e-3.yaml xtra_b_800ep_in1k_lin xtra_b_800ep-eval-lr_1.5e-3-100ep-in1k_lin
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_lin/lr_2e-3.yaml xtra_b_800ep_in1k_lin xtra_b_800ep-eval-lr_2e-3-100ep-in1k_lin
bash scripts/lsf/train_non_ddp.sh ./config/eval/xtra_b/in1k_lin/lr_4e-3.yaml xtra_b_800ep_in1k_lin xtra_b_800ep-eval-lr_4e-3-100ep-in1k_lin
