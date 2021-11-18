#!/bin/bash
# the env now have the base pos + end pos as the subgoal spaces. 
# in `run_train_hrl4in_tabletop` we only have the base pos as the subgoal space. 
gpu="0"
reward_type="dense"
pos="fixed"
irs="30.0"
sgr="0.0"
lr="1e-4"
meta_lr="1e-5"        # 1e-4, 1e-5
fr_lr="0"             # 0, 100
death="30.0"
init_std_dev_xy="0.6" # 0.6, 1.2
init_std_dev_z="0.1"
failed_pnt="0.0"      # 0.0, -0.2
num_steps="1024"
ext_col="0.0"         # 0.0, 0.5, 1.0, 2.0
name="exp"
run="0"

log_dir="hrl4in_tabletop_6dim_subgoal_ll_pretrained"
echo $log_dir

python -u train_hrl4in_tabletop_6dim_subgoal_ll_pretrained.py \
   --use-gae \
   --sim-gpu-id $gpu \
   --pth-gpu-id $gpu \
   --lr $lr \
   --meta-lr $meta_lr \
   --freeze-lr-n-updates $fr_lr \
   --clip-param 0.1 \
   --value-loss-coef 0.5 \
   --num-train-processes 3 \
   --num-eval-processes 1 \
   --num-steps $num_steps \
   --num-mini-batch 1 \
   --num-updates 50000 \
   --use-linear-lr-decay \
   --use-linear-clip-decay \
   --entropy-coef 0.01 \
   --log-interval 1 \
   --experiment-folder "ckpt/"$log_dir \
   --time-scale 50 \
   --intrinsic-reward-scaling $irs \
   --subgoal-achieved-reward $sgr \
   --subgoal-init-std-dev $init_std_dev_xy $init_std_dev_xy $init_std_dev_z \
   --subgoal-failed-penalty $failed_pnt \
   --use-action-masks \
   --meta-agent-normalize-advantage \
   --extrinsic-collision-reward-weight $ext_col \
   --meta-gamma 0.99 \
   --use-pretrained-ll-policy \
   --pretrained-ll-policy-path "ppo_tiago_tabletop_hrl4in_ss/ckpt/ckpt.25730.pth" \
   --checkpoint-interval 10 \
   --checkpoint-index -1 \
   --config-file "tiago_tabletop_hrl4in_6dim_subgoal.yaml" \
   --num-eval-episodes 1
