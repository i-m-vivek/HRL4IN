#!/usr/bin/env python3

from time import time
from collections import deque
import random
import numpy as np
import argparse
import gym

import torch
from torch.utils.tensorboard import SummaryWriter

import hrl4in
from hrl4in.utils.logging import logger
from hrl4in.rl.ppo import PPO, Policy, RolloutStorage, MetaPolicy, AsyncRolloutStorage
from hrl4in.utils.utils import *
from hrl4in.utils.args import *

import gibson2

# from gibson2.envs.parallel_env import ParallelNavEnvironment
# from gibson2.envs.locomotor_env import (
#     NavigateEnv,
#     NavigateRandomEnv,
#     InteractiveNavigateEnv,
# )
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.envs.parallel_env import ParallelNavEnv

from IPython import embed
import matplotlib.pyplot as plt


def evaluate(
    args,
    envs,
    meta_actor_critic,
    base_actor_critic,
    arm_actor_critic,
    action_mask_choices,
    subgoal_mask_choices,
    subgoal_tolerance,
    device,
    writer,
    update=0,
    count_steps=0,
    eval_only=False,
):
    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs._num_envs, 1, device=device)
    episode_success_rates = torch.zeros(envs._num_envs, 1, device=device)
    episode_lengths = torch.zeros(envs._num_envs, 1, device=device)
    episode_collision_steps = torch.zeros(envs._num_envs, 1, device=device)
    # episode_total_energy_costs = torch.zeros(envs._num_envs, 1, device=device)
    # episode_avg_energy_costs = torch.zeros(envs._num_envs, 1, device=device)
    # episode_stage_open_door = torch.zeros(envs._num_envs, 1, device=device)
    # episode_stage_to_target = torch.zeros(envs._num_envs, 1, device=device)

    episode_counts = torch.zeros(envs._num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs._num_envs, 1, device=device)

    subgoal_rewards = torch.zeros(envs._num_envs, 1, device=device)
    subgoal_success_rates = torch.zeros(envs._num_envs, 1, device=device)
    subgoal_lengths = torch.zeros(envs._num_envs, 1, device=device)
    subgoal_counts = torch.zeros(envs._num_envs, 1, device=device)
    current_subgoal_reward = torch.zeros(envs._num_envs, 1, device=device)

    current_meta_recurrent_hidden_states = torch.zeros(
        envs._num_envs, args.hidden_size, device=device
    )
    next_meta_recurrent_hidden_states = torch.zeros(
        envs._num_envs, args.hidden_size, device=device
    )
    recurrent_hidden_states = torch.zeros(
        envs._num_envs, args.hidden_size, device=device
    )
    subgoals_done = torch.zeros(envs._num_envs, 1, device=device)
    masks = torch.zeros(envs._num_envs, 1, device=device)

    current_subgoals = torch.zeros(batch["sensor"].shape, device=device)
    current_subgoals_steps = torch.zeros(envs._num_envs, 1, device=device)
    current_subgoal_masks = torch.zeros(batch["sensor"].shape, device=device)

    action_dim = envs.action_space.shape[0]
    current_action_masks = torch.zeros(envs._num_envs, action_dim, device=device)

    step = 0
    while episode_counts.sum() < args.num_eval_episodes:
        with torch.no_grad():
            (
                _,
                subgoals,
                _,
                action_mask_indices,
                _,
                meta_recurrent_hidden_states,
            ) = meta_actor_critic.act(
                batch,
                current_meta_recurrent_hidden_states,
                masks,
                deterministic=False,
            )
            print(action_mask_indices)
            if meta_actor_critic.use_action_masks:
                action_masks = action_mask_choices.index_select(
                    0, action_mask_indices.squeeze(1)
                )
                subgoal_masks = subgoal_mask_choices.index_select(
                    0, action_mask_indices.squeeze(1)
                )
            else:
                action_masks = torch.ones_like(current_action_masks)
                subgoal_masks = torch.ones_like(current_subgoal_masks)

            should_use_new_subgoals = (current_subgoals_steps == 0.0).float()
            current_subgoals = (
                should_use_new_subgoals * subgoals
                + (1 - should_use_new_subgoals) * current_subgoals
            )
            current_subgoal_masks = (
                should_use_new_subgoals * subgoal_masks.float()
                + (1 - should_use_new_subgoals) * current_subgoal_masks
            )
            current_subgoals *= current_subgoal_masks
            current_action_masks = (
                should_use_new_subgoals * action_masks
                + (1 - should_use_new_subgoals) * current_action_masks
            )
            next_meta_recurrent_hidden_states = (
                should_use_new_subgoals * meta_recurrent_hidden_states
                + (1 - should_use_new_subgoals) * next_meta_recurrent_hidden_states
            )
            ideal_next_state = batch["sensor"] + current_subgoals

            # if eval_only: NOT IMPLEMENTED FOR NEW iGIBSON ENVS
            #     envs.set_subgoal(ideal_next_state.cpu().numpy())
            #     base_only = (current_subgoal_masks[:, 2] == 0).cpu().numpy()
            #     envs.set_subgoal_type(base_only)

            roll = batch["auxiliary_sensor"][:, 3] * np.pi
            pitch = batch["auxiliary_sensor"][:, 4] * np.pi
            yaw = batch["auxiliary_sensor"][:, 84] * np.pi

            temp_current_subgoals = current_subgoals.clone()
            temp_current_subgoals = temp_current_subgoals.view(-1, 2, 3)
            temp_action_mask_indices = action_mask_indices.clone()
            temp_action_mask_indices = torch.cat([1 - temp_action_mask_indices, temp_action_mask_indices], 1)
            temp_action_mask_indices = temp_action_mask_indices.unsqueeze(-1)
            
            temp_current_subgoals = temp_action_mask_indices*temp_current_subgoals
            temp_current_subgoals = torch.sum(temp_current_subgoals, 1)

            robot_pos =batch["auxiliary_sensor"][:, 0:3]
            local_subgoal = rotate_torch_vector(temp_current_subgoals - robot_pos, roll, pitch, yaw)
            temp_current_subgoals = rotate_torch_vector(temp_current_subgoals, roll, pitch, yaw)

            batch["auxiliary_sensor"][:, 87:90] = local_subgoal
            batch["auxiliary_sensor"][:, 90:93] = temp_current_subgoals

            base_batch = batch.copy()
            base_batch["sensor"] = base_batch["sensor"][:, 0:3]
            arm_batch = batch.copy()
            arm_batch["sensor"] = arm_batch["sensor"][:, 3:]

            # (
            #     values,
            #     actions,
            #     action_log_probs,
            #     recurrent_hidden_states,
            # ) = actor_critic.act(
            #     batch,
            #     recurrent_hidden_states,
            #     1 - subgoals_done,
            #     deterministic=False,
            #     update=0,
            # )
            
            (
                base_values,
                base_actions,
                base_action_log_probs,
                base_recurrent_hidden_states,
            ) = base_actor_critic.act(
                base_batch,
                recurrent_hidden_states,
                1 - subgoals_done,
                deterministic=False,
                update=0,
            )
            (
                arm_values,
                arm_actions,
                arm_action_log_probs,
                arm_recurrent_hidden_states,
            ) = arm_actor_critic.act(
                arm_batch,
                recurrent_hidden_states,
                1 - subgoals_done,
                deterministic=False,
                update=0,
            )

            values = (1 - action_mask_indices)*base_values + action_mask_indices*arm_values
            actions = (1 - action_mask_indices)*base_actions + action_mask_indices*arm_actions
            action_log_probs = (1 - action_mask_indices)*base_action_log_probs + action_mask_indices*arm_action_log_probs
            recurrent_hidden_states = (1 - action_mask_indices)*base_recurrent_hidden_states+ action_mask_indices*arm_recurrent_hidden_states

            actions_masked = actions * current_action_masks

        actions_np = actions_masked.cpu().numpy()
        outputs = envs.step(actions_np)

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        next_obs = [
            info["last_observation"] if done else obs
            for obs, done, info in zip(observations, dones, infos)
        ]

        prev_batch = batch

        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        next_obs_batch = batch_obs(next_obs)
        for sensor in next_obs_batch:
            next_obs_batch[sensor] = next_obs_batch[sensor].to(device)

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=device,
        )
        success_masks = torch.tensor(
            [
                [float(info["success"])] if done and "success" in info else [0.0]
                for done, info in zip(dones, infos)
            ],
            dtype=torch.float,
            device=device,
        )
        lengths = torch.tensor(
            [
                [float(info["episode_length"])]
                if done and "episode_length" in info
                else [0.0]
                for done, info in zip(dones, infos)
            ],
            dtype=torch.float,
            device=device,
        )
        collision_steps = torch.tensor(
            [
                [float(info["collision_step"])]
                if done and "collision_step" in info
                else [0.0]
                for done, info in zip(dones, infos)
            ],
            dtype=torch.float,
            device=device,
        )
        # total_energy_cost = torch.tensor(
        #     [[float(info["energy_cost"])] if done and "energy_cost" in info else [0.0]
        #      for done, info in zip(dones, infos)],
        #     dtype=torch.float,
        #     device=device
        # )
        # avg_energy_cost = torch.tensor(
        #     [[float(info["energy_cost"]) / float(info["episode_length"])]
        #      if done and "energy_cost" in info and "episode_length" in info
        #      else [0.0]
        #      for done, info in zip(dones, infos)],
        #     dtype=torch.float,
        #     device=device
        # )
        # stage_open_door = torch.tensor(
        #     [[float(info["stage"] >= 1)] if done and "stage" in info else [0.0]
        #      for done, info in zip(dones, infos)],
        #     dtype=torch.float,
        #     device=device
        # )
        # stage_to_target = torch.tensor(
        #     [[float(info["stage"] >= 2)] if done and "stage" in info else [0.0]
        #      for done, info in zip(dones, infos)],
        #     dtype=torch.float,
        #     device=device
        # )
        collision_rewards = torch.tensor(
            [
                [float(info["collision_reward"])]
                if "collision_reward" in info
                else [0.0]
                for info in infos
            ],
            dtype=torch.float,
            device=device,
        )

        current_episode_reward += rewards
        episode_rewards += (1 - masks) * current_episode_reward
        episode_success_rates += success_masks
        episode_lengths += lengths
        episode_collision_steps += collision_steps
        # episode_total_energy_costs += total_energy_cost
        # episode_avg_energy_costs += avg_energy_cost
        # episode_stage_open_door += stage_open_door
        # episode_stage_to_target += stage_to_target
        episode_counts += 1 - masks
        current_episode_reward *= masks

        current_subgoals_steps += 1

        subgoals_diff = (
            ideal_next_state - next_obs_batch["sensor"]
        ) * current_subgoal_masks
        subgoals_distance = torch.abs(subgoals_diff)

        subgoals_achieved = torch.all(
            subgoals_distance < subgoal_tolerance, dim=1, keepdim=True
        )

        subgoals_done = (
            subgoals_achieved  # subgoals achieved
            | (current_subgoals_steps == args.time_scale)  # subgoals time up
            | (1.0 - masks).bool()  # episode is done
        )
        subgoals_done = subgoals_done.float()
        subgoals_achieved = subgoals_achieved.float()

        prev_potential = ideal_next_state - prev_batch["sensor"]
        prev_potential = torch.norm(
            prev_potential * current_subgoal_masks, dim=1, keepdim=True
        )

        current_potential = ideal_next_state - next_obs_batch["sensor"]
        current_potential = torch.norm(
            current_potential * current_subgoal_masks, dim=1, keepdim=True
        )

        intrinsic_reward = 0.0
        intrinsic_reward += (
            prev_potential - current_potential
        ) * args.intrinsic_reward_scaling
        intrinsic_reward += subgoals_achieved.float() * args.subgoal_achieved_reward
        intrinsic_reward += collision_rewards * args.extrinsic_collision_reward_weight
        intrinsic_reward += rewards * args.extrinsic_reward_weight

        current_subgoal_reward += intrinsic_reward
        subgoal_rewards += subgoals_done * current_subgoal_reward
        subgoal_success_rates += subgoals_achieved
        subgoal_lengths += subgoals_done * current_subgoals_steps
        subgoal_counts += subgoals_done
        current_subgoal_reward *= 1 - subgoals_done

        current_subgoals = (
            ideal_next_state - next_obs_batch["sensor"]
        ) * current_subgoal_masks
        current_subgoals_steps = (1 - subgoals_done) * current_subgoals_steps
        current_meta_recurrent_hidden_states = (
            subgoals_done * next_meta_recurrent_hidden_states
            + (1 - subgoals_done) * current_meta_recurrent_hidden_states
        )
        step += 1

    episode_reward_mean = (episode_rewards.sum() / episode_counts.sum()).item()
    episode_success_rate_mean = (
        episode_success_rates.sum() / episode_counts.sum()
    ).item()
    episode_length_mean = (episode_lengths.sum() / episode_counts.sum()).item()
    episode_collision_step_mean = (
        episode_collision_steps.sum() / episode_counts.sum()
    ).item()
    # episode_total_energy_cost_mean = (
    #     episode_total_energy_costs.sum() / episode_counts.sum()
    # ).item()
    # episode_avg_energy_cost_mean = (
    #     episode_avg_energy_costs.sum() / episode_counts.sum()
    # ).item()
    # episode_stage_open_door_mean = (
    #     episode_stage_open_door.sum() / episode_counts.sum()
    # ).item()
    # episode_stage_to_target_mean = (
    #     episode_stage_to_target.sum() / episode_counts.sum()
    # ).item()

    subgoal_reward_mean = (subgoal_rewards.sum() / subgoal_counts.sum()).item()
    subgoal_success_rate_mean = (
        subgoal_success_rates.sum() / subgoal_counts.sum()
    ).item()
    subgoal_length_mean = (subgoal_lengths.sum() / subgoal_counts.sum()).item()

    if eval_only:
        print(
            "EVAL: num_eval_episodes: {}\treward: {:.3f}\t"
            "success_rate: {:.3f}\tepisode_length: {:.3f}\tcollision_step: {:.3f}\t".format(
                args.num_eval_episodes,
                episode_reward_mean,
                episode_success_rate_mean,
                episode_length_mean,
                episode_collision_step_mean,
            )
        )
        print(
            "EVAL: num_eval_episodes: {}\tsubgoal_reward: {:.3f}\t"
            "subgoal_success_rate: {:.3f}\tsubgoal_length: {:.3f}".format(
                args.num_eval_episodes,
                subgoal_reward_mean,
                subgoal_success_rate_mean,
                subgoal_length_mean,
            )
        )
    else:
        logger.info(
            "EVAL: num_eval_episodes: {}\tupdate: {}\t"
            "reward: {:.3f}\tsuccess_rate: {:.3f}\tepisode_length: {:.3f}\tcollision_step: {:.3f}".format(
                args.num_eval_episodes,
                update,
                episode_reward_mean,
                episode_success_rate_mean,
                episode_length_mean,
                episode_collision_step_mean,
            )
        )
        logger.info(
            "EVAL: num_eval_episodes: {}\tupdate: {}\t"
            "subgoal_reward: {:.3f}\tsubgoal_success_rate: {:.3f}\tsubgoal_length: {:.3f}".format(
                args.num_eval_episodes,
                update,
                subgoal_reward_mean,
                subgoal_success_rate_mean,
                subgoal_length_mean,
            )
        )
        writer.add_scalar(
            "eval/updates/reward", episode_reward_mean, global_step=update
        )
        writer.add_scalar(
            "eval/updates/success_rate", episode_success_rate_mean, global_step=update
        )
        writer.add_scalar(
            "eval/updates/episode_length", episode_length_mean, global_step=update
        )
        writer.add_scalar(
            "eval/updates/collision_step",
            episode_collision_step_mean,
            global_step=update,
        )
        # writer.add_scalar(
        #     "eval/updates/total_energy_cost",
        #     episode_total_energy_cost_mean,
        #     global_step=update,
        # )
        # writer.add_scalar(
        #     "eval/updates/avg_energy_cost",
        #     episode_avg_energy_cost_mean,
        #     global_step=update,
        # )
        # writer.add_scalar(
        #     "eval/updates/stage_open_door",
        #     episode_stage_open_door_mean,
        #     global_step=update,
        # )
        # writer.add_scalar(
        #     "eval/updates/stage_to_target",
        #     episode_stage_to_target_mean,
        #     global_step=update,
        # )

        writer.add_scalar(
            "eval/env_steps/reward", episode_reward_mean, global_step=count_steps
        )
        writer.add_scalar(
            "eval/env_steps/success_rate",
            episode_success_rate_mean,
            global_step=count_steps,
        )
        writer.add_scalar(
            "eval/env_steps/episode_length",
            episode_length_mean,
            global_step=count_steps,
        )
        writer.add_scalar(
            "eval/env_steps/collision_step",
            episode_collision_step_mean,
            global_step=count_steps,
        )
        # writer.add_scalar(
        #     "eval/env_steps/total_energy_cost",
        #     episode_total_energy_cost_mean,
        #     global_step=count_steps,
        # )
        # writer.add_scalar(
        #     "eval/env_steps/avg_energy_cost",
        #     episode_avg_energy_cost_mean,
        #     global_step=count_steps,
        # )
        # writer.add_scalar(
        #     "eval/env_steps/stage_open_door",
        #     episode_stage_open_door_mean,
        #     global_step=count_steps,
        # )
        # writer.add_scalar(
        #     "eval/env_steps/stage_to_target",
        #     episode_stage_to_target_mean,
        #     global_step=count_steps,
        # )

        writer.add_scalar(
            "eval/updates/subgoal_reward", subgoal_reward_mean, global_step=update
        )
        writer.add_scalar(
            "eval/updates/subgoal_success_rate",
            subgoal_success_rate_mean,
            global_step=update,
        )
        writer.add_scalar(
            "eval/updates/subgoal_length", subgoal_length_mean, global_step=update
        )
        writer.add_scalar(
            "eval/env_steps/subgoal_reward",
            subgoal_reward_mean,
            global_step=count_steps,
        )
        writer.add_scalar(
            "eval/env_steps/subgoal_success_rate",
            subgoal_success_rate_mean,
            global_step=count_steps,
        )
        writer.add_scalar(
            "eval/env_steps/subgoal_length",
            subgoal_length_mean,
            global_step=count_steps,
        )


def wrap_to_one(theta):
    return theta - 2.0 * torch.floor((theta + 1.0) / 2.0)


def plot_action_mask(plot_env, meta_actor_critic, hidden_size, device):
    meta_recurrent_hidden_states = torch.zeros(1, hidden_size, device=device)
    masks = torch.zeros(1, 1, device=device)
    base_heat_map = np.zeros((plot_env.height, plot_env.width))
    arm_heat_map = np.zeros((plot_env.height, plot_env.width))

    plot_env.reset()
    plot_env.agent_orientation = 3
    plot_env.target_pos = np.array([3, 5])
    for col in range(1, plot_env.width - 1):
        if col == plot_env.width // 2:
            continue
        if col > plot_env.width // 2:
            plot_env.door_state = plot_env.door_max_state
        for row in range(1, plot_env.height - 1):
            plot_env.agent_pos = np.array([row, col])
            observations = [plot_env.get_state()]
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            with torch.no_grad():
                _, _, _, _, _, _, action_mask_probs = meta_actor_critic.act(
                    batch,
                    meta_recurrent_hidden_states,
                    masks,
                )
            base_heat_map[row, col] = (
                action_mask_probs[0][0].item() + action_mask_probs[0][2].item()
            )
            arm_heat_map[row, col] = (
                action_mask_probs[0][1].item() + action_mask_probs[0][2].item()
            )
            print(row, col, action_mask_probs)
    print(arm_heat_map)
    plt.figure(0)
    plt.imshow(base_heat_map, cmap="hot", interpolation="nearest")
    plt.figure(1)
    plt.imshow(arm_heat_map, cmap="hot", interpolation="nearest")
    plt.show()
    assert False


def main():
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    add_hrl_args(parser)
    args = parser.parse_args()

    (
        ckpt_folder,
        ckpt_path,
        start_epoch,
        start_env_step,
        summary_folder,
        log_file,
    ) = set_up_experiment_folder(
        args.experiment_folder, args.checkpoint_index, args.use_checkpoint
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:{}".format(args.pth_gpu_id))
    logger.add_filehandler(log_file)

    if not args.eval_only:
        writer = SummaryWriter(log_dir=summary_folder)
    else:
        writer = None

    for p in sorted(list(vars(args))):
        logger.info("{}: {}".format(p, getattr(args, p)))

    config_file = os.path.join(
        os.path.dirname(gibson2.__file__), "../igibson_usage/new_configs", args.config_file
    )
    assert os.path.isfile(config_file), "config file does not exist: {}".format(
        config_file
    )

    env_config = parse_config(config_file)
    for (k, v) in env_config.items():
        logger.info("{}: {}".format(k, v))

    def load_env(env_mode, device_idx):
        return iGibsonEnv(
            config_file=config_file,
            mode=env_mode,
            action_timestep=args.action_timestep,
            physics_timestep=args.physics_timestep,
            automatic_reset=True,
            device_idx=device_idx,
        )

    sim_gpu_id = [int(gpu_id) for gpu_id in args.sim_gpu_id.split(",")]
    env_id_to_which_gpu = np.linspace(
        0,
        len(sim_gpu_id),
        num=args.num_train_processes + args.num_eval_processes,
        dtype=np.int,
        endpoint=False,
    )
    # train_envs = [
    #     lambda device_idx=sim_gpu_id[env_id_to_which_gpu[env_id]]: load_env(
    #         "headless", device_idx
    #     )
    #     for env_id in range(args.num_train_processes)
    # ]
    # train_envs = ParallelNavEnv(train_envs, blocking=False)
    eval_envs = [
        lambda device_idx=sim_gpu_id[env_id_to_which_gpu[env_id]]: load_env(
            "gui", device_idx
        )
        for env_id in range(
            args.num_train_processes,
            args.num_train_processes + args.num_eval_processes - 1,
        )
    ]
    eval_envs += [lambda: load_env("gui", sim_gpu_id[env_id_to_which_gpu[-1]])]
    eval_envs = ParallelNavEnv(eval_envs, blocking=False)

    # logger.info(train_envs.observation_space)
    # logger.info(train_envs.action_space)

    action_dim = eval_envs.action_space.shape[0]
    
    # action_mask = base only & arm only
    action_mask_choices = torch.zeros(2, action_dim, device=device)
    action_mask_choices[0, 0:2] = 1.0
    action_mask_choices[1, 2:] = 1.0

    cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]

    # meta observation space, 6D subgoals
    meta_observation_space = eval_envs.observation_space
    sensor_space = eval_envs.observation_space.spaces["sensor"]
    subgoal_space = gym.spaces.Box(
        low=-2.0, high=2.0, shape=sensor_space.shape, dtype=np.float32
    )

    # subgoal mask - base subgoals or arm subgoals
    subgoal_mask_choices = torch.zeros(2, sensor_space.shape[0], device=device)
    subgoal_mask_choices[0, 0:3] = 1.0
    subgoal_mask_choices[1, 3:] = 1.0

    # observation_space - used for rollouts, 6D subgoals 
    observation_space = gym.spaces.Dict(eval_envs.observation_space.spaces.copy())

    # ll_observation space -> change the sensor dim to 3D, used for our base and arm policy
    ll_observation_space = eval_envs.observation_space.spaces.copy()
    ll_observation_space["sensor"] = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(3, ), dtype=np.float32
    )
    ll_observation_space = gym.spaces.Dict(ll_observation_space)
    print("\n\n LL Obs Space: ", ll_observation_space)
    subgoal_tolerance = torch.tensor(1.0 / 3.0, dtype=torch.float32, device=device)

    meta_actor_critic = MetaPolicy(
        observation_space=meta_observation_space,
        subgoal_space=subgoal_space,
        use_action_masks=args.use_action_masks,
        action_masks_dim=action_mask_choices.shape[0],
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
        stddev_transform=torch.nn.functional.softplus,
    )
    meta_actor_critic.to(device)

    meta_agent = PPO(
        meta_actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.meta_lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        is_meta_agent=True,
        normalize_advantage=args.meta_agent_normalize_advantage,
    )

    base_actor_critic = Policy(
        observation_space=ll_observation_space,
        action_space=eval_envs.action_space,
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
        initial_stddev=args.action_init_std_dev,
        min_stddev=args.action_min_std_dev,
        stddev_anneal_schedule=args.action_std_dev_anneal_schedule,
        stddev_transform=torch.nn.functional.softplus,
    )
    arm_actor_critic = Policy(
        observation_space=ll_observation_space,
        action_space=eval_envs.action_space,
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
        initial_stddev=args.action_init_std_dev,
        min_stddev=args.action_min_std_dev,
        stddev_anneal_schedule=args.action_std_dev_anneal_schedule,
        stddev_transform=torch.nn.functional.softplus,
    )
    base_actor_critic.to(device)
    arm_actor_critic.to(device)

    base_agent = PPO(
        base_actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        is_meta_agent=False,
        normalize_advantage=True,
    )
    arm_agent = PPO(
        arm_actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        is_meta_agent=False,
        normalize_advantage=True,
    )

    # load pretrained LL & HL policy
    
    base_ckpt_path = "wts/hrl4in_base_arm_only_run2/base/ckpt.12000.pth"
    arm_ckpt_path = "wts/hrl4in_base_arm_only_run1/arm/ckpt.2700.pth"
    meta_ckpt_path = "wts/hrl4in_base_arm_only_run2/meta_ckpt.5800.pth"
    
    ckpt = torch.load(base_ckpt_path, map_location=device)
    base_agent.load_state_dict(ckpt["state_dict"])
    logger.info("loaded base checkpoint: {}".format(base_ckpt_path))

    ckpt = torch.load(arm_ckpt_path, map_location=device)
    arm_agent.load_state_dict(ckpt["state_dict"])
    logger.info("loaded arm checkpoint: {}".format(arm_ckpt_path))

    ckpt = torch.load(meta_ckpt_path, map_location=device)
    meta_agent.load_state_dict(ckpt["state_dict"])
    logger.info("loaded checkpoint: {}".format(ckpt_path))

    # Freeze params
    for p in base_actor_critic.net.parameters():
        p.requires_grad = False
    for p in arm_actor_critic.net.parameters():
        p.requires_grad = False

    # elif ckpt_path is not None:
    #     ckpt = torch.load(ckpt_path, map_location=device)
    #     arm_agent.load_state_dict(ckpt["state_dict"])
    #     logger.info("loaded checkpoint: {}".format(ckpt_path))

    #     ckpt_path = os.path.join(
    #         os.path.dirname(ckpt_path),
    #         os.path.basename(ckpt_path).replace("ckpt", "meta_ckpt"),
    #     )
    #     ckpt = torch.load(ckpt_path, map_location=device)
    #     meta_agent.load_state_dict(ckpt["state_dict"])
    #     logger.info("loaded checkpoint: {}".format(ckpt_path))

    logger.info(
        "base LL agent number of parameters: {}".format(
            sum(param.numel() for param in base_agent.parameters())
        )
    )
    logger.info(
        "Arm LL agent number of parameters: {}".format(
            sum(param.numel() for param in arm_agent.parameters())
        )
    )
    logger.info(
        "meta agent number of parameters: {}".format(
            sum(param.numel() for param in meta_agent.parameters())
        )
    )

    if args.eval_only:
        evaluate(
            args,
            eval_envs,
            meta_actor_critic,
            base_actor_critic,
            arm_actor_critic,
            action_mask_choices,
            subgoal_mask_choices,
            subgoal_tolerance,
            device,
            writer,
            update=0,
            count_steps=0,
            eval_only=True,
        )
        return


if __name__ == "__main__":
    main()
