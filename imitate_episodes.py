"""
模仿学习训练和评估脚本

该脚本用于训练和评估模仿学习算法,支持ACT和CNNMLP两种策略。
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # 数据加载函数
from utils import sample_box_pose, sample_insertion_pose # 机器人相关函数
from utils import compute_dict_mean, set_seed, detach_dict # 辅助函数
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    """
    主函数，用于解析命令行参数和执行训练或评估
    """
    print("args : ", args)
    # 设置随机种子，确保结果可复现
    set_seed(1)
    # 解析命令行参数
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # 获取任务参数
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        # TODO: fix real data task config
        # from aloha_scripts.constants import TASK_CONFIGS
        TASK_CONFIGS = {
            'piper_pick_and_place':{
                'dataset_dir': 'piper_pick_and_place',  # 相对于data_dir的路径
                'episode_len': 50,
                'camera_names': ['cam_right_wrist']
                # 'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
            },
        }
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # 固定参数
    state_dim = 14  # 机器人关节状态的维度，joint position
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        # 策略配置参数
        policy_config = {
            'lr': args['lr'],                     # 学习率
            'num_queries': args['chunk_size'],    # 查询数量,对应动作序列长度
            'kl_weight': args['kl_weight'],       # KL散度权重,用于VAE训练
            'hidden_dim': args['hidden_dim'],     # 隐藏层维度
            'dim_feedforward': args['dim_feedforward'],  # 前馈网络维度
            'lr_backbone': lr_backbone,           # 主干网络学习率
            'backbone': backbone,                 # 主干网络类型
            'enc_layers': enc_layers,             # 编码器层数
            'dec_layers': dec_layers,             # 解码器层数
            'nheads': nheads,                     # 注意力头数
            'camera_names': camera_names,         # 使用的相机名称列表
        }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    # 评估
    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # 训练
    # 1. 创建数据加载器，而不是立即加载所有离线数据
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # 2.保存数据集统计信息
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # 3.训练模型
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # 4.保存最佳模型
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    """
    根据策略类别创建策略对象
    """
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    """
    根据策略类别创建优化器
    """
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    """
    获取当前时间步的相机图像，并转换为模型输入格式
    """
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    # 将多个相机图像堆叠为单个数组
    curr_image = np.stack(curr_images, axis=0)  # shape: (num_cameras, C, H, W)
    
    # 归一化图像并转换为PyTorch张量，然后移至GPU
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda()
    
    # 添加批次维度
    curr_image = curr_image.unsqueeze(0)  # shape: (1, num_cameras, C, H, W)
    
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    """
    评估行为克隆策略的性能
    """
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # 加载策略和统计信息
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    # create policy
    policy = make_policy(policy_class, policy_config)
    # load policy parameters
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 减去均值并除以标准差，使数据分布更均匀，有利于模型学习
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    # 将模型输出的动作转换回原始尺度
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # 加载环境
    if real_robot: # 真机
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else: # 仿真
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    # 设置查询频率
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        # 如果使用时间聚合，则每步查询一次
        query_frequency = 1
        # 保存原始的查询数量
        num_queries = policy_config['num_queries']

    # episode_len
    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    # 评估50次
    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        # 随机操作物体的初始位置
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        # 重置环境并获取初始时间步
        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        # temporal_agg用于控制动作预测的时间聚合
        # 当启用时，模型在每个时间步都会预测整个动作序列，而不是每隔固定间隔预测一次
        # 这允许模型考虑更长期的时间依赖，可能提高预测的连贯性和准确性
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda() # shape: (1, episode_len, state_dim)
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    # 可视化
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                # 预处理: 将原始关节位置数据标准化
                qpos = pre_process(qpos_numpy) # shape: (14,)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0) # shape: (1, 14)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names) # shape: (1, num_cameras, C, H, W)

                ### query policy
                if config['policy_class'] == "ACT":
                    # 如果当前时间步是查询频率的倍数，则重新预测整个动作序列
                    if t % query_frequency == 0:
                        # qpos shape: (1, 14)
                        # curr_image shape: (1, num_cameras, C, H, W)
                        # all_actions shape: (1, episode_len, 14)
                        all_actions = policy(qpos, curr_image)
                    
                    if temporal_agg:
                        # 将预测的动作序列存储到all_time_actions中
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        # 获取当前时间步之前的所有预测动作
                        actions_for_curr_step = all_time_actions[:, t]  # shape: (max_timesteps, 14)
                        # 找出已经填充了动作的时间步
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)  # shape: (max_timesteps,)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]  # shape: (num_populated, 14)
                        
                        # 计算指数衰减权重
                        # w_i = exp(-k * i) / sum(exp(-k * i))
                        # 这里的i是时间步索引，k是衰减率
                        k = 0.01  # 衰减率
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()  # 归一化权重
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1) # shape: (num_populated, 1)
                        
                        # 使用加权平均计算最终动作
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True) # shape: (1, 14)
                    else:
                        # 如果不使用时间聚合，直接选择当前时间步对应的动作
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy() # shape: (14,)
                # 后处理: 将动作换回原始尺度
                action = post_process(raw_action) # shape: (14,)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    """
    前向传播函数，用于训练和评估
    """
    # image_data: (batch_size, C, H, W)
    # qpos_data: (batch_size, 14)
    # action_data: (batch_size, episode_len, 14)
    # is_pad: (batch_size, episode_len)
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # policy: ACTPolicy or CNNMLPPolicy
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    """
    训练行为克隆策略
    """
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    # create policy
    policy = make_policy(policy_class, policy_config) # ACTPolicy or CNNMLPPolicy
    policy.cuda()
    # create optimizer
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    # 训练num_epochs个epoch, 每个epoch包括训练和验证
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            # data: (image_data, qpos_data, action_data, is_pad)
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy) # forward_dict: {'loss': loss, 'action': action}
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            # 更新最小验证损失
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        # 计算平均损失
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # 100个epoch保存一次
        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """
    绘制训练和验证曲线
    """
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # traning or evaluation
    parser.add_argument('--eval', action='store_true')
    # 是否在屏幕上显示渲染结果
    parser.add_argument('--onscreen_render', action='store_true')
    # 保存ckpt的目录
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    # 策略类: act or cnnmlp
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    # 任务名称
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # batch_size
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    # seed
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    # 训练轮数
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    # 学习率
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT policy
    # VAE中的KL权重
    parser.add_argument('--kl_weight', action='store', type=int, help='KL 权重', required=False)
    # 动作块的长度
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    # transformer的encoder输入维度
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    # transformer 前馈网络FFN的维度
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    # 时序聚合 
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
