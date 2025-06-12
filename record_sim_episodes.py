# 导入必要的库
import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

# 导入自定义常量和环境
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy

import IPython
e = IPython.embed


"""
在仿真环境中生成示范数据。
主要步骤：
1. 首先在ee_sim_env（末端执行器仿真环境）中执行策略（在末端执行器空间中定义），获取关节轨迹
2. 用命令的关节位置替换夹持器关节位置
3. 在sim_env中重放这个关节轨迹（作为动作序列），并记录所有观察数据
4. 保存这个数据片段，继续收集下一个数据片段

支持的任务：
- sim_transfer_cube_scripted: 使用脚本化策略进行立方体转移任务
- sim_insertion_scripted: 使用脚本化策略进行插入任务

数据收集过程：
1. 首先在末端执行器控制的环境(ee_sim_env)中执行任务
2. 记录机器人的关节轨迹和夹持器状态
3. 在关节控制的环境(sim_env)中重放这些轨迹
4. 收集并保存所有的观察数据（图像、关节状态等）
"""


def main(args):
    """
    在仿真环境中生成示范数据。
    首先在末端执行器空间的仿真环境中执行策略，获取关节轨迹。
    用命令的关节位置替换夹持器关节位置。
    在仿真环境中重放这个关节轨迹（作为动作序列），并记录所有观察数据。
    保存这个数据片段，然后继续收集下一个数据片段。

    参数：
        args: 字典，包含以下键值：
            - task_name: 任务名称，支持'sim_transfer_cube_scripted'和'sim_insertion_scripted'
            - dataset_dir: 数据集保存目录
            - num_episodes: 要收集的数据片段数量
            - onscreen_render: 是否在屏幕上显示渲染结果
    """

    # 从配置中获取任务相关参数
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name = 'angle'

    # 如果数据集目录不存在，创建它
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    # 从任务配置中获取episode长度和相机名称
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    # 根据任务名称选择对应的策略类
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # 设置环境
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        # 设置可视化
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        # 计算episode的总回报和最大回报
        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        # 获取关节轨迹和夹持器控制轨迹
        joint_traj = [ts.observation['qpos'] for ts in episode]
        # 用夹持器控制信号替换夹持器位置
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        # 保存初始状态信息（步骤0时的物体位置）
        subtask_info = episode[0].observation['env_state'].copy()

        # 清理不再使用的变量以释放内存
        del env
        del episode
        del policy

        # 设置仿真环境进行轨迹重放
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info  # 确保sim_env与ee_sim_env有相同的物体配置
        ts = env.reset()

        episode_replay = [ts]
        # 设置可视化
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        # 计算episode的总回报和最大回报
        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        """
        每个时间步的数据格式：
        observations（观察数据）:
        - images（图像）
            - each_cam_name（每个相机名称）    (480, 640, 3) 'uint8'类型
        - qpos（关节位置）                    (14,)         'float64'类型
        - qvel（关节速度）                    (14,)         'float64'类型

        action（动作）                        (14,)         'float64'类型
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # 因为重放过程会导致比原始轨迹多一个时间步，这里进行截断以保持一致
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # joint_traj（动作）的长度：max_timesteps
        # episode_replay（时间步）的长度：max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # 使用HDF5格式保存数据
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            # 创建图像数据集，使用分块存储以提高性能
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # 创建其他观察和动作数据集
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            # 将数据写入HDF5文件
            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('--task_name', action='store', type=str, help='任务名称', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='数据集保存目录', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='数据收集的episode数量', required=False)
    parser.add_argument('--onscreen_render', action='store_true', help='是否在屏幕上显示渲染结果')
    
    main(vars(parser.parse_args()))
