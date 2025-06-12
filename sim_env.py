"""
双臂机器人仿真环境模块

这个模块实现了基于MuJoCo的双臂机器人仿真环境，支持以下功能：
1. 双臂ViperX机器人的关节位置控制
2. 物体转移和插入任务
3. 基于接触的奖励计算
4. 多视角相机观察
"""

import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None]  # 物体初始位姿，需要从外部设置

def make_sim_env(task_name):
    """
    创建仿真环境，支持双臂机器人操作任务
    
    动作空间:  
        - 左臂关节位置 (6维)
        - 左夹爪位置 (1维, 归一化: 0表示关闭, 1表示打开)
        - 右臂关节位置 (6维)
        - 右夹爪位置 (1维, 归一化: 0表示关闭, 1表示打开)

    观察空间:
        - qpos: 关节位置状态
            - 左臂关节位置 (6维)
            - 左夹爪位置 (1维, 归一化)
            - 右臂关节位置 (6维)
            - 右夹爪位置 (1维, 归一化)
        - qvel: 关节速度状态
            - 左臂关节速度 (6维, 弧度/秒)
            - 左夹爪速度 (1维, 归一化, 正值表示打开, 负值表示关闭)
            - 右臂关节速度 (6维, 弧度/秒)
            - 右夹爪速度 (1维, 归一化, 正值表示打开, 负值表示关闭)
        - images: 相机观察图像
            - main: RGB图像 (480x640x3)
    """
    if 'sim_transfer_cube' in task_name:
        # 创建物体转移任务环境
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        # 创建插入任务环境
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    """双臂ViperX机器人任务的基类，实现基本的控制和状态获取功能"""
    
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        """执行动作前的预处理，将归一化的夹爪动作转换为实际控制信号"""
        # 分解动作向量
        left_arm_action = action[:6]  # 左臂关节动作
        right_arm_action = action[7:7+6]  # 右臂关节动作
        normalized_left_gripper_action = action[6]  # 归一化的左夹爪动作
        normalized_right_gripper_action = action[7+6]  # 归一化的右夹爪动作

        # 转换夹爪动作到实际控制范围
        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        # 生成完整的夹爪动作（考虑对称性）
        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        # 组合完整的动作向量
        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """初始化每个回合的环境状态"""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        """获取当前的关节位置状态"""
        qpos_raw = physics.data.qpos.copy()
        # 分离左右臂的关节状态
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        # 提取臂部关节位置
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        # 提取并归一化夹爪位置
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        """获取当前的关节速度状态"""
        qvel_raw = physics.data.qvel.copy()
        # 分离左右臂的速度状态
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        # 提取臂部关节速度
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        # 提取并归一化夹爪速度
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        """获取环境状态（在子类中实现）"""
        raise NotImplementedError

    def get_observation(self, physics):
        """获取完整的观察信息，包括关节状态和相机图像"""
        obs = collections.OrderedDict()
        # 获取关节状态
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        # 获取多个视角的相机图像
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        return obs

    def get_reward(self, physics):
        """计算奖励值（在子类中实现）"""
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    """物体转移任务：机器人需要用右臂抓取物体并传递给左臂"""
    
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4  # 最大奖励值

    def initialize_episode(self, physics):
        """初始化回合，设置机器人和物体的初始状态"""
        # 注意：该函数不随机化环境配置，物体位置需要从外部设置
        with physics.reset_context():
            # 设置机器人初始姿态
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            # 设置物体初始位置
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        """获取环境状态（物体的位置和姿态）"""
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        """计算奖励值，基于物体与机器人的接触状态
        
        奖励等级：
        0: 初始状态
        1: 右夹爪接触物体
        2: 右夹爪抓起物体（物体离开桌面）
        3: 左夹爪接触物体（尝试传递）
        4: 左夹爪成功接住物体（完成传递）
        """
        # 获取所有接触对
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # 检查关键接触状态
        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        # 计算阶段性奖励
        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # 物体被抓起
            reward = 2
        if touch_left_gripper:  # 开始传递
            reward = 3
        if touch_left_gripper and not touch_table:  # 成功传递
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    """插入任务：机器人需要完成物体的精确插入操作"""
    
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4  # 最大奖励值

    def initialize_episode(self, physics):
        """初始化回合，设置机器人和物体的初始状态"""
        # 注意：该函数不随机化环境配置，物体位置需要从外部设置
        with physics.reset_context():
            # 设置机器人初始姿态
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            # 设置两个物体的初始位置
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0]
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        """获取环境状态（两个物体的位置和姿态）"""
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        """计算奖励值，基于插入任务的完成程度
        
        奖励等级：
        0: 初始状态
        1: 右夹爪接触插销
        2: 插销离开初始位置
        3: 插销接近目标位置
        4: 成功完成插入
        """
        # 获取所有接触对
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # 检查关键接触状态
        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        # 计算阶段性奖励
        reward = 0
        if touch_left_gripper and touch_right_gripper:  # 开始操作
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table):  # 插销离开初始位置
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # 插销接近目标位置
            reward = 3
        if pin_touched:  # 成功完成插入
            reward = 4
        return reward


def get_action(master_bot_left, master_bot_right):
    """根据主机器人状态生成控制动作"""
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """测试仿真环境下的远程控制"""
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()
