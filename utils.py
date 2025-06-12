"""
工具函数模块

该模块包含数据加载、数据预处理、环境采样等工具函数。
主要功能包括：
1. 数据集加载和预处理
2. 数据标准化
3. 环境状态采样
4. 辅助函数
"""

import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    """情节式数据集类，用于加载和预处理机器人轨迹数据
    
    特点：
    - 支持按情节id加载数据
    - 处理图像和状态数据
    - 实现数据标准化
    """
    
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        """初始化数据集参数
        
        Args:
            episode_ids: 情节ID列表
            dataset_dir: 数据集目录
            camera_names: 相机名称列表
            norm_stats: 标准化统计信息
        """
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0)  # 初始化self.is_sim

    def __len__(self):
        """返回数据集中的情节数量"""
        return len(self.episode_ids)

    def __getitem__(self, index):
        """获取单个数据样本
        
        Args:
            index: 样本索引
            
        Returns:
            tuple: (图像数据, 关节位置数据, 动作数据, 填充标志)
        """
        sample_full_episode = False  # 是否采样完整情节

        # 加载数据文件
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            
            # 选择起始时间步
            if sample_full_episode:
                start_ts = 0
            else:
                # 随机选择起始点
                start_ts = np.random.choice(episode_len)
                
            # 仅获取start_ts时刻的观察数据
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts] # 注意：这里提取了qvel但未使用
            
            # 仅获取start_ts时刻的相机图像
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                
            # 获取动作数据
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                # TODO: 真实机器人数据从上一个时刻的action开始?
                action = root['/action'][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim
        
        # 处理动作序列数据，添加填充
        # 创建一个全零数组，大小与原始动作序列相同
        padded_action = np.zeros(original_action_shape, dtype=np.float32)  # 形状(400, 14)
        # 将实际动作数据复制到前action_len个时间步
        padded_action[:action_len] = action  # 只填充有效部分，后面保持为0
        # 创建padding掩码
        is_pad = np.zeros(episode_len) # 全部初始化为0（表示有效数据）
        is_pad[action_len:] = 1 # 将padding部分标记为1

        # 堆叠多个相机的图像
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # 转换为torch张量
        image_data = torch.from_numpy(all_cam_images) #（K, H, W, C)
        qpos_data = torch.from_numpy(qpos).float() # (14,)
        action_data = torch.from_numpy(padded_action).float() # (400, 14)
        is_pad = torch.from_numpy(is_pad).bool() # (400,)

        # 调整图像通道顺序（NCHW格式）
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # 数据标准化
        image_data = image_data / 255.0  # 图像归一化
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    """计算数据集的标准化统计信息
    
    Args:
        dataset_dir: 数据集目录
        num_episodes: 演示数据数量
        
    Returns:
        dict: 包含均值和标准差的统计信息
    """
    # 初始化列表用于存储所有episode的qpos和action数据
    all_qpos_data = []
    all_action_data = []
    
    # 遍历所有episode
    for episode_idx in range(num_episodes):
        # 构建每个episode的数据文件路径
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        
        # 使用h5py读取HDF5文件
        with h5py.File(dataset_path, 'r') as root:
            # 提取qpos, qvel和action数据
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]  # 注意：这里提取了qvel但未使用
            action = root['/action'][()]
        
        # 将numpy数组转换为torch张量并添加到列表中
        # list(tensor[T, D]) -> tensor[N, T, D]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    
    # 将所有episode的数据堆叠成一个大的张量
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    
    # 计算action数据的均值和标准差
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    # 将标准差的最小值限制在1e-2，以避免除以零的情况
    action_std = torch.clip(action_std, 1e-2, np.inf)
    
    # 计算qpos数据的均值和标准差
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    # 同样限制qpos标准差的最小值
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)
    
    # 构建包含所有统计信息的字典
    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos  # 保存一个qpos样例，可能用于后续处理
    }
    
    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    """加载并准备训练和验证数据
    
    Args:
        dataset_dir: 数据集目录
        num_episodes: 演示数据数量
        camera_names: 相机名称列表
        batch_size_train: 训练批次大小
        batch_size_val: 验证批次大小
        
    Returns:
        tuple: (训练数据加载器, 验证数据加载器, 标准化统计信息, 是否为仿真数据)
    """
    print(f'\nData from: {dataset_dir}\n')
    
    # 训练集和验证集划分
    train_ratio = 0.8
    # numpy.random.permutation 函数随机打乱索引顺序
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # 计算数据标准化统计信息
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # 构建数据集和数据加载器
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    
    # 创建数据加载器，设置批处理和多进程加载
    # 如果想提高训练速度，可以考虑：
    # num_workers=4,        # 增加工作进程
    # prefetch_factor=2     # 增加预加载因子
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True,     # 打乱数据
        pin_memory=True,  # 使用固定内存，加快数据传输
        num_workers=1,    # 数据加载的工作进程数
        prefetch_factor=1 # 预加载因子
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size_val, 
        shuffle=True, 
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### 环境工具函数

def sample_box_pose():
    """采样物体的初始位姿
    
    Returns:
        np.ndarray: 包含位置和四元数的姿态数组
    """
    # 定义位置范围
    x_range = [0.0, 0.2]   # x方向范围
    y_range = [0.4, 0.6]   # y方向范围
    z_range = [0.05, 0.05] # z方向固定高度

    # 在指定范围内随机采样位置
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1]) # shape: (3,)

    # 设置固定的朝向（四元数表示）
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat]) # shape: (7,)

def sample_insertion_pose():
    """采样插入任务中插销和插座的初始位姿
    
    Returns:
        tuple: (插销位姿, 插座位姿)
    """
    # 采样插销位置
    x_range = [0.1, 0.2]   # 插销在工作空间右侧
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1]) # shape: (3,)

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat]) # shape: (7,)

    # 采样插座位置
    x_range = [-0.2, -0.1] # 插座在工作空间左侧
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1]) # shape: (3,)

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat]) # shape: (7,)

    return peg_pose, socket_pose

### 辅助函数

def compute_dict_mean(epoch_dicts):
    """计算多个字典中相同键的平均值
    
    Args:
        epoch_dicts: 字典列表
        
    Returns:
        dict: 平均值字典
    """
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    """分离字典中的PyTorch张量, 用于减少内存使用
    
    Args:
        d: 输入字典
        
    Returns:
        dict: 分离后的字典
    """
    new_d = dict()
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_d[k] = v.detach()
        else:
            new_d[k] = v
    return new_d

def set_seed(seed):
    """设置随机种子以确保结果可重现
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
