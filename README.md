# Software Dependency

- Ubuntu 20.04 LTS
- ROS Noetic

# ACT: Action Chunking with Transformers

## [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
  if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

## Project Website: https://tonyzhaozh.github.io/aloha/

## 详细使用文档 [ALOHA ACT算法训练、推理](https://nxjux7a2aq.feishu.cn/wiki/WZX7wu1xvi31CWkwsELcMKTYngf)

# 1. Software installation

## 1.1 install nvidia-driver (If you haven't installed it before)
  查看适用的nvidia-driver:
  ```sh
  ubuntu-drivers devices
  ```

  安装推荐的版本,比如:
  ```sh
  sudo apt install nvidia-driver-570
  ```

  重启电脑后, 查看驱动是否安装成功:
  ```sh
  nvidia-smi
  ```

## 1.2 Create conda environment:
  ```sh
  conda create -n aloha python=3.8.10
  conda activate aloha
  pip install -r requirements.txt

  cd detr && pip install -e .
  ```

# 2.tranning
  ```sh
  bash train.sh
  ```

# 3.evaluate dataset
  ```sh
  bash eval_dataset.sh
  ```

# 4.evaluate real robot
## 4.1 source the ros env first, because we get model input data from rostopic.
if you use python sdk:
  ```sh
  cd y1_sdk_python/y1_ros
  source devel/setup.bash
  ```
if you use c++ sdk:
  ```sh
  cd y1_sdk
  source devel/setup.bash
  ```

## 4.2 model inference
  ```sh
  bash eval_real_robot.sh
  ```