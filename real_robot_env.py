from y1_msg.msg import ArmJointState
from y1_msg.msg import ArmJointPositionControl
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
# import rospy
import numpy as np
import torch
from typing import Union
import rclpy
from rclpy.node import Node
from task_config import TASK_CONFIGS

class RealRobotEnv(Node):
  def __init__(self, args):
    # 确保rclpy已初始化
    if not rclpy.ok():
      rclpy.init()
    
    super().__init__('aloha_act_node')
    self.task_config = TASK_CONFIGS[args['task_name']]
    self.bridge = CvBridge()
    self.right_puppet_arm_state = None
    self.left_puppet_arm_state = None
    self.img_dict = {}
    self.left_arm_joint_position_control_pub_ = None
    self.right_arm_joint_position_control_pub_ = None
    # 初始化订阅
    self.init_subscriptions()

  def destroy(self):
    self.destroy_node()
    rclpy.shutdown()
    
  def init_subscriptions(self):
    # subscribe
    # robotic arm data
    state_dim = self.task_config['state_dim']
    if state_dim == 7:
      # one arm, default right arm
      self.create_subscription(
            ArmJointState,
            "/puppet_arm_right/joint_states",
            self.puppet_arm_right_callback,
            1)
      
      # control right arm
      self.right_arm_joint_position_control_pub_ = self.create_publisher(
            ArmJointPositionControl,
            '/master_arm_right/joint_states',
            1)
      
    elif state_dim == 14:
      # two arm
      self.create_subscription(
            ArmJointState,
            "/puppet_arm_right/joint_states",
            self.puppet_arm_right_callback,
            1)
      
      self.create_subscription(
          ArmJointState,
          "/puppet_arm_left/joint_states",
          self.puppet_arm_left_callback,
          1)
      
      # control left and right arm
      self.left_arm_joint_position_control_pub_ = self.create_publisher(
          ArmJointPositionControl,
          '/master_arm_left/joint_states',
          1)
      self.right_arm_joint_position_control_pub_ = self.create_publisher(
          ArmJointPositionControl,
          '/master_arm_right/joint_states',
          1)   
    else:
      raise Exception(f"state dim {state_dim} not support, only support 7 or 14")
  
    # subscribe camera rgb data
    camera_names = self.task_config['camera_names']
    for cam_name in camera_names:
      if cam_name == "cam_right_wrist":
        # right arm wrist camera rgb image
        self.create_subscription(
            Image, "/camera_right/color/image_raw", self.img_right_callback, 1)
      elif cam_name == "cam_left_wrist":
        # left arm wrist camera rgb image
        self.create_subscription(
            Image, "/camera_left/color/image_raw", self.img_left_callback, 1)
      elif cam_name == "cam_high":
        # high camera rgb image
        self.create_subscription(
            Image, "/camera_high/color/image_raw", self.img_high_callback, 1)
      elif cam_name == "cam_low":
        # low camera rgb image
        self.create_subscription(
            Image, "/camera_low/color/image_raw", self.img_low_callback, 1)
      else:
        raise Exception(f"camera name {cam_name} not found")

  def puppet_arm_right_callback(self, msg: ArmJointState):
    """right arm"""
    self.right_puppet_arm_state = msg 
    
  def puppet_arm_left_callback(self, msg: ArmJointState):
    """left arm"""
    self.left_puppet_arm_state = msg 
    
  def img_right_callback(self, msg: Image):
    """right arm wrist camera rgb image"""
    self.img_dict["cam_right_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def img_left_callback(self, msg: Image):
    """left arm wrist camera rgb image"""
    self.img_dict["cam_left_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def img_high_callback(self, msg: Image):
    """high camera rgb image"""
    self.img_dict["cam_high"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def img_low_callback(self, msg: Image):
    """low camera rgb image"""
    self.img_dict["cam_low"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def get_observation(self):
    observation = {}

    # state
    state_dim = self.task_config['state_dim']
    if state_dim == 7:
      # single arm
      if self.right_puppet_arm_state is None:
        print("not receive right arm data")
        return None
      else:
        joint_state = np.array(self.right_puppet_arm_state.joint_position)
        observation["state"] = joint_state
    elif state_dim == 14:
      # double arm
      if self.right_puppet_arm_state is None:
        print("not receive right arm data")
        return None

      if self.left_puppet_arm_state is None:
        print("not receive left arm data")
        return None
      
      observation["state"] = np.concatenate([self.left_puppet_arm_state.joint_position,
                                 self.right_puppet_arm_state.joint_position])
      
    else:
      raise Exception(f"state dim {state_dim} not support, only support 7 or 14")
    
    # image
    image_list = []
    for cam_name in self.task_config['camera_names']:
      if cam_name not in  self.img_dict:
        print(f"not receive {cam_name} image data")
        return None
      image_list.append(self.img_dict[cam_name])
      
    observation["images"] = image_list
    
    return observation
    
  def step(self, action: Union[list, np.ndarray, torch.Tensor]):
    if self.task_config['state_dim'] == 7:
      # single arm, default right arm
      joint_control_msg = ArmJointPositionControl()
      joint_control_msg.header.stamp = self.get_clock().now().to_msg()
      joint_control_msg.joint_position = action[0:6]
      joint_control_msg.joint_velocity = 3
      joint_control_msg.gripper_stroke = action[6]
      joint_control_msg.gripper_velocity = 3
      self.right_arm_joint_position_control_pub_.publish(joint_control_msg)

    elif self.task_config['state_dim'] == 14:
      # action[0:6]  -> left arm control
      left_arm_control_msg = ArmJointPositionControl()
      left_arm_control_msg.header.stamp = self.get_clock().now().to_msg()
      left_arm_control_msg.joint_position = action[0:6]
      left_arm_control_msg.joint_velocity = 3
      left_arm_control_msg.gripper_stroke = action[6]
      left_arm_control_msg.gripper_velocity = 3
      self.left_arm_joint_position_control_pub_.publish(left_arm_control_msg)

      # action[7:13] -> right arm control
      right_arm_control_msg = ArmJointPositionControl()
      right_arm_control_msg.header.stamp = self.get_clock().now().to_msg()
      right_arm_control_msg.joint_position = action[7:13]
      right_arm_control_msg.joint_velocity = 3
      right_arm_control_msg.gripper_stroke = action[13]
      right_arm_control_msg.gripper_velocity = 3
      self.right_arm_joint_position_control_pub_.publish(right_arm_control_msg)
      
    else:
      raise Exception(f"state dim {self.task_config['state_dim']} not support, only support 7 or 14")
