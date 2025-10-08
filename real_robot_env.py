from y1_msg.msg import ArmJointState
from y1_msg.msg import ArmJointPositionControl
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
import numpy as np
import torch
from typing import Union
from task_config import TASK_CONFIGS

class RealRobotEnv:
  def __init__(self, args):
    self.task_config = TASK_CONFIGS[args['task_name']]
    
    self.bridge = CvBridge()
    self.right_puppet_arm_state = None
    self.left_puppet_arm_state = None
    self.img_dict = {}
    self.left_arm_joint_position_control_pub_ = None
    self.right_arm_joint_position_control_pub_ = None
    self.init_topic()
    
  def init_topic(self):
    rospy.init_node("eval_real_robot")
    
    # subscribe
    # robotic arm data
    state_dim = self.task_config['state_dim']
    if state_dim == 7:
      # one arm, default right arm
      rospy.Subscriber("/y1/arm_joint_state",
          ArmJointState, self.puppet_arm_right_callback, queue_size=1, tcp_nodelay=True)
      # control right arm
      self.right_arm_joint_position_control_pub_ = rospy.Publisher('/y1/arm_joint_position_control', 
                                                                   ArmJointPositionControl, queue_size=1)
      
    elif state_dim == 14:
      # two arm
      rospy.Subscriber("/arm_joint_state",
            ArmJointState, self.puppet_arm_right_callback, queue_size=1, tcp_nodelay=True)
      rospy.Subscriber("/puppet_arm_left/joint_states",
            ArmJointState, self.puppet_arm_left_callback, queue_size=1, tcp_nodelay=True)
      
      # control left and right arm
      self.left_arm_joint_position_control_pub_ = rospy.Publisher('/puppet_arm_left/joint_states', 
                                                                  ArmJointPositionControl, queue_size=1)
      self.right_arm_joint_position_control_pub_ = rospy.Publisher('/puppet_arm_right/joint_states', 
                                                                   ArmJointPositionControl, queue_size=1)    
    else:
      raise Exception(f"state dim {state_dim} not support, only support 7 or 14")
  
    # subscribe camera rgb data
    camera_names = self.task_config['camera_names']
    for cam_name in camera_names:
      if cam_name == "cam_right_wrist":
        # right arm wrist camera rgb image
        rospy.Subscriber("/camera_right/color/image_raw", 
          Image, self.img_right_callback, queue_size=1, tcp_nodelay=True)
      elif cam_name == "cam_left_wrist":
        # left arm wrist camera rgb image
        rospy.Subscriber("/camera_left/color/image_raw", 
          Image, self.img_left_callback, queue_size=1, tcp_nodelay=True)
      elif cam_name == "cam_front":
        # front camera rgb image
        rospy.Subscriber("/camera_front/color/image_raw", 
          Image, self.img_front_callback, queue_size=1, tcp_nodelay=True)
      elif cam_name == "cam_top":
        # top camera rgb image
        rospy.Subscriber("/camera_top/color/image_raw", 
          Image, self.img_top_callback, queue_size=1, tcp_nodelay=True)
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
    
  def img_front_callback(self, msg: Image):
    """front camera rgb image"""
    self.img_dict["cam_front"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def img_top_callback(self, msg: Image):
    """top camera rgb image"""
    self.img_dict["cam_top"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def get_observation(self):
    observation = {}
    # state
    if self.right_puppet_arm_state is None:
      print("not receive right arm data")
      return None
    else:
      joint_state = np.array(self.right_puppet_arm_state.joint_position)
      observation["state"] = joint_state
      # observation["state"] = torch.from_numpy(joint_state).float()
      
    # TODO: add left arm joint state
    
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
    joint_control_msg = ArmJointPositionControl()
    joint_control_msg.header.stamp = rospy.Time.now()
    joint_control_msg.arm_joint_position = action[0:6]
    joint_control_msg.gripper = action[6]
    
    self.right_arm_joint_position_control_pub_.publish(joint_control_msg)