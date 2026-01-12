import time
from abc import ABC
from pathlib import Path

import atlas
import cv2
import link
import numpy as np
from link._link import Node
from manip_shared_msg.locomotion.robot_state_pb2 import RobotState
from manip_shared_msg.locomotion.servo_effector_pb2 import ServoEffector
from manip_shared_msg.locomotion.servo_joint_pb2 import ServoJoint

# "front right left"
# front_img = cv2.imread("assets/images/right_arm_pick_bottle/front-color.png")
# wrist_right_img = cv2.imread("assets/images/right_arm_pick_bottle/wrist_right-color.png")
# wrist_left_img = cv2.imread("assets/images/dual_move_thing_to_cup/wrist_left-color.png")

# "right front left"
# _ASSETS_DIR = Path(__file__).resolve().parent / "assets"
# front_img = cv2.imread(str(_ASSETS_DIR / "images/dual_arm_pick_bottle_300/front-color.png"))
# wrist_right_img = cv2.imread(str(_ASSETS_DIR / "images/dual_arm_pick_bottle_300/wrist_right-color.png"))
# wrist_left_img = cv2.imread(str(_ASSETS_DIR / "images/dual_arm_pick_bottle_300/wrist_left-color.png"))
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
# front_img = cv2.imread(str(_ASSETS_DIR / "images/dual_arm_pick_water_from_box/front-color.png"))
# wrist_right_img = cv2.imread(str(_ASSETS_DIR / "images/dual_arm_pick_water_from_box/wrist_right-color.png"))
# wrist_left_img = cv2.imread(str(_ASSETS_DIR / "images/dual_arm_pick_water_from_box/wrist_left-color.png"))

front_img = cv2.imread("image_assets/images/dual_arm_pick_box_100/front-color.png")
wrist_right_img = cv2.imread("image_assets/images/dual_arm_pick_box_100/wrist_right-color.png")
wrist_left_img = cv2.imread("image_assets/images/dual_arm_pick_box_100/wrist_left-color.png")

def now_ros_stamp():
    ns = time.time_ns()
    return ns // 1_000_000_000, ns % 1_000_000_000

def _require_img(img, name: str):
    if img is None:
        raise FileNotFoundError(f"加载图像失败: {name}. 请确认 assets/images 路径存在，并在 middleware 目录运行: python pub_first_img.py")


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    # cv2.imread 读取为 BGR，这里转成 RGB8 以匹配 link.ImageFormat.RGB8
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class BaseNode(ABC):
    node_cpp: Node = None

    def _init_node(self, node_name):
        link.Node.Initialize(node_name)
        self.node_cpp = link.GetNode()


class FirstImgPubNode(BaseNode):
    def __init__(self, node_name):
        self._init_node(node_name)

        # R2V2 初始位置
        self.robot_state_pos_right = np.array([0.1, 0.8, 0.0, 1.6, 0.0, 0.35, -0.55])
        self.robot_state_pos_left = np.array([0.1, 0.8, 0.0, 1.6, 0.0, 0.35, -0.55])

        self.wrist_right_pub = self.node_cpp.CreatePublisher("/right_arm/manip_t/sensor/camera/color", link.SImage)
        self.wrist_left_pub = self.node_cpp.CreatePublisher("/left_arm/manip_t/sensor/camera/color", link.SImage)
        self.front_pub = self.node_cpp.CreatePublisher("/embodied/head/manip_t/sensor/camera/color", link.SImage)

        self.right_joint_pub = self.node_cpp.CreatePublisher("/right_arm/manip_t/controller/joint/servo", ServoJoint)
        self.right_effector_pub = self.node_cpp.CreatePublisher("/right_arm/manip_t/controller/gripper/servo", ServoEffector)

        self.left_joint_pub = self.node_cpp.CreatePublisher("/left_arm/manip_t/controller/joint/servo", ServoJoint)
        self.left_effector_pub = self.node_cpp.CreatePublisher("/left_arm/manip_t/controller/gripper/servo", ServoEffector)

        self.state_pub_right = self.node_cpp.CreatePublisher("/right_arm/manip_t/controller/robot/state", RobotState)
        self.state_pub_left = self.node_cpp.CreatePublisher("/left_arm/manip_t/controller/robot/state", RobotState)

    def spin_loop(self):
        idx = 0
        # req_right = ServoJoint()
        # req_left = ServoJoint()
        # req_gripper_right = ServoEffector()
        # req_gripper_left = ServoEffector()
        # joints_right = Joints()
        # joints_left = Joints()
        # self.initial_pos_right = self.initial_pos_right.tolist()
        # self.initial_pos_left = self.initial_pos_left.tolist()
        # self.gripper_command = 1.000

        # joints_right.position.extend(self.initial_pos_right)
        # req_right.target.CopyFrom(joints_right)

        # joints_left.position.extend(self.initial_pos_left)
        # req_left.target.CopyFrom(joints_left)

        # effort = EffectorCommand()
        # adv_command = AdvancedCommand()

        # adv_command.ratio = self.gripper_command
        # adv_command.speed = 0.5
        # adv_command.force = 0.5
        # effort.adv.CopyFrom(adv_command)

        # req_gripper_right.command.CopyFrom(effort)
        # req_gripper_left.command.CopyFrom(effort)
        # # req.speed = accelerations or []

        # effort_status = EffectorStatus()
        # effort_status.gripper.motion.ratio = self.gripper_command

        # # 确保首帧: 先下发一次关节/夹爪初始位姿
        # self.right_joint_pub.Publish(req_right)
        # self.right_effector_pub.Publish(req_gripper_right)
        # self.left_joint_pub.Publish(req_left)
        # self.left_effector_pub.Publish(req_gripper_left)

        _require_img(front_img, "front")
        _require_img(wrist_right_img, "wrist_right")
        _require_img(wrist_left_img, "wrist_left")

        front_rgb = _bgr_to_rgb(front_img)
        wrist_right_rgb = _bgr_to_rgb(wrist_right_img)
        wrist_left_rgb = _bgr_to_rgb(wrist_left_img)

        while True:
            idx += 1
            # self.joints_list = np.random.uniform(-1.0, 1.0, 7).tolist()

            self.gripper_command = 1.000

            state_right = RobotState()
            sec, nsec = now_ros_stamp()
            state_right.header.timestamp.seconds = int(sec)
            state_right.header.timestamp.nanos = int(nsec)
            state_right.joints.position.extend(self.robot_state_pos_right)
            state_right.effector.gripper.motion.ratio = self.gripper_command

            state_left = RobotState()
            state_left.header.timestamp.seconds = int(sec)
            state_left.header.timestamp.nanos = int(nsec)
            state_left.joints.position.extend(self.robot_state_pos_left)
            state_left.effector.gripper.motion.ratio = self.gripper_command

            front_img_atlas = atlas.utils.numpy_to_image(front_rgb, link.ImageFormat.RGB8)
            wrist_right_img_atlas = atlas.utils.numpy_to_image(wrist_right_rgb, link.ImageFormat.RGB8)
            wrist_left_img_atlas = atlas.utils.numpy_to_image(wrist_left_rgb, link.ImageFormat.RGB8)

            front_img_atlas.header.frame_id = f"front_{idx}"
            front_img_atlas.header.stamp.sec = int(sec)
            front_img_atlas.header.stamp.nanosec = int(nsec)

            wrist_right_img_atlas.header.frame_id = f"wrist_right_{idx}"
            wrist_right_img_atlas.header.stamp.sec = int(sec)
            wrist_right_img_atlas.header.stamp.nanosec = int(nsec)

            wrist_left_img_atlas.header.frame_id = f"wrist_left_{idx}"
            wrist_left_img_atlas.header.stamp.sec = int(sec)
            wrist_left_img_atlas.header.stamp.nanosec = int(nsec)

            print(f"【front】发布图像数据...{front_img_atlas.header.stamp.sec}.{front_img_atlas.header.stamp.nanosec}")
            print(f"【wrist_right】发布图像数据...{wrist_right_img_atlas.header.stamp.sec}.{wrist_right_img_atlas.header.stamp.nanosec}")
            print(f"【wrist_left】发布图像数据...{wrist_left_img_atlas.header.stamp.sec}.{wrist_left_img_atlas.header.stamp.nanosec}")
            print("关节位置:", self.robot_state_pos_right, self.robot_state_pos_left)
            print("夹爪指令:", self.gripper_command)

            # 同步发布 robot/state（datacenter 会订阅并使用 header.timestamp）
            self.state_pub_right.Publish(state_right)
            self.state_pub_left.Publish(state_left)

            self.wrist_right_pub.Publish(wrist_right_img_atlas)
            self.wrist_left_pub.Publish(wrist_left_img_atlas)
            self.front_pub.Publish(front_img_atlas)

            print(f"发布第{idx}帧数据...")
            time.sleep(0.033)


def main():
    sub_node = FirstImgPubNode("sub_test")
    sub_node.spin_loop()


if __name__ == "__main__":
    main()
    # pub_node = FirstImgPubNode("joints")
    # pub_node.spin_loop()