import time
from abc import ABC

import atlas
import cv2
import link
import numpy as np
from link._link import Node
from manip_shared_msg.base.effector_pb2 import EffectorCommand, EffectorStatus
from manip_shared_msg.base.gripper_pb2 import AdvancedCommand
from manip_shared_msg.base.joint_pb2 import Joints
from manip_shared_msg.locomotion.servo_effector_pb2 import ServoEffector
from manip_shared_msg.locomotion.servo_joint_pb2 import ServoJoint

front_img = cv2.imread("assets/images/front-color.png")
wrist_right_img = cv2.imread("assets/images/wrist_right-color.png")
wrist_left_img = cv2.imread("assets/images/wrist_left-color.png")


class BaseNode(ABC):
    node_cpp: Node = None

    def _init_node(self, node_name):
        link.Node.Initialize(node_name)
        self.node_cpp = link.GetNode()


class FirstImgPubNode(BaseNode):
    def __init__(self, node_name):
        self._init_node(node_name)

        # R2V2 初始位置
        self.initial_pos = np.array([0.1, 0.8, 0.0, 1.6, 0.0, 0.35, -0.55])

        self.wrist_right_pub = self.node_cpp.CreatePublisher("/right_arm/manip_t/sensor/camera/color", link.SImage)
        self.wrist_left_pub = self.node_cpp.CreatePublisher("/left_arm/manip_t/sensor/camera/color", link.SImage)
        self.front_pub = self.node_cpp.CreatePublisher("/embodied/head/manip_t/sensor/camera/color", link.SImage)

        self.right_joint_pub = self.node_cpp.CreatePublisher("/right_arm/manip_t/controller/joint/servo", ServoJoint)
        self.right_effector_pub = self.node_cpp.CreatePublisher("/right_arm/manip_t/controller/gripper/servo", ServoEffector)

        self.left_joint_pub = self.node_cpp.CreatePublisher("/left_arm/manip_t/controller/joint/servo", ServoJoint)
        self.left_effector_pub = self.node_cpp.CreatePublisher("/left_arm/manip_t/controller/gripper/servo", ServoEffector)

    def spin_loop(self):
        idx = 0

        while True:
            idx += 1
            self.joints_list = np.random.uniform(-1.0, 1.0, 7).tolist()
            self.gripper_command = 0.0

            front_img_atlas = atlas.utils.numpy_to_image(front_img, link.ImageFormat.RGB8)
            wrist_right_img_atlas = atlas.utils.numpy_to_image(wrist_right_img, link.ImageFormat.RGB8)
            wrist_left_img_atlas = atlas.utils.numpy_to_image(wrist_left_img, link.ImageFormat.RGB8)

            front_img_atlas.header.frame_id = f"front_{idx}"
            front_img_atlas.header.stamp.sec = int(time.time())
            front_img_atlas.header.stamp.nanosec = int((time.time() % 1) * 1e9)

            wrist_right_img_atlas.header.frame_id = f"wrist_right_{idx}"
            wrist_right_img_atlas.header.stamp.sec = int(time.time())
            wrist_right_img_atlas.header.stamp.nanosec = int((time.time() % 1) * 1e9)

            wrist_left_img_atlas.header.frame_id = f"wrist_left_{idx}"
            wrist_left_img_atlas.header.stamp.sec = int(time.time())
            wrist_left_img_atlas.header.stamp.nanosec = int((time.time() % 1) * 1e9)

            print(f"【front】发布图像数据...{front_img_atlas.header.stamp.sec}.{front_img_atlas.header.stamp.nanosec}")
            print(f"【wrist_right】发布图像数据...{wrist_right_img_atlas.header.stamp.sec}.{wrist_right_img_atlas.header.stamp.nanosec}")
            print(f"【wrist_left】发布图像数据...{wrist_left_img_atlas.header.stamp.sec}.{wrist_left_img_atlas.header.stamp.nanosec}")

            self.wrist_right_pub.Publish(wrist_right_img_atlas)
            self.wrist_left_pub.Publish(wrist_left_img_atlas)
            self.front_pub.Publish(front_img_atlas)

            req = ServoJoint()
            req_gripper = ServoEffector()
            joints = Joints()
            joints.position.extend(self.joints_list)
            req.target.CopyFrom(joints)

            effort = EffectorCommand()
            adv_command = AdvancedCommand()

            adv_command.ratio = self.gripper_command
            adv_command.speed = 0.5
            adv_command.force = 0.5
            effort.adv.CopyFrom(adv_command)

            req_gripper.command.CopyFrom(effort)
            # req.speed = accelerations or []

            effort_status = EffectorStatus()
            effort_status.gripper.motion.ratio = self.gripper_command

            # self.right_joint_pub.Publish(req)
            # self.right_effector_pub.Publish(req_gripper)
            # self.left_joint_pub.Publish(req)
            # self.left_effector_pub.Publish(req_gripper)

            print(f"发布第{idx}帧数据...")
            time.sleep(0.033)


def main():
    sub_node = FirstImgPubNode("sub_test")
    sub_node.spin_loop()


if __name__ == "__main__":
    main()
    # pub_node = FirstImgPubNode("joints")
    # pub_node.spin_loop()
