import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py


# def load_hdf5(dataset_path):
#     if not os.path.isfile(dataset_path):
#         print(f"Dataset does not exist at \n{dataset_path}\n")
#         exit()

#     with h5py.File(dataset_path, "r") as root:
#         left_gripper, left_arm = (
#             root["/joint_action/left_gripper"][()],
#             root["/joint_action/left_arm"][()],
#         )
#         right_gripper, right_arm = (
#             root["/joint_action/right_gripper"][()],
#             root["/joint_action/right_arm"][()],
#         )
#         vector = root["/joint_action/vector"][()]
#         # pointcloud = root["/pointcloud"][()]
#         image_dict = dict()
#         # for cam_name in root[f"/observation/"].keys():
#         #     image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

#         # 针对第三人称视角图像的修改
#         if "/third_view_rgb" in root:
#             third_view_rgb = root["/third_view_rgb"][()]  # (65,) |S14705
#             image_dict["third_view"] = third_view_rgb
#         else:
#             image_dict["third_view"] = np.array([])

#     # return left_gripper, left_arm, right_gripper, right_arm, vector, pointcloud, image_dict
#     return left_gripper, left_arm, right_gripper, right_arm, vector, image_dict

# def main():
#     parser = argparse.ArgumentParser(description="Process some episodes.")
#     parser.add_argument(
#         "task_name",
#         type=str,
#         help="The name of the task (e.g., beat_block_hammer)",
#     )
#     parser.add_argument("task_config", type=str)
#     parser.add_argument(
#         "expert_data_num",
#         type=int,
#         help="Number of episodes to process (e.g., 50)",
#     )
#     args = parser.parse_args()

#     task_name = args.task_name
#     task_config = args.task_config
#     num = args.expert_data_num

#     load_dir = "../../data/" + str(task_name)

#     total_count = 0

#     save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)

#     current_ep = 0

#     zarr_root = zarr.group(save_dir)
#     zarr_data = zarr_root.create_group("data")
#     zarr_meta = zarr_root.create_group("meta")

#     point_cloud_arrays = []
#     head_camera_arrays = []
#     episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = (
#         [],
#         [],
#         [],
#         [],
#     )

#     while current_ep < num:
#         print(f"processing episode: {current_ep + 1} / {num}", end="\r")

#         load_path = os.path.join(load_dir, f"episode_{current_ep:06d}.hdf5")
#         (
#             left_gripper_all,
#             left_arm_all,
#             right_gripper_all,
#             right_arm_all,
#             vector_all,
#             # pointcloud_all,
#             image_dict_all,
#         ) = load_hdf5(load_path)

#         for j in range(0, left_gripper_all.shape[0]):

#             # pointcloud = pointcloud_all[j]
#             joint_state = vector_all[j]

#             # head_img_bit = image_dict_all["head_camera"][j]

#             third_view_bit = image_dict_all["third_view"][j]


#             if j != left_gripper_all.shape[0] - 1:
#                 # point_cloud_arrays.append(pointcloud)
#                 state_arrays.append(joint_state)
#                 # head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)

#                 head_img = cv2.imdecode(np.frombuffer(third_view_bit, np.uint8), cv2.IMREAD_COLOR)

#                 head_camera_arrays.append(head_img)
#             if j != 0:
#                 joint_action_arrays.append(joint_state)

#         current_ep += 1
#         total_count += left_gripper_all.shape[0] - 1
#         episode_ends_arrays.append(total_count)

#     print()
#     try:
#         episode_ends_arrays = np.array(episode_ends_arrays)
#         state_arrays = np.array(state_arrays)
#         # point_cloud_arrays = np.array(point_cloud_arrays)
#         joint_action_arrays = np.array(joint_action_arrays)
#         head_camera_arrays = np.array(head_camera_arrays)
#         head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
    
#         compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
#         state_chunk_size = (100, state_arrays.shape[1])
#         joint_chunk_size = (100, joint_action_arrays.shape[1])
#         # point_cloud_chunk_size = (100, point_cloud_arrays.shape[1])
#         head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
        
#         zarr_data.create_dataset(
#             "head_camera",
#             data=head_camera_arrays,
#             chunks=head_camera_chunk_size,
#             overwrite=True,
#             compressor=compressor,
#         )
#         # zarr_data.create_dataset(
#         #     "point_cloud",
#         #     data=point_cloud_arrays,
#         #     chunks=point_cloud_chunk_size,
#         #     overwrite=True,
#         #     compressor=compressor,
#         # )
#         zarr_data.create_dataset(
#             "state",
#             data=state_arrays,
#             chunks=state_chunk_size,
#             dtype="float32",
#             overwrite=True,
#             compressor=compressor,
#         )
#         zarr_data.create_dataset(
#             "action",
#             data=joint_action_arrays,
#             chunks=joint_chunk_size,
#             dtype="float32",
#             overwrite=True,
#             compressor=compressor,
#         )
#         zarr_meta.create_dataset(
#             "episode_ends",
#             data=episode_ends_arrays,
#             dtype="int64",
#             overwrite=True,
#             compressor=compressor,
#         )
#     except ZeroDivisionError as e:
#         print("If you get a `ZeroDivisionError: division by zero`, check that `data/pointcloud` in the task config is set to true.")
#         raise 
#     except Exception as e:
#         print(f"An unexpected error occurred ({type(e).__name__}): {e}")
#         raise

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    # 初始化存储数据的变量
    action = None
    qpos = None
    image_dict = dict()

    # 读取hdf5文件
    with h5py.File(dataset_path, "r") as root:
        # 1. 读取action（根目录下的action数据集）
        if "action" in root:
            action = root["action"][()]  # shape=(602, 16), dtype=float32
        else:
            print("Warning: 'action' dataset not found in root!")

        # 2. 读取qpos（observations组下的qpos数据集）
        if "observations" in root and "qpos" in root["observations"]:
            qpos = root["observations/qpos"][()]  # shape=(602, 16), dtype=float32
        else:
            print("Warning: 'observations/qpos' dataset not found!")

        # 3. 读取图像信息（observations/images下的三个摄像头）
        if "observations" in root and "images" in root["observations"]:
            images_group = root["observations/images"]
            # 遍历指定的三个摄像头（也可遍历所有摄像头）
            cam_names = ["head_camera", "wrist_left_camera", "wrist_right_camera"]
            for cam_name in cam_names:
                if cam_name in images_group:
                    # 读取图像数据（shape=(602,), dtype=object）
                    image_dict[cam_name] = images_group[cam_name][()]
                else:
                    print(f"Warning: '{cam_name}' not found in observations/images!")
                    image_dict[cam_name] = None
        else:
            print("Warning: 'observations/images' group not found!")

    # 返回读取到的核心数据
    return action, qpos, image_dict

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = args.expert_data_num

    load_dir = "../../data/" + str(task_name)

    total_count = 0

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays = []
    episode_ends_arrays, state_arrays, joint_action_arrays = [], [], []

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"episode_{current_ep:06d}.hdf5")
        action_all, qpos_all, image_dict_all = load_hdf5(load_path)

        # 检查核心数据是否存在，避免后续报错
        if action_all is None or qpos_all is None:
            print(f"\nWarning: Episode {current_ep} missing action/qpos data, skipping!")
            current_ep += 1
            continue
        
        # 获取当前episode的总帧数
        frame_num = action_all.shape[0]

        for j in range(frame_num):
            # 2. 映射数据：qpos对应state，action对应joint_action
            joint_state = qpos_all[j]
            joint_action = action_all[j]

            # 3. 修正摄像头名称
            head_img_bit = image_dict_all.get("head_camera", [None]*frame_num)[j]

            if j != frame_num - 1:
                # 存储状态（qpos）
                state_arrays.append(joint_state)
                
                # 解码头部摄像头图像（增加异常处理）
                if head_img_bit is not None:
                    try:
                        head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                        head_camera_arrays.append(head_img)
                    except Exception as e:
                        print(f"\nWarning: Episode {current_ep} frame {j} decode failed: {e}")
                        # 填充默认空图像（避免数组长度不一致）
                        head_camera_arrays.append(np.zeros((480, 640, 3), dtype=np.uint8))
                else:
                    head_camera_arrays.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            # 非第一帧存储动作（action）
            if j != 0:
                joint_action_arrays.append(joint_action)

        current_ep += 1
        total_count += frame_num - 1  # 有效帧数=总帧数-1
        episode_ends_arrays.append(total_count)

    print()
    try:
        # 转换为numpy数组（处理空数组边界情况）
        episode_ends_arrays = np.array(episode_ends_arrays, dtype="int64")
        state_arrays = np.array(state_arrays, dtype="float32") if state_arrays else np.array([], dtype="float32")
        joint_action_arrays = np.array(joint_action_arrays, dtype="float32") if joint_action_arrays else np.array([], dtype="float32")
        head_camera_arrays = np.array(head_camera_arrays, dtype=np.uint8) if head_camera_arrays else np.array([], dtype=np.uint8)
        
        # 图像格式转换：NHWC → NCHW（保持原逻辑）
        if head_camera_arrays.size > 0:
            head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)
        
        # 配置压缩器
        compressor = Blosc(cname="zstd", clevel=3, shuffle=1)
        
        # 定义分块大小（兼容空数组）
        state_chunk_size = (100, state_arrays.shape[1]) if state_arrays.ndim > 1 else (100,)
        joint_chunk_size = (100, joint_action_arrays.shape[1]) if joint_action_arrays.ndim > 1 else (100,)
        head_camera_chunk_size = (100, *head_camera_arrays.shape[1:]) if head_camera_arrays.ndim > 1 else (100,)
        
        # 写入Zarr数据集
        # 1. 头部摄像头图像
        zarr_data.create_dataset(
            "head_camera",
            data=head_camera_arrays,
            chunks=head_camera_chunk_size,
            overwrite=True,
            compressor=compressor,
            dtype=np.uint8
        )
        
        # 2. 状态（qpos）
        zarr_data.create_dataset(
            "state",
            data=state_arrays,
            chunks=state_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        
        # 3. 动作（action）
        zarr_data.create_dataset(
            "action",
            data=joint_action_arrays,
            chunks=joint_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        
        # 4. 元数据：episode结束位置
        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_arrays,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
        
        print(f"Success! Zarr data saved to {save_dir}")
        
    except ZeroDivisionError as e:
        print("If you get a `ZeroDivisionError: division by zero`, check that `data/pointcloud` in the task config is set to true.")
        raise 
    except Exception as e:
        print(f"An unexpected error occurred ({type(e).__name__}): {e}")
        raise




if __name__ == "__main__":
    main()
