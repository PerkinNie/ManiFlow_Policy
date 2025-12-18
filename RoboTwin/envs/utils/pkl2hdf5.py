import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    # images_to_video(np.array(data_list["observation"]["head_camera"]["rgb"]), out_path=video_path)

    print("data_list顶层键：", list(data_list.keys()))  # 包含third_view_rgb
    print("third_view_rgb列表长度：", len(data_list["third_view_rgb"]))  # 等于pkl文件数量
    print("单帧形状：", data_list["third_view_rgb"][0].shape)  # (480,640,3)
    print("合并后数组形状：", np.array(data_list["third_view_rgb"]).shape)
    try:
        print(f"现在开始使用转换第三人称相机图像为视频")
        rgb_frames = np.array(data_list["third_view_rgb"])
        print(f"third_view_rgb 数组维度：{len(rgb_frames.shape)}，形状：{rgb_frames.shape}")      
        images_to_video(rgb_frames, out_path=video_path)
        print(f"✅ 第三人称视频生成成功：{video_path}")
    except KeyError as e:
        print(f"❌ 未找到 third_view_rgb 键：{e}")
        try:
            print(f"现在开始使用转换头部相机图像为视频")
            rgb_frames = np.array(data_list["observation"]["head_camera"]["rgb"])
            images_to_video(rgb_frames, out_path=video_path)
        except KeyError as e2:
            print(f"❌ 未找到 head_camera 键：{e2}")
            print(f"Warning: No RGB data (third_view_rgb/head_camera) found, skip video generation for {video_path}")
            return
    except Exception as e3:
        # 捕获其他异常
        print(f"❌ 视频生成失败：{e3}")
        return

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path)
