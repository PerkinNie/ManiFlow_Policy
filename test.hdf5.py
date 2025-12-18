import h5py
import numpy as np

def build_hdf5_structure_dict(h5_obj):
    """
    递归构建HDF5文件的嵌套字典结构（键名+数据类型）
    :param h5_obj: HDF5组/文件对象
    :return: 嵌套字典，叶子节点为数据类型字符串（如 'ndarray (65,240,320,3) uint8'）
    """
    structure_dict = {}
    
    # 遍历当前层级所有子项
    for name in h5_obj:
        item = h5_obj[name]
        
        # 若是组（Group）→ 递归构建子字典
        if isinstance(item, h5py.Group):
            structure_dict[name] = build_hdf5_structure_dict(item)
        
        # 若是数据集（Dataset）→ 记录数据类型+形状
        elif isinstance(item, h5py.Dataset):
            # 拼接数据类型信息：数组类型 + 形状 + 元素类型
            dtype_info = f"ndarray {item.shape} {item.dtype}"
            structure_dict[name] = dtype_info
        
        # 其他类型（如链接）→ 标注类型
        else:
            structure_dict[name] = f"unknown type: {type(item).__name__}"
    
    return structure_dict

def print_nested_dict(d, indent=0):
    """
    格式化打印嵌套字典（模仿你要的结构风格）
    :param d: 嵌套字典
    :param indent: 缩进级别
    """
    indent_str = "    " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            # 嵌套字典（组）
            print(f"{indent_str}'{key}': {{")
            print_nested_dict(value, indent + 1)
            print(f"{indent_str}}},")
        else:
            # 叶子节点（数据集/其他类型）
            print(f"{indent_str}'{key}': '{value}',")

def read_hdf5_structure_as_dict(hdf5_path):
    """
    读取HDF5文件并返回嵌套字典结构，同时格式化打印
    :param hdf5_path: HDF5文件路径
    :return: 嵌套字典
    """
    try:
        with h5py.File(hdf5_path, "r") as h5_file:
            # 构建结构字典
            struct_dict = build_hdf5_structure_dict(h5_file)
            
            # 打印标题
            print("=== HDF5文件结构（嵌套字典格式）===")
            print("data_list = {")
            print_nested_dict(struct_dict, indent=1)
            print("}")
            
            return struct_dict
    
    except FileNotFoundError:
        print(f"❌ 错误：文件 {hdf5_path} 不存在！")
        return None
    except PermissionError:
        print(f"❌ 错误：无权限读取文件 {hdf5_path}！")
        return None
    except Exception as e:
        print(f"❌ 读取失败：{str(e)}")
        return None

if __name__ == "__main__":
    # ===================== 配置项 =====================
    # 替换为你的HDF5文件路径
    HDF5_FILE_PATH = "/root/workspace/ManiFlow_Policy/RoboTwin/beat_block_hammer_data_ur5-wsg/beat_block_hammer/efort_beat_block_hammer/data/episode0.hdf5"
    # ==================================================

    # 读取并打印嵌套字典结构
    hdf5_struct = read_hdf5_structure_as_dict(HDF5_FILE_PATH)