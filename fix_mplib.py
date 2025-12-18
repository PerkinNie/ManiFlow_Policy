#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动修改 mplib 0.2.1 版本的两处关键代码
"""
import os
import sys
import shutil
import mplib


def get_planner_file_path():
    """
    获取 mplib/planner.py 的绝对路径
    """
    # 获取 mplib 库的安装目录（__init__.py 路径）
    mplib_init_path = mplib.__file__
    mplib_dir = os.path.dirname(mplib_init_path)
    planner_path = os.path.join(mplib_dir, "planner.py")
    
    # 验证文件是否存在
    if not os.path.exists(planner_path):
        raise FileNotFoundError(f"未找到 planner.py 文件：{planner_path}")
    return planner_path


def backup_original_file(file_path):
    """
    备份原文件（避免修改出错无法恢复）
    """
    backup_path = f"{file_path}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"✅ 已备份原文件至：{backup_path}")
    else:
        print(f"ℹ️ 原文件备份已存在：{backup_path}")


def modify_planner_file(file_path):
    """
    核心修改逻辑：
    1. 移除 ArticulatedModel 中的 convex=True 参数
    2. 移除第848行的 or collide 逻辑
    """
    modified_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, 1):  # 行号从1开始
        
        # -------------------------- 第一处修改：移除 convex=True --------------------------
        # 匹配包含 "ArticulatedModel" 且包含 "convex=True" 的行（兼容行号偏移）
        if "ArticulatedModel" in line and "convex=True" in line:
            # 注释掉 convex=True 或直接删除该行（这里选择注释，保留痕迹）
            modified_line = line.replace("convex=True,", "# convex=True,  # 自动注释：移除该参数")
            modified_lines.append(modified_line)
            print(f"✅ 第{idx}行：已移除 convex=True 参数")
        # -------------------------- 第二处修改：移除 or collide --------------------------
        # 匹配包含 "np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit" 的行
        elif "np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit" in line:
            # 替换掉 "or collide" 部分
            modified_line = line.replace(
                "np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit",
                "np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit"
            )
            modified_lines.append(modified_line)
            print(f"✅ 第{idx}行：已移除 or collide 逻辑")
        else:
            modified_lines.append(line)

    # 写入修改后的内容
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(modified_lines)
    print(f"✅ 已完成所有修改，文件路径：{file_path}")


def verify_modification(file_path):
    """
    验证修改是否成功（可选，增加可靠性）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 检查是否还存在错误内容
    errors = []
    if "convex=True" in content and not "# convex=True" in content:
        errors.append("convex=True 未被正确注释/移除")
    if "np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit" in content:
        errors.append("or collide 未被正确移除")
    
    if errors:
        raise RuntimeError(f"❌ 修改验证失败：{'; '.join(errors)}")
    else:
        print("✅ 修改验证通过！mplib 库已可正常使用")


if __name__ == "__main__":
    try:
        # 1. 定位 planner.py 文件
        planner_path = get_planner_file_path()
        print(f"ℹ️ 找到 mplib planner.py 文件：{planner_path}")
        
        # 2. 备份原文件
        backup_original_file(planner_path)
        
        # 3. 执行修改
        modify_planner_file(planner_path)
        
        # 4. 验证修改结果
        verify_modification(planner_path)
        
    except Exception as e:
        print(f"❌ 执行失败：{str(e)}", file=sys.stderr)
        sys.exit(1)
    
    print("\n🎉 所有修改完成！mplib 0.2.1 已修复可正常使用")
