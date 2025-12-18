#!/bin/bash
set -euo pipefail

# ===================== 基础配置 =====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # 重置颜色

# Conda环境名
CONDA_ENV="maniflow"
# RoboTwin根目录（请确保脚本在该目录下执行，或修改为绝对路径）
ROOT_DIR=$(pwd)

# ===================== 工具函数 =====================
# 信息日志
info() {
    echo -e "${GREEN}[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1${NC}"
}
# 警告日志
warn() {
    echo -e "${YELLOW}[WARN] $(date +'%Y-%m-%d %H:%M:%S') - $1${NC}"
}
# 错误日志（退出脚本）
error() {
    echo -e "${RED}[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1${NC}"
    exit 1
}
# 检查目录是否存在
check_dir() {
    if [ ! -d "$1" ]; then
        error "目录不存在：$1，请检查文件结构"
    fi
}

# 检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        error "文件不存在：$1，请检查文件结构"
    fi
}

# ===================== 核心步骤 =====================
# 1. 安装Vulkan依赖
info "===== 步骤1：安装Vulkan依赖 ====="
sudo apt update -y || error "APT源更新失败"
sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools || error "Vulkan依赖安装失败"
info "Vulkan依赖安装完成"

# 检查conda是否安装
info "===== 检查Conda环境 ====="
if ! command -v conda &> /dev/null; then
    error "未检测到conda！请先安装Anaconda/Miniconda并配置环境变量"
fi

conda init bash
exec bash

# 初始化conda（解决脚本中conda activate失效问题）
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 检查maniflow环境是否存在
if conda info --envs | grep -q "^${CONDA_ENV}\s"; then
    info "检测到${CONDA_ENV}环境已存在，直接激活"
    conda activate "${CONDA_ENV}" || error "激活${CONDA_ENV}环境失败"
    # 环境存在时，仅激活并提示完成，不执行后续步骤
    info "======================================"
    info "✅ ${CONDA_ENV}环境已激活，无需执行后续配置！"
    info "📌 当前激活的conda环境：${CONDA_ENV}"
    info "📌 Python版本：$(python --version | awk '{print $2}')"
    info "======================================"
    exit 0  # 退出脚本，不执行后续步骤
else
    info "未检测到${CONDA_ENV}环境，开始创建"
    
    # 2. 创建conda环境并安装基础依赖
    info "===== 步骤2：创建${CONDA_ENV}环境 ====="
    check_dir "${ROOT_DIR}/RoboTwin/policy/ManiFlow/scripts"
    cd "${ROOT_DIR}/RoboTwin/policy/ManiFlow/scripts" || error "进入脚本目录失败"
    
    conda create -n "${CONDA_ENV}" python=3.10 -y || error "创建conda环境失败"
    conda activate "${CONDA_ENV}" || error "激活${CONDA_ENV}环境失败"
    
    check_file "requirements.txt"
    pip install -r requirements.txt || error "安装requirements.txt依赖失败"
    info "${CONDA_ENV}环境创建及基础依赖安装完成"
fi

# 3. 安装PyTorch3D（无论环境是否新建，确保依赖完整）
info "===== 步骤3：安装PyTorch3D ====="
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation || error "PyTorch3D安装失败"
info "PyTorch3D安装完成"

# 4. 安装RoboTwin2.0基础环境
info "===== 步骤4：安装RoboTwin2.0基础环境 ====="
cd "${ROOT_DIR}/RoboTwin" || error "返回RoboTwin根目录失败"
check_file "script/_install.sh"
bash script/_install.sh || error "执行_install.sh失败"
check_file "script/_download_assets.sh"
bash script/_download_assets.sh || error "执行_download_assets.sh失败"
info "RoboTwin2.0基础环境安装完成"

# 5. 自动修改sapien/mplib/curobo代码
info "===== 步骤5：修改sapien/mplib/curobo代码 ====="
MODIFY_SCRIPT="${ROOT_DIR}/policy/ManiFlow/scripts/modify_code.sh"  # 若为绝对路径请修改此处
if [ -f "${MODIFY_SCRIPT}" ]; then
    bash "${MODIFY_SCRIPT}" || error "执行modify_code.sh失败"
    info "代码自动修改完成"
else
    warn "修改脚本不存在：${MODIFY_SCRIPT}，跳过此步骤"
fi

# 6. 安装ManiFlow包（步骤6缺失，忽略）
info "===== 步骤6：安装ManiFlow包 ====="
check_dir "${ROOT_DIR}/RoboTwin/policy/ManiFlow/ManiFlow"
cd "${ROOT_DIR}/RoboTwin/policy/ManiFlow/ManiFlow" || error "进入ManiFlow目录失败"
pip install -e . || error "安装ManiFlow包失败"
cd .. || error "返回上级目录失败"
info "ManiFlow包安装完成"

# 7. 安装第三方包
info "===== 步骤7：安装第三方包 ====="
check_dir "${ROOT_DIR}/RoboTwin/third_party"
cd "${ROOT_DIR}/RoboTwin/third_party" || error "进入third_party目录失败"

# 安装gym-0.21.0
check_dir "gym-0.21.0"
cd gym-0.21.0 || error "进入gym-0.21.0目录失败"
pip install -e . || error "安装gym-0.21.0失败"
cd .. || error "返回third_party目录失败"

# 安装Metaworld
check_dir "Metaworld"
cd Metaworld || error "进入Metaworld目录失败"
pip install -e . || error "安装Metaworld失败"
cd .. || error "返回third_party目录失败"

# 安装rrl-dependencies
check_dir "rrl-dependencies"
cd rrl-dependencies || error "进入rrl-dependencies目录失败"
check_dir "mj_envs"
pip install -e mj_envs/. || error "安装mj_envs失败"
check_dir "mjrl"
pip install -e mjrl/. || error "安装mjrl失败"
cd ../ || error "返回third_party目录失败"

# 安装r3m
rm -rf r3m || warn "删除原有r3m目录失败（非致命）"
git clone https://github.com/facebookresearch/r3m.git || error "克隆r3m仓库失败"
cd r3m || error "进入r3m目录失败"
pip install -e . || error "安装r3m失败"
cd ../.. || error "返回RoboTwin根目录失败"
info "第三方包安装完成"

# 8. 修改mplib 0.2.1代码
info "===== 步骤8：修改mplib代码 ====="
check_file "${ROOT_DIR}/RoboTwin/fix_mplib.py"
python "${ROOT_DIR}/RoboTwin/fix_mplib.py" || error "执行fix_mplib.py失败"
info "mplib代码修改完成"

# ===================== 完成提示 =====================
info "======================================"
info "✅ 所有环境搭建步骤执行完成！"
info "📌 当前激活的conda环境：${CONDA_ENV}"
info "📌 Python版本：$(python --version | awk '{print $2}')"
info "======================================"
