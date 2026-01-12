# ManiFlow推理框架实现计划

## 目标
创建一个独立于RoboTwin2.0的ManiFlow推理框架，包含服务端和客户端。

## 文件结构
```
inference_framework/
├── server.py              # 推理服务端
├── client.py              # 推理客户端
├── config.py              # 配置参数管理
└── utils.py               # 工具函数
```

## 核心功能模块

### 1. 服务端 (server.py)
- **模型加载**：参考 `deploy_policy.py` 的 `get_model()` 函数
  - 加载配置文件 `maniflow_image_timm_policy_robotwin2.yaml`
  - 从checkpoint加载训练好的模型：`latest.ckpt`
  - 初始化ManiFlow策略和runner
  
- **推理服务**：
  - 接收客户端发送的观测数据
  - 调用模型进行推理
  - 返回预测的动作序列

- **通信接口**：
  - 使用socket或gRPC实现服务端-客户端通信
  - 支持批量观测处理

### 2. 客户端 (client.py)
- **数据采集**：参考 `datacenter.py` 的 `InteractionDataCenter`
  - 使用 `get_observation()` 获取当前状态（图像+关节位置）
  - 使用 `datacenter_obs_to_maniflow()` 转换观测格式
  
- **动作执行**：
  - 发送观测到服务端
  - 接收服务端返回的动作
  - 使用 `publish_action()` 逐步发送动作到机器人
  
- **状态管理**：
  - 维护观测缓存
  - 处理动作序列的逐步执行

### 3. 配置管理 (config.py)
- 模型路径配置
- 服务端/客户端通信参数（IP、端口）
- 观测参数（图像尺寸、关节数量等）
- 动作参数（n_obs_steps, n_action_steps）

### 4. 工具函数 (utils.py)
- 观测数据编码/解码
- 动作数据序列化
- 通信协议封装

## 实现步骤
1. 创建配置文件，定义所有需要的参数
2. 实现服务端：模型加载 + 推理服务 + 通信接口
3. 实现客户端：数据采集 + 动作执行 + 通信接口
4. 实现工具函数：数据转换和通信协议
5. 测试服务端-客户端通信
6. 测试完整的推理流程