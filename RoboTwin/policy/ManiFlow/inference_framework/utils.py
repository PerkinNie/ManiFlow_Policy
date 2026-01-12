import json
import socket
import pickle
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict


logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("inference_utils")


@dataclass
class ObservationRequest:
    observation: Dict[str, Any]
    task_name: str


@dataclass
class ActionResponse:
    action: np.ndarray
    success: bool
    message: str = ""


def encode_image_array(img: np.ndarray) -> List:
    if img is None:
        return []
    return img.astype(np.uint8).tolist()


def decode_image_array(img_list: List) -> np.ndarray:
    if not img_list:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return np.array(img_list, dtype=np.uint8)


def serialize_observation(observation: Dict[str, Any], task_name: str = "dual_arm_pick_box") -> bytes:
    request = ObservationRequest(observation=observation, task_name=task_name)
    return pickle.dumps(request)


def deserialize_observation(data: bytes) -> ObservationRequest:
    return pickle.loads(data)


def serialize_action(action: np.ndarray, success: bool = True, message: str = "") -> bytes:
    response = ActionResponse(action=action, success=success, message=message)
    return pickle.dumps(response)


def deserialize_action(data: bytes) -> ActionResponse:
    return pickle.loads(data)


def send_data(sock: socket.socket, data: bytes) -> bool:
    try:
        length = len(data)
        sock.sendall(length.to_bytes(8, byteorder='big'))
        sock.sendall(data)
        return True
    except Exception as e:
        logger.error(f"发送数据失败: {e}")
        return False


def receive_data(sock: socket.socket, timeout: Optional[float] = None) -> Optional[bytes]:
    try:
        if timeout:
            sock.settimeout(timeout)
        
        length_bytes = sock.recv(8)
        if len(length_bytes) < 8:
            logger.error("接收数据长度失败")
            return None
        
        length = int.from_bytes(length_bytes, byteorder='big')
        
        data = b''
        while len(data) < length:
            chunk = sock.recv(min(4096, length - len(data)))
            if not chunk:
                logger.error("接收数据连接中断")
                return None
            data += chunk
        
        sock.settimeout(None)
        return data
    except socket.timeout:
        logger.error(f"接收数据超时 ({timeout}s)")
        return None
    except Exception as e:
        logger.error(f"接收数据失败: {e}")
        return None


def create_server_socket(host: str, port: int, max_clients: int = 1) -> Optional[socket.socket]:
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(max_clients)
        logger.info(f"服务端已启动: {host}:{port}")
        return server_socket
    except Exception as e:
        logger.error(f"创建服务端socket失败: {e}")
        return None


def create_client_socket(host: str, port: int, timeout: int = 30) -> Optional[socket.socket]:
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(timeout)
        client_socket.connect((host, port))
        logger.info(f"客户端已连接到服务端: {host}:{port}")
        return client_socket
    except Exception as e:
        logger.error(f"创建客户端socket失败: {e}")
        return None


def close_socket(sock: socket.socket):
    try:
        sock.close()
        logger.info("Socket已关闭")
    except Exception as e:
        logger.error(f"关闭socket失败: {e}")
