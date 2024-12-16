import json
import os
import threading
import time

import pandas as pd

# Directories for logs, config, and status
LOG_DIR = "./logs"
CONFIG_DIR = "./config"
STATUS_DIR = "./status"
DEFAULT_UPDATE_INTERVAL = 10
configs = {}
clients = {}


class LatestDataProvider:
    def __init__(self, status_dir, update_interval=10):
        """
        初始化 LatestDataProvider 类，并设置后台更新线程
        :param status_dir: 状态文件所在的目录路径
        :param update_interval: 后台更新的时间间隔（秒）
        """
        self.status_dir = status_dir
        self.update_interval = update_interval
        self.latest_status = {}  # 用于存储所有服务器的最新状态
        self.lock = threading.Lock()  # 用于线程安全
        self._start_background_update()

    def _start_background_update(self):
        """启动后台线程，每隔 `update_interval` 秒更新一次数据"""
        thread = threading.Thread(target=self._background_update, daemon=True)
        thread.start()

    def _background_update(self):
        """后台更新数据"""
        while True:
            for server_file in os.listdir(self.status_dir):
                if server_file.endswith(".jsonl"):
                    server_name = server_file.replace(".jsonl", "")
                    status = self.load_last_status(server_name)
                    with self.lock:  # 确保线程安全
                        self.latest_status[server_name] = self.process_status(status)
            time.sleep(self.update_interval)

    def load_last_status(self, server_name):
        """
        从指定服务器的状态文件中读取最后一条记录
        :param server_name: 服务器名称
        :return: 最后一条状态记录的字典
        """
        status_file = os.path.join(self.status_dir, f"{server_name}.jsonl")
        if not os.path.exists(status_file):
            return {}

        with open(status_file, 'rb') as f:
            # 从文件末尾读取最后一行
            f.seek(0, os.SEEK_END)  # 移动到文件末尾
            position = f.tell()  # 获取当前位置
            while position >= 0:
                f.seek(position)
                position -= 1
                if f.read(1) == b'\n':  # 找到换行符
                    break
            last_line = f.readline().decode().strip()

        return json.loads(last_line)

    def process_status(self, raw_status):
        """
        对原始状态数据进行处理，转换为更直观的格式
        :param raw_status: 原始状态数据字典
        :return: 处理后的状态数据字典
        """
        processed_status = raw_status.copy()  # 对于不需要处理的数据，保持原样

        # 显式处理需要格式改变的字段
        if "gpu_usage_per_user" in raw_status:
            gpu_usage_raw = raw_status["gpu_usage_per_user"].strip()
            gpu_usage_dict = {}
            for line in gpu_usage_raw.split("\n"):
                if "Error" in line:
                    break
                if " " in line:
                    user, usage = line.split(" ", 1)
                    gpu_usage_dict[user] = usage
            processed_status["gpu_usage_per_user"] = gpu_usage_dict

        if "gpu_using_users" in raw_status:
            processed_status["gpu_using_users"] = raw_status["gpu_using_users"].strip().split("\n")
        if "gpu_load" in raw_status:
            processed_status["gpu_load"] = raw_status["gpu_load"].strip().split(", ")
            processed_status["gpu_load"]=processed_status["gpu_load"] if len(processed_status["gpu_load"])>1 else None
        if "gpu_mem" in raw_status:
            processed_status["gpu_mem"] = raw_status["gpu_mem"].strip().split(", ")
            processed_status["gpu_mem"]=processed_status["gpu_mem"] if len(processed_status["gpu_mem"])>1 else None

            # 检查，如果任意一个字段出现Error，则返回空
            for key in processed_status:
                if isinstance(processed_status[key], str) and "Error" in processed_status[key]:
                    return {}


        return processed_status

    def get_server_status(self, server_name):
        """
        提供对外接口，获取服务器的最新状态数据
        :param server_name: 服务器名称
        :return: 处理后的状态数据字典
        """
        # 若传入的是一个dict，则取出其中的server_name
        if isinstance(server_name, dict):
            server_name = server_name.get("name", "")
        with self.lock:  # 确保线程安全
            return self.latest_status.get(server_name, self.load_last_status(server_name))


class FullDataProvider:
    def __init__(self, status_dir, latestDataProvider, update_interval=60):
        """
        初始化 FullDataProvider 类
        :param status_dir: 状态文件所在的目录路径
        :param update_interval: 后台更新线程的时间间隔（秒）
        """
        self.status_dir = status_dir
        self.update_interval = update_interval
        self.latestDataProvider = latestDataProvider
        self.server_data = {}  # 缓存每台服务器的完整数据，键为服务器名称，值为 DataFrame
        self.lock = threading.Lock()  # 用于线程安全
        self._start_background_update()

    def _start_background_update(self):
        """启动后台线程，定期更新所有服务器的数据"""
        thread = threading.Thread(target=self._background_update, daemon=True)
        thread.start()

    def _background_update(self):
        """后台更新线程，定期从硬盘读取所有服务器的数据"""
        while True:
            for server_file in os.listdir(self.status_dir):
                if server_file.endswith(".jsonl"):
                    server_name = server_file.replace(".jsonl", "")
                    df = self._load_all_status(server_name)
                    with self.lock:
                        self.server_data[server_name] = df
            time.sleep(self.update_interval)

    def _load_all_status(self, server_name):
        """
        从指定服务器的状态文件中读取所有记录，并返回一个 DataFrame
        :param server_name: 服务器名称
        :return: 包含所有记录的 Pandas DataFrame
        """
        status_file = os.path.join(self.status_dir, f"{server_name}.jsonl")
        if not os.path.exists(status_file):
            return pd.DataFrame()  # 返回空 DataFrame

        records = []
        with open(status_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue  # 跳过格式错误的行

        df = pd.DataFrame(records)
        df = df.apply(lambda row: pd.Series(self.latestDataProvider.process_status(row.to_dict())), axis=1)

        return df

    def get_server_data(self, server_name):
        """
        提供对外接口，获取指定服务器的完整数据记录
        :param server_name: 服务器名称
        :return: 缓存中的 Pandas DataFrame
        """
        with self.lock:
            return self.server_data.get(server_name, pd.DataFrame())

    def get_all_servers_data(self):
        """
        提供对外接口，获取所有服务器的完整数据记录
        :return: 包含所有服务器记录的 Pandas DataFrame，添加服务器名称字段
        """
        with self.lock:
            all_records = []
            for server_name, df in self.server_data.items():
                if not df.empty:
                    df["server_name"] = server_name
                    all_records.append(df)
            if all_records:
                return pd.concat(all_records, ignore_index=True)
            return pd.DataFrame()  # 返回空 DataFrame


last = LatestDataProvider(STATUS_DIR, 10)

all = FullDataProvider(STATUS_DIR, last, 60)
