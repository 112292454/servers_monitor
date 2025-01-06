import json
import os
import threading
import time

import pandas as pd

# Directories for logs, config, and status
LOG_DIR = "./logs"
CONFIG_DIR = "./config"
STATUS_DIR = "./status"
DEFAULT_UPDATE_INTERVAL = 60
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
        #如果status_file结尾有空行，会导致读取最后一行时出错，所以先去掉空行，要高效操作，所以用二进制读写
        with open(status_file, 'rb+') as f:
            f.seek(-1, os.SEEK_END)
            while f.read(1) == b'\n':
                f.seek(-2, os.SEEK_CUR)
            # f.seek(1, os.SEEK_CUR)
            f.truncate()


        with open(status_file, 'rb') as f:
            # 从文件末尾读取最后一行
            f.seek(0, os.SEEK_END)  # 移动到文件末尾
            position = f.tell()  # 获取当前位置
            # start=position
            while position >= 0:
                f.seek(position)
                position -= 1
                if f.read(1) == b'\n':  # 找到换行符
                    break
            last_line = f.readline().decode().strip()
        try:
            res=json.loads(last_line)
            return res
        except json.JSONDecodeError:
            print(f"解析json文件失败：{last_line}")
        return {}

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
            processed_status["gpu_load"] = processed_status["gpu_load"] if len(
                processed_status["gpu_load"]) > 1 else None
        if "gpu_mem" in raw_status:
            processed_status["gpu_mem"] = raw_status["gpu_mem"].strip().split(", ")
            processed_status["gpu_mem"] = processed_status["gpu_mem"] if len(processed_status["gpu_mem"]) > 1 else None

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


class UserDataProvider:
    def __init__(self, config_dir):
        """
        初始化 UserDataProvider 类
        :param config_dir: 配置文件所在的目录路径
        """
        self.config_dir = config_dir
        self.users = {}  # 用于存储所有用户的配置信息
        self.lock = threading.Lock()  # 用于线程安全
        self._load_all_users()

    def _load_all_users(self):
        """加载所有用户的配置信息"""
        # 从username_mapper.csv加载，共两列，第一列的用户名，第二列是用户的真名,用df形式
        user_mapper = pd.read_csv(os.path.join(self.config_dir, f"username_mapper.csv"))
        # 以username为key，其他所有属性在一起作为一个dict，作为value
        self.users = user_mapper.set_index('username').apply(lambda x: x.to_dict(), axis=1).to_dict()

    def get_user_realname(self, user):
        """
        提供对外接口，获取指定用户的配置信息
        :param user_name: 用户名称
        :return: 用户的配置信息字典
        """
        with self.lock:
            return self.users.get(user, {})

    def get_user_by_realname(self, realname):
        with self.lock:
            for k, v in self.users.items():
                if v['realname'] == realname:
                    return k
            return None

    def get_user_name_mapper(self):
        """只返回用户名和真名的映射关系，去除其他属性"""
        with self.lock:
            return {k: v['realname'] for k, v in self.users.items()}


class FullDataProvider:
    def __init__(self, status_dir, latest_data_provider, user_data_provider, update_interval=60):
        """
        初始化 FullDataProvider 类
        :param status_dir: 状态文件所在的目录路径
        :param update_interval: 后台更新线程的时间间隔（秒）
        """
        self.status_dir = status_dir
        self.update_interval = update_interval
        self.last_data_provider = latest_data_provider
        self.user_data_provider = user_data_provider
        self.server_data = {}  # 缓存每台服务器的完整数据，键为服务器名称，值为 DataFrame
        self._start_background_update()

    def _start_background_update(self):
        """启动后台线程，定期更新所有服务器的数据"""
        thread = threading.Thread(target=self._background_update, daemon=True)
        thread.start()

    def _background_update(self):
        """后台更新线程，定期从硬盘读取所有服务器的数据"""
        while True:
            st=time.time()
            for server_file in os.listdir(self.status_dir):
                if server_file.endswith(".jsonl"):
                    server_name = server_file.replace(".jsonl", "")
                    df = self._load_all_status(server_name)
                    self.server_data[server_name] = df
            et=time.time()
            print(f"\nload full data used: {et-st} s\n")
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
                if not line.strip():
                    continue
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue  # 跳过格式错误的行

        df = pd.DataFrame(records)
        df = df.apply(lambda row: pd.Series(self.last_data_provider.process_status(row.to_dict())), axis=1)
        # 检查所有行，对于is_ssh_up为空，或nan，或false的行，删除
        df = df.dropna(subset=['is_ssh_up'])
        #从config解析保存期限，只保留期限内的数据行，如果没有则默认为7天，形式如“7d”,"30d","60d"
        # save_period = self.user_data_provider.users.get(server_name, {}).get("save_period", "7d")
        save_period= "7d"
        save_period = int(save_period[:-1]) if save_period[-1] == 'd' else 60
        df = df[df['timestamp_ms']/1000 >= time.time() - save_period * 24 * 3600]
        def fill_gpu_usage_and_replace_usernames(gpu_usage_dict, user_name_mapper):
            # 创建一个新的字典来保存结果，键为真名
            filled_gpu_usage = {}

            # 对每个用户填充
            for user in set(user_name_mapper.keys()):
                true_name = user_name_mapper.get(user, user)  # 获取真名，若无则默认使用用户名
                filled_gpu_usage[true_name] = gpu_usage_dict.get(user, 0)  # 填充显存，若用户没有记录则为0

            return filled_gpu_usage

        def get_top_users_gpu_usage(df, user_name_mapper):
            # Step 1: 聚合每个用户的显存总占用量
            user_gpu_usage = {}

            for gpu_usage_dict in df['gpu_usage_per_user']:
                for user, usage in gpu_usage_dict.items():
                    usage_value = int(usage.replace('MB', '').strip()) if isinstance(usage, str) else 0
                    true_name = user_name_mapper.get(user, user)
                    if true_name in user_gpu_usage:
                        user_gpu_usage[true_name] += usage_value
                    else:
                        user_gpu_usage[true_name] = usage_value

            # Step 2: 排序并保留前十个用户
            top_users = sorted(user_gpu_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            top_users = [user for user in top_users if user[1] > 0]
            top_users_set = set([user for user, _ in top_users])

            # Step 3: 更新每行的 gpu_usage_per_user，只保留前十个用户
            def filter_top_users(gpu_usage_dict):
                return {user: usage for user, usage in gpu_usage_dict.items() if user in top_users_set}

            df['gpu_usage_per_user'] = df['gpu_usage_per_user'].apply(filter_top_users)

            return df

        user_name_mapper = self.user_data_provider.get_user_name_mapper()
        df['gpu_usage_per_user'] = df['gpu_usage_per_user'].fillna({})

        df['gpu_usage_per_user'] = df['gpu_usage_per_user'].apply(
            lambda x: fill_gpu_usage_and_replace_usernames(x, user_name_mapper)
        )
        get_top_users_gpu_usage(df, user_name_mapper)

        return df

    def get_server_data(self, server_name):
        """
        提供对外接口，获取指定服务器的完整数据记录
        :param server_name: 服务器名称
        :return: 缓存中的 Pandas DataFrame
        """
        if isinstance(server_name, dict):
            server_name = server_name.get("name", "")
        return self.server_data.get(server_name, pd.DataFrame())

    def get_all_servers_data(self):
        """
        提供对外接口，获取所有服务器的完整数据记录
        :return: 包含所有服务器记录的 Pandas DataFrame，添加服务器名称字段
        """
        all_records = []
        for server_name, df in self.server_data.items():
            if not df.empty:
                df["server_name"] = server_name
                all_records.append(df)
        if all_records:
            return pd.concat(all_records, ignore_index=True)
        return pd.DataFrame()  # 返回空 DataFrame


user = UserDataProvider(CONFIG_DIR)

last = LatestDataProvider(STATUS_DIR, 10)

all = FullDataProvider(STATUS_DIR, last, user, 60)
