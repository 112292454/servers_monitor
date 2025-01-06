import matplotlib.pyplot as plt
import numpy as np

import data_layer as dl
from data_layer import last, all, DEFAULT_UPDATE_INTERVAL


def plot_gpu_utilization(server):
    """
    绘制单个服务器的 GPU 利用率和内存使用情况。
    :param server: 服务器名称，用于获取状态数据
    :return: Matplotlib 图表对象，用于 Gradio 的 gr.Plot 显示
    """
    # 获取 GPU 利用率和内存使用情况
    status = last.get_server_status(server)
    gpu_load = status.get("gpu_load", "N/A")
    gpu_mem = status.get("gpu_mem", "N/A")

    try:
        gpu_load = [float(x.strip('%')) for x in gpu_load]
        gpu_mem = [float(x.strip(' MB')) for x in gpu_mem]
        # 处理数据
        if gpu_load == "N/A" or gpu_mem == "N/A":
            return None  # 返回空图
    except ValueError:
        return None  # 返回空图

    # 确定 GPU 数量
    num_gpus = len(gpu_load)
    gpu_indices = np.arange(num_gpus)  # GPU 索引

    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 6))

    # GPU Util 柱状图
    bar_width = 0.4
    ax.bar(gpu_indices - bar_width / 2, gpu_load, width=bar_width, color='blue', alpha=0.6, label='GPU Util (%)')

    # GPU Memory 柱状图
    ax.bar(gpu_indices + bar_width / 2, gpu_mem, width=bar_width, color='green', alpha=0.6, label='GPU Memory (MB)')

    # 设置图表属性
    ax.set_xlabel('GPU Index')
    ax.set_xticks(gpu_indices)
    ax.set_xticklabels([f"GPU {i}" for i in gpu_indices])
    ax.set_ylabel('Usage')
    ax.set_title('GPU Utilization and Memory Usage')
    ax.legend()

    plt.tight_layout()
    return fig  # 返回 Matplotlib 图表对象


def plot_native_gpu_utilization(server):
    """
    Generates a bar plot for GPU utilization and memory usage using Gradio's native plotting capabilities.
    :param server: Dictionary containing server status information.
    :return: gr.BarPlot object displaying GPU utilization and memory usage.
    """
    max_mem = dl.configs[server['name']]['show_info'].get('max_gpu_mem', '8G').strip('G')
    # Extract GPU load and memory information from the server status
    status = last.get_server_status(server)
    gpu_load = status.get("gpu_load", "N/A")
    gpu_mem = status.get("gpu_mem", "N/A")

    # Check if data is available
    if gpu_load == "N/A" or gpu_mem == "N/A":
        return None  # Data not available

    try:
        # Convert the GPU data from strings to lists of floats
        gpu_load = [float(x.strip('%')) for x in gpu_load]
        gpu_mem = [float(x.strip(' MB')) / 1024 for x in gpu_mem]
    except Exception as e:
        print(status)
        return None  # Data format is incorrect

    # Create a DataFrame to hold the GPU data
    data = {
        "GPU Index": [f"GPU {i}" for i in range(len(gpu_load))],
        "GPU Utilization (%)": gpu_load,
        "GPU Memory Usage (MB)": gpu_mem
    }
    df = pd.DataFrame(data)

    # Create a BarPlot using Gradio's native plotting capabilities
    with gr.Row():
        utilization_plot = gr.BarPlot(
            value=df,
            x="GPU Index",
            y="GPU Utilization (%)",
            title="显卡功率",
            y_title="Percentage",
            y_lim=[0, 100],
            colors=["blue", "green"],
            every=DEFAULT_UPDATE_INTERVAL,
        )
        memory_plot = gr.BarPlot(
            value=df,
            x="GPU Index",
            y="GPU Memory Usage (MB)",
            title="显卡占用率",
            y_lim=[0, max_mem],
            y_title="GB",
            colors=["blue", "green"],
            every=DEFAULT_UPDATE_INTERVAL,
        )

    return utilization_plot, memory_plot


def plot_native_cpu_utilization():
    """
    Generates a line plot for CPU utilization across multiple servers with formatted time.
    :return: gr.LinePlot object displaying CPU utilization for multiple servers.
    """
    all_data = all.get_all_servers_data()

    # Check if data is available
    if all_data.empty or "cpu_load" not in all_data.columns:
        return None  # No data available

    # Process CPU load data
    all_data["cpu_load"] = all_data["cpu_load"].str.rstrip('%').astype(float)
    all_data["timestamp_ms"] = pd.to_datetime(all_data["timestamp_ms"], unit="ms")
    all_data["Time"] = all_data["timestamp_ms"]

    # Overall data downsampling to 100 points (if needed)
    max_points = 100
    if len(all_data) > max_points:
        all_data = all_data.groupby("server_name").apply(lambda group: group.iloc[::len(group) // max_points or 1])

    # Prepare plot data
    cpu_plot_data = all_data[["server_name", "Time", "cpu_load"]]
    cpu_plot_data = cpu_plot_data.rename(columns={"server_name": "Server", "cpu_load": "CPU Load (%)"})

    # Create a LinePlot using Gradio's native plotting capabilities
    fig = gr.LinePlot(
        value=cpu_plot_data,
        x="Time",
        y="CPU Load (%)",
        color="Server",
        title="各服务器CPU使用率",
        y_title="CPU Load (%)",
        x_title="Time",
        y_lim=[0, 100],
        every=10,  # Update interval in seconds
    )

    def select_region(selection: gr.SelectData):
        min_w, max_w = min(selection.index), max(selection.index)
        return gr.LinePlot(x_lim=[min_w, max_w])

    fig.select(select_region, None, fig)
    fig.double_click(lambda: gr.LinePlot(x_lim=None), None, fig)
    return fig


def plot_native_memory_utilization():
    """
    Generates a line plot for memory utilization across multiple servers with formatted time.
    :return: gr.LinePlot object displaying memory utilization for multiple servers.
    """
    all_data = all.get_all_servers_data()

    # Check if data is available
    if all_data.empty or "memory_load" not in all_data.columns:
        return None  # No data available

    # Process memory load data
    memory_data = all_data["memory_load"].str.split("/", expand=True).astype(str).map(lambda a: a.strip('GB'))
    memory_data.columns = ["Used Memory (GB)", "Total Memory (GB)"]
    memory_data["Used Memory (GB)"] = memory_data["Used Memory (GB)"].astype(float)
    memory_data["Total Memory (GB)"] = memory_data["Total Memory (GB)"].astype(float)
    memory_data["Memory Utilization (%)"] = (memory_data["Used Memory (GB)"] / memory_data["Total Memory (GB)"]) * 100

    all_data["Memory Utilization (%)"] = memory_data["Memory Utilization (%)"]
    all_data["timestamp_ms"] = pd.to_datetime(all_data["timestamp_ms"], unit="ms")
    all_data["Time"] = all_data["timestamp_ms"]

    # Overall data downsampling to 100 points (if needed)
    max_points = 100
    if len(all_data) > max_points:
        all_data = all_data.groupby("server_name").apply(lambda group: group.iloc[::len(group) // max_points or 1])

    # Prepare plot data
    memory_plot_data = all_data[["server_name", "Time", "Memory Utilization (%)"]]
    # memory_plot_data.rename(columns={"server_name": "Server"}, inplace=True)
    memory_plot_data = memory_plot_data.rename(columns={"server_name": "Server"})

    # Create a LinePlot using Gradio's native plotting capabilities
    fig = gr.LinePlot(
        value=memory_plot_data,
        x="Time",
        y="Memory Utilization (%)",
        color="Server",
        title="各服务器内存占用率",
        y_title="Memory Utilization (%)",
        x_title="Time",
        y_lim=[0, 100],
        every=10,  # Update interval in seconds
    )

    def select_region(selection: gr.SelectData):
        min_w, max_w = min(selection.index), max(selection.index)
        return gr.LinePlot(x_lim=[min_w, max_w])

    fig.select(select_region, None, fig)
    fig.double_click(lambda: gr.LinePlot(x_lim=None), None, fig)
    return fig


def plot_native_gpu_utilization_by_server():
    """
    Generates a line plot for GPU utilization (averaged per server) across multiple servers.
    :return: gr.LinePlot object displaying GPU utilization for multiple servers.
    """
    all_data = all.get_all_servers_data()

    # Check if data is available
    if all_data.empty or "gpu_load" not in all_data.columns:
        return None  # No data available

    # Process GPU load data
    def process_gpu_load(gpu_load):
        gpu_values = [float(x.strip('%')) for x in gpu_load]
        return sum(gpu_values) / len(gpu_values)  # Average GPU utilization

    all_data = all_data[all_data['gpu_load'].notnull()]
    all_data["GPU Utilization (%)"] = all_data["gpu_load"].apply(process_gpu_load)
    all_data["timestamp_ms"] = pd.to_datetime(all_data["timestamp_ms"], unit="ms")
    all_data["Time"] = all_data["timestamp_ms"]

    # Downsample data to 100 points if needed
    max_points = 100
    if len(all_data) > max_points:
        all_data = all_data.groupby("server_name").apply(lambda group: group.iloc[::len(group) // max_points or 1])

    # Prepare plot data
    gpu_util_plot_data = all_data[["server_name", "Time", "GPU Utilization (%)"]]
    gpu_util_plot_data = gpu_util_plot_data.rename(columns={"server_name": "Server"})

    # Create a LinePlot
    fig = gr.LinePlot(
        value=gpu_util_plot_data,
        x="Time",
        y="GPU Utilization (%)",
        color="Server",
        title="各服务器GPU算力使用率",
        y_title="GPU Utilization (%)",
        x_title="Time",
        y_lim=[0, 100],
        every=10,  # Update interval in seconds
    )

    def select_region(selection: gr.SelectData):
        min_w, max_w = min(selection.index), max(selection.index)
        return gr.LinePlot(x_lim=[min_w, max_w])

    fig.select(select_region, None, fig)
    fig.double_click(lambda: gr.LinePlot(x_lim=None), None, fig)
    return fig


def plot_native_gpu_memory_utilization_by_server():
    """
    Generates a line plot for GPU memory utilization (averaged per server) across multiple servers.
    :return: gr.LinePlot object displaying GPU memory utilization for multiple servers.
    """
    all_data = all.get_all_servers_data()

    # Check if data is available
    if all_data.empty or "gpu_mem" not in all_data.columns:
        return None  # No data available

    # Process GPU memory data
    def process_gpu_memory(line):
        mem_values = [float(x.strip(' MB')) for x in line['gpu_mem']]
        server_name = line['server_name']
        num_gpus = len(mem_values)
        max_mem_per_gpu = float(dl.configs[server_name]['show_info'].get('max_gpu_mem', '8G').strip('G')) * 1024
        return (sum(mem_values) / (max_mem_per_gpu * num_gpus)) * 100  # Percentage

    all_data = all_data[all_data['gpu_mem'].notnull()]
    all_data["GPU Memory Utilization (%)"] = all_data.apply(process_gpu_memory, axis=1)
    all_data["timestamp_ms"] = pd.to_datetime(all_data["timestamp_ms"], unit="ms")
    all_data["Time"] = all_data["timestamp_ms"]

    # Downsample data to 100 points if needed
    max_points = 100
    if len(all_data) > max_points:
        all_data = all_data.groupby("server_name").apply(lambda group: group.iloc[::len(group) // max_points or 1])

    # Prepare plot data
    gpu_mem_plot_data = all_data[["server_name", "Time", "GPU Memory Utilization (%)"]]
    gpu_mem_plot_data = gpu_mem_plot_data.rename(columns={"server_name": "Server"})

    # Create a LinePlot
    fig = gr.LinePlot(
        value=gpu_mem_plot_data,
        x="Time",
        y="GPU Memory Utilization (%)",
        color="Server",
        title="各服务器显存使用率",
        y_title="GPU Memory Utilization (%)",
        x_title="Time",
        y_lim=[0, 100],
        every=10,  # Update interval in seconds
    )

    def select_region(selection: gr.SelectData):
        min_w, max_w = min(selection.index), max(selection.index)
        return gr.LinePlot(x_lim=[min_w, max_w])

    fig.select(select_region, None, fig)
    fig.double_click(lambda: gr.LinePlot(x_lim=None), None, fig)
    return fig


def plot_native_gpu_usage_per_user(server_name):
    """
    Generates a line plot for GPU memory usage history for each user on a specific server.
    :param server_name: Name of the server to display GPU usage for each user.
    :return: gr.LinePlot object displaying GPU usage per user over time.
    """
    if isinstance(server_name, dict):
        server_name = server_name.get("name", "")
    server_data = all.get_server_data(server_name)

    # Check if data is available
    if server_data.empty or "gpu_usage_per_user" not in server_data.columns:
        return None  # No data available

    # Process GPU usage per user
    usage_records = []
    server_data["timestamp_ms"] = pd.to_datetime(server_data["timestamp_ms"], unit="ms")
    server_data["Time"] = server_data["timestamp_ms"]

    for _, row in server_data.iterrows():
        if pd.isna(row["gpu_usage_per_user"]):
            continue
        usage_dict = row["gpu_usage_per_user"]
        for user, usage in usage_dict.items():
            try:
                usage_records.append({
                    "User": user,
                    "Time": row["Time"],
                    "GPU显存使用量 (GB)": float(str(usage).strip("MB")) / 1024
                })
            except Exception as e:
                print(row)
                raise e

    # Convert the processed records into a DataFrame
    usage_df = pd.DataFrame(usage_records)
    # 如果不含有user列，直接返回None
    if 'User' not in usage_df.columns:
        return None

    # Downsample data to 100 points if needed
    max_points = 500
    window = 10

    def smooth_gpu_memory(group):
        group = group.copy()
        group["GPU Memory Usage (GB)"] = group["GPU显存使用量 (GB)"].rolling(
            window=window, min_periods=1
        ).mean()
        return group

    def remove_middle_same(group):
        group = group.copy()
        gpu_mem = group["GPU显存使用量 (GB)"].values
        keep = [True]  # 保留首行

        # 遍历每一行，标记需要保留的行
        for i in range(1, len(gpu_mem) - 1):
            if gpu_mem[i] == gpu_mem[i - 1] and gpu_mem[i] == gpu_mem[i + 1]:
                keep.append(False)  # 中间重复的值，不保留
            else:
                keep.append(True)
        keep.append(True)  # 保留尾行

        return group[keep]

    usage_df = usage_df.groupby("User").apply(smooth_gpu_memory).reset_index(drop=True)

    if len(usage_df) > max_points:
        usage_df = usage_df.groupby("User").apply(
            lambda group: group.iloc[::len(group) // max_points or 1]).reset_index(drop=True)

    # 应用处理逻辑
    window = 7
    usage_df = usage_df.groupby("User").apply(smooth_gpu_memory).reset_index(drop=True)
    usage_df = usage_df.groupby("User").apply(remove_middle_same).reset_index(drop=True)
    window = 2
    usage_df = usage_df.groupby("User").apply(smooth_gpu_memory).reset_index(drop=True)

    # Create a LinePlot
    fig = gr.LinePlot(
        value=usage_df,
        x="Time",
        y="GPU显存使用量 (GB)",
        color="User",
        title=f"{server_name} - GPU显存使用量",
        y_title="GPU Memory Usage (GB)",
        x_title="Time",
        every=10,  # Update interval in seconds
    )

    def select_region(selection: gr.SelectData):
        min_w, max_w = min(selection.index), max(selection.index)
        return gr.LinePlot(x_lim=[min_w, max_w])

    fig.select(select_region, None, fig)
    fig.double_click(lambda: gr.LinePlot(x_lim=None), None, fig)
    return fig


def plot_gpu_memory_usage_ranking(server_name):
    """
    Generates a bar plot for GPU memory usage ranking of all users on a specific server.
    :param server_name: Name of the server to display GPU memory usage ranking.
    :return: gr.BarPlot object displaying GPU memory usage ranking by user.
    """
    # 获取服务器名称
    if isinstance(server_name, dict):
        server_name = server_name.get("name", "")

    # 获取服务器数据
    server_data = all.get_server_data(server_name)

    # 检查数据是否可用
    if server_data.empty or "gpu_usage_per_user" not in server_data.columns:
        return None  # 数据不可用

    # 解析 GPU 显存使用记录
    usage_records = []
    server_data["timestamp_ms"] = pd.to_datetime(server_data["timestamp_ms"], unit="ms")
    server_data["Time"] = server_data["timestamp_ms"]

    for _, row in server_data.iterrows():
        if pd.isna(row["gpu_usage_per_user"]):
            continue
        usage_dict = row["gpu_usage_per_user"]
        for user, usage in usage_dict.items():
            try:
                usage_records.append({
                    "User": user,
                    "Time": row["Time"],
                    "GPU显存使用量 (GB)": float(str(usage).strip("MB")) / 1024
                })
            except Exception as e:
                print(row)
                # raise e

    # 转换为 DataFrame
    usage_df = pd.DataFrame(usage_records)
    # 如果不含有user列，直接返回None
    if 'User' not in usage_df.columns:
        return None

    # 按用户分组，计算平均显存使用量
    user_memory_usage = (
        usage_df.groupby("User")["GPU显存使用量 (GB)"]
        .mean()
        .reset_index()
        .sort_values(by="GPU显存使用量 (GB)", ascending=True)
    )

    # 添加排名信息
    user_memory_usage["Rank"] = range(1, len(user_memory_usage) + 1)

    # 绘制条形图
    fig = gr.BarPlot(
        value=user_memory_usage,
        x="User",
        y="GPU显存使用量 (GB)",
        color="Rank",
        sort='-y',
        title=f"{server_name} - 用户显存使用排名",
        y_title="Average GPU Memory Usage (GB)",
        x_title="User",
        every=10  # 更新间隔（秒）
    )

    return fig


import pandas as pd
import gradio as gr
import re

max_points = 100  # 限制最大点数


def plot_disk_io(server_name):
    """
    Generates a line plot for disk read/write speeds using Gradio's LinePlot.
    :param server_name: str, 服务器名称
    :return: gr.LinePlot object displaying disk read/write speeds.
    """
    # 从服务器数据中获取历史数据
    server_data = all.get_server_data(server_name)

    # 检查数据是否可用
    if server_data.empty or "disk_IO" not in server_data.columns:
        return None

    # 解析 'disk_IO' 列，提取 Read 和 Write 数据
    def parse_disk_io(io_string):
        match = re.search(r"Read/Write: ([\d.]+)/([\d.]+)", io_string)
        if match:
            return float(match.group(1)), float(match.group(2))
        return 0.0, 0.0

    disk_parsed = server_data["disk_IO"].apply(parse_disk_io)
    server_data["Read (MB/s)"], server_data["Write (MB/s)"] = zip(*disk_parsed)

    # 处理时间戳
    server_data["Time"] = pd.to_datetime(server_data["timestamp_ms"], unit="ms")

    # 下采样到最大点数
    if len(server_data) > max_points:
        server_data = server_data.iloc[::len(server_data) // max_points or 1]

    # 构建长格式数据
    plot_data = pd.melt(
        server_data,
        id_vars=["Time"],
        value_vars=["Read (MB/s)", "Write (MB/s)"],
        var_name="Type",
        value_name="Speed (MB/s)"
    )

    # 绘制图表
    return gr.LinePlot(
        value=plot_data,
        x="Time",
        y="Speed (MB/s)",
        color="Type",
        title=f"{server_name} - 硬盘 I/O 读写速率",
        x_title="Time",
        y_title="Speed (MB/s)",
        every=10
    )


def plot_network_io(server_name):
    """
    Generates a line plot for network upload/download speeds using Gradio's LinePlot.
    :param server_name: str, 服务器名称
    :return: gr.LinePlot object displaying network upload/download speeds.
    """
    # 从服务器数据中获取历史数据
    server_data = all.get_server_data(server_name)

    # 检查数据是否可用
    if server_data.empty or "network_IO" not in server_data.columns:
        return None

    # 解析 'network_IO' 列，提取 Up 和 Down 数据
    def parse_network_io(io_string):
        match = re.search(r"Up/Down: ([\d.]+)/([\d.]+)", io_string)
        if match:
            return float(match.group(1)), float(match.group(2))
        return 0.0, 0.0

    network_parsed = server_data["network_IO"].apply(parse_network_io)
    server_data["Upload (MB/s)"], server_data["Download (MB/s)"] = zip(*network_parsed)

    # 处理时间戳
    server_data["Time"] = pd.to_datetime(server_data["timestamp_ms"], unit="ms")

    # 下采样到最大点数
    if len(server_data) > max_points:
        server_data = server_data.iloc[::len(server_data) // max_points or 1]

    # 构建长格式数据
    plot_data = pd.melt(
        server_data,
        id_vars=["Time"],
        value_vars=["Upload (MB/s)", "Download (MB/s)"],
        var_name="Type",
        value_name="Speed (MB/s)"
    )

    # 绘制图表
    return gr.LinePlot(
        value=plot_data,
        x="Time",
        y="Speed (MB/s)",
        color="Type",
        title=f"{server_name} - 网络 I/O 上行/下行速率",
        x_title="Time",
        y_title="Speed (MB/s)",
        every=10
    )
