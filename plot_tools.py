import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_layer import last,all, DEFAULT_UPDATE_INTERVAL, configs
import data_layer as dl

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
    max_mem=dl.configs[server['name']]['show_info'].get('max_gpu_mem', '8G').strip('G')
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
        gpu_mem = [float(x.strip(' MB'))/1024 for x in gpu_mem]
    except ValueError:
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
    all_data["Time"]=all_data["timestamp_ms"]

    # Overall data downsampling to 100 points (if needed)
    max_points = 100
    if len(all_data) > max_points:
        all_data = all_data.groupby("server_name").apply(lambda group: group.iloc[::len(group) // max_points or 1])


    # Prepare plot data
    cpu_plot_data = all_data[["server_name", "Time","cpu_load"]]
    cpu_plot_data.rename(columns={"server_name": "Server", "cpu_load": "CPU Load (%)"}, inplace=True)

    # Create a LinePlot using Gradio's native plotting capabilities
    fig = gr.LinePlot(
        value=cpu_plot_data,
        x="Time",
        y="CPU Load (%)",
        color="Server",
        title="CPU Utilization Over Time",
        y_title="CPU Load (%)",
        x_title="Time",
        y_lim=[0, 100],
        every=10,  # Update interval in seconds
    )

    def select_region(selection: gr.SelectData):
        min_w,max_w = min(selection.index),max(selection.index)
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
    all_data["Time"]=all_data["timestamp_ms"]

    # Overall data downsampling to 100 points (if needed)
    max_points = 100
    if len(all_data) > max_points:
        all_data = all_data.groupby("server_name").apply(lambda group: group.iloc[::len(group) // max_points or 1])

    # Prepare plot data
    memory_plot_data = all_data[["server_name", "Time", "Memory Utilization (%)"]]
    memory_plot_data.rename(columns={"server_name": "Server"}, inplace=True)

    # Create a LinePlot using Gradio's native plotting capabilities
    fig = gr.LinePlot(
        value=memory_plot_data,
        x="Time",
        y="Memory Utilization (%)",
        color="Server",
        title="Memory Utilization Over Time",
        y_title="Memory Utilization (%)",
        x_title="Time",
        y_lim=[0, 100],
        every=10,  # Update interval in seconds
    )

    def select_region(selection: gr.SelectData):
        min_w,max_w = min(selection.index),max(selection.index)
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

    all_data=all_data[all_data['gpu_load'].notnull()]
    all_data["GPU Utilization (%)"] = all_data["gpu_load"].apply(process_gpu_load)
    all_data["timestamp_ms"] = pd.to_datetime(all_data["timestamp_ms"], unit="ms")
    all_data["Time"] = all_data["timestamp_ms"]

    # Downsample data to 100 points if needed
    max_points = 100
    if len(all_data) > max_points:
        all_data = all_data.groupby("server_name").apply(lambda group: group.iloc[::len(group) // max_points or 1])

    # Prepare plot data
    gpu_util_plot_data = all_data[["server_name", "Time", "GPU Utilization (%)"]]
    gpu_util_plot_data.rename(columns={"server_name": "Server"}, inplace=True)

    # Create a LinePlot
    fig = gr.LinePlot(
        value=gpu_util_plot_data,
        x="Time",
        y="GPU Utilization (%)",
        color="Server",
        title="Average GPU Utilization Over Time",
        y_title="GPU Utilization (%)",
        x_title="Time",
        y_lim=[0, 100],
        every=10,  # Update interval in seconds
    )

    def select_region(selection: gr.SelectData):
        min_w,max_w = min(selection.index),max(selection.index)
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

    all_data=all_data[all_data['gpu_mem'].notnull()]
    all_data["GPU Memory Utilization (%)"] = all_data.apply(process_gpu_memory,axis=1)
    all_data["timestamp_ms"] = pd.to_datetime(all_data["timestamp_ms"], unit="ms")
    all_data["Time"] = all_data["timestamp_ms"]

    # Downsample data to 100 points if needed
    max_points = 100
    if len(all_data) > max_points:
        all_data = all_data.groupby("server_name").apply(lambda group: group.iloc[::len(group) // max_points or 1])

    # Prepare plot data
    gpu_mem_plot_data = all_data[["server_name", "Time", "GPU Memory Utilization (%)"]]
    gpu_mem_plot_data.rename(columns={"server_name": "Server"}, inplace=True)

    # Create a LinePlot
    fig = gr.LinePlot(
        value=gpu_mem_plot_data,
        x="Time",
        y="GPU Memory Utilization (%)",
        color="Server",
        title="Average GPU Memory Utilization Over Time",
        y_title="GPU Memory Utilization (%)",
        x_title="Time",
        y_lim=[0, 100],
        every=10,  # Update interval in seconds
    )

    def select_region(selection: gr.SelectData):
        min_w,max_w = min(selection.index),max(selection.index)
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
    server_data=all.get_server_data(server_name)

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
                "GPU Memory Usage (GB)": float(usage.strip("MB"))/1024
            })
            except Exception as e:
                print(row)
                raise e

    # Convert the processed records into a DataFrame
    usage_df = pd.DataFrame(usage_records)

    # Downsample data to 100 points if needed
    max_points = 100
    if len(usage_df) > max_points:
        usage_df = usage_df.groupby("User").apply(lambda group: group.iloc[::len(group) // max_points or 1])

    # Create a LinePlot
    fig = gr.LinePlot(
        value=usage_df,
        x="Time",
        y="GPU Memory Usage (GB)",
        color="User",
        title=f"GPU Memory Usage Per User on {server_name}",
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
