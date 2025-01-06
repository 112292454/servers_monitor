import signal
from datetime import datetime

import yaml

import display as dp
import data_layer
from data_layer import *
from plot_tools import *
from status_tools import *


# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(STATUS_DIR, exist_ok=True)


def load_server_configs():
    """Load server configurations from the config directory."""
    configs = {}
    for file_name in os.listdir(CONFIG_DIR):
        if file_name.endswith(".yml") or file_name.endswith(".yaml"):
            with open(os.path.join(CONFIG_DIR, file_name), 'r', encoding='utf-8') as f:
                server_config = yaml.safe_load(f)

                # Check if 'ssh_connect_cmd' exists in show_info and perform replacement
                if "show_info" in server_config and "ssh_connect_cmd" in server_config["show_info"]:
                    ssh_cmd_template = server_config["show_info"]["ssh_connect_cmd"]

                    # Replace {ip} with the server's IP and {port} with the given port (if any)
                    ssh_cmd = ssh_cmd_template.format(ip=server_config.get("ip", ""),
                                                      port=server_config["show_info"].get("ssh_port", "22"))

                    # Update the ssh_connect_cmd with the filled-in command
                    server_config["show_info"]["ssh_connect_cmd"] = ssh_cmd

                configs[server_config['name']]=server_config
    return configs


def log_event(server_name, event):
    """Log events to a file in the logs directory."""
    log_file = os.path.join(LOG_DIR, f"{server_name}.log")
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now()}] {event}\n")


def monitor_server(server, ssh_client):
    """Monitor a server and collect all metrics, regardless of display configuration."""
    status = {}
    status['is_up'] = is_server_up(server['ip'])
    status['is_ssh_up'] = is_ssh_up(ssh_client)
    if not status['is_up']:
        return status

    # Collect metrics for all items, whether displayed or not
    for key in server['monitoring']:
        # For each metric, call the appropriate function to fetch the data
        if key == "cpu_load":
            status["cpu_load"] = get_cpu_load(ssh_client)
        elif key == "memory_load":
            status["memory_load"] = get_memory_load(ssh_client)
        elif key == "gpu_load":
            status["gpu_load"] = get_gpu_load(ssh_client)
        elif key == "gpu_mem":
            status["gpu_mem"] = get_gpu_mem(ssh_client)
        elif key == "gpu_using_users":
            status["gpu_using_users"] = get_gpu_using_users(ssh_client)
        elif key == "gpu_usage_per_user":
            status["gpu_usage_per_user"] = get_gpu_load_per_user(ssh_client)
        elif key == "network_IO":
            status["network_IO"] = get_network_IO_bandwidth(ssh_client)
        elif key == "disk_IO":
            status["disk_IO"] = get_disk_IO(ssh_client)

    status["last_checked"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return status


def background_monitor(interval=DEFAULT_UPDATE_INTERVAL):
    # configs, clients

    """Run the monitoring process in the background, updating server statuses every `interval` seconds."""
    while True:
        # 获取每台服务器的状态并保存到文件
        for name,server in configs.items():
            # 获取服务器的监控状态
            status = monitor_server(server, clients.get(name,None))
            # 添加时间戳标识
            status["last_checked"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            status["timestamp_ms"] = int(time.time() * 1000)

            # 将状态保存到status目录中的JSON文件
            status_file = os.path.join(STATUS_DIR, f"{name}.jsonl")
            with open(status_file, 'a') as f:  # Append mode
                f.write('\n' + json.dumps(status).strip('\0'))  # Write each status as a separate line

            # 记录日志（可选）
            log_event(name, f"Status updated: {status}")

        # 每隔 `interval` 秒执行一次监控
        time.sleep(interval)
    print("ERROR: Background monitor stopped.")


def main():
    data_layer.configs = load_server_configs()
    data_layer.clients = create_ssh_clients(data_layer.configs)
    global configs, clients
    configs = data_layer.configs
    clients = data_layer.clients

    # Start the background monitor
    threading.Thread(target=background_monitor, daemon=True).start()

    def server_tab(server):
        """Generate the tab for each server to display its monitoring status and info."""
        with gr.Tab(server["name"]):
            # Fetch the server's current status from the status directory
            # status = last.get_server_status(server)

            # def update_status(status_ref):
            #     while True:
            #         status_ref[0] = last.get_server_status(server)
            #         time.sleep(10)
            #
            # status_ref = [status]
            # thread = threading.Thread(target=update_status, args=(status_ref,))
            # thread.daemon = True  # 确保程序退出时线程自动结束
            # thread.start()

            # status_outputs = []
            # # Set the server up flag to green if up, otherwise red
            with gr.Row():
                # 显示服务器和 SSH 状态，动态更新
                # status_outputs = []

                gr.HTML(
                    value=lambda: dp.display_is_up(server),
                    label="Server Status",
                    every=10,
                )
                gr.HTML(
                    value=lambda: dp.display_ssh_status(server),
                    label="SSH Status",
                    every=10,
                )

            # 动态绑定监控项数据
            if server["monitoring"].get("cpu_load", False):
                gr.Textbox(
                    value=lambda: dp.display_cpu_load(server),
                    label="CPU Load",
                    every=DEFAULT_UPDATE_INTERVAL,
                    interactive=False,
                )
            if server["monitoring"].get("memory_load", False):
                gr.Textbox(
                    value=lambda: dp.display_memory_load(server),
                    label="Memory Load",
                    every=DEFAULT_UPDATE_INTERVAL,
                    interactive=False,
                )

            # def plot_example():
            #     import matplotlib.pyplot as plt
            #     import numpy as np
            #
            #     x = np.linspace(0, 10, 100)
            #     y = np.sin(x)
            #
            #     plt.plot(x, y)
            #     plt.title("Sine Wave")
            #     return plt.gcf()  # 返回当前的图表
            # gr.Plot(plot_example,every=10)
            if server["monitoring"].get("gpu_load", False) and server["monitoring"].get("gpu_mem", False):
                # gr.Plot(
                #     lambda: plot_native_gpu_utilization(server),
                #     label="GPU Utilization",
                #     every=DEFAULT_UPDATE_INTERVAL,)
                    plot_native_gpu_utilization(server)

            # if server["monitoring"].get("gpu_load", False):
            #     gr.Textbox(
            #         value=lambda: dp.display_gpu_load(server),
            #         label="GPU Load",
            #         every=DEFAULT_UPDATE_INTERVAL,
            #         interactive=False,
            #     )
            # if server["monitoring"].get("gpu_mem", False):
            #     gr.Textbox(
            #         value=lambda: dp.display_gpu_mem(server),
            #         label="GPU Memory",
            #         every=DEFAULT_UPDATE_INTERVAL,
            #         interactive=False,
            #     )
            if server["monitoring"].get("gpu_using_users", False):
                # gr.Textbox(
                #     value=lambda: dp.display_gpu_using_users(server),
                #     label="Using GPU Users",
                #     every=DEFAULT_UPDATE_INTERVAL,
                #     interactive=False,
                # )
                plot_gpu_memory_usage_ranking(server['name'])
            if server["monitoring"].get("gpu_usage_per_user", False):
                # gr.Textbox(
                #     value=lambda: dp.display_gpu_usage_per_user(server),
                #     label="GPU Usage per User",
                #     every=DEFAULT_UPDATE_INTERVAL,
                #     interactive=False,
                # )
                plot_native_gpu_usage_per_user(server)

            if server["monitoring"].get("network_IO", False):
                # gr.Textbox(
                #     value=lambda: dp.display_network_io(server),
                #     label="Network I/O",
                #     every=DEFAULT_UPDATE_INTERVAL,
                #     interactive=False,
                # )
                plot_network_io(server['name'])
            if server["monitoring"].get("disk_IO", False):
                # gr.Textbox(
                #     value=lambda: dp.display_disk_io(server),
                #     label="Disk I/O",
                #     every=DEFAULT_UPDATE_INTERVAL,
                #     interactive=False,
                # )
                plot_disk_io(server['name'])
    
            gr.Textbox(
                value=lambda: dp.display_last_checked(server),
                label="上次更新时间",
                every=DEFAULT_UPDATE_INTERVAL,
                interactive=False,
            )

            # Display the server's additional info (like disk usage) in a DataFrame
            info_data = [[key, value] for key, value in server["show_info"].items()]

            info_output = gr.DataFrame(
                value=pd.DataFrame(info_data, columns=["Key", "Value"]),
                headers=["Key", "Value"],
                datatype=["markdown", "markdown"],
                label="Server Info",
            )
            gr.HTML(
                "<p style='color:green; font-size:24px;'>上面俩绿着，能获取到信息就说明服务器没挂，连不上的拿ssh_connect_cmd试试（user改成自己）</p>")

            #
            # # Refresh button to update data
            # def update_server_tab():
            #     """更新每个服务器的tab，读取status文件中的最新监控数据并更新显示"""
            #     # 获取服务器的状态文件路径
            #     status = last.get_server_status(server)
            #
            #     # 更新监控项显示
            #     updated_status = []
            #     if status.get('is_up', False):
            #         updated_status.append(gr.HTML(value=green("Server is UP"), label='Server Status'))
            #     else:
            #         updated_status.append(gr.HTML(value=red("Server is DOWN"), label='Server Status'))
            #     # Set the SSH up flag to green if SSH is available, otherwise red
            #     if status.get('is_ssh_up', False):
            #         updated_status.append(gr.HTML(value=green("SSH is UP"), label='SSH Status'))
            #     else:
            #         updated_status.append(gr.HTML(value=red("SSH is DOWN"), label='SSH Status'))
            #
            #     # if server["monitoring"].get("cpu_load", False):
            #     #     updated_status.append(gr.update(value=status.get("cpu_load", "N/A")))
            #     if server["monitoring"].get("memory_load", False):
            #         updated_status.append(gr.update(value=status.get("memory_load", "N/A")))
            #     if server["monitoring"].get("gpu_load", False):
            #         updated_status.append(gr.update(value=status.get("gpu_load", "N/A")))
            #     if server["monitoring"].get("gpu_mem", False):
            #         updated_status.append(gr.update(value=status.get("gpu_mem", "N/A")))
            #     if server["monitoring"].get("gpu_using_users", False):
            #         updated_status.append(gr.update(value=status.get("gpu_using_users", "N/A")))
            #     if server["monitoring"].get("gpu_usage_per_user", False):
            #         updated_status.append(gr.update(value=status.get("gpu_usage_per_user", "N/A")))
            #     if server["monitoring"].get("network_IO", False):
            #         updated_status.append(gr.update(value=status.get("network_IO", "N/A")))
            #     if server["monitoring"].get("disk_IO", False):
            #         updated_status.append(gr.update(value=status.get("disk_IO", "N/A")))
            #
            #     updated_status.append(gr.update(value=status.get("last_checked", "N/A")))
            #
            #     # info_data 固定内容，不需要每次更新
            #     updated_info = gr.update(value=info_data)  # 固定不变的服务器附加信息
            #
            #     return updated_status + [updated_info]
            #
            # refresh_button = gr.Button("Refresh")
            # refresh_button.click(update_server_tab, outputs=status_outputs + [info_output], every=DEFAULT_UPDATE_INTERVAL)
            # gr.Timer(10)
            # gr.Timer.tick(fn=update_server_tab)

            # write a thread call each 10s: gr.Button("Refresh").click(update_server_tab, outputs=status_outputs + [info_output])

    with gr.Blocks() as app:
        gr.Markdown("""# Server Status Monitor""")

        # with gr.Row():
        # Define tabs for each server
        with gr.Tab('OVERALL'):
            with gr.Row():
                plot_native_cpu_utilization()
                plot_native_memory_utilization()
            # plot_native_cpu_utilization()
            # plot_native_memory_utilization()
            plot_native_gpu_utilization_by_server()
            plot_native_gpu_memory_utilization_by_server()
            with gr.Row():
                for name,server in configs.items():
                    if server["show_info"].get("max_gpu_mem", False):
                        plot_native_gpu_usage_per_user(server)
            pass

        for name,server in configs.items():
            server_tab(server)

        app.launch(server_name="0.0.0.0",server_port=18000)



# Function to stop the application after 2 hours
def stop_app_after_delay(delay_seconds):
    time.sleep(delay_seconds)
    print("Shutting down the application after 2 hours...")
    os.kill(os.getpid(), signal.SIGINT)  # Send SIGINT to terminate gracefully

if __name__ == "__main__":
    # Start the shutdown timer in a separate thread
    delay_seconds = 2 * 60 * 60  # 2 hours in seconds
    shutdown_thread = threading.Thread(target=stop_app_after_delay, args=(delay_seconds,))
    shutdown_thread.daemon = True  # Make sure this thread does not block the program from exiting
    shutdown_thread.start()

    # Start the application
    main()