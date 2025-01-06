
from data_layer import last,all


def display_is_up(server):
    """
    返回服务器是否在线的状态
    """
    status = last.get_server_status(server)
    return f"<p style='color:green; font-size:36px;'><strong>&#x2705;</strong> Server is UP</p>" if status.get("is_up", False) \
        else f"<p style='color:red; font-size:36px;'><strong>&#10060;</strong> Server is DOWN</p>"

def display_ssh_status(server):
    """
    返回 SSH 是否可访问的状态
    """
    status = last.get_server_status(server)
    return f"<p style='color:green; font-size:36px;'><strong>&#x2705;</strong> SSH is UP</p>" if status.get("is_ssh_up", False) \
        else f"<p style='color:red; font-size:36px;'><strong>&#10060;</strong> SSH is DOWN</p>"

def display_cpu_load(server):
    # def inner():
    status = last.get_server_status(server)
    return status.get("cpu_load","N/A")
    # return inner

def display_memory_load(server):
    """
    返回内存使用情况
    """
    status = last.get_server_status(server)
    return status.get("memory_load", "N/A")

def display_gpu_load(server):
    """
    返回 GPU 使用率
    """
    status = last.get_server_status(server)
    return status.get("gpu_load", "N/A")

def display_gpu_mem(server):
    """
    返回 GPU 内存使用情况
    """
    status = last.get_server_status(server)
    return status.get("gpu_mem", "N/A")

def display_gpu_using_users(server):
    """
    返回当前正在使用 GPU 的用户列表
    """
    status = last.get_server_status(server)
    users = status.get("gpu_using_users", [])
    if isinstance(users, list):
        return ", ".join(users)
    return users

def display_gpu_usage_per_user(server):
    """
    返回每个用户的 GPU 使用情况
    """
    status = last.get_server_status(server)
    usage = status.get("gpu_usage_per_user", {})
    if isinstance(usage, dict):
        return "; ".join([f"{user}: {int(usage[user].strip('MB'))//1024}GB" for user in usage])
    return usage

def display_network_io(server):
    """
    返回网络 I/O 使用情况
    """
    status = last.get_server_status(server)
    return status.get("network_IO", "N/A")

def display_disk_io(server):
    """
    返回磁盘 I/O 使用情况
    """
    status = last.get_server_status(server)
    return status.get("disk_IO", "N/A")

def display_last_checked(server):
    """
    返回上次更新时间
    """
    status = last.get_server_status(server)
    return status.get("last_checked", "N/A")
