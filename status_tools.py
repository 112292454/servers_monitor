import subprocess
import time

import paramiko


class MYSSHClient:
    def __init__(self, host, port=22, username="root", password=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client = None
        self.ssh = None

    def connect(self):
        """Establish an SSH connection to the server."""
        if self.client is None:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(self.host, port=self.port, username=self.username, password=self.password)

    def execute_command(self, command):
        """Execute a command on the remote server."""
        if not self.client:
            self.connect()
        return self.client.exec_command(command)
        # stdin, stdout, stderr = self.client.exec_command(command)
        # return stdout.read().decode()

    def close(self):
        """Close the SSH connection."""
        if self.client:
            self.client.close()
            self.client = None


# 设计一个map：对于每个server，存储一个SSHClient对象，通过一个函数初始化，传入config
def create_ssh_clients(configs):
    """Create SSH clients for each server in the configuration."""
    ssh_clients = {}
    for name, server in configs.items():
        ssh_clients[name] = MYSSHClient(server["ip"], server["show_info"].get("ssh_port", 22),password=server.get('passwd',None),username=server.get('user','root'))
        ssh_clients[name].connect()
    return ssh_clients


def is_server_up(host):
    if is_ssh_up:
        return True
    """Check if the server is up and running by pinging the host."""
    try:
        # Run the ping command to check if the server is reachable
        command = ["ping", "-c", "1", host]  # Send 1 ping request
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # If ping is successful, the server is up
        if result.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        return False


def is_ssh_up(ssh_client: MYSSHClient):
    """Check if the server is up and running by pinging the host."""
    return ssh_client.client is not None


def get_cpu_load(ssh_client):
    """Fetch CPU load from the remote server (as a percentage)."""
    try:
        # Command to get the load averages directly from uptime
        command = "uptime | awk -F'load average: ' '{print $2}' | cut -d',' -f1"

        # Run the command on the remote server
        stdin, stdout, stderr = ssh_client.execute_command(command)
        load_1min = stdout.read().decode().strip()

        # Return the load as a percentage (add % sign)
        return f"{load_1min.strip()}%"

    except Exception as e:
        return f"Error: {str(e)}"


def get_memory_load(ssh_client):
    """Fetch memory usage from the remote server (used/total)."""
    try:
        # Command to get the memory usage
        command = "free -g | sed -n '2p' | awk '{print $3 \"/\" $2}'"

        # Run the command on the remote server
        stdin, stdout, stderr = ssh_client.execute_command(command)
        memory_usage = stdout.read().decode().strip()

        return memory_usage + ' GB'

    except Exception as e:
        return f"Error: {str(e)}"


def get_gpu_load(ssh_client):
    """Fetch GPU usage from the remote server."""
    try:
        # Command to get GPU usage using nvidia-smi
        command = "nvidia-smi -i $(nvidia-smi --list-gpus | grep -oP 'GPU \K\d+' | tr '\n' ',' | sed 's/,$//') --query-gpu=utilization.gpu --format=csv,noheader,nounits"

        # Run the command on the remote server
        stdin, stdout, stderr = ssh_client.execute_command(command)
        gpu_usages = stdout.read().decode().strip().splitlines()

        # Return GPU utilization as a percentage for each GPU
        return ", ".join([f"{usage}%" for usage in gpu_usages])

    except Exception as e:
        return f"Error: {str(e)}"


def get_gpu_mem(ssh_client):
    """Fetch GPU memory usage from the remote server."""
    try:
        # Command to get GPU memory usage using nvidia-smi
        command = "nvidia-smi -i $(nvidia-smi --list-gpus | grep -oP 'GPU \K\d+' | tr '\n' ',' | sed 's/,$//') --query-gpu=memory.used --format=csv,noheader,nounits"

        # Run the command on the remote server
        stdin, stdout, stderr = ssh_client.execute_command(command)
        gpu_memories = stdout.read().decode().strip().splitlines()

        # Return the used memory for each GPU
        return ", ".join([f"{mem}MB" for mem in gpu_memories])

    except Exception as e:
        return f"Error: {str(e)}"


def get_gpu_using_users(ssh_client):
    """Fetch GPU users from the remote server by checking the processes using the GPU."""
    try:
        # Combine nvidia-smi and ps to get the users in one step
        command = """
        nvidia-smi -i $(nvidia-smi --list-gpus | grep -oP 'GPU \K\d+' | tr '\n' ',' | sed 's/,$//') --query-compute-apps=pid --format=csv,noheader | 
        xargs -I {} ps -o user= -p {} | sort | uniq
        """
        stdin, stdout, stderr = ssh_client.execute_command(command)
        users = stdout.read().decode().strip()

        # If there are no users, return a default message
        if not users:
            return "Error: No users using"

        return users

    except Exception as e:
        return f"Error: {str(e)}"


def get_gpu_load_per_user(ssh_client):
    """Fetch GPU users from the remote server by checking the processes using the GPU."""
    try:
        # Combine nvidia-smi and ps to get the users in one step
        command = """
        nvidia-smi -i $(nvidia-smi --list-gpus | grep -oP 'GPU \K\d+' | tr '\n' ',' | sed 's/,$//') \
        --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits |sed 's/,//g' |awk '{print $1 , $NF}'|\
        awk '{cmd = "ps -o user= -p " $1; cmd | getline user; close(cmd); print $1, $2, user}' |\
        awk '{mem[$3] += $2} END {for (user in mem) print user, mem[user]"MB"}'    
        """
        stdin, stdout, stderr = ssh_client.execute_command(command)
        usage = stdout.read().decode().strip()

        # If there are no users, return a default message
        if not usage:
            return "no_user 0MB"

        return usage

    except Exception as e:
        return f"Error: {str(e)}"


def get_network_IO(ssh_client):
    """Fetch network I/O bandwidth from the remote server."""
    try:
        # Command to get network stats from /proc/net/dev
        command = "cat /proc/net/dev"

        # Read the stats from /proc/net/dev
        stdin, stdout, stderr = ssh_client.execute_command(command)
        net_stats = stdout.read().decode().strip().split("\n")

        # Parse the network stats
        interfaces = {}
        for line in net_stats[2:]:  # Skip header lines
            fields = line.split(":")
            interface_name = fields[0].strip()
            stats = fields[1].split()
            recv_bytes = int(stats[0])  # Received bytes
            sent_bytes = int(stats[8])  # Sent bytes

            interfaces[interface_name] = {
                "recv_bytes": recv_bytes,
                "sent_bytes": sent_bytes
            }

        return interfaces

    except Exception as e:
        return f"Error: {str(e)}"


def get_network_IO_bandwidth(ssh_client):
    """Fetch network I/O bandwidth (in MB/s) for upload (sent) and download (recv) from the remote server."""
    try:
        # First read to get initial stats
        interfaces1 = get_network_IO(ssh_client)
        time.sleep(1)  # Wait for 1 second to measure the change

        # Second read to get updated stats
        interfaces2 = get_network_IO(ssh_client)

        max_bandwidth = 0
        max_interface = None
        upload_bandwidth = 0
        download_bandwidth = 0

        for interface in interfaces1:
            if interface == "lo":  # Skip the loopback interface
                continue

            # Calculate the difference in received and sent bytes between the two readings
            recv_bytes_diff = interfaces2[interface]["recv_bytes"] - interfaces1[interface]["recv_bytes"]
            sent_bytes_diff = interfaces2[interface]["sent_bytes"] - interfaces1[interface]["sent_bytes"]

            # Convert to MB/s (bytes -> MB)
            interface_download_bandwidth = recv_bytes_diff / 1024 / 1024  # MB/s
            interface_upload_bandwidth = sent_bytes_diff / 1024 / 1024  # MB/s

            # Total bandwidth (download + upload)
            interface_bandwidth = interface_download_bandwidth + interface_upload_bandwidth

            # If this interface has the largest total bandwidth, we update the values
            if interface_bandwidth > max_bandwidth:
                max_bandwidth = interface_bandwidth
                max_interface = interface
                upload_bandwidth = interface_upload_bandwidth
                download_bandwidth = interface_download_bandwidth

        # return {
        #     "interface": max_interface,
        #     "upload_bandwidth_MB_s": round(upload_bandwidth, 2),  # Upload in MB/s
        #     "download_bandwidth_MB_s": round(download_bandwidth, 2),  # Download in MB/s
        # }
        return f"Up/Down: {round(upload_bandwidth, 2)}/{round(download_bandwidth, 2)} (MB/s)"

    except Exception as e:
        return f"Error: {str(e)}"


def get_disk_IO(ssh_client):
    """Fetch the total disk I/O read/write rates (in MB/s) from the remote server, excluding loop devices."""
    try:
        # Run the iostat command to get disk stats and exclude loop devices
        command = """
        iostat -d | grep -v 'loop' | awk '{if(NR>3) print $1, $3, $4}' 
        """
        stdin, stdout, stderr = ssh_client.execute_command(command)
        disk_io_data = stdout.read().decode().strip()

        # If no data is returned (e.g., no physical devices), return a default message
        if not disk_io_data:
            return "No disk information available"

        total_read_rate = 0
        total_write_rate = 0

        # Process the disk I/O data, summing up the read and write rates
        for line in disk_io_data.splitlines():
            device, read_rate, write_rate = line.split()

            # Convert read and write rates from KB/s to MB/s
            total_read_rate += float(read_rate) / 1024  # KB to MB
            total_write_rate += float(write_rate) / 1024  # KB to MB

        # Return the total I/O read and write rates in MB/s
        # return {
        #     "total_read_rate": f"{total_read_rate:.2f} MB/s",
        #     "total_write_rate": f"{total_write_rate:.2f} MB/s"
        # }
        return f"Read/Write: {total_read_rate:.2f}/{total_write_rate:.2f} (MB/s)"


    except Exception as e:
        return f"Error: {str(e)}"
