import paramiko
import tqdm
import time

usr = 24

for i in range(usr):
    ip = '192.168.1.1'+str(i+11)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=ip, port=22, username='pi', password='ljnnnnnn')
    # stdin, stdout, stderr = ssh.exec_command('python /home/pi/Desktop/client_rasp.py')
    invoke = ssh.invoke_shell()
    invoke.send("python /home/pi/Desktop/client_rasp.py \n")


while True:
    n = invoke.recv(1024).decode('utf-8','ignore')
    print('\r',n)
