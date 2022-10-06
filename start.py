import paramiko
from time import sleep
usr = 8

for i in range(usr):
    ip = '192.168.1.1'+str(i+11)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=ip, port=22, username='pi', password='ljnnnnnn')
    stdin, stdout, stderr = ssh.exec_command('python /home/pi/Desktop/client_rasp_test.py')

# from fabric import Connection
# usr = 8
# # 建议将ssh连接所需参数变量化
# user = 'pi'
# password = 'ljnnnnnn'
# for i in range(usr):
#     host = '192.168.1.1'+str(i+11)
#     # 利用fabric.Connection快捷创建连接
#     c = Connection(host=f'{user}@{host}',
#                    connect_kwargs=dict(
#                        password=password
#                    ))
#
#     # 利用run方法直接执行传入的命令
#     c.run('python /home/pi/Desktop/client_rasp_test.py', hide=True)

