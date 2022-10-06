# CIFAR10_FederatedLearning_in_Clients

This project is used for distributed deep learning, the dataset used is [CIFAR_10](http://www.cs.toronto.edu/~kriz/cifar.html), and the distributed deep learning platform is composed of a computer and Raspberry Pis. This project uses 8 Raspberry Pi as clients and builds a Linux environment. To reflect the heterogeneous nature of the data between each clients, I will assign a unique training set to each clients in order, ensuring a different distribution of datasets between clients.

## Repository summary

- `server.ipynb`  is required to be run on the host and used as a server. the role of integrating and returning the weight obtained from the clients training through the FedAvg model. This is the first program that needs to be run.
- `client_rasp.py`is the program that trains the model in each clients and runs after the server program starts.
- `start.py` can connect to all clients at the same time via ssh and execute `client_rasp.py` under their respective client files at the same time. `usr` in the program can be configured according to the actual number of clients.

## Preparation

First we need to configure the environment for the Raspberry Pi and the computer separately

### Requirement(server)

```
pip install -r requirement_server.txt
```

### Requirement(clients)

```
pip install -r requirement_clients.txt
```

Each clients need to set static IP address, The specific method can be found [here](https://www.makeuseof.com/raspberry-pi-set-static-ip/). The following table shows the IP address correspondence between the clients used in this project:

| CLIENTS  |  IP ADDRESS   |
| :------: | :-----------: |
| Client1  | 192.168.1.111 |
| Client2  | 192.168.1.112 |
|   ...    |      ...      |
| Client32 | 192.168.1.142 |



## How to use

### 1. Run server 

Run  `~server.ipynb` file, set `users`,`round`

### 2. Run clients

You __need__ to use `~startr.py` file, set `usr`,`ip`.`username`,`password`, what you need to do is running this file in the ==python console==

![image-20221006164836497](image-20221006164836497.png)



