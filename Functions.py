import struct
import numpy as np
import pickle
import copy
import torch
from torch import nn

HEADERSIZE = 10  # Bytes
FORMAT = 'utf-8'

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def send_msg(sock, msg):
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    msglen = recvall(sock, 4)
    if not msglen:
        return None
    msg_len = struct.unpack('>I', msglen)[0]
    return recvall(sock, msg_len)

def recvall(sock, message_length):
    recv_data = bytearray()
    while len(recv_data) < message_length:
        packet = sock.recv(message_length - len(recv_data))
        if not packet:
            return None
        recv_data.extend(packet)
    return recv_data

