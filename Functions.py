import struct
import numpy as np
import pickle
import copy
import torch
from torch import nn
# from tqdm import tqdm
import socket

HEADERSIZE = 10  # Bytes
FORMAT = 'utf-8'
SIZE = 1024

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# def send_msg(sock, msg):
#     print(msg, len(msg))
#     msg_pickle = pickle.dumps(msg)
#     sock.sendall(struct.pack(">I", len(msg_pickle)))
#     sock.sendall(msg_pickle)
#     print(msg[0], 'sent to', sock.getpeername())
#
#
# def recv_msg(sock, expect_msg_type=None):
#     msg_len = struct.unpack(">I", sock.recv(4))[0]
#     print(msg_len)
#     msg = sock.recv(msg_len, socket.MSG_WAITALL)
#     print(len(msg))
#     # recv_data = bytearray()
#     # while len(recv_data) < msg_len:
#     #     packet = sock.recv(msg_len - len(recv_data), socket.MSG_WAITALL)
#     #     if not packet:
#     #         return None
#     #     recv_data.extend(packet)
#     #     print('recv_data: ', len(recv_data))
#     msg = pickle.loads(msg)
#     print(msg[0], 'received from', sock.getpeername())
#
#     # if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
#     #     raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
#     return msg

# def send_msg(sock, msg):
#     # print('send_msg 1 : ', msg, len(msg))
#     msg = pickle.dumps(msg)
#     # print('send_msg 2 : ', msg, len(msg))
#     msg = struct.pack('>I', len(msg)) + msg
#     # print('send_msg 3 : ', msg, len(msg))
#     sock.sendall(msg)
#
# def recv_msg(sock):
#     msglen = recvall(sock, 4)
#     print(msglen)
#     if not msglen:
#         return None
#     msg_len = struct.unpack('>I', msglen)[0]
#     print(msg_len)
#     return recvall(sock, msg_len)
#
# def recvall(sock, message_length):
#     print(message_length)
#     recv_data = bytearray()
#     while len(recv_data) < message_length:
#         packet = sock.recv(message_length - len(recv_data), socket.MSG_WAITALL)
#         if not packet:
#             return None
#         recv_data.extend(packet)
#         print(recv_data, len(recv_data))
#     return recv_data

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    print('pickle msg length: ', len(msg_pickle))
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.send(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())


def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    print('msg length: ', msg_len)
    # msg = sock.recv(msg_len)
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    print('length: ', len(msg))
    msg = pickle.loads(msg)
    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


