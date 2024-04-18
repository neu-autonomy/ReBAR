import argparse
import os

import torch
import numpy as np

from model.model import two_agent_orig
from model.model import POLICY_DIM
from model.train.generate_data import generate_sample
from model.train.trainer import train


def parse_arguments():

    parser = argparse.ArgumentParser(description='Train a NNC for collision avoidance using DDPG')
    parser.add_argument('--goal_1_x', type=float, default=9.0, help="Agent 1's goal x coord. [0, 10]")
    parser.add_argument('--goal_1_y', type=float, default=5.0, help="Agent 1's goal y coord. [0, 10]")
    parser.add_argument('--goal_2_x', type=float, default=5.0, help="Agent 2's goal x coord. [0, 10]")
    parser.add_argument('--goal_2_y', type=float, default=9.0, help="Agent 2's goal y coord. [0, 10]")
    parser.add_argument('--save_path', type=str, default="./model/checkpoints/single_integ", help="directory to save the checkpoint")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size during training")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate of training")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay used in optimizer')
    parser.add_argument('--epoch', type=int, default=20, help='number of epoch for training')
    parser.add_argument('--double_integ', action="store_true")
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_arguments()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    state_space_lbs = [0, 0, -1, -1, 0, 0, -1, -1]
    state_space_ubs = [10, 10, 1, 1, 10, 10, 1, 1]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    BATCH_SIZE = args.batch_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    MAX_EPOCH = args.epoch

    X_1, Y_1, X_2, Y_2 = generate_sample(
        state_space_lbs, 
        state_space_ubs, 
        [1.0, 1.0],
        POLICY_DIM,
        [args.goal_1_x, args.goal_1_y],
        [args.goal_2_x, args.goal_2_y],
        double_integ = args.double_integ
    )

    agent_1 = two_agent_orig().to(device)
    agent_2 = two_agent_orig().to(device)

    train(
        agent_1,
        X_1, Y_1,
        BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCH, 
        args.save_path,
        'avoid_agent_1',
        device
    )

    train(
        agent_2,
        X_2, Y_2,
        BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCH, 
        args.save_path,
        'avoid_agent_2',
        device
    )