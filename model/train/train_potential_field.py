import argparse
import os

import torch
import numpy as np

from model.model import two_agent_orig
from model.model import POLICY_DIM
from model.train.generate_data import generate_potential_field_sample
from model.train.trainer import train


def parse_arguments():

    parser = argparse.ArgumentParser(description='Train a NNC using a potential field')
    parser.add_argument('--hidden_dim_1', type=int, default=10, help="first hidden dim of controller")
    parser.add_argument('--hidden_dim_2', type=int, default=10, help='second hidden dim of controller')
    parser.add_argument('--save_path', type=str, default="./model/checkpoints/potential_field", help="directory to save the checkpoint")
    parser.add_argument('--total_agent_num', type=int, default=10, help="total number of agents in the system")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size during training")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate of training")
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay used in optimizer')
    parser.add_argument('--epoch', type=int, default=20, help='number of epoch for training')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_arguments()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    state_space_lbs = [-10, -10, -1, -1] * args.total_agent_num
    state_space_ubs = [10, 10, 1, 1] * args.total_agent_num
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    BATCH_SIZE = args.batch_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    MAX_EPOCH = args.epoch

    for current_agent_id in range(args.total_agent_num):

        agent = two_agent_orig(
            HIDDEN1_DIM=args.hidden_dim_1, 
            HIDDEN2_DIM=args.hidden_dim_2, 
            n=args.total_agent_num
        ).to(device)

        X, Y = generate_potential_field_sample(
            state_space_lbs,
            state_space_ubs,
            args.total_agent_num,
            current_agent_id
        )

        train(
            agent,
            X, Y,
            BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCH,
            args.save_path,
            'agent_' + str(current_agent_id),
            device
        )