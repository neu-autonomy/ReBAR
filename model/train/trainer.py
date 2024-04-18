import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


'''
Train the model with the given data with given hyperparams
'''
def train(
    model,
    X, Y,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    MAX_EPOCH,
    save_path,
    agent_name,
    device = torch.device('cuda:0')
):
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1)

    trainingloader = DataLoader(
        TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)),
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    testloader = DataLoader(
        TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)),
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss(reduce="mean")

    train_loss_list = []
    val_loss_list = []

    for epoch in range(MAX_EPOCH):

        print("epoch %d / %d" % (epoch+1, MAX_EPOCH))
        model.train()

        temp_loss_list = list()
        for x, y in tqdm(trainingloader):
            x = x.type(torch.float32).to(device).view(x.shape[0], -1)
            y = y.type(torch.float32).to(device).view(y.shape[0], -1)

            optimizer.zero_grad()

            score = model(x)
            loss = criterion(input=score, target=y)
            loss.backward()

            optimizer.step()

            temp_loss_list.append(loss.detach().cpu().numpy())

        train_loss_list.append(np.average(temp_loss_list))

        # validation
        model.eval()

        temp_loss_list = list()
        for x, y in tqdm(testloader):
            x = x.type(torch.float32).to(device).view(x.shape[0], -1)
            y = y.type(torch.float32).to(device).view(y.shape[0], -1)

            score = model(x)
            loss = criterion(input=score, target=y)

            temp_loss_list.append(loss.detach().cpu().numpy())

        val_loss_list.append(np.average(temp_loss_list))

        print("\ttrain loss: %.5f" % train_loss_list[-1])
        print("\tval loss: %.5f" % val_loss_list[-1])
        # print(y_val, score)
        if epoch % 5 == 0 or epoch == MAX_EPOCH - 1:
            filename = save_path + '/' + agent_name + '_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), filename)