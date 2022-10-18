from torch_geometric.data import DataLoader
from spec_model import *
import random
from tqdm import tqdm
from torch import optim
from arg_helper import *
import os
import copy
from utils import *
import copy
import matplotlib.pyplot as plt

def train(config, seed):
    data = torch.load(config.data)
    set_seed(seed)
    random.shuffle(data)

    train_set = data[0:71]
    valid_set = data[71:81]
    test_set = data[81:]
    print(len(train_set), len(valid_set), len(test_set))


    train_loader = DataLoader(train_set, batch_size = 1, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size = 1)
    test_loader = DataLoader(test_set, batch_size = 1)

    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    model = eval(config.model)(config).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25, eta_min=1e-10)
    spec_edge = build_fully_connected_edge_idx(config.K).to('cuda:0')

    train_loss_history = []
    valid_loss_history = []
    best_loss = 10e10

    save_dir = os.path.join(config.save_dir, 'seed{0}'.format(seed))
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(300)):

        model.train()
        train_loss = 0
        valid_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data,spec_edge)

            weight = torch.bincount(data.y) / len(data.y)
            weight = 1 / weight
            weight = weight / weight.sum()

            criterion = torch.nn.CrossEntropyLoss(weight=weight)

            loss = criterion(out, data.y)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        scheduler.step() 
        
        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)

                out = model(data,spec_edge)

                weight = torch.bincount(data.y) / len(data.y)
                weight = 1 / weight
                weight = weight / weight.sum()

                criterion = torch.nn.CrossEntropyLoss(weight=weight)

                loss = criterion(out, data.y)
                valid_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        print(f'Epoch: {epoch:03d} Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}')

    torch.save(train_loss_history, os.path.join(save_dir, 'train_loss.txt'))
    torch.save(valid_loss_history, os.path.join(save_dir, 'valid_loss.txt'))

    best_model.eval()
    with torch.no_grad():
        test_dice = 0

        for data in test_loader:
            data = data.to(device)
            out = best_model(data, spec_edge)
            pred = out.argmax(dim=1)

            D = dice(pred, data.y)

            test_dice += D
        test_dice /= len(test_set)
    print(test_dice)

    plt.plot(train_loss_history)
    plt.plot(valid_loss_history)
    plt.plot(valid_loss_history.index(min(valid_loss_history)),min(valid_loss_history),marker="v",color='red')
    plt.title(f'test dice = {test_dice}')
    plt.savefig(os.path.join(save_dir,'test.jpg'))
    plt.close()

    return test_dice


