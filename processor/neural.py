import torch
from torch import nn
import math
import item_calculator
from item_calculator import update_stats, choose_items
from more_itertools import chunked

device = "cpu"
print(f"Using {device} device")

BASE_ONE = {
    'hp': 1,
    'mp': 1,
    'movespeed': 1,
    'armor': 1,
    'spellblock': 1,
    'attackrange': 1,
    'hpregen': 1,
    'mpregen': 1,
    'crit': 1,
    'attackdamage': 1,
    'attackspeed': 1.0
}

# HYPER PARAMETERS
LEARNING_RATE = 1e-3 # 0.0001
EPOCHS = 2
MAX_ITEMS = 447113.0
MIN_ITEMS = 1001.0
BATCH_SIZE = 1000

def exilu(x):
    return x if x >= MIN_ITEMS and x <= MAX_ITEMS else 1001.0

class ExiLU(nn.Module):
    '''Applies Existent Linear Unit activation for existent items'''
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor):
        return tensor.detach().apply_(lambda x: exilu(x))

class LeBRS(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embedding = nn.Embedding(num_embeddings=446112, embedding_dim=256)
        self.fully_connected_1 = nn.Linear(6, 8, dtype=torch.long)
        self.fully_connected_2 = nn.Linear(8, 6)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.exilu = ExiLU()
        self.linear_stack = nn.Sequential(
            nn.Linear(6, 8),
            #nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 6),
            #ExiLU(),
            #nn.ReLU(),
            ExiLU()
        )

    def forward(self, x):
        x = x.to(torch.int)
        item_embedded = self.item_embedding(x)
        x = self.relu(self.fully_connected_1(item_embedded))
        x = self.dropout(x)
        output = self.fully_connected_2(x)
        #items = self.linear_stack(x)
        return output

def train_loop(data, model, optimizer, item_df, epoch):
    chunks = list(chunked(data, BATCH_SIZE))
    model.train()
    batch_count = 0

    for chunk in chunks:
        for batch, (X, y) in enumerate(chunk):
            pred = model(X)
            l = stat_loss_fn(pred, y, item_df)
            l.backward()

            optimizer.step()
            optimizer.zero_grad()

            if batch == BATCH_SIZE - 1:
                print(f"X: {X}")
                print("---------------------")
                li = l.item()
                print(f"loss: {l.item}")
                print("---------------------")
                with open(f'loss_{epoch}.txt', 'a') as f:
                    f.write(f"{li:>2f}\n")
                batch_count = batch_count + 1
                print(f"BATCH [{batch_count}/{len(chunks)}] COMPLETED")
                print(f"================================")
                print(f"LOSS: {li:>7f}")
                print(f"PRED: {pred}")
                print(f"================================")
    path = f"epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': l.item()
    }, path)

def test_loop(data, model, item_df, epoch):
    chunks = list(chunked(data, BATCH_SIZE))
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for chunk in chunks:
            for batch, (X, y) in enumerate(chunk):
                pred = model(X)
                test_loss += stat_loss_fn(pred, y, item_df).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(chunks)
    correct /= len(chunks)
    print(f"Test error: \nAccuracy: {100*correct:>0.1f},\nAvg loss: {test_loss:>8f}\n")

def stat_loss_fn(y_pred, y, item_df):
    y_pred_tensor = torch.tensor(update_stats(BASE_ONE, choose_items(y_pred.tolist(), item_df)), requires_grad=True)
    y_tensor = torch.tensor(update_stats(BASE_ONE, choose_items(y.tolist(), item_df)), requires_grad=True)

    # previous: CrossEntropyLoss
    loss = nn.HuberLoss() # EUREKA: PROBABLY IS THIS
    result = loss(y_pred_tensor, y_tensor)
    return result
