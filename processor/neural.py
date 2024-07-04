import torch
from torch import nn
import math
import item_calculator
from item_calculator import update_stats, choose_items
from more_itertools import chunked

device = "xpu"
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
LEARNING_RATE = 1e-3
EPOCHS = 2
MAX_ITEMS = 447113.0
MIN_ITEMS = 1001.0
BATCH_SIZE = 5000

def exilu(x):
    return x if x >= MIN_ITEMS and x <= MAX_ITEMS else 0.0

class ExiLU(nn.Module):
    '''Applies Existent Linear Unit activation for existent items'''
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor):
        return tensor.detach().apply_(lambda x: exilu(x))

class LeBRS(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(6, 8),
            #ExiLU(),
            nn.ReLU(),
            nn.Linear(8, 8),
            #ExiLU(),
            nn.ReLU(),
            nn.Linear(8, 6),
            #ExiLU(),
            nn.ReLU(),
        )

    def forward(self, x):
        items = self.linear_stack(x)
        return items

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
                li = l.item()
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
    }, path)
def stat_loss_fn(y_pred, y, item_df):
    y_tensor = torch.tensor(update_stats(BASE_ONE, choose_items(y.tolist(), item_df)), requires_grad=True)
    y_pred_tensor = torch.tensor(update_stats(BASE_ONE, choose_items(y_pred.tolist(), item_df)), requires_grad=True)

    loss = nn.MSELoss()
    result = loss(y_pred_tensor, y_tensor)
    return result
