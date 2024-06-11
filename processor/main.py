import pandas as pd
from pandas import DataFrame
from typing import Tuple
from dataclasses import dataclass
import item_calculator
import torch
from torch import nn

#device = (
#    "cuda"
#    if torch.cuda.is_available()
#    else "mps"
#    if torch.backends.mps.is_available()
#    else "cpu"
#)
device = "cpu"
print(f"Using {device} device")

# HYPER PARAMETERS
LEARNING_RATE = 1e-3
EPOCHS = 10
MAX_ITEMS = 447113
MIN_ITEMS = 1001
BATCH_SIZE = 64

class ExiLU(nn.Module):
    '''Applies Existent Linear Unit activation for existent items'''
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor):
        return tensor.detach().apply_(lambda x: x if x >= MIN_ITEMS or x <= MAX_ITEMS else 0)

class LeBRS(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(6, 8),
            ExiLU(),
            nn.Linear(8, 8),
            ExiLU(),
            nn.Linear(8, 6),
        )

    def forward(self, x):
        items = self.linear_stack(x)
        return items

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

def update_stats(boc: dict, items: list[item_calculator.Effect]) -> dict:
    updated_boc = {**boc}
    for item in items:
        if item.operator == item_calculator.EffectMod.ADD:
            updated_boc[item.stat_field] = boc[item.stat_field] + item.value
        if item.operator == item_calculator.EffectMod.TIMES:
            updated_boc[item.stat_field] = boc[item.stat_field] * item.value

    return updated_boc

def split_matches(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    x = df.loc[df['win'] == 0]
    y = df.loc[df['win'] == 1]

    x = x[['item1', 'item2', 'item3', 'item4', 'item5', 'item6']].astype(float)
    y = y[['item1', 'item2', 'item3', 'item4', 'item5', 'item6']].astype(float)

    return x, y

def clean_matches(df: DataFrame) -> DataFrame:
    return df[['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'win']]

def choose_items(items: list, df: DataFrame) -> list[item_calculator.Effect]:
    effects = []
    for item in items:
        item_stats = df.loc[df['id'] == item]['stats']
        if len(item_stats.values) > 0:
            for key, value in item_stats.values[0].items():
                effects.append(item_calculator.get_item_effect(key, value))
    return effects

def to_ndarray(boc: dict) -> list:
    arr = []
    for _, value in boc.items():
        arr.append(value)

    return arr

def train_loop(data, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(data):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            li = loss.item()
            print(f"loss: {li:>7f} ")

def main():
    item_df = pd.read_json("../data/item_processed.json")
    item_df = item_df[['id', 'name', 'stats']]

    matches_df = pd.read_csv("../data/stats1.csv")
    x, y = split_matches(clean_matches(matches_df))
    model = LeBRS().to(device)

    x_train = torch.tensor(x.sample(frac=0.8, random_state=200).values) \
        .type(torch.float32) \
        .to(device)

    x_test = torch.tensor(x.sample(frac=0.2, random_state=200).values) \
        .type(torch.float32) \
        .to(device)

    y_train = torch.tensor(y.sample(frac=0.8, random_state=200).values) \
        .type(torch.float32) \
        .to(device)
    y_test = torch.tensor(y.sample(frac=0.2, random_state=200).values) \
        .type(torch.float32) \
        .to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for t in range(EPOCHS):
        print(f"Epoch: {t+1}\n--------------------------")
        train_loop(list(zip(x_train, y_train)), model, loss_fn, optimizer)

    print("done!")
    # TODO: THIS WILL COME, WAIT FOR THE GLORIOUS DAY
    chosen_items = choose_items([1001, 1054, 3047, 3067, 2045, 2001], item_df)
    base_one = update_stats(BASE_ONE, chosen_items)


if __name__ == '__main__':
    main()

