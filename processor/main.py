import pandas as pd
from pandas import DataFrame
from typing import Tuple
from dataclasses import dataclass
import torch
from torch import nn
import math

import item_calculator
import neural

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
            if item_stats.values[0] is not None:
                for key, value in item_stats.values[0].items():
                    effects.append(item_calculator.get_item_effect(key, value))
    return effects

def to_ndarray(boc: dict) -> list:
    arr = []
    for _, value in boc.items():
        arr.append(value)

    return arr

def stat_loss_fn(y_pred, y, item_df):
    pred_mapped = y_pred.detach().apply_(lambda x: math.floor(x))

    t = y.detach().apply_(lambda x: )
    #t = map(lambda x: update_stats(BASE_ONE, choose_items(x.tolist(), item_df)), y)
    #y_choose_items = choose_items(y[0].tolist(), item_df)
    #y_base_one = update_stats(BASE_ONE, y_choose_items)
    print(t)

def main():
    item_df = pd.read_json("../data/item_processed.json")
    item_df = item_df[['id', 'name', 'stats']]

    matches_df = pd.read_csv("../data/stats1.csv")
    x, y = split_matches(clean_matches(matches_df))
    model = neural.LeBRS().to(device)

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

    print('========== DEBUG =========')
    pred = model(x_train)
    neural.stat_loss_fn(pred, y_train, item_df)

    #for t in range(EPOCHS):
        #print(f"Epoch: {t+1}\n--------------------------")
        #train_loop(list(zip(x_train, y_train)), model, loss_fn, optimizer)

    #print("done!")
    #chosen_items = choose_items([1001, 1054, 3047, 3067, 2045, 2001], item_df)
    #base_one = update_stats(BASE_ONE, chosen_items)


if __name__ == '__main__':
    main()

