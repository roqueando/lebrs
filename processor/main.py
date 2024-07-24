import pandas as pd
from pandas import DataFrame
from typing import Tuple
from dataclasses import dataclass
import torch
from torch import nn
import math

import item_calculator
import process_items
import neural

def split_matches(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    x = df.loc[df['win'] == 0]
    y = df.loc[df['win'] == 1]
    #stats = ['hp','mp','movespeed','armor','spellblock','attackrange','hpregen','mpregen','crit','attackdamage','attackspeed']
    stats = ['item1','item2','item3','item4','item5','item6']


    x = x[stats].astype(float)
    y = y[stats].astype(float)

    return x, y

def to_ndarray(boc: dict) -> list:
    arr = []
    for _, value in boc.items():
        arr.append(value)

    return arr

def main():
    item_df = pd.read_json("../data/item_processed.json")
    item_df = item_df[['id', 'name', 'stats']]

    matches_df = pd.read_csv("../data/stats1.csv")
    x, y = split_matches(process_items.clean_matches(matches_df))

    model = neural.LeBRS().to(neural.device)
    #model = ipex.optimize(model, torch.float32)

    x_train = torch.tensor(x.sample(frac=0.1, random_state=200).values) \
        .type(torch.float32) \
        .to(neural.device)

    x_test = torch.tensor(x.sample(frac=0.2, random_state=200).values) \
        .type(torch.float32) \
        .to(neural.device)

    y_train = torch.tensor(y.sample(frac=0.1, random_state=200).values) \
        .type(torch.float32) \
        .to(neural.device)
    y_test = torch.tensor(y.sample(frac=0.2, random_state=200).values) \
        .type(torch.float32) \
        .to(neural.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=neural.LEARNING_RATE)

    print('========== DEBUG =========')

    for t in range(neural.EPOCHS):
        print(f"Epoch: {t+1}\n--------------------------")
        neural.train_loop(list(zip(x_train, y_train)), model, optimizer, item_df, t)
        #neural.test_loop(list(zip(x_test, y_test)), model, item_df, t)

    print('==========================')
    print("done!")


if __name__ == '__main__':
    main()

