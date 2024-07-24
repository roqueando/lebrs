import numpy as np
import pandas as pd
from pandas import DataFrame

# hp, mp, atk
inputs = np.array([10.0, 12.0, 6.0])
h1_fixed_ws = np.array([0.495, 0.862, 0.801, 0.659, 0.193])
b = 1

hl1 = []
for i in inputs:
    hl1.append((i * h1_fixed_ws + b).sum())

df = pd.read_csv("../data/stats1.csv")

# Help Functions
def separate_line():
    print("="*50)

def clean_unused_columns(df: DataFrame) -> DataFrame:
    return df[['item1','item2', 'item3', 'item4', 'item5', 'item6', 'totdmgdealt', 'magicdmgdealt', 'physicaldmgdealt', 'totdmgtaken', 'magicdmgtaken', 'physdmgtaken', 'champlvl']]

debug_data = df.sample(frac=0.0001, random_state=200)
print(debug_data.head())
separate_line()
print(f"total size: {debug_data.size} rows")
separate_line()

winners = clean_unused_columns(debug_data.loc[debug_data['win'] == 0]) # will be our Y
losers = clean_unused_columns(debug_data.loc[debug_data['win'] == 1]) # will be our X

print(f"winners: \n{winners.head()}")
print(f"losers: \n{losers.head()}")
separate_line()

