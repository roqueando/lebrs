import pandas as pd
from pandas import DataFrame, Series
import item_calculator
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import tqdm

OUTPUT_FOLDER = 'data/'

BATCH_SIZE = 300
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
item_df = pd.read_json("../data/item_processed.json")

def work(row, chunk):
    for row in chunk.iterrows():
        items = item_calculator.choose_items(row[1].values, item_df)
        stats = item_calculator.update_stats(BASE_ONE, items)

        win = int(df.iloc[row[0]]['win'])
        new_dict = {
            'hp': stats[0],
            'mp': stats[1],
            'movespeed': stats[2],
            'armor': stats[3],
            'spellblock': stats[4],
            'attackrange': stats[5],
            'hpregen': stats[6],
            'mpregen': stats[7],
            'crit': stats[8],
            'attackdamage': stats[9],
            'attackspeed': round(stats[10], 2),
            'win': win
        }

def clean_matches(df: DataFrame) -> DataFrame:
    return df[['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'win']]

def choose(df: DataFrame) -> list:
    dict_frame = []
    for row in df.iterrows():
        items = item_calculator.choose_items(row[1].values, item_df)
        stats = item_calculator.update_stats(BASE_ONE, items)

        #win = int(df.iloc[row[0]]['win'])
        new_dict = {
            'hp': stats[0],
            'mp': stats[1],
            'movespeed': stats[2],
            'armor': stats[3],
            'spellblock': stats[4],
            'attackrange': stats[5],
            'hpregen': stats[6],
            'mpregen': stats[7],
            'crit': stats[8],
            'attackdamage': stats[9],
            'attackspeed': round(stats[10], 2),
            #'win': win
        }
        dict_frame.append(new_dict)
    return pd.DataFrame(dict_frame)


# transform and return X, Y (data and labels)
def transform_into_base_one_champs(df: DataFrame):
    data_list = []
    with Pool(processes=mp.cpu_count()) as pool:
        split = np.array_split(df, mp.cpu_count() * 2)
        split = tqdm.tqdm(split)
        ret_list = pool.map(choose, split)
        ret_list = tqdm.tqdm(ret_list)
        output = pd.concat(ret_list)

    return output

def main():
    matches_df = clean_matches(pd.read_csv("../data/stats1.csv"))
    print("transforming...")
    data = transform_into_base_one_champs(matches_df)
    data.to_csv('data_processed.csv', index=True)
    #print("saving X")
    #X.to_csv('data_processed.csv', index=False)

    #print("saving Labels")
    #Y.to_csv('label_processed.csv', index=False)

if __name__ == '__main__':
    main()
