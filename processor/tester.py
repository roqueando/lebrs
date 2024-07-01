from neural import LeBRS, device, LEARNING_RATE
from torch import optim
import torch
import pandas as pd
from main import split_matches, clean_matches

INPUT_EPOCH = 0
model = LeBRS().to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

checkpoint = torch.load(f"epoch_{INPUT_EPOCH}.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

model.eval()

with torch.no_grad():
    matches_df = pd.read_csv("../data/stats1.csv")
    x, y = split_matches(clean_matches(matches_df))

    x_test = torch.tensor(x.sample(frac=0.2, random_state=200).values) \
        .type(torch.float32) \
        .to(device)

    y_test = torch.tensor(y.sample(frac=0.2, random_state=200).values) \
        .type(torch.float32) \
        .to(device)
    zipped = list(zip(x_test, y_test))
    for X, y in zipped:
        pred = model(X)
        print(pred)
