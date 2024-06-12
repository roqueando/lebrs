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

def train_loop(data, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(data):
        pred = model(X)
        # stat_loss_fn(pred, y)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            li = loss.item()
            print(f"loss: {li:>7f} ")

def stat_loss_fn(y_pred, y, item_df):
    pred_mapped = y_pred.detach().apply_(lambda x: math.floor(x))

    t = y.detach().apply_(lambda x: )
    #t = map(lambda x: update_stats(BASE_ONE, choose_items(x.tolist(), item_df)), y)
    #y_choose_items = choose_items(y[0].tolist(), item_df)
    #y_base_one = update_stats(BASE_ONE, y_choose_items)
    print(t)
