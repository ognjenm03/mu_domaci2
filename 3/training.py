import torch as th
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

x = np.load(BASE_DIR / 'observations.npy', allow_pickle=True)
y = np.load(BASE_DIR / 'actions.npy', allow_pickle=True)

x = np.stack(list(x)).reshape(len(x), -1).astype(np.float64)
y = np.asarray(y, dtype=np.int64)

train_x = th.tensor(x)
train_y = th.tensor(y)

nb_train = len(train_y)

if th.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

lr = 0.001
nb_epochs = 100
batch_size = 32
nb_batches = max(1, nb_train // batch_size)

nb_input = 108
nb_hidden = 64
nb_classes = 7

class SimpleNet(th.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = th.nn.Linear(nb_input, nb_hidden, dtype=th.float64)
        self.fc2 = th.nn.Linear(nb_hidden, nb_hidden, dtype=th.float64)
        self.out = th.nn.Linear(nb_hidden, nb_classes, dtype=th.float64)

    def forward(self, x):
        x = th.nn.functional.relu(self.fc1(x))
        x = th.nn.functional.relu(self.fc2(x))
        return self.out(x)

model = SimpleNet().to(device)

model.train()

loss_fn = th.nn.CrossEntropyLoss()

optimizer = th.optim.Adam(params=model.parameters(), lr=lr)

for epoch in range(nb_epochs):

    epoch_loss = 0

    perm = th.randperm(nb_train)

    for batch in range(nb_batches):

        bidx = perm[batch*batch_size : (batch+1)*batch_size]
        x = train_x[bidx].to(device)
        y = train_y[bidx].to(device)

        out = model.forward(x)

        loss = loss_fn(out, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= nb_batches

    print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

th.save(model.state_dict(), BASE_DIR / 'model.pth')
print('Model sacuvan u model.pth')
