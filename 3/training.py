import torch as th
import numpy as np
from pathlib import Path

# Fajlove citamo/cuvamo pored ove skripte (u folderu 3).
BASE_DIR = Path(__file__).resolve().parent

x = np.load(BASE_DIR / 'observations.npy', allow_pickle=True)
y = np.load(BASE_DIR / 'actions.npy', allow_pickle=True)

# Ravnamo svako stanje (FullyObs slika 6x6x3) u vektor od 108 i prebacujemo u tenzore.
x = np.stack(list(x)).reshape(len(x), -1).astype(np.float64)
y = np.asarray(y, dtype=np.int64)

train_x = th.tensor(x)
train_y = th.tensor(y)

nb_train = len(train_y)

# Izbor uredjaja / akceleratora
if th.cuda.is_available():  # nVidia
    device = "cuda"
else:
    device = "cpu"

# Parametri mreze
lr = 0.001
nb_epochs = 100
batch_size = 32
nb_batches = max(1, nb_train // batch_size)

# Parametri arhitekture
nb_input = 108    # 6*6*3 (FullyObs slika okruzenja)
nb_hidden = 64    # broj neurona u skrivenim slojevima
nb_classes = 7    # broj akcija u MiniGrid okruzenju (Discrete(7))

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

# Odabir funkcije greske
loss_fn = th.nn.CrossEntropyLoss()

# Odabir optimizatora
optimizer = th.optim.Adam(params=model.parameters(), lr=lr)

# Izvrsavamo epohe treninga (epoha = prolaz kroz sve podatke).
for epoch in range(nb_epochs):

    epoch_loss = 0

    perm = th.randperm(nb_train)  # mesamo podatke u svakoj epohi

    for batch in range(nb_batches):

        bidx = perm[batch*batch_size : (batch+1)*batch_size]
        x = train_x[bidx].to(device)
        y = train_y[bidx].to(device)

        out = model.forward(x)

        # Racunanje greske
        loss = loss_fn(out, y)

        # Resetovanje starih gradijenata
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimizacija modela
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= nb_batches

    # U svakoj epohi ispisujemo prosecan loss.
    print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

# Cuvamo istreniran model.
th.save(model.state_dict(), BASE_DIR / 'model.pth')
print('Model sacuvan u model.pth')
