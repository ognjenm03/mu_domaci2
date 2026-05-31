import numpy as np
import torch as th
import pandas as pd
from pathlib import Path

np.random.seed(42)
th.manual_seed(42)

DATA_PATH = Path(__file__).resolve().parent.parent / 'crop.csv'
df = pd.read_csv(DATA_PATH)

X = df.drop('Crop', axis=1).values.astype(np.float64)
labels = df['Crop'].values
classes = np.unique(labels)
y = np.searchsorted(classes, labels)

idx = np.random.permutation(len(y))
X, y = X[idx], y[idx]
split = int(0.8 * len(y))
train_x, test_x = X[:split], X[split:]
train_y, test_y = y[:split], y[split:]

mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

nb_train = len(train_y)
nb_test = len(test_y)
print(nb_train, nb_test)

if th.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

lr = 0.001
nb_epochs = 100
batch_size = 64
nb_batches = int(nb_train / batch_size)

nb_input = train_x.shape[1]
nb_hidden1 = 64
nb_hidden2 = 32
nb_classes = len(classes)

w = {
    '1': th.randn([nb_input, nb_hidden1], dtype=th.float64, requires_grad=True, device=device),
    '2': th.randn([nb_hidden1, nb_hidden2], dtype=th.float64, requires_grad=True, device=device),
    'out': th.randn([nb_hidden2, nb_classes], dtype=th.float64, requires_grad=True, device=device)
}

b = {
    '1': th.zeros([nb_hidden1], dtype=th.float64, requires_grad=True, device=device),
    '2': th.zeros([nb_hidden2], dtype=th.float64, requires_grad=True, device=device),
    'out': th.zeros([nb_classes], dtype=th.float64, requires_grad=True, device=device)
}

f = {
    '1': th.nn.ReLU(),
    '2': th.nn.ReLU(),
    'out': th.softmax
}

th.nn.init.kaiming_uniform_(w['1'], a=th.math.sqrt(5), mode='fan_in', nonlinearity='relu')
th.nn.init.kaiming_uniform_(w['2'], a=th.math.sqrt(5), mode='fan_in', nonlinearity='relu')
th.nn.init.kaiming_uniform_(w['out'], a=th.math.sqrt(5), mode='fan_in', nonlinearity='relu')

def forward(x):
    z1 = x @ w['1'] + b['1']
    a1 = f['1'](z1)
    z2 = a1 @ w['2'] + b['2']
    a2 = f['2'](z2)
    z_out = a2 @ w['out'] + b['out']
    out = f['out'](z_out, dim=1)

    pred = th.argmax(out, 1)

    return pred, out

for epoch in range(nb_epochs):

    epoch_loss = 0

    for batch in range(nb_batches):

        x = th.tensor(train_x[batch*batch_size : (batch+1)*batch_size, :], device=device)
        y = th.tensor(train_y[batch*batch_size : (batch+1)*batch_size], device=device)
        y_onehot = th.nn.functional.one_hot(y, nb_classes).double()

        pred, out = forward(x)

        hyp_masked = y_onehot * th.log(out + 1e-8)
        loss = -hyp_masked.sum()

        loss.backward()

        with th.no_grad():
            w['1'] -= lr * w['1'].grad
            w['2'] -= lr * w['2'].grad
            w['out'] -= lr * w['out'].grad
            b['1'] -= lr * b['1'].grad
            b['2'] -= lr * b['2'].grad
            b['out'] -= lr * b['out'].grad

        w['1'].grad.zero_()
        w['2'].grad.zero_()
        w['out'].grad.zero_()
        b['1'].grad.zero_()
        b['2'].grad.zero_()
        b['out'].grad.zero_()

        epoch_loss += loss.item()

    epoch_loss /= nb_batches

    print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

x = th.tensor(test_x, device=device)
y = th.tensor(test_y, device=device)

pred, _ = forward(x)
pred_correct = th.eq(pred, y).float()
accuracy = th.mean(pred_correct)

print(f'Test acc: {accuracy*100:.2f}%')
