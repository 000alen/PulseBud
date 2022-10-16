# %%
import pandas
import numpy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from random import shuffle


# %%
table = list([float(_) for _ in row] for row in csv.reader(open("ecg.csv", "r")))
normal = list(map(lambda row: row[:-1], filter(lambda row: row[-1] == 0, table)))
abnormal = list(map(lambda row: row[:-1], filter(lambda row: row[-1] == 1, table)))
dataset = [*[(0, _) for _ in normal], *[(1, _) for _ in abnormal]]
shuffle(dataset)


# %%
class Block(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size).float()
        self.c2o = nn.Linear(input_size + hidden_size, output_size).float()

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.sigmoid(self.i2h(combined.float()))
        output = self.c2o(combined)
        return output, hidden

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))


# %%
def ResampleLinear1D(original, targetLen):
    original = numpy.array(original, dtype=numpy.float32)
    index_arr = numpy.linspace(0, len(original) - 1, num=targetLen, dtype=numpy.float32)
    index_floor = numpy.array(index_arr, dtype=numpy.int64)  # Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor  # Remain
    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0 - index_rem) + val2 * index_rem
    assert len(interp) == targetLen
    return interp


# %%
EPOCHS = 1
RESAMPLING_SIZE = 100

model = Block(RESAMPLING_SIZE, RESAMPLING_SIZE, 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for epoch in range(EPOCHS):
    running_loss = 0.0
    hidden_state = model.init_hidden()
    for i, (label, inputs) in enumerate(dataset):
        print(label)

        inputs = ResampleLinear1D(inputs, RESAMPLING_SIZE)
        optimizer.zero_grad()
        outputs, hidden_state = model(torch.tensor(inputs).reshape(1, RESAMPLING_SIZE), hidden_state)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")


# %%



