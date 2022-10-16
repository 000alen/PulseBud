import pandas
import numpy
import torch
import torch.nn as nn
import torch.optim as optim


SIZE = 100
EPOCHS = 25


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        y_hat = self.l1(
            x,
        )
        h = torch.sigmoid(y_hat)
        y_hat = self.l2(h)
        y_hat = torch.sigmoid(y_hat)
        return y_hat, h

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size, dtype=torch.float32)


def ResampleLinear1D(x, target_size):
    x = numpy.array(x)
    index_array = numpy.linspace(0, len(x) - 1, num=target_size)
    index_floor = numpy.array(index_array, dtype=numpy.int64)
    index_ceil = index_floor + 1
    index_remaining = index_array - index_floor
    lower = x[index_floor]
    upper = x[index_ceil % len(x)]
    interpolation = lower * (1.0 - index_remaining) + upper * index_remaining
    return interpolation


dataframe = pandas.read_csv("ecg.csv")
model = Model(SIZE, SIZE, 1)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


i = 0
for epoch in range(EPOCHS):
    h = model.init_hidden()
    for _, (*x, y) in dataframe.iterrows():
        x = ResampleLinear1D(x, SIZE)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y).unsqueeze(0)
        y_hat, h = model(x, h)
        loss = criterion(y_hat, y)

        if i % 1000 == 0:
            print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        i += 1
