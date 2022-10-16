import pandas
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


SIZE = 100
EPOCHS = 10
LEARNING_RATE = 2e-5
MOMENTUM = 0.9
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1


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


def fit(id, dataframe, model, optimizer, criterion, writer):
    h = model.init_hidden()
    i = 0
    for _, (*x, y) in dataframe.iterrows():
        x = ResampleLinear1D(x, SIZE)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y).unsqueeze(0)
        y_hat, h = model(x, h)
        loss = criterion(y_hat, y)

        if i % 10 == 0:
            writer.add_scalar(f"{id}/loss", loss, i)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        i += 1


def evaluate(id, dataframe, model, writer):
    x = [x_i for _, (*x_i, _) in dataframe.iterrows()]
    x = [ResampleLinear1D(x_i, SIZE) for x_i in x]
    x = [torch.tensor(x_i, dtype=torch.float32) for x_i in x]
    y = [y_i for _, (*_, y_i) in dataframe.iterrows()]
    y = [torch.tensor(y_i, dtype=torch.float32) for y_i in y]

    total, correct = 0, 0
    for x_i, y_i in zip(x, y):
        y_hat, _ = model(x_i, model.init_hidden())
        y_hat = torch.round(y_hat)
        total += 1
        correct += int(y_hat == y_i)
    accuracy = correct / total
    writer.add_scalar(f"{id}/accuracy", accuracy, 0)


# alternative
#tf.math.confusion_matrix([real_labels], [predictions])


dataframe = pandas.read_csv("ecg.csv")
train_dataframe = dataframe.sample(frac=TRAIN_SPLIT, random_state=0)
test_dataframe = dataframe.drop(train_dataframe.index)
validation_dataframe = test_dataframe.sample(
    frac=VALIDATION_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT), random_state=0
)
test_dataframe = test_dataframe.drop(validation_dataframe.index)

model = Model(SIZE, SIZE, 1)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
writer = SummaryWriter()

TOTAL = EPOCHS * len(dataframe)
for epoch in range(EPOCHS):
    # fit("train", train_dataframe, model, optimizer, criterion, writer)
    evaluate("test", test_dataframe, model, writer)
