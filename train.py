import torch.optim as optim
import model
import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

dataset = data.NamesDataset("../data/names/")
train_sampler = SubsetRandomSampler(
        range(len(dataset.names)))
        
train_loader = DataLoader(
    dataset, batch_size=1, sampler=train_sampler
)

model = model.ClassifierModel(
    dataset.n_letters, 128, len(dataset.languages), 300)
criterion = nn.NLLLoss()
learning_rate = 0.0003
optimizer = optim.Adam(model.parameters(), learning_rate)

print_every = 5000
plot_every = 1000


def categoryFromOutput(output):
    _, top_i = output.topk(1)
    category_i = top_i[0].item()
    return dataset.languages[category_i]


def train(input, target):
    hidden = model.initHidden()
    optimizer.zero_grad()
    output, hidden = model(input, hidden)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def run():
    all_losses = []
    start = time.time()
    n_epochs = 3
    current_loss = 0
    for e in range(n_epochs):
        for i, (input, target) in enumerate(train_loader):
            output, loss = train(input[0], target[0])
            current_loss += loss
            if i % print_every == 0:
                guess = categoryFromOutput(output)
                truth = dataset.languages[target[0]]
                correct = '✓' if guess == truth else '✗ (%s)' % truth
                print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / dataset.__len__() *
                                                        100, timeSince(start), loss, dataset.tensor_to_name(input), guess, correct))
            if i % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()

    torch.save(model, "classifier.pt")


run()
