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

model = model.GeneratorModel(
    dataset.n_letters, 128, dataset.n_letters, 300, 300, len(dataset.languages))
criterion = nn.NLLLoss()
learning_rate = 0.0003
optimizer = optim.Adam(model.parameters(), learning_rate)

print_every = 5000
plot_every = 1000


def add_sos_to_input(input):
    sos = torch.LongTensor([dataset.letter_to_index('<SOS>')])
    res = torch.cat((sos, input), 0)
    return res


def add_eos_to_target(target):
    eos = torch.LongTensor([dataset.letter_to_index('<EOS>')])
    res = torch.cat((target, eos), 0)
    return res


def train(input, language, target):
    hidden = model.initHidden()
    optimizer.zero_grad()
    loss = 0
    target.unsqueeze_(-1)
    for i in range(input.size(0)):
        output, hidden = model(input[i], language[0], hidden)
        loss += criterion(output, target[i])
    loss.backward()
    optimizer.step()

    return output, loss.item() / input.size(0)


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
            output, loss = train(add_sos_to_input(
                input[0]), target[0], add_eos_to_target(input[0]))
            current_loss += loss

            if i % print_every == 0:
                print('%d %d%% (%s) %.4f' % (i, i / dataset.__len__() *
                                             100, timeSince(start), loss))

            if i % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()

    torch.save(model, "generator.pt")


run()
