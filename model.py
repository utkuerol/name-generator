from os import name
import torch.nn as nn
import torch


class ClassifierModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, name_embedding_dim):
        super(ClassifierModel, self).__init__()
        self.hidden_size = hidden_size
        self.name_embedding = nn.Embedding(
            input_size, name_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=name_embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=False,
        )
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        name_embeds = self.name_embedding(input)
        name_embeds = name_embeds.unsqueeze(1)
        lstm_out, hidden = self.lstm(name_embeds, hidden)
        output = self.out(lstm_out[-1])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size), torch.zeros(2, 1, self.hidden_size))


class GeneratorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, name_embedding_dim, language_embedding_dim, num_languages):
        super(GeneratorModel, self).__init__()
        self.hidden_size = hidden_size
        self.name_embedding = nn.Embedding(
            input_size, name_embedding_dim)
        self.language_embedding = nn.Embedding(
            num_languages, language_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=name_embedding_dim + language_embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3,
            batch_first=False,
        )
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, language, hidden):
        name_embeds = self.name_embedding(input)
        language_embeds = self.language_embedding(language)
        input_combined = torch.cat(
            (name_embeds, language_embeds), 0).unsqueeze(0).unsqueeze(0)
        lstm_out, hidden = self.lstm(input_combined, hidden)
        output = self.out(lstm_out[-1])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size), torch.zeros(2, 1, self.hidden_size))
