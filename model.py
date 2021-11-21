import torch.nn as nn
import torch


class GeneratorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, name_embedding_dim, lang_embedding_dim, gender_embedding_dim, num_languages):
        super(GeneratorModel, self).__init__()
        self.hidden_size = hidden_size
        self.name_embedding = nn.Embedding(
            input_size, name_embedding_dim)
        self.language_embedding = nn.Embedding(
            num_languages, lang_embedding_dim)
        self.gender_embedding = nn.Embedding(2, gender_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=name_embedding_dim + lang_embedding_dim + gender_embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=False,
        )
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, language, gender, hidden):
        name_embeds = self.name_embedding(input)
        language_embeds = self.language_embedding(language)
        gender_embeds = self.gender_embedding(gender)
        input_combined = torch.cat(
            (name_embeds, language_embeds, gender_embeds), 0).unsqueeze(0).unsqueeze(0)
        lstm_out, hidden = self.lstm(input_combined, hidden)
        output = self.out(lstm_out[-1])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size), torch.zeros(2, 1, self.hidden_size))
