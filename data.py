import string
import unicodedata
import torch
from torch.utils.data import Dataset
import pandas as pd
import langid

origins = {
    "English": ["GB"],
    "Greek": ["GR"],
    "Germanic": ["DE", "AT", "SE", "NL", "NO", "BE", "DK"],
    "East-Asian": ["CN", "KP", "VN", "JP"],
    "Turkic": ["TR"],
    "Hispanic": ["ES", "PT", "MX", "BR"],
    "Slavic": ["CZ", "UA", "RU"],
    "French": ["FR"],
    "Italian": ["IT"],
    "Arabic": ["SA", "EG", "MA", "SY", "JO"],
    "Hebrew": ["IL"],
    "Japanese": ["JP"],
    "German": ["DE", "AT"],
    "Scandinavian": ["SE", "NO", "DK"]
}


def is_eng(s):
    if s["origin"] == "English":
        if langid.classify(s["name"])[0] == "en":
            return s
        else:
            s["name"] = ""
            return s
    else:
        return s


class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.all_letters = list(string.ascii_letters + " .,;'")
        self.all_letters += ['<SOS>', '<EOS>']
        self.data = self.prep_data(data_dir)
        self.languages = list(origins.keys())
        self.n_letters = len(self.all_letters)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.name_to_tensor(self.data.iloc[index, 0]), self.lang_to_tensor(
            self.data.iloc[index, 1]), self.gender_to_tensor(self.data.iloc[index, 2])

    def letter_to_index(self, letter):
        return self.all_letters.index(letter)

    def to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def prep_data(self, data_dir):
        file = open(data_dir + "names.csv", encoding="utf-8")
        df = pd.read_csv(file)
        df = df.iloc[:, [0, 1, 2]]
        df["name"] = df["name"].transform(lambda x: self.to_ascii(str(x)))
        df["name"] = df["name"].transform(lambda x: x.title())
        df = df.dropna()
        df = df[(df.name != "")]
        df = df[(df.code != "")]
        df = df[(df.gender != "")]
        df = df[(df.gender != "?")]

        new_df = pd.DataFrame(columns=["name", "origin", "gender"])
        for o in origins:
            selected = df[(df.code.isin(origins[o]))]
            selected["origin"] = o
            selected = selected.drop("code", 1)
            oversample_factor = int(
                len(df[(df.code.isin(origins["English"]))]) / len(selected))
            for _ in range(oversample_factor):
                new_df = new_df.append(selected)

        new_df = new_df.apply(is_eng, 1)
        new_df = new_df[(new_df.name != "")]
        new_df = new_df.sample(n=100000)
        return new_df

    def name_to_tensor(self, name):
        encoded = []
        for c in name:
            encoded.append(self.letter_to_index(c))
        return torch.LongTensor(encoded)

    def tensor_to_name(self, tensor):
        decoded = ""
        for i in range(tensor.size()[1]):
            encoded = tensor[0][i]
            decoded += self.all_letters[encoded]
        return decoded

    def lang_to_tensor(self, lang):
        lang_idx = self.languages.index(lang)
        return torch.LongTensor([lang_idx])

    def gender_to_tensor(self, gender):
        if gender == "M":
            return torch.LongTensor([0])
        else:
            return torch.LongTensor([1])
