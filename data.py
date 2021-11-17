import string
import unicodedata
import torch
from torch.utils.data import Dataset
import glob
import os


class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = glob.glob(data_dir + "*.txt")
        self.all_letters = list(string.ascii_letters + " .,;'")
        self.all_letters += ['<SOS>', '<EOS>']
        self.languages, self.names = self.prep_data(self.files)
        self.n_letters = len(self.all_letters)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        return self.name_to_tensor(self.names[index][0]), self.lang_to_tensor(
            self.names[index][1])

    def to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def read_lines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.to_ascii(line) for line in lines]

    def prep_data(self, files):
        languages = []
        names = []
        for filename in files:
            lang = os.path.splitext(os.path.basename(filename))[0]
            languages.append(lang)
            lines = self.read_lines(filename)
            names.extend([(line, lang) for line in lines])
        return languages, names

    def letter_to_index(self, letter):
        return self.all_letters.index(letter)

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
