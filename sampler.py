import torch
from data import NamesDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
import sys
import random
from pprint import pprint

model = torch.load("generator.pt")
model.eval()

dataset = NamesDataset("data/names/")
max_length = 20


def add_sos_to_input(input):
    sos = torch.LongTensor([dataset.letter_to_index('<SOS>')])
    res = torch.cat((sos, input), 0)
    return res


def add_eos_to_target(target):
    eos = torch.LongTensor([dataset.letter_to_index('<EOS>')])
    res = torch.cat((target, eos), 0)
    return res


def sample(lang, lang2, var=1):
    with torch.no_grad():
        input = dataset.name_to_tensor("")
        input = add_sos_to_input(input)
        hidden = model.initHidden()

        output_name = ""

        for i in range(max_length):
            lang_select = random.randint(0, 1)
            if lang_select == 0:
                lang_tensor = dataset.lang_to_tensor(lang)
            else:
                lang_tensor = dataset.lang_to_tensor(lang2)
            output, hidden = model(input[0], lang_tensor[0], hidden)
            if i == 0:
                topv, topi = output.topk(30)
                topi = topi[0][random.randint(0, 20)]
            else:
                topv, topi = output.topk(10)
                topi = topi[0][random.randint(0, var)]

            if topi == len(dataset.all_letters) - 1:
                break
            else:
                letter = dataset.all_letters[topi]
                output_name += letter
            input = dataset.name_to_tensor(letter)

        return output_name


def main():
    names = set()
    while len(names) < 1000:
        print(len(names))
        names.add(sample(sys.argv[1], sys.argv[2], int(sys.argv[3])) +
                  " " + sample(sys.argv[1], sys.argv[2], int(sys.argv[3])))

    filename = sys.argv[1] + "-" + sys.argv[2] + ".txt"
    with open(filename, 'wt') as out:
        pprint(names, stream=out)


if __name__ == "__main__":
    main()
