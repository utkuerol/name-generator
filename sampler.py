import torch
from data import NamesDataset
import sys
import random
from pprint import pprint
import pycountry

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


def sample(lang, gender, var=1):
    with torch.no_grad():
        input = dataset.name_to_tensor("")
        input = add_sos_to_input(input)
        hidden = model.initHidden()
        lang_tensor = dataset.lang_to_tensor(lang)
        gender = dataset.gender_to_tensor(gender)
        output_name = ""

        for i in range(max_length):
            output, hidden = model(input[0], lang_tensor[0], gender[0], hidden)
            if i == 0:
                topv, topi = output.topk(30)
                topi = topi[0][random.randint(0, 20)]
            else:
                topv, topi = output.topk(10)
                topi = topi[0][random.randint(0, var)]

            if topi == dataset.n_letters - 1:
                break
            else:
                letter = dataset.all_letters[topi]
                output_name += letter
            input = dataset.name_to_tensor(letter)

        return output_name


def main():
    # lang = sys.argv[1]
    # gender = sys.argv[2]
    # var = int(sys.argv[3])

    for o in dataset.languages:
        names_m = set()
        while len(names_m) < 3000:
            first = sample(o, "M", 1)
            last = ""
            if len(first.split(" ")) < 2:
                last = sample(o, "M", 1)
            names_m.add(first.title() + " " + last.title())

        filename = "generated/new/" + o + "-M.txt"
        with open(filename, 'wt') as out:
            print(*names_m, sep="\n", file=out)

        names_f = set()
        while len(names_f) < 3000:
            first = sample(o, "F", 1)
            last = ""
            if len(first.split(" ")) < 2:
                last = sample(o, "F", 1)
            names_f.add(first.title() + " " + last.title())

        filename = "generated/new/" + o + "-F.txt"
        with open(filename, 'wt') as out:
            print(*names_f, sep="\n", file=out)


if __name__ == "__main__":
    main()
