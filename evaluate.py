import torch
from data import NamesDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
import sys 

model = torch.load("classifier.pt")
model.eval()

dataset = NamesDataset("../data/names/")


def evaluate(input):
    hidden = model.initHidden()
    output, hidden = model(input, hidden)

    return output


def predict(input, n_predictions=3):
    print('\n> %s' % input)
    with torch.no_grad():
        print(input)
        output = evaluate(dataset.name_to_tensor(input))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' %
                  (value, dataset.languages[category_index]))
            predictions.append(
                [value, dataset.languages[category_index]])


def categoryFromOutput(output):
    _, top_i = output.topk(1)
    category_i = top_i[0].item()
    return dataset.languages[category_i]


def main():
    # test_sampler = SubsetRandomSampler(
    #     range(len(dataset.names)))
    # test_loader = DataLoader(
    #     dataset, batch_size=1, sampler=test_sampler
    # )
    # with torch.no_grad():
    #     true = 0
    #     for i, (input, target) in enumerate(test_loader):
    #         if categoryFromOutput(evaluate(input[0])) == dataset.languages[target[0]]:
    #             true += 1
    #     print("%.0f%%" % (100 * float(true)/i+1))
    
    with torch.no_grad():
        predict(sys.argv[1])



if __name__ == "__main__":
    main()
