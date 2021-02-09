from pathlib import Path
import logging

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn
import torch.utils.data as tud
import torchvision
import torchvision.transforms as transforms

from .args import ArgumentParserBuilder, opt
from howl.settings import SETTINGS
from howl.model import RegisteredModel, Workspace
from howl.utils.random import set_seed


def expand(x):
    return x.expand(3, -1, -1)


def main():
    def evaluate():
        model.eval()
        num_correct = 0
        num_total = 0
        for inputs, labels in tqdm(test_dl, total=len(test_dl)):
            inputs = inputs.to(device)
            labels = torch.tensor([x % 10 for x in labels.tolist()])
            labels = labels.to(device)
            scores = model(inputs, None)
            num_correct += (scores.max(1)[1] == labels).float().sum().item()
            num_total += scores.size(0)
        logging.info(f'{num_correct / num_total}')
        ws.increment_model(model, num_correct / num_total / 100)

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--model', type=str, choices=RegisteredModel.registered_names(), default='las'),
                    opt('--workspace', type=str, default=str(Path('workspaces') / 'default')),
                    opt('--load-weights', action='store_true'))
    args = apb.parser.parse_args()

    ws = Workspace(Path(args.workspace))
    writer = ws.summary_writer
    set_seed(SETTINGS.training.seed)

    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset1 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset1 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    trainset2 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset2 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    expand])
    test_transform = transforms.Compose([transforms.Pad((2, 2)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,)),
                                         expand])
    trainset3 = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset3 = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)
    train_dl = tud.DataLoader(tud.ConcatDataset([trainset1, trainset2, trainset3]),
                              batch_size=SETTINGS.training.batch_size,
                              shuffle=True)
    test_dl = tud.DataLoader(tud.ConcatDataset([testset1, testset2, testset3]),
                             batch_size=SETTINGS.training.batch_size,
                             shuffle=False)

    device = torch.device(SETTINGS.training.device)
    model = RegisteredModel.find_registered_class(args.model)().to(device)
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, SETTINGS.training.learning_rate, weight_decay=SETTINGS.training.weight_decay)
    logging.info(f'{sum(p.numel() for p in params)} parameters')
    criterion = nn.CrossEntropyLoss()

    ws.write_args(args)
    ws.write_settings(SETTINGS)
    writer.add_scalar('Meta/Parameters', sum(p.numel() for p in params))
    for epoch_idx in trange(SETTINGS.training.num_epochs, position=0, leave=True):
        model.train()
        pbar = tqdm(train_dl,
                    total=len(train_dl),
                    position=1,
                    desc='Training',
                    leave=True)
        for inputs, labels in pbar:
            optimizer.zero_grad()
            model.zero_grad()
            labels = torch.tensor([x % 10 for x in labels.tolist()])
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = model(inputs, None)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))
            writer.add_scalar('Training/Loss', loss.item(), epoch_idx)

        for group in optimizer.param_groups:
            group['lr'] *= 0.9
        evaluate()


if __name__ == '__main__':
    main()
