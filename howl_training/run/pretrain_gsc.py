from collections import Counter
from functools import partial
from pathlib import Path
import logging

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn

from .args import ArgumentParserBuilder, opt
from howl.data.dataset import GoogleSpeechCommandsDatasetLoader
from howl.data.dataloader import StandardAudioDataLoaderBuilder
from howl.data.transform import compose, ZmuvTransform, StandardAudioTransform,\
    NoiseTransform, batchify, TimeshiftTransform, truncate_length
from howl.settings import SETTINGS
from howl.model import RegisteredModel, Workspace
from howl.utils.random import set_seed


def main():
    def evaluate_accuracy(data_loader, prefix: str, save: bool = False):
        std_transform.eval()
        model.eval()
        pbar = tqdm(data_loader, desc=prefix, leave=True, total=len(data_loader))
        num_corr = 0
        num_tot = 0
        counter = Counter()
        for idx, batch in enumerate(pbar):
            batch = batch.to(device)
            scores = model(zmuv_transform(std_transform(batch.audio_data)),
                           std_transform.compute_lengths(batch.lengths))
            num_tot += scores.size(0)
            labels = batch.labels.to(device)
            counter.update(labels.tolist())
            num_corr += (scores.max(1)[1] == labels).float().sum().item()
            acc = num_corr / num_tot
            pbar.set_postfix(accuracy=f'{acc:.4}')
        if save and not args.eval:
            writer.add_scalar(f'{prefix}/Metric/acc', acc, epoch_idx)
            ws.increment_model(model, acc / 10)
        elif args.eval:
            tqdm.write(str(counter))
            tqdm.write(str(acc))

        return num_corr / num_tot

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--model', type=str, choices=RegisteredModel.registered_names(), default='las'),
                    opt('--workspace', type=str, default=str(Path('workspaces') / 'default')),
                    opt('--load-weights', action='store_true'),
                    opt('--eval', action='store_true'))
    args = apb.parser.parse_args()

    ws = Workspace(Path(args.workspace), delete_existing=not args.eval)
    writer = ws.summary_writer
    set_seed(SETTINGS.training.seed)
    loader = GoogleSpeechCommandsDatasetLoader(SETTINGS.training.vocab)
    sr = SETTINGS.audio.sample_rate
    ds_kwargs = dict(sr=sr, mono=SETTINGS.audio.use_mono)
    train_ds, dev_ds, test_ds = loader.load_splits(Path(SETTINGS.dataset.dataset_path), **ds_kwargs)

    sr = SETTINGS.audio.sample_rate
    device = torch.device(SETTINGS.training.device)
    std_transform = StandardAudioTransform().to(device).eval()
    zmuv_transform = ZmuvTransform().to(device)
    batchifier = partial(batchify, label_provider=lambda x: x.label)
    truncater = partial(truncate_length, length=int(SETTINGS.training.max_window_size_seconds * sr))
    train_comp = compose(truncater,
                         TimeshiftTransform().train(),
                         NoiseTransform().train(),
                         batchifier)
    prep_dl = StandardAudioDataLoaderBuilder(train_ds, collate_fn=batchifier).build(1)
    prep_dl.shuffle = True
    train_dl = StandardAudioDataLoaderBuilder(train_ds, collate_fn=train_comp).build(SETTINGS.training.batch_size)
    dev_dl = StandardAudioDataLoaderBuilder(dev_ds, collate_fn=compose(truncater, batchifier)).build(SETTINGS.training.batch_size)
    test_dl = StandardAudioDataLoaderBuilder(test_ds, collate_fn=compose(truncater, batchifier)).build(SETTINGS.training.batch_size)

    model = RegisteredModel.find_registered_class(args.model)(30).to(device)
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, SETTINGS.training.learning_rate, weight_decay=SETTINGS.training.weight_decay)
    logging.info(f'{sum(p.numel() for p in params)} parameters')
    criterion = nn.CrossEntropyLoss()

    if (ws.path / 'zmuv.pt.bin').exists():
        zmuv_transform.load_state_dict(torch.load(str(ws.path / 'zmuv.pt.bin')))
    else:
        for idx, batch in enumerate(tqdm(prep_dl, desc='Constructing ZMUV')):
            batch.to(device)
            zmuv_transform.update(std_transform(batch.audio_data))
            if idx == 2000:  # TODO: quick debugging, remove later
                break
        logging.info(dict(zmuv_mean=zmuv_transform.mean, zmuv_std=zmuv_transform.std))
    torch.save(zmuv_transform.state_dict(), str(ws.path / 'zmuv.pt.bin'))

    if args.load_weights:
        ws.load_model(model, best=True)
    if args.eval:
        ws.load_model(model, best=True)
        evaluate_accuracy(dev_dl, 'Dev')
        evaluate_accuracy(test_dl, 'Test')
        return

    ws.write_args(args)
    ws.write_settings(SETTINGS)
    writer.add_scalar('Meta/Parameters', sum(p.numel() for p in params))
    dev_acc = 0
    for epoch_idx in trange(SETTINGS.training.num_epochs, position=0, leave=True):
        model.train()
        std_transform.train()
        pbar = tqdm(train_dl,
                    total=len(train_dl),
                    position=1,
                    desc='Training',
                    leave=True)
        for batch in pbar:
            batch.to(device)
            audio_data = zmuv_transform(std_transform(batch.audio_data))
            scores = model(audio_data, std_transform.compute_lengths(batch.lengths))
            optimizer.zero_grad()
            model.zero_grad()
            labels = batch.labels.to(device)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))
            writer.add_scalar('Training/Loss', loss.item(), epoch_idx)

        for group in optimizer.param_groups:
            group['lr'] *= SETTINGS.training.lr_decay
        dev_acc = evaluate_accuracy(dev_dl, 'Dev', save=True)
    test_acc = evaluate_accuracy(test_dl, 'Test')

    print("model: ", args.model)
    print("dev_acc: ", dev_acc)
    print("test_acc: ", test_acc)


if __name__ == '__main__':
    main()
