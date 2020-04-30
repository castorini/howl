import logging

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn
import torch.utils.data as tud

from .args import ArgumentParserBuilder, opt
from ww4ff.data.dataset import FlatWavDatasetLoader
from ww4ff.data.dataloader import AudioDataLoader
from ww4ff.settings import SETTINGS
from ww4ff.model import MobileNetClassifier, MNClassifierConfig
from ww4ff.utils.torch_utils import prepare_device


def main():
    def evaluate(data_loader, prefix: str):
        model.eval()

        pbar = tqdm(data_loader, total=len(data_loader), position=1, desc=prefix, leave=True)
        for (data, targets) in pbar:
            data.to(device)
            scores = model(data).squeeze()
            loss = criterion(scores, targets)
            loss.backward()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))

    def filter_fn(x):
        return sha256_int(x.path.stem) % 100 < args.filter_pct

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--words', type=str,  nargs='+', default=['hey firefox'], help=''),
                    opt('--num_epochs', type=int,  default=2, help=''),
                    opt('--lr', type=float, default=1e-3),
                    opt('--num_gpu', type=int, default=1))
    args = apb.parser.parse_args()

    device, gpu_device_ids = prepare_device(args.num_gpu)

    logging.info(args.words)

    loader = FlatWavDatasetLoader()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    ww_train_ds, ww_dev_ds, ww_test_ds = loader.load_splits(SETTINGS.dataset.dataset_path, **ds_kwargs)

    logging.info(f'train dataset size {len(ww_train_ds)}')
    logging.info(f'dev dataset size {len(ww_dev_ds)}')
    logging.info(f'test dataset size {len(ww_test_ds)}')

    dl_kwargs = dict(batch_size=4)
    ww_train_dl = AudioDataLoader(ww_train_ds, **dl_kwargs)
    ww_dev_dl = AudioDataLoader(ww_dev_ds, **dl_kwargs)
    ww_test_dl = AudioDataLoader(ww_test_ds, **dl_kwargs)

    num_labels = len(args.words)

    config = MNClassifierConfig(num_labels)
    model = MobileNetClassifier(config)

    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, args.lr)
    criterion = nn.MSELoss()

    for epoch_idx in trange(args.num_epochs, position=0, leave=True):
        model.train()

        pbar = tqdm(ww_train_dl, total=len(ww_train_dl), position=1, desc='Training', leave=True)
        for (data, targets) in pbar:
            data.to(device)
            scores = model(data).squeeze()
            optimizer.zero_grad()
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))

    evaluate(ww_test_dl, 'Dev')

if __name__ == '__main__':
    main()
