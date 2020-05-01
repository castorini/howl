from collections import Counter
from functools import partial
import logging

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn

from .args import ArgumentParserBuilder, opt
from .preprocess_dataset import print_stats
from ww4ff.data.dataset import FlatWavDatasetLoader, WakeWordTrainingDataset, WakeWordEvaluationDataset
from ww4ff.data.dataloader import StandardAudioDataLoaderBuilder
from ww4ff.data.transform import compose, ZmuvTransform, StandardAudioTransform, batchify, random_slice
from ww4ff.settings import SETTINGS
from ww4ff.model import find_model, model_names, InferenceEngine
from ww4ff.utils.random import set_seed


def main():
    def evaluate(dataset: WakeWordEvaluationDataset, prefix: str):
        model.eval()
        ds_iter = iter(dataset)
        c = Counter()
        engine = InferenceEngine(model, zmuv_transform)
        last_idx = None
        with tqdm(position=1, desc=prefix, leave=True) as pbar:
            while True:
                try:
                    example = next(ds_iter)
                except StopIteration:
                    break
                if last_idx != ds_iter.curr_file_idx:
                    engine.reset()
                    last_idx = ds_iter.curr_file_idx
                pred = engine.infer(example.audio_data.to(device))
                label = example.contains_wake_word
                if pred and label:
                    c['tp'] += 1
                elif pred and not label:
                    c['fp'] += 1
                elif not pred and label:
                    c['fn'] += 1
                elif not pred and not label:
                    c['tn'] += 1
                pbar.set_postfix(dict(measure=f'{c}'))
        logging.info(f'{c}')

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--model', type=str, choices=model_names(), default='las'))
    args = apb.parser.parse_args()

    set_seed(SETTINGS.training.seed)
    ww = SETTINGS.training.wake_word
    logging.info(f'Using {ww}')
    loader = FlatWavDatasetLoader()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    ww_train_ds, ww_dev_ds, ww_test_ds = loader.load_splits(SETTINGS.dataset.dataset_path, **ds_kwargs)
    print_stats('Wake word dataset', ww_train_ds, ww_dev_ds, ww_test_ds)

    sr = SETTINGS.audio.sample_rate
    ws = int(SETTINGS.training.eval_window_size_seconds * sr)
    ss = int(SETTINGS.training.eval_stride_size_seconds * sr)
    ww_train_ds = WakeWordTrainingDataset(ww_train_ds, ww)
    ww_dev_ds = WakeWordEvaluationDataset(WakeWordTrainingDataset(ww_dev_ds, ww), ws, ss)
    ww_test_ds = WakeWordEvaluationDataset(WakeWordTrainingDataset(ww_test_ds, ww), ws, ss)

    device = torch.device(SETTINGS.training.device)
    std_transform = StandardAudioTransform(sr).to(device)
    zmuv_transform = ZmuvTransform().to(device)
    train_comp = compose(partial(random_slice, max_window_size=int(SETTINGS.training.max_window_size_seconds * sr)),
                         batchify)
    prep_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=batchify).build(1)
    train_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=train_comp).build(SETTINGS.training.batch_size)

    model = find_model(args.model)().to(device)
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, SETTINGS.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(prep_dl, desc='Constructing ZMUV'):
        batch.to(device)
        zmuv_transform.update(std_transform(batch.audio_data))

    for _ in trange(SETTINGS.training.num_epochs, position=0, leave=True):
        model.train()
        pbar = tqdm(train_dl,
                    total=len(train_dl),
                    position=1,
                    desc='Training',
                    leave=True)
        for batch in pbar:
            batch.to(device)
            scores = model(zmuv_transform(std_transform(batch.audio_data)),
                           std_transform.compute_lengths(batch.lengths))
            optimizer.zero_grad()
            model.zero_grad()
            loss = criterion(scores, batch.labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))
        evaluate(ww_dev_ds, 'Dev')
    evaluate(ww_test_ds, 'Test')


if __name__ == '__main__':
    main()
