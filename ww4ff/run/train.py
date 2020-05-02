from functools import partial
from pathlib import Path
import logging

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn

from .args import ArgumentParserBuilder, opt
from .preprocess_dataset import print_stats
from ww4ff.data.dataset import FlatWavDatasetLoader, WakeWordTrainingDataset, WakeWordEvaluationDataset, DatasetType
from ww4ff.data.dataloader import StandardAudioDataLoaderBuilder
from ww4ff.data.transform import compose, ZmuvTransform, StandardAudioTransform, batchify, random_slice,\
    NoiseTransform, TimestretchTransform, TimeshiftTransform, trim
from ww4ff.settings import SETTINGS
from ww4ff.model import find_model, model_names, Workspace, ConfusionMatrix
from ww4ff.model.inference import InferenceEngine
from ww4ff.utils.random import set_seed


def main():
    def evaluate(dataset: WakeWordEvaluationDataset, prefix: str):
        std_transform.eval()
        ds_iter = iter(dataset)
        engine = InferenceEngine(model, zmuv_transform)
        model.eval()
        conf_matrix = ConfusionMatrix()
        with tqdm(position=1, desc=prefix, leave=True) as pbar:
            while True:
                try:
                    example = next(ds_iter)
                    pbar.update(1)
                except StopIteration:
                    break
                example = trim([example])[0]
                pred = engine.infer(example.audio_data.to(device))
                label = example.contains_wake_word
                conf_matrix.increment(pred, label)
                if pbar.n % 10 == 9:
                    pbar.set_postfix(dict(mcc=f'{conf_matrix.mcc}'))
        logging.info(f'{conf_matrix}')
        if dataset.set_type == DatasetType.DEV and not args.eval:
            writer.add_scalar(f'{prefix}/Metric/mcc', conf_matrix.mcc, epoch_idx)
            ws.increment_model(model, conf_matrix.mcc)

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--model', type=str, choices=model_names(), default='las'),
                    opt('--workspace', type=str, default=str(Path('workspaces') / 'default')),
                    opt('--load-weights', action='store_true'),
                    opt('--eval', action='store_true'))
    args = apb.parser.parse_args()

    ws = Workspace(Path(args.workspace), delete_existing=not args.eval)
    writer = ws.summary_writer
    set_seed(SETTINGS.training.seed)
    ww = SETTINGS.training.wake_word
    logging.info(f'Using {ww}')
    loader = FlatWavDatasetLoader()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    ww_train_ds, ww_dev_ds, ww_test_ds = loader.load_splits(SETTINGS.dataset.dataset_path, **ds_kwargs)
    print_stats('Wake word dataset', ww_train_ds, ww_dev_ds, ww_test_ds)

    sr = SETTINGS.audio.sample_rate
    wind_sz = int(SETTINGS.training.eval_window_size_seconds * sr)
    stri_sz = int(SETTINGS.training.eval_stride_size_seconds * sr)
    ww_train_ds = WakeWordTrainingDataset(ww_train_ds, ww)
    ww_dev_ds = WakeWordEvaluationDataset(WakeWordTrainingDataset(ww_dev_ds, ww), wind_sz, stri_sz)
    ww_test_ds = WakeWordEvaluationDataset(WakeWordTrainingDataset(ww_test_ds, ww), wind_sz, stri_sz)

    device = torch.device(SETTINGS.training.device)
    std_transform = StandardAudioTransform(sr).to(device).eval()
    zmuv_transform = ZmuvTransform().to(device)
    train_comp = compose(trim,
                         partial(random_slice, max_window_size=int(SETTINGS.training.max_window_size_seconds * sr)),
                         TimeshiftTransform().train(),
                         TimestretchTransform().train(),
                         NoiseTransform().train(),
                         batchify)
    prep_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=compose(trim, batchify)).build(1)
    prep_dl.shuffle = True
    train_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=train_comp).build(SETTINGS.training.batch_size)

    model = find_model(args.model)().to(device)
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, SETTINGS.training.learning_rate, weight_decay=SETTINGS.training.weight_decay)
    logging.info(f'{sum(p.numel() for p in params)} parameters')
    criterion = nn.CrossEntropyLoss()

    for idx, batch in enumerate(tqdm(prep_dl, desc='Constructing ZMUV')):
        batch.to(device)
        zmuv_transform.update(std_transform(batch.audio_data))
        if idx == 2000:  # TODO: quick debugging, remove later
            break

    if args.load_weights:
        ws.load_model(model, best=True)
    if args.eval:
        ws.load_model(model, best=True)
        evaluate(ww_dev_ds, 'Dev')
        evaluate(ww_test_ds, 'Test')
        return

    ws.write_args(args)
    ws.write_setting(SETTINGS)
    writer.add_scalar('Meta/Parameters', sum(p.numel() for p in params))
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
            scores = model(zmuv_transform(std_transform(batch.audio_data)),
                           std_transform.compute_lengths(batch.lengths))
            optimizer.zero_grad()
            model.zero_grad()
            loss = criterion(scores, batch.labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))
            writer.add_scalar('Training/Loss', loss.item(), epoch_idx)

        for group in optimizer.param_groups:
            group['lr'] *= 0.75
        evaluate(ww_dev_ds, 'Dev')
    evaluate(ww_test_ds, 'Test')


if __name__ == '__main__':
    main()
