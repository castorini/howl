from functools import partial
from pathlib import Path
import logging

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn

from .args import ArgumentParserBuilder, opt
from .preprocess_dataset import print_stats
from ww4ff.data.dataset import WakeWordDatasetLoader, WakeWordDataset
from ww4ff.data.dataloader import StandardAudioDataLoaderBuilder
from ww4ff.data.transform import compose, ZmuvTransform, StandardAudioTransform, WakeWordBatchifier,\
    NoiseTransform, batchify, TimestretchTransform
from ww4ff.settings import SETTINGS
from ww4ff.model import find_model, model_names, Workspace, ConfusionMatrix
from ww4ff.model.inference import InferenceEngine, InferenceEngineSettings
from ww4ff.utils.audio import stride
from ww4ff.utils.random import set_seed


def main():
    def evaluate_engine(dataset: WakeWordDataset,
                        prefix: str,
                        save: bool = False,
                        positive_set: bool = False):
        std_transform.eval()

        engine = InferenceEngine(model, zmuv_transform, negative_label=num_labels - 1)
        model.eval()
        conf_matrix = ConfusionMatrix()
        pbar = tqdm(dataset, desc=prefix)
        for idx, ex in enumerate(pbar):
            audio_data = ex.audio_data.to(device)
            engine.reset()
            seq_present = False
            curr_time = 0
            for window in stride(audio_data,
                                 SETTINGS.training.max_window_size_seconds * 1000,
                                 SETTINGS.training.eval_stride_size_seconds * 1000,
                                 SETTINGS.audio.sample_rate):
                if window.size(-1) < 1000:
                    break
                pred = engine.infer(window.squeeze(0), curr_time=curr_time)
                engine.append_label(pred, curr_time=curr_time)
                seq_present = seq_present or engine.sequence_present(curr_time=curr_time)
                curr_time += SETTINGS.training.eval_stride_size_seconds * 1000
            if args.error_file and seq_present != positive_set:
                with open(args.error_file, 'a') as f:
                    f.write(f'{ex.metadata.transcription.transcription}\t{int(seq_present)}\t{int(positive_set)}\t{ex.metadata.path}\n')
            conf_matrix.increment(seq_present, positive_set)
            pbar.set_postfix(dict(mcc=f'{conf_matrix.mcc}', c=f'{conf_matrix}'))

        logging.info(f'{conf_matrix}')
        if save and not args.eval:
            writer.add_scalar(f'{prefix}/Metric/tp', conf_matrix.tp, epoch_idx)
            ws.increment_model(model, conf_matrix.tp)

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--model', type=str, choices=model_names(), default='las'),
                    opt('--workspace', type=str, default=str(Path('workspaces') / 'default')),
                    opt('--load-weights', action='store_true'),
                    opt('--load-last', action='store_true'),
                    opt('--vocab', type=str, nargs='+', default=[' hey', 'fire fox']),
                    opt('--error-file', type=str),
                    opt('--eval', action='store_true'))
    args = apb.parser.parse_args()

    num_labels = len(args.vocab) + 1

    ws = Workspace(Path(args.workspace), delete_existing=not args.eval)
    writer = ws.summary_writer
    set_seed(SETTINGS.training.seed)
    ww = SETTINGS.training.wake_word
    logging.info(f'Using {ww}')
    loader = WakeWordDatasetLoader()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, words=args.vocab)
    ww_train_ds, ww_dev_ds, ww_test_ds = loader.load_splits(SETTINGS.dataset.dataset_path, **ds_kwargs)
    print_stats('Wake word dataset', ww_train_ds, ww_dev_ds, ww_test_ds)

    sr = SETTINGS.audio.sample_rate
    wind_sz = int(SETTINGS.training.eval_window_size_seconds * sr)
    stri_sz = int(SETTINGS.training.eval_stride_size_seconds * sr)

    inference_wakeword = InferenceEngineSettings().make_wakeword(args.vocab)
    ww_dev_all_pos_ds = ww_dev_ds.filter(lambda x: x.compute_frame_labels(args.vocab), clone=True)
    ww_dev_pos_ds = ww_dev_ds.filter(lambda x: x.compute_frame_labels([inference_wakeword]), clone=True)
    ww_dev_neg_ds = ww_dev_ds.filter(lambda x: not x.compute_frame_labels([inference_wakeword]), clone=True)
    ww_test_pos_ds = ww_test_ds.filter(lambda x: x.compute_frame_labels([inference_wakeword]), clone=True)
    ww_test_neg_ds = ww_test_ds.filter(lambda x: not x.compute_frame_labels([inference_wakeword]), clone=True)

    device = torch.device(SETTINGS.training.device)
    std_transform = StandardAudioTransform().to(device).eval()
    zmuv_transform = ZmuvTransform().to(device)
    batchifier = WakeWordBatchifier(num_labels - 1,
                                    window_size_ms=int(SETTINGS.training.max_window_size_seconds * 1000))
    train_comp = compose(NoiseTransform().train(), batchifier)
    prep_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=batchify).build(1)
    prep_dl.shuffle = True
    train_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=train_comp).build(SETTINGS.training.batch_size)

    model = find_model(args.model)().to(device)
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
        ws.load_model(model, best=not args.load_last)
    if args.eval:
        ws.load_model(model, best=not args.load_last)
        evaluate_engine(ww_dev_pos_ds, 'Dev positive', positive_set=True)
        evaluate_engine(ww_dev_neg_ds, 'Dev negative', positive_set=False)
        evaluate_engine(ww_test_pos_ds, 'Test positive', positive_set=True)
        evaluate_engine(ww_test_neg_ds, 'Test negative', positive_set=False)
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
            group['lr'] *= SETTINGS.training.lr_decay
        evaluate_engine(ww_dev_pos_ds, 'Dev positive', positive_set=True, save=True)

    evaluate_engine(ww_dev_pos_ds, 'Dev positive', positive_set=True)
    evaluate_engine(ww_dev_neg_ds, 'Dev negative', positive_set=False)
    evaluate_engine(ww_test_pos_ds, 'Test positive', positive_set=True)
    evaluate_engine(ww_test_neg_ds, 'Test negative', positive_set=False)


if __name__ == '__main__':
    main()
