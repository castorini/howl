from pathlib import Path
import logging

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn

from .args import ArgumentParserBuilder, opt
from .create_raw_dataset import print_stats
from howl.data.dataset import WakeWordDatasetLoader, WakeWordDataset, RecursiveNoiseDatasetLoader, Sha256Splitter,\
    DatasetType, PronunciationDictionary, WordFrameLabeler, PhoneticFrameLabeler, PhonePhrase
from howl.data.dataloader import StandardAudioDataLoaderBuilder
from howl.data.transform import compose, ZmuvTransform, StandardAudioTransform, WakeWordBatchifier,\
    NoiseTransform, batchify, TimestretchTransform, DatasetMixer
from howl.settings import SETTINGS
from howl.model import Workspace, ConfusionMatrix, RegisteredModel
from howl.model.inference import InferenceEngine, InferenceEngineSettings, LabelColoring, WordTranscriptSearcher,\
    PhoneticTranscriptSearcher
from howl.utils.audio import stride
from howl.utils.random import set_seed


def main():
    def evaluate_engine(dataset: WakeWordDataset,
                        prefix: str,
                        save: bool = False,
                        positive_set: bool = False,
                        write_errors: bool = True,
                        mixer: DatasetMixer = None):
        std_transform.eval()

        engine = InferenceEngine(model, zmuv_transform, negative_label=negative_label, coloring=coloring)
        model.eval()
        conf_matrix = ConfusionMatrix()
        pbar = tqdm(dataset, desc=prefix)
        if write_errors:
            with (ws.path / 'errors.tsv').open('a') as f:
                print(prefix, file=f)
        for idx, ex in enumerate(pbar):
            if mixer is not None:
                ex, = mixer([ex])
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
            if seq_present != positive_set and write_errors:
                with (ws.path / 'errors.tsv').open('a') as f:
                    f.write(f'{ex.metadata.transcription}\t{int(seq_present)}\t{int(positive_set)}\t{ex.metadata.path}\n')
            conf_matrix.increment(seq_present, positive_set)
            pbar.set_postfix(dict(mcc=f'{conf_matrix.mcc}', c=f'{conf_matrix}'))

        logging.info(f'{conf_matrix}')
        if save and not args.eval:
            writer.add_scalar(f'{prefix}/Metric/tp', conf_matrix.tp, epoch_idx)
            ws.increment_model(model, conf_matrix.tp)

    def do_evaluate():
        evaluate_engine(ww_dev_pos_ds, 'Dev positive', positive_set=True)
        evaluate_engine(ww_dev_neg_ds, 'Dev negative', positive_set=False)
        if SETTINGS.training.use_noise_dataset:
            evaluate_engine(ww_dev_pos_ds, 'Dev noisy positive', positive_set=True, mixer=dev_mixer)
            evaluate_engine(ww_dev_neg_ds, 'Dev noisy negative', positive_set=False, mixer=dev_mixer)
        evaluate_engine(ww_test_pos_ds, 'Test positive', positive_set=True)
        evaluate_engine(ww_test_neg_ds, 'Test negative', positive_set=False)
        if SETTINGS.training.use_noise_dataset:
            evaluate_engine(ww_test_pos_ds, 'Test noisy positive', positive_set=True, mixer=test_mixer)
            evaluate_engine(ww_test_neg_ds, 'Test noisy negative', positive_set=False, mixer=test_mixer)

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--model', type=str, choices=RegisteredModel.registered_names(), default='las'),
                    opt('--workspace', type=str, default=str(Path('workspaces') / 'default')),
                    opt('--load-weights', action='store_true'),
                    opt('--load-last', action='store_true'),
                    opt('--no-dev-per-epoch', action='store_false', dest='dev_per_epoch'),
                    opt('--dataset-paths', '-i', type=str, nargs='+', default=[SETTINGS.dataset.dataset_path]),
                    opt('--vocab', type=str, nargs='+', default=[' hey', 'fire fox']),
                    opt('--eval', action='store_true'))
    args = apb.parser.parse_args()

    labeler = WordFrameLabeler(args.vocab)
    negative_label = len(args.vocab)
    coloring = None
    if SETTINGS.training.use_phone:
        coloring = LabelColoring()
        pd = PronunciationDictionary.from_file(SETTINGS.training.phone_dictionary)
        new_vocab = []
        for color, word in enumerate(args.vocab):
            phone_phrases = pd.encode(word)
            logging.info(f'Using phonemes {str(phone_phrases)} for word {word}.')
            new_vocab.extend(x.text for x in phone_phrases)
            coloring.extend_sequence(len(phone_phrases))
        args.vocab = new_vocab
        phone_phrases = [PhonePhrase.from_string(x) for x in new_vocab]
        labeler = PhoneticFrameLabeler(phone_phrases)
    num_labels = len(args.vocab) + 1

    ws = Workspace(Path(args.workspace), delete_existing=not args.eval)
    writer = ws.summary_writer
    set_seed(SETTINGS.training.seed)
    loader = WakeWordDatasetLoader()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, frame_labeler=labeler)

    ww_train_ds, ww_dev_ds, ww_test_ds = WakeWordDataset(metadata_list=[], set_type=DatasetType.TRAINING, **ds_kwargs),\
                                         WakeWordDataset(metadata_list=[], set_type=DatasetType.DEV, **ds_kwargs),\
                                         WakeWordDataset(metadata_list=[], set_type=DatasetType.TEST, **ds_kwargs)
    for ds_path in args.dataset_paths:
        ds_path = Path(ds_path)
        train_ds, dev_ds, test_ds = loader.load_splits(ds_path, **ds_kwargs)
        ww_train_ds.extend(train_ds)
        ww_dev_ds.extend(dev_ds)
        ww_test_ds.extend(test_ds)
    print_stats(f'Wake word dataset', ww_train_ds, ww_dev_ds, ww_test_ds)

    searcher = PhoneticTranscriptSearcher(phone_phrases, coloring) if SETTINGS.training.use_phone else WordTranscriptSearcher(args.vocab)
    ww_dev_pos_ds = ww_dev_ds.filter(lambda x: searcher.search(x.transcription), clone=True)
    ww_dev_neg_ds = ww_dev_ds.filter(lambda x: not searcher.search(x.transcription), clone=True)
    ww_test_pos_ds = ww_test_ds.filter(lambda x: searcher.search(x.transcription), clone=True)
    ww_test_neg_ds = ww_test_ds.filter(lambda x: not searcher.search(x.transcription), clone=True)

    device = torch.device(SETTINGS.training.device)
    std_transform = StandardAudioTransform().to(device).eval()
    zmuv_transform = ZmuvTransform().to(device)
    batchifier = WakeWordBatchifier(num_labels - 1,
                                    window_size_ms=int(SETTINGS.training.max_window_size_seconds * 1000))
    train_comp = (NoiseTransform().train(), batchifier)

    if SETTINGS.training.use_noise_dataset:
        noise_ds = RecursiveNoiseDatasetLoader().load(SETTINGS.raw_dataset.noise_dataset_path,
                                                      sr=SETTINGS.audio.sample_rate,
                                                      mono=SETTINGS.audio.use_mono)
        logging.info(f'Loaded {len(noise_ds.metadata_list)} noise files.')
        noise_ds_train, noise_ds_dev = noise_ds.split(Sha256Splitter(80))
        noise_ds_dev, noise_ds_test = noise_ds_dev.split(Sha256Splitter(50))
        train_comp = (DatasetMixer(noise_ds_train).train(),) + train_comp
        dev_mixer = DatasetMixer(noise_ds_dev, seed=0, do_replace=False)
        test_mixer = DatasetMixer(noise_ds_test, seed=0, do_replace=False)
    train_comp = compose(*train_comp)

    prep_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=batchify).build(1)
    prep_dl.shuffle = True
    train_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=train_comp).build(SETTINGS.training.batch_size)

    model = RegisteredModel.find_registered_class(args.model)().to(device)
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
        do_evaluate()
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
        if args.dev_per_epoch:
            evaluate_engine(ww_dev_pos_ds, 'Dev positive', positive_set=True, save=True, write_errors=False)
    do_evaluate()


if __name__ == '__main__':
    main()
