import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from tqdm import tqdm, trange

from howl.context import InferenceContext
from howl.data.common.tokenizer import WakeWordTokenizer
from howl.data.dataloader import StandardAudioDataLoaderBuilder
from howl.data.dataset.dataset import DatasetType, WakeWordDataset
from howl.data.dataset.dataset_loader import RecursiveNoiseDatasetLoader, WakeWordDatasetLoader
from howl.data.transform.batchifier import AudioSequenceBatchifier, WakeWordFrameBatchifier
from howl.data.transform.operator import ZmuvTransform, batchify, compose
from howl.data.transform.transform import DatasetMixer, NoiseTransform, StandardAudioTransform
from howl.model import ConfusionMatrix, ConvertedStaticModel, RegisteredModel, Workspace
from howl.model.inference import FrameInferenceEngine, InferenceEngine
from howl.settings import SETTINGS
from howl.utils import hash_utils, logging_utils, random
from training.run.deprecated.create_raw_dataset import print_stats

from .args import ArgumentParserBuilder, opt


def main():
    """Train or evaluate howl model"""
    # TODO: train.py needs to be refactored
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=duplicate-code

    def evaluate_engine(
        dataset: WakeWordDataset,
        prefix: str,
        save: bool = False,
        positive_set: bool = False,
        write_errors: bool = True,
        mixer: DatasetMixer = None,
    ):
        """Evaluate the current model on the given dataset"""
        audio_transform.eval()

        if use_frame:
            engine = FrameInferenceEngine(
                int(SETTINGS.training.max_window_size_seconds * 1000),
                int(SETTINGS.training.eval_stride_size_seconds * 1000),
                model,
                zmuv_transform,
                ctx,
            )
        else:
            engine = InferenceEngine(model, zmuv_transform, ctx)
        model.eval()
        conf_matrix = ConfusionMatrix()
        pbar = tqdm(dataset, desc=prefix)
        if write_errors:
            with (workspace.path / "errors.tsv").open("a") as error_file:
                print(prefix, file=error_file)
        for _, ex in enumerate(pbar):
            if mixer is not None:
                (ex,) = mixer([ex])
            audio_data = ex.audio_data.to(device)
            engine.reset()
            seq_present = engine.infer(audio_data)
            if seq_present != positive_set and write_errors:
                with (workspace.path / "errors.tsv").open("a") as error_file:
                    error_file.write(
                        f"{ex.metadata.transcription}\t{int(seq_present)}\t{int(positive_set)}\t{ex.metadata.path}\n"
                    )
            conf_matrix.increment(seq_present, positive_set)
            pbar.set_postfix(dict(mcc=f"{conf_matrix.mcc}", c=f"{conf_matrix}"))

        logger.info(f"{conf_matrix}")
        if save and not args.eval:
            # TODO: evaluate_engine must be moved outside of the main
            # pylint: disable=undefined-loop-variable
            writer.add_scalar(f"{prefix}/Metric/tp", conf_matrix.tp, epoch_idx)
            workspace.increment_model(model, conf_matrix.tp)
        if args.eval:
            threshold = engine.threshold
            with (workspace.path / (str(round(threshold, 2)) + "_results.csv")).open("a") as result_file:
                result_file.write(
                    f"{prefix},{threshold},{conf_matrix.tp},{conf_matrix.tn},{conf_matrix.fp},{conf_matrix.fn}\n"
                )

    def do_evaluate():
        """Run evaluation on different datasets"""
        evaluate_engine(ww_dev_pos_ds, "Dev positive", positive_set=True)
        evaluate_engine(ww_dev_neg_ds, "Dev negative", positive_set=False)
        if SETTINGS.training.use_noise_dataset:
            evaluate_engine(ww_dev_pos_ds, "Dev noisy positive", positive_set=True, mixer=dev_mixer)
            evaluate_engine(ww_dev_neg_ds, "Dev noisy negative", positive_set=False, mixer=dev_mixer)
        evaluate_engine(ww_test_pos_ds, "Test positive", positive_set=True)
        evaluate_engine(ww_test_neg_ds, "Test negative", positive_set=False)
        if SETTINGS.training.use_noise_dataset:
            evaluate_engine(
                ww_test_pos_ds, "Test noisy positive", positive_set=True, mixer=test_mixer,
            )
            evaluate_engine(
                ww_test_neg_ds, "Test noisy negative", positive_set=False, mixer=test_mixer,
            )

    apb = ArgumentParserBuilder()
    apb.add_options(
        opt("--model", type=str, choices=RegisteredModel.registered_names(), default="las",),
        opt("--workspace", type=str, default=str(Path("workspaces") / "default")),
        opt("--load-weights", action="store_true"),
        opt("--load-last", action="store_true"),
        opt("--no-dev-per-epoch", action="store_false", dest="dev_per_epoch"),
        opt("--dataset-paths", "-i", type=str, nargs="+", default=[SETTINGS.dataset.dataset_path],),
        opt("--eval", action="store_true"),
    )
    args = apb.parser.parse_args()

    # region prepare training environment
    random.set_seed(SETTINGS.training.seed)
    logger = logging_utils.setup_logger(os.path.basename(__file__))
    use_frame = SETTINGS.training.objective == "frame"
    workspace = Workspace(Path(args.workspace), delete_existing=not args.eval)
    writer = workspace.summary_writer
    device = torch.device(SETTINGS.training.device)
    # endregion prepare training environment

    # region load datasets
    ctx = InferenceContext(SETTINGS.training.vocab, token_type=SETTINGS.training.token_type, use_blank=not use_frame,)
    loader = WakeWordDatasetLoader()
    ds_kwargs = dict(sample_rate=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, frame_labeler=ctx.labeler,)

    ww_train_ds, ww_dev_ds, ww_test_ds = (
        WakeWordDataset(metadata_list=[], set_type=DatasetType.TRAINING, **ds_kwargs),
        WakeWordDataset(metadata_list=[], set_type=DatasetType.DEV, **ds_kwargs),
        WakeWordDataset(metadata_list=[], set_type=DatasetType.TEST, **ds_kwargs),
    )
    for ds_path in args.dataset_paths:
        ds_path = Path(ds_path)
        train_ds, dev_ds, test_ds = loader.load_splits(ds_path, **ds_kwargs)
        ww_train_ds.extend(train_ds)
        ww_dev_ds.extend(dev_ds)
        ww_test_ds.extend(test_ds)
    print_stats("Wake word dataset", ctx, ww_train_ds, ww_dev_ds, ww_test_ds)

    ww_dev_pos_ds = ww_dev_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)
    print_stats("Dev pos dataset", ctx, ww_dev_pos_ds)
    ww_dev_neg_ds = ww_dev_ds.filter(lambda x: not ctx.searcher.search(x.transcription), clone=True)
    print_stats("Dev neg dataset", ctx, ww_dev_neg_ds)
    ww_test_pos_ds = ww_test_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)
    print_stats("Test pos dataset", ctx, ww_test_pos_ds)
    ww_test_neg_ds = ww_test_ds.filter(lambda x: not ctx.searcher.search(x.transcription), clone=True)
    print_stats("Test neg dataset", ctx, ww_test_neg_ds)
    # endregion load datasets

    # region create dataloaders with audio preprocessor
    audio_transform = StandardAudioTransform().to(device).eval()
    zmuv_transform = ZmuvTransform().to(device)
    if use_frame:
        batchifier = WakeWordFrameBatchifier(
            ctx.negative_label, window_size_ms=int(SETTINGS.training.max_window_size_seconds * 1000),
        )
    else:
        tokenizer = WakeWordTokenizer(ctx.vocab, ignore_oov=False)
        batchifier = AudioSequenceBatchifier(ctx.negative_label, tokenizer)
    train_comp = (NoiseTransform().train(), batchifier)

    if SETTINGS.training.use_noise_dataset:
        noise_ds = RecursiveNoiseDatasetLoader().load(
            Path(SETTINGS.training.noise_dataset_path),
            sample_rate=SETTINGS.audio.sample_rate,
            mono=SETTINGS.audio.use_mono,
        )
        logger.info(f"Loaded {len(noise_ds.metadata_list)} noise files.")
        noise_ds_train, noise_ds_dev = noise_ds.split(hash_utils.Sha256Splitter(80))
        noise_ds_dev, noise_ds_test = noise_ds_dev.split(hash_utils.Sha256Splitter(50))
        train_comp = (DatasetMixer(noise_ds_train).train(),) + train_comp
        dev_mixer = DatasetMixer(noise_ds_dev, seed=0, do_replace=False)
        test_mixer = DatasetMixer(noise_ds_test, seed=0, do_replace=False)
    train_comp = compose(*train_comp)

    prep_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=batchify).build(1)
    prep_dl.shuffle = True
    train_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=train_comp).build(SETTINGS.training.batch_size)
    # endregion initialize audio pre-processors

    # region prepare model for zmuv normalization
    if (workspace.path / "zmuv.pt.bin").exists():
        zmuv_transform.load_state_dict(torch.load(str(workspace.path / "zmuv.pt.bin")))
    else:
        for idx, batch in enumerate(tqdm(prep_dl, desc="Constructing ZMUV")):
            batch.to(device)
            zmuv_transform.update(audio_transform(batch.audio_data))
            if idx == 2000:  # TODO: quick debugging, remove later
                break
        logger.info(dict(zmuv_mean=zmuv_transform.mean, zmuv_std=zmuv_transform.std))
    torch.save(zmuv_transform.state_dict(), str(workspace.path / "zmuv.pt.bin"))
    # endregion prepare model for zmuv normalization

    # region prepare model training
    model = RegisteredModel.find_registered_class(args.model)(ctx.num_labels).to(device).streaming()
    if SETTINGS.training.convert_static:
        model = ConvertedStaticModel(model, 40, 10)

    if use_frame:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CTCLoss(ctx.blank_label)

    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, SETTINGS.training.learning_rate, weight_decay=SETTINGS.training.weight_decay,)
    logger.info(f"{sum(p.numel() for p in params)} parameters")

    if args.load_weights:
        workspace.load_model(model, best=not args.load_last)
    # endregion prepare model training

    if args.eval:
        workspace.load_model(model, best=not args.load_last)
        do_evaluate()
        return

    # region train model
    workspace.write_args(args)
    workspace.write_settings(SETTINGS)
    writer.add_scalar("Meta/Parameters", sum(p.numel() for p in params))
    for epoch_idx in trange(SETTINGS.training.num_epochs, position=0, leave=True):
        model.train()
        audio_transform.train()
        model.streaming_state = None
        pbar = tqdm(train_dl, total=len(train_dl), position=1, desc="Training", leave=True)
        total_loss = torch.Tensor([0.0]).to(device)
        for batch in pbar:
            batch.to(device)
            if use_frame:
                scores = model(
                    zmuv_transform(audio_transform(batch.audio_data)), audio_transform.compute_lengths(batch.lengths),
                )
                loss = criterion(scores, batch.labels)
            else:
                lengths = audio_transform.compute_lengths(batch.audio_lengths)
                scores = model(zmuv_transform(audio_transform(batch.audio_data)), lengths)
                scores = F.log_softmax(scores, -1)  # [num_frames x batch_size x num_labels]
                lengths = torch.tensor([model.compute_length(x.item()) for x in lengths]).to(device)
                loss = criterion(scores, batch.labels, lengths, batch.label_lengths)
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f"{loss.item():.3}"))
            with torch.no_grad():
                total_loss += loss

        for group in optimizer.param_groups:
            group["lr"] *= SETTINGS.training.lr_decay

        mean = total_loss / len(train_dl)
        writer.add_scalar("Training/Loss", mean.item(), epoch_idx)
        # TODO: group["lr"] is invalid
        # pylint: disable=undefined-loop-variable
        writer.add_scalar("Training/LearningRate", group["lr"], epoch_idx)

        if args.dev_per_epoch:
            evaluate_engine(
                ww_dev_pos_ds, "Dev positive", positive_set=True, save=True, write_errors=False,
            )

    do_evaluate()
    # endregion train model


if __name__ == "__main__":
    main()
