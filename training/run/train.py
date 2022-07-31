from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from tqdm import tqdm, trange

from howl.context import InferenceContext
from howl.data.common.tokenizer import WakeWordTokenizer
from howl.data.dataloader import StandardAudioDataLoaderBuilder
from howl.data.dataset.dataset import DatasetSplit, DatasetType, WakeWordDataset
from howl.data.dataset.dataset_loader import RecursiveNoiseDatasetLoader, WakeWordDatasetLoader
from howl.data.transform.batchifier import AudioSequenceBatchifier, WakeWordFrameBatchifier
from howl.data.transform.operator import ZmuvTransform, batchify, compose
from howl.data.transform.transform import (
    DatasetMixer,
    NoiseTransform,
    SpecAugmentTransform,
    StandardAudioTransform,
    TimeshiftTransform,
    TimestretchTransform,
)
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.model import ConfusionMatrix, ConvertedStaticModel, RegisteredModel
from howl.model.inference import FrameInferenceEngine, InferenceEngine
from howl.settings import SETTINGS
from howl.utils import hash_utils, random_utils
from howl.utils.args_utils import ArgOption, ArgumentParserBuilder
from howl.utils.logger import Logger
from howl.workspace import Workspace


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

        Logger.info(f"{conf_matrix}")
        if save and not args.eval and positive_set:
            # TODO: evaluate_engine must be moved outside of the main
            # pylint: disable=undefined-loop-variable
            writer.add_scalar(f"{prefix}/Metric/tp_rate", conf_matrix.tp / len(dataset), epoch_idx)
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
        ArgOption("--model", type=str, choices=RegisteredModel.registered_names(), default="las",),
        ArgOption("--workspace", type=str, default=str(Path("workspaces") / "default")),
        ArgOption("--load-weights", action="store_true"),
        ArgOption("--load-last", action="store_true"),
        ArgOption("--dataset-paths", "-i", type=str, nargs="+", default=[SETTINGS.dataset.dataset_path],),
        ArgOption("--eval-freq", type=int, default=10),
        ArgOption("--eval", action="store_true"),
        ArgOption("--use-stitched-datasets", action="store_true"),
    )
    args = apb.parser.parse_args()

    # region prepare training environment
    random_utils.set_random_seed(SETTINGS.training.seed)
    use_frame = SETTINGS.training.objective == "frame"
    workspace = Workspace(Path(args.workspace), delete_existing=not args.eval)
    writer = workspace.summary_writer
    device = torch.device(SETTINGS.training.device)
    # endregion prepare training environment

    # region load datasets
    Logger.heading("Loading datasets")
    ctx = InferenceContext(
        vocab=SETTINGS.training.vocab, token_type=SETTINGS.training.token_type, use_blank=not use_frame,
    )
    loader = WakeWordDatasetLoader()
    ds_kwargs = dict(sample_rate=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, frame_labeler=ctx.labeler,)

    ww_train_ds, ww_dev_ds, ww_test_ds = (
        WakeWordDataset(
            metadata_list=[], set_type=DatasetType.TRAINING, dataset_split=DatasetSplit.TRAINING, **ds_kwargs
        ),
        WakeWordDataset(metadata_list=[], set_type=DatasetType.DEV, dataset_split=DatasetSplit.DEV, **ds_kwargs),
        WakeWordDataset(metadata_list=[], set_type=DatasetType.TEST, dataset_split=DatasetSplit.TEST, **ds_kwargs),
    )
    for ds_path in args.dataset_paths:
        ds_path = Path(ds_path)
        train_ds, dev_ds, test_ds = loader.load_splits(ds_path, **ds_kwargs)
        ww_train_ds.extend(train_ds)
        ww_dev_ds.extend(dev_ds)
        ww_test_ds.extend(test_ds)

    ww_train_ds.print_stats(word_searcher=ctx.searcher, compute_length=True)
    ww_dev_ds.print_stats(word_searcher=ctx.searcher, compute_length=True)
    ww_test_ds.print_stats(word_searcher=ctx.searcher, compute_length=True)

    if args.use_stitched_datasets:
        Logger.heading("Loading stitched datasets")
        ds_kwargs.pop("frame_labeler")
        ds_kwargs["labeler"] = ctx.labeler
        for ds_path in args.dataset_paths:
            ds_path = Path(ds_path)
            dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.STITCHED, ds_path)
            try:
                train_ds, dev_ds, test_ds = dataset_loader.load_splits(**ds_kwargs)
                ww_train_ds.extend(train_ds)
                ww_dev_ds.extend(dev_ds)
                ww_test_ds.extend(test_ds)
            except FileNotFoundError as file_not_found_error:
                Logger.error(f"Stitched dataset is missing for {ds_path}: {file_not_found_error}")

        header = "w/ stitched"
        ww_train_ds.print_stats(header=header, word_searcher=ctx.searcher, compute_length=True)
        ww_dev_ds.print_stats(header=header, word_searcher=ctx.searcher, compute_length=True)
        ww_test_ds.print_stats(header=header, word_searcher=ctx.searcher, compute_length=True)

    ww_dev_pos_ds = ww_dev_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)
    ww_dev_pos_ds.print_stats(header="dev_pos", word_searcher=ctx.searcher, compute_length=True)
    ww_dev_neg_ds = ww_dev_ds.filter(lambda x: not ctx.searcher.search(x.transcription), clone=True)
    ww_dev_pos_ds.print_stats(header="dev_neg", word_searcher=ctx.searcher, compute_length=True)
    ww_test_pos_ds = ww_test_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)
    ww_test_pos_ds.print_stats(header="test_neg", word_searcher=ctx.searcher, compute_length=True)
    ww_test_neg_ds = ww_test_ds.filter(lambda x: not ctx.searcher.search(x.transcription), clone=True)
    ww_test_neg_ds.print_stats(header="test_neg", word_searcher=ctx.searcher, compute_length=True)

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

    audio_augmentations = (
        TimestretchTransform().train(),
        TimeshiftTransform().train(),
        NoiseTransform().train(),
        batchifier,
    )

    if SETTINGS.training.use_noise_dataset:
        noise_ds = RecursiveNoiseDatasetLoader().load(
            Path(SETTINGS.training.noise_dataset_path),
            sample_rate=SETTINGS.audio.sample_rate,
            mono=SETTINGS.audio.use_mono,
        )
        Logger.info(f"Loaded {len(noise_ds.metadata_list)} noise files.")
        noise_ds_train, noise_ds_dev = noise_ds.split(hash_utils.Sha256Splitter(80))
        noise_ds_dev, noise_ds_test = noise_ds_dev.split(hash_utils.Sha256Splitter(50))
        audio_augmentations = (DatasetMixer(noise_ds_train).train(),) + audio_augmentations
        dev_mixer = DatasetMixer(noise_ds_dev, seed=0, do_replace=False)
        test_mixer = DatasetMixer(noise_ds_test, seed=0, do_replace=False)
    audio_augmentations = compose(*audio_augmentations)

    prep_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=batchify).build(1)
    prep_dl.shuffle = True
    train_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=audio_augmentations).build(
        SETTINGS.training.batch_size
    )
    # endregion initialize audio pre-processors

    # region prepare model for zmuv normalization
    Logger.heading("ZMUV normalization")
    if (workspace.path / "zmuv.pt.bin").exists():
        zmuv_transform.load_state_dict(torch.load(str(workspace.path / "zmuv.pt.bin")))
    else:
        for idx, batch in enumerate(tqdm(prep_dl, desc="Constructing ZMUV")):
            batch.to(device)
            zmuv_transform.update(audio_transform(batch.audio_data))
            if idx == 2000:  # TODO: quick debugging, remove later
                break
        Logger.info(dict(zmuv_mean=zmuv_transform.mean, zmuv_std=zmuv_transform.std))
    torch.save(zmuv_transform.state_dict(), str(workspace.path / "zmuv.pt.bin"))
    # endregion prepare model for zmuv normalization

    # region prepare model training
    Logger.heading("Model preparation")
    model = RegisteredModel.find_registered_class(args.model)(ctx.num_labels).to(device).streaming()
    if SETTINGS.training.convert_static:
        model = ConvertedStaticModel(model, 40, 10)

    if use_frame:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CTCLoss(ctx.blank_label)

    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, SETTINGS.training.learning_rate, weight_decay=SETTINGS.training.weight_decay,)
    Logger.info(f"{sum(p.numel() for p in params)} parameters")

    if args.load_weights:
        workspace.load_model(model, best=not args.load_last)
    # endregion prepare model training

    if args.eval:
        Logger.heading("Model evaluation")
        workspace.load_model(model, best=not args.load_last)
        Logger.info(SETTINGS)
        do_evaluate()
        return

    # region train model
    Logger.heading("Model training")
    workspace.write_args(args)
    workspace.save_settings(SETTINGS)
    Logger.info(SETTINGS)
    writer.add_scalar("Meta/Parameters", sum(p.numel() for p in params))

    spectrogram_augmentations = (SpecAugmentTransform().train(),)
    spectrogram_augmentations = compose(*spectrogram_augmentations)

    pbar = trange(SETTINGS.training.num_epochs, position=0, desc="Training", leave=True)
    for epoch_idx in pbar:
        model.train()
        audio_transform.train()
        model.streaming_state = None
        total_loss = torch.Tensor([0.0]).to(device)
        for batch in train_dl:
            batch.to(device)
            audio_length = audio_transform.compute_lengths(batch.lengths)
            zmuv_audio_data = zmuv_transform(audio_transform(batch.audio_data))
            augmented_audio_data = spectrogram_augmentations(zmuv_audio_data)
            if use_frame:
                scores = model(augmented_audio_data, audio_length)
                loss = criterion(scores, batch.labels)
            else:
                scores = model(augmented_audio_data, audio_length)
                scores = F.log_softmax(scores, -1)  # [num_frames x batch_size x num_labels]
                audio_length = torch.tensor([model.compute_length(x.item()) for x in audio_length]).to(device)
                loss = criterion(scores, batch.labels, audio_length, batch.label_lengths)
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                total_loss += loss

        for group in optimizer.param_groups:
            group["lr"] *= SETTINGS.training.lr_decay

        mean_loss = total_loss / len(train_dl)
        pbar.set_postfix(dict(loss=f"{mean_loss.item():.3}"))

        writer.add_scalar("Training/Loss", mean_loss.item(), epoch_idx)
        # TODO: group["lr"] is invalid
        # pylint: disable=undefined-loop-variable
        writer.add_scalar("Training/LearningRate", group["lr"], epoch_idx)

        if epoch_idx % args.eval_freq == 0 and epoch_idx != 0:
            evaluate_engine(
                ww_dev_pos_ds, "Dev positive", positive_set=True, save=True, write_errors=False,
            )

    Logger.heading("Model evaluation")
    do_evaluate()
    # endregion train model


if __name__ == "__main__":
    main()
