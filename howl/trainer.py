import logging
from pathlib import Path

import howl
from howl.config import TrainingConfig
from howl.context import InferenceContext
from howl.utils import logging_utils
from howl.workspace import Workspace

# WIP; please use train.py


class Trainer:
    """Class which defines training logics"""

    def __init__(
        self, training_cfg: TrainingConfig, logger: logging.Logger = None,
    ):
        """Initialize trainer

        Args:
            training_cfg (TrainingConfig): training config that defines how to load datasets and train the model
            logger (logging.Logger): logger
        """
        self.training_cfg = training_cfg
        self.context_cfg = training_cfg.context_config
        self.context = InferenceContext.load_from_config(self.context_cfg)

        if logger is None:
            self.logger = logging_utils.setup_logger(self.__class__.__name__)

        if self.training_cfg.workspace_path is None:
            self.training_cfg.workspace_path = howl.workspaces_path() / self.context.wake_word.replace(" ", "_")

        self.workspace = Workspace(Path(self.training_cfg.workspace_path))

    def train(self):
        """
        Train the model on train datasets.
        """

        print(self.training_cfg.noise_datasets)
        print(self.training_cfg.train_datasets)
        print(self.training_cfg.val_datasets)
        print(self.training_cfg.test_datasets)

        loader = WakeWordDatasetLoader()
        ds_kwargs = dict(
            sample_rate=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, frame_labeler=ctx.labeler,
        )

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

        # def validation(self):
        #     """
        #     Validate the model on validation datasets.
        #     """
        #     raise NotImplementedError()
