from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from devtools import debug as print_debug
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm, trange

import howl
from howl.config import DatasetConfig, TrainingConfig
from howl.context import InferenceContext
from howl.data.common.tokenizer import WakeWordTokenizer
from howl.data.dataloader import StandardAudioDataLoaderBuilder
from howl.data.dataset.dataset import AudioClipDataset, DatasetSplit, DatasetType, WakeWordDataset
from howl.data.dataset.dataset_loader import RecursiveNoiseDatasetLoader
from howl.data.transform.batchifier import AudioSequenceBatchifier, WakeWordFrameBatchifier
from howl.data.transform.operator import ZmuvTransform, batchify, compose
from howl.data.transform.transform import (
    AugmentModule,
    DatasetMixer,
    NoiseTransform,
    SpecAugmentTransform,
    StandardAudioTransform,
    TimeshiftTransform,
    TimestretchTransform,
)
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.model import RegisteredModel
from howl.utils import hash_utils
from howl.utils.logger import Logger
from howl.workspace import Workspace

# WIP; please use train.py


class Trainer:
    """Class which defines training logics"""

    def __init__(
        self, training_cfg: TrainingConfig,
    ):
        """Initialize trainer

        Args:
            training_cfg (TrainingConfig): training config that defines how to load datasets and train the model
        """
        Logger.info(training_cfg)
        # TODO: Config should be printed out in a cleaner format on terminal
        print_debug(training_cfg)
        self.training_cfg = training_cfg
        self.context_cfg = training_cfg.context_config
        self.context = InferenceContext.load_from_config(self.context_cfg)

        self.device = torch.device(self.training_cfg.device)

        # TODO: Ideally, WakeWordDataset needs to be deprecated
        self.datasets: Dict[str, WakeWordDataset] = {
            DatasetSplit.TRAINING: WakeWordDataset(
                metadata_list=[],
                set_type=DatasetType.TRAINING,
                dataset_split=DatasetSplit.TRAINING,
                frame_labeler=self.context.labeler,
            ),
            DatasetSplit.DEV: WakeWordDataset(
                metadata_list=[],
                set_type=DatasetType.DEV,
                dataset_split=DatasetSplit.TRAINING,
                frame_labeler=self.context.labeler,
            ),
            DatasetSplit.TEST: WakeWordDataset(
                metadata_list=[],
                set_type=DatasetType.TEST,
                dataset_split=DatasetSplit.TRAINING,
                frame_labeler=self.context.labeler,
            ),
        }

        self.noise_datasets: Dict[str, AudioClipDataset] = {
            DatasetSplit.TRAINING: None,
            DatasetSplit.DEV: None,
            DatasetSplit.TEST: None,
        }

        self.use_frame = self.training_cfg.objective == "frame"
        self.ctx = InferenceContext(
            vocab=self.context_cfg.vocab, token_type=self.context_cfg.token_type, use_blank=(not self.use_frame),
        )

        self.audio_transform: StandardAudioTransform = None
        self.zmuv_transform: ZmuvTransform = None
        self.audio_augmentations: List[AugmentModule] = []
        self.spectrogram_augmentations: List[AugmentModule] = []

        self.model: nn.Module = None

    def _load_dataset(self, dataset_split: DatasetSplit, dataset_cfg: DatasetConfig):
        """Load a dataset given dataset config"""
        dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.ALIGNED, Path(dataset_cfg.path).expanduser())
        ds_kwargs = dataset_cfg.audio_config.dict()
        ds_kwargs["labeler"] = self.ctx.labeler
        dataset = dataset_loader.load_split(dataset_split, **ds_kwargs)
        # dataset.print_stats(header=dataset_cfg.path, word_searcher=self.ctx.searcher, compute_length=True)

        return dataset

    def _prepare_dataset(self, dataset_split: DatasetSplit = DatasetSplit.TRAINING):
        """Load datasets for the given dataset split type"""
        dataset_cfgs = self.training_cfg.train_datasets
        if dataset_split == DatasetSplit.DEV:
            dataset_cfgs = self.training_cfg.val_datasets
        elif dataset_split == DatasetSplit.TEST:
            dataset_cfgs = self.training_cfg.test_datasets

        Logger.info(f"Loading {dataset_split.value} datasets")

        for dataset_cfg in dataset_cfgs:
            dataset = self._load_dataset(dataset_split, dataset_cfg)

            self.datasets[dataset_split].extend(dataset)

    def _prepare_noise_dataset(self):
        """Load noise dataset for audio augmentation"""

        for idx, noise_dataset_cfg in enumerate(self.training_cfg.noise_datasets):
            noise_ds = RecursiveNoiseDatasetLoader().load(
                Path(noise_dataset_cfg.path).expanduser(),
                sample_rate=noise_dataset_cfg.audio_config.sample_rate,
                mono=noise_dataset_cfg.audio_config.mono,
            )
            # 80, 10, 10 split
            noise_ds_train, noise_ds_dev_test = noise_ds.split(hash_utils.Sha256Splitter(80))
            noise_ds_dev, noise_ds_test = noise_ds_dev_test.split(hash_utils.Sha256Splitter(90))

            if idx == 0:
                self.noise_datasets[DatasetSplit.TRAINING] = noise_ds_train
                self.noise_datasets[DatasetSplit.DEV] = noise_ds_dev
                self.noise_datasets[DatasetSplit.TEST] = noise_ds_test
            else:
                self.noise_datasets[DatasetSplit.TRAINING].extend(noise_ds_train)
                self.noise_datasets[DatasetSplit.DEV].extend(noise_ds_dev)
                self.noise_datasets[DatasetSplit.TEST].extend(noise_ds_test)

        for dataset_split in [DatasetSplit.TRAINING, DatasetSplit.DEV, DatasetSplit.TEST]:
            Logger.info(
                f"Loaded {len(self.noise_datasets[dataset_split].metadata_list)} noise files for {dataset_split}"
            )

    def _prepare_audio_augmentations(self):
        """Instantiate a set of audio augmentations"""
        self.audio_transform = StandardAudioTransform().to(self.device).eval()
        self.zmuv_transform = ZmuvTransform().to(self.device)

        if self.use_frame:
            batchifier = WakeWordFrameBatchifier(
                self.ctx.negative_label, window_size_ms=self.training_cfg.inference_engine_config.window_ms
            )
        else:
            tokenizer = WakeWordTokenizer(self.ctx.vocab, ignore_oov=False)
            batchifier = AudioSequenceBatchifier(self.ctx.negative_label, tokenizer)

        if self.training_cfg.use_noise_dataset:
            self.audio_augmentations = [DatasetMixer(self.noise_datasets[DatasetSplit.TRAINING]).train()]

        self.audio_augmentations.extend(
            [TimestretchTransform().train(), TimeshiftTransform().train(), NoiseTransform().train(), batchifier]
        )

    def _prepare_spectrogram_augmentations(self):
        """Instantiate a set of spectrogram augmentations"""
        self.spectrogram_augmentations = [SpecAugmentTransform().train()]

    def _train_zmuv_model(self, workspace: Workspace, num_batch_to_consider: int = 2000):
        """Train or load ZMUV model"""
        zmuv_dl = StandardAudioDataLoaderBuilder(self.datasets[DatasetSplit.TRAINING], collate_fn=batchify).build(1)
        zmuv_dl.shuffle = True

        load_pretrained_model = Path(workspace.zmuv_model_path()).exists()

        if load_pretrained_model:
            self.zmuv_transform.load_state_dict(torch.load(workspace.zmuv_model_path()))
        else:
            for idx, batch in enumerate(tqdm(zmuv_dl, desc="Constructing ZMUV model", total=num_batch_to_consider)):
                batch.to(self.device)
                self.zmuv_transform.update(self.audio_transform(batch.audio_data))

                # We just need to approximate mean and variance
                if idx == num_batch_to_consider:
                    break

        zmuv_mean = self.zmuv_transform.mean.item()
        workspace.summary_writer.add_scalar("Meta/ZMUV_mean", zmuv_mean)

        zmuv_std = self.zmuv_transform.std.item()
        workspace.summary_writer.add_scalar("Meta/ZMUV_std", zmuv_std)

        Logger.info(f"zmuv_mean: {zmuv_mean}, zmuv_std: {zmuv_std}")

        if not load_pretrained_model:
            torch.save(self.zmuv_transform.state_dict(), workspace.zmuv_model_path())

    def train(self, load_dataset: bool = True, continue_training: bool = False, debug: bool = False):
        """
        Train the model on train datasets.
        """
        # pylint: disable=too-many-statements

        if debug:
            self.training_cfg.workspace_path = (
                f"{howl.workspaces_path()}/{self.context.wake_word.replace(' ', '_')}-debug"
            )

        if self.training_cfg.workspace_path is None:
            if continue_training:
                raise RuntimeError("workspace_path should be specified when continue_training flag enabled")
            curr_date_time = datetime.now()
            self.training_cfg.workspace_path = (
                f"{howl.workspaces_path()}/{self.context.wake_word.replace(' ', '_')}-"
                f"{curr_date_time.strftime('%m_%d_%H_%M')}"
            )

        Logger.info(f"Workspace: {self.training_cfg.workspace_path}")

        workspace = Workspace(Path(self.training_cfg.workspace_path), delete_existing=(not debug))
        writer = workspace.summary_writer

        # Prepare datasets
        Logger.heading("Dataset preparation")
        if load_dataset:
            self._prepare_dataset(DatasetSplit.TRAINING)
            self._prepare_dataset(DatasetSplit.DEV)
            self._prepare_dataset(DatasetSplit.TEST)

        if self.training_cfg.use_noise_dataset:
            self._prepare_noise_dataset()

        # Audio data augmentation
        self._prepare_audio_augmentations()
        audio_aug_comp = compose(*self.audio_augmentations)

        # Spectrogram augmentation
        self._prepare_spectrogram_augmentations()
        spec_aug_comp = compose(*self.spectrogram_augmentations)

        # model for normalization
        Logger.heading("ZMUV model preparation")
        self._train_zmuv_model(workspace)

        # model for kws
        Logger.heading("KWS model preparation")
        self.model = (
            RegisteredModel.find_registered_class(self.training_cfg.model_config.architecture)(self.ctx.num_labels)
            .to(self.device)
            .streaming()
        )

        if continue_training:
            workspace.load_model(self.model, best=False)

        # Training kws model
        Logger.heading("Model training")

        if self.use_frame:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CTCLoss(self.ctx.blank_label)

        params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        optimizer = AdamW(params, self.training_cfg.learning_rate, weight_decay=self.training_cfg.weight_decay)
        lr_scheduler = ExponentialLR(optimizer, gamma=self.training_cfg.lr_decay)

        Logger.info(f"Total number of parameters: {sum(p.numel() for p in params)}")

        train_dl = StandardAudioDataLoaderBuilder(
            self.datasets[DatasetSplit.TRAINING], collate_fn=audio_aug_comp
        ).build(self.training_cfg.batch_size)

        workspace.save_config(self.training_cfg)
        writer.add_scalar("Meta/Parameters", sum(p.numel() for p in params))

        pbar = trange(self.training_cfg.num_epochs, position=0, desc="Training", leave=True)
        for epoch_idx in pbar:
            self.model.train()
            audio_aug_comp.train()
            self.model.streaming_state = None
            total_loss = torch.Tensor([0.0]).to(self.device)
            for batch in train_dl:
                batch.to(self.device)
                audio_length = self.audio_transform.compute_lengths(batch.lengths)
                zmuv_audio_data = self.zmuv_transform(self.audio_transform(batch.audio_data))
                augmented_audio_data = spec_aug_comp(zmuv_audio_data)
                if self.use_frame:
                    scores = self.model(augmented_audio_data, audio_length)
                    loss = criterion(scores, batch.labels)
                else:
                    scores = self.model(augmented_audio_data, audio_length)
                    scores = F.log_softmax(scores, -1)  # [num_frames x batch_size x num_labels]
                    audio_length = torch.tensor([self.model.compute_length(x.item()) for x in audio_length]).to(
                        self.device
                    )
                    loss = criterion(scores, batch.labels, audio_length, batch.label_lengths)
                optimizer.zero_grad()
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    total_loss += loss

            lr_scheduler.step()
            writer.add_scalar("Training/lr", lr_scheduler.get_last_lr()[0], epoch_idx)

            mean_loss = total_loss / len(train_dl)
            pbar.set_postfix(dict(loss=f"{mean_loss.item():.3}"))
            writer.add_scalar("Training/Loss", mean_loss.item(), epoch_idx)

            # TODO: evaluate the performance on dev set
            # if epoch_idx % args.eval_freq == 0 and epoch_idx != 0:
            #     evaluate_engine(
            #         ww_dev_pos_ds, "Dev positive", positive_set=True, save=True, write_errors=False,
            #     )

        Logger.heading("Model evaluation")
        # endregion train model
        # TODO: evaluate the final model
