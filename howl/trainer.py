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
from howl.dataset.audio_dataset_constants import AudioDatasetType, SampleType
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.model import ConfusionMatrix, RegisteredModel
from howl.model.inference import FrameInferenceEngine, InferenceEngine
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
        self.inference_engine_cfg = training_cfg.inference_engine_config
        self.device = torch.device(self.training_cfg.device)

        self.inference_engine_cfg.per_frame = self.training_cfg.objective == "frame"
        self.context_cfg.use_blank = self.training_cfg.objective == "ctc"
        self.context = InferenceContext.load_from_config(self.context_cfg)

        # TODO: Ideally, WakeWordDataset needs to be deprecated
        self.train_dataset: WakeWordDataset = WakeWordDataset(
            metadata_list=[],
            set_type=DatasetType.TRAINING,
            dataset_split=DatasetSplit.TRAINING,
            frame_labeler=self.context.labeler,
        )
        self.dev_pos_dataset: WakeWordDataset = WakeWordDataset(
            metadata_list=[],
            set_type=DatasetType.DEV,
            dataset_split=DatasetSplit.TRAINING,
            frame_labeler=self.context.labeler,
        )
        self.dev_neg_dataset: WakeWordDataset = WakeWordDataset(
            metadata_list=[],
            set_type=DatasetType.DEV,
            dataset_split=DatasetSplit.TRAINING,
            frame_labeler=self.context.labeler,
        )
        self.test_pos_dataset: WakeWordDataset = WakeWordDataset(
            metadata_list=[],
            set_type=DatasetType.TEST,
            dataset_split=DatasetSplit.TEST,
            frame_labeler=self.context.labeler,
        )
        self.test_neg_dataset: WakeWordDataset = WakeWordDataset(
            metadata_list=[],
            set_type=DatasetType.TEST,
            dataset_split=DatasetSplit.TEST,
            frame_labeler=self.context.labeler,
        )

        self.noise_datasets: Dict[str, AudioClipDataset] = {
            DatasetSplit.TRAINING: None,
            DatasetSplit.DEV: None,
            DatasetSplit.TEST: None,
        }

        self.inference_engine: InferenceEngine = None

        self.audio_transform: StandardAudioTransform = None
        self.zmuv_transform: ZmuvTransform = None
        self.audio_augmentations: List[AugmentModule] = []
        self.spectrogram_augmentations: List[AugmentModule] = []

        self.model: nn.Module = None

    def _load_dataset(self, dataset_split: DatasetSplit, dataset_cfg: DatasetConfig):
        """Load a dataset given dataset config"""
        dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.ALIGNED, Path(dataset_cfg.path).expanduser())
        ds_kwargs = dataset_cfg.audio_config.dict()
        ds_kwargs["labeler"] = self.context.labeler
        dataset = dataset_loader.load_split(dataset_split, **ds_kwargs)
        # dataset.print_stats(header=dataset_cfg.path, word_searcher=self.context.searcher, compute_length=True)

        return dataset

    def _prepare_train_dataset(self):
        """Load train datasets"""
        for dataset_cfg in self.training_cfg.train_datasets:
            dataset = self._load_dataset(DatasetSplit.TRAINING, dataset_cfg)
            self.train_dataset.extend(dataset)

    def _prepare_dev_dataset(self):
        """Load dev datasets and store them into appropriate variable (positive, negative)"""
        for dataset_cfg in self.training_cfg.train_datasets:
            dataset = self._load_dataset(DatasetSplit.DEV, dataset_cfg)

            if SampleType.POSITIVE in dataset_cfg.path:
                self.dev_pos_dataset.extend(dataset)
            elif SampleType.NEGATIVE in dataset_cfg.path:
                self.dev_neg_dataset.extend(dataset)
            else:
                dev_pos_dataset = dataset.filter(lambda x: self.context.searcher.search(x.transcription), clone=True)
                self.dev_pos_dataset.extend(dev_pos_dataset)
                dev_neg_dataset = dataset.filter(
                    lambda x: not self.context.searcher.search(x.transcription), clone=True
                )
                self.dev_neg_dataset.extend(dev_neg_dataset)

    def _prepare_test_dataset(self):
        """Load test datasets and store them into appropriate variable (positive, negative)"""
        for dataset_cfg in self.training_cfg.train_datasets:
            dataset = self._load_dataset(DatasetSplit.DEV, dataset_cfg)

            if SampleType.POSITIVE in dataset_cfg.path:
                self.test_pos_dataset.extend(dataset)
            elif SampleType.NEGATIVE in dataset_cfg.path:
                self.test_neg_dataset.extend(dataset)
            else:
                test_pos_dataset = dataset.filter(lambda x: self.context.searcher.search(x.transcription), clone=True)
                self.test_pos_dataset.extend(test_pos_dataset)
                test_neg_dataset = dataset.filter(
                    lambda x: not self.context.searcher.search(x.transcription), clone=True
                )
                self.test_neg_dataset.extend(test_neg_dataset)

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

        if self.training_cfg.objective == "frame":
            batchifier = WakeWordFrameBatchifier(
                self.context.negative_label, window_size_ms=self.inference_engine_cfg.window_ms
            )
        else:
            tokenizer = WakeWordTokenizer(self.context.vocab, ignore_oov=False)
            batchifier = AudioSequenceBatchifier(self.context.negative_label, tokenizer)

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
        zmuv_dl = StandardAudioDataLoaderBuilder(self.train_dataset, collate_fn=batchify).build(1)
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

    def _prepare_models(self, workspace: Workspace, load_pretrained_model: bool = False):
        # model for normalization
        self._train_zmuv_model(workspace)

        # model for kws
        self.model = (
            RegisteredModel.find_registered_class(self.training_cfg.model_config.architecture)(self.context.num_labels)
            .to(self.device)
            .streaming()
        )

        if load_pretrained_model:
            workspace.load_model(self.model, best=False)

    def train(self, load_dataset: bool = True, continue_training: bool = False, debug: bool = False):
        """
        Train the model on train datasets.
        """
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches

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
            self._prepare_train_dataset()
            self._prepare_dev_dataset()
            self._prepare_test_dataset()

        # TODO: print dataset stats
        # ww_dev_pos_ds.print_stats(header="dev_pos", word_searcher=ctx.searcher, compute_length=True)

        if self.training_cfg.use_noise_dataset:
            self._prepare_noise_dataset()

        # Audio data augmentation
        self._prepare_audio_augmentations()
        audio_aug_comp = compose(*self.audio_augmentations)

        # Spectrogram augmentation
        self._prepare_spectrogram_augmentations()
        spec_aug_comp = compose(*self.spectrogram_augmentations)

        # prepare_models
        Logger.heading("Model preparation")
        self._prepare_models(workspace, load_pretrained_model=continue_training)

        # prepare inference engine
        if self.inference_engine_cfg.per_frame:
            self.inference_engine = FrameInferenceEngine(
                self.inference_engine_cfg.window_ms,
                self.inference_engine_cfg.stride_ms,
                self.model,
                self.zmuv_transform,
                self.context,
            )
        else:
            self.inference_engine = InferenceEngine(self.model, self.zmuv_transform, self.context)

        # Training kws model
        Logger.heading("Model training")

        if self.training_cfg.objective == "frame":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CTCLoss(self.context.blank_label)

        params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        optimizer = AdamW(params, self.training_cfg.learning_rate, weight_decay=self.training_cfg.weight_decay)
        lr_scheduler = ExponentialLR(optimizer, gamma=self.training_cfg.lr_decay)

        Logger.info(f"Total number of parameters: {sum(p.numel() for p in params)}")

        train_dl = StandardAudioDataLoaderBuilder(self.train_dataset, collate_fn=audio_aug_comp).build(
            self.training_cfg.batch_size
        )

        workspace.save_config(self.training_cfg)
        writer.add_scalar("Meta/Parameters", sum(p.numel() for p in params))

        pbar = trange(self.training_cfg.num_epochs, position=0, desc="Training", leave=True)
        for epoch_idx in pbar:
            self.model.train()
            self.audio_transform.train()
            self.model.streaming_state = None
            total_loss = torch.Tensor([0.0]).to(self.device)
            for batch in train_dl:
                batch.to(self.device)
                audio_length = self.audio_transform.compute_lengths(batch.lengths)
                zmuv_audio_data = self.zmuv_transform(self.audio_transform(batch.audio_data))
                augmented_audio_data = spec_aug_comp(zmuv_audio_data)
                if self.training_cfg.objective == "frame":
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

            if epoch_idx % self.training_cfg.eval_frequency == 0 and epoch_idx != 0:
                prefix = "Dev positive"
                conf_matrix = self.evaluate_on_dataset(self.dev_pos_dataset, workspace, prefix, positive_set=True)

                writer.add_scalar(f"{prefix}/Metric/tp_rate", conf_matrix.tp / len(self.dev_pos_dataset), epoch_idx)
                workspace.increment_model(self.model, conf_matrix.tp)

        Logger.heading("Model evaluation")
        self.evaluate(workspace, evaluate_on_noisy_dataset=True)

    def evaluate(self, workspace: Workspace, evaluate_on_noisy_dataset: bool = False):
        """Evaluate the model on every dev/test dataset"""
        self.evaluate_on_dataset(
            self.dev_pos_dataset, workspace, "Dev positive", positive_set=True, record_false_detections=True
        )
        self.evaluate_on_dataset(self.dev_neg_dataset, workspace, "Dev negative", positive_set=False)
        if evaluate_on_noisy_dataset:
            dev_mixer = DatasetMixer(self.noise_datasets[DatasetSplit.DEV], seed=0, do_replace=False)
            self.evaluate_on_dataset(
                self.dev_pos_dataset,
                workspace,
                "Dev noisy positive",
                positive_set=True,
                mixer=dev_mixer,
                record_false_detections=True,
            )
            self.evaluate_on_dataset(
                self.dev_neg_dataset, workspace, "Dev noisy negative", positive_set=False, mixer=dev_mixer
            )
        self.evaluate_on_dataset(
            self.test_pos_dataset, workspace, "Test positive", positive_set=True, record_false_detections=True
        )
        self.evaluate_on_dataset(self.test_neg_dataset, workspace, "Test negative", positive_set=False)
        if evaluate_on_noisy_dataset:
            test_mixer = DatasetMixer(self.noise_datasets[DatasetSplit.TEST], seed=0, do_replace=False)
            self.evaluate_on_dataset(
                self.test_pos_dataset,
                workspace,
                "Test noisy positive",
                positive_set=True,
                mixer=test_mixer,
                record_false_detections=True,
            )
            self.evaluate_on_dataset(
                self.test_neg_dataset, workspace, "Test noisy negative", positive_set=False, mixer=test_mixer,
            )

    def evaluate_on_dataset(
        self,
        dataset,
        workspace: Workspace,
        prefix: str,
        positive_set: bool = False,
        mixer: DatasetMixer = None,
        record_false_detections: bool = False,
    ):
        """Evaluate the model on the given dataset"""
        self.audio_transform.eval()
        self.model.eval()

        conf_matrix = ConfusionMatrix()
        pbar = tqdm(dataset, desc=prefix)

        for _, sample in enumerate(pbar):
            if mixer is not None:
                (sample,) = mixer([sample])
            audio_data = sample.audio_data.to(self.device)
            self.inference_engine.reset()
            seq_present = self.inference_engine.infer(audio_data)
            if seq_present != positive_set and record_false_detections:
                with (workspace.path / f"{prefix}_errors.tsv").open("a") as error_file:
                    error_file.write(
                        f"{sample.metadata.transcription}"
                        f"\t{int(seq_present)}"
                        f"\t{int(positive_set)}"
                        f"\t{sample.metadata.path}\n"
                    )
            conf_matrix.increment(seq_present, positive_set)
            pbar.set_postfix(dict(mcc=f"{conf_matrix.mcc}", c=f"{conf_matrix}"))

        Logger.info(f"{conf_matrix}")
        return conf_matrix
