from pathlib import Path

import torch

from howl.client import HowlClient
from howl.context import InferenceContext
from howl.data.transform.operator import ZmuvTransform
from howl.model import RegisteredModel
from howl.model.inference import FrameInferenceEngine, InferenceEngine
from howl.utils.args_utils import ArgOption, ArgumentParserBuilder
from howl.utils.logger import Logger
from howl.utils.logging_utils import setup_logger
from howl.workspace import Workspace


def main():
    """Link live stream of audio with the trained model within given workspace to demonstrate wake word detection"""
    apb = ArgumentParserBuilder()
    apb.add_options(
        ArgOption("--model", type=str, choices=RegisteredModel.registered_names(), default="las"),
        ArgOption("--workspace", type=str, default=str(Path("workspaces") / "default")),
    )
    args = apb.parser.parse_args()

    logger = setup_logger("howl-demo")
    workspace = Workspace(Path(args.workspace), delete_existing=False)
    settings = workspace.load_settings()
    Logger.info(settings)

    use_frame = settings.training.objective == "frame"
    ctx = InferenceContext(
        vocab=settings.training.vocab, token_type=settings.training.token_type, use_blank=not use_frame
    )

    device = torch.device(settings.training.device)
    zmuv_transform = ZmuvTransform().to(device)
    print(f"args.model: {args.model}")
    print(f"zmuv.model: {str(workspace.path / 'zmuv.pt.bin')}")
    print(f"use_frame: {use_frame}")
    print(f"max_window_size_ms: {int(settings.training.max_window_size_seconds * 1000)}")  # 500
    print(f"eval_stride_size_ms: {int(settings.training.eval_stride_size_seconds * 1000)}")  # 63

    model = RegisteredModel.find_registered_class(args.model)(ctx.num_labels).to(device).eval()
    zmuv_transform.load_state_dict(torch.load(str(workspace.path / "zmuv.pt.bin"), map_location=device))

    workspace.load_model(model, best=True)
    model.streaming()
    if use_frame:
        engine = FrameInferenceEngine(
            int(settings.training.max_window_size_seconds * 1000),
            int(settings.training.eval_stride_size_seconds * 1000),
            model,
            zmuv_transform,
            ctx,
        )
    else:
        engine = InferenceEngine(model, zmuv_transform, ctx)

    from howl.data.dataset.dataset_loader import WakeWordDatasetLoader

    loader = WakeWordDatasetLoader()
    ds_path = Path("datasets/fire_fox/positive")
    print(settings.audio.sample_rate)
    print(settings.audio.use_mono)
    ds_kwargs = dict(sample_rate=settings.audio.sample_rate, mono=settings.audio.use_mono, frame_labeler=ctx.labeler,)
    train_ds, dev_ds, test_ds = loader.load_splits(ds_path, **ds_kwargs)

    print(len(dev_ds))
    # ww_dev_pos_ds = dev_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)

    # print(len(ww_dev_pos_ds))
    detected = 0
    for idx, ex in enumerate(dev_ds):
        print(ex.metadata.transcription)
        audio_data = ex.audio_data.to(device)
        engine.reset()
        seq_present = engine.infer(audio_data)
        print(idx, seq_present)
        if seq_present:
            detected += 1

    print(f"positive dataset: {detected}/{len(dev_ds)}")

    from howl.dataset.audio_dataset_constants import AudioDatasetType
    from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader

    loader = HowlAudioDatasetLoader(AudioDatasetType.STITCHED, ds_path)

    ds_kwargs.pop("frame_labeler")
    ds_kwargs["labeler"] = ctx.labeler
    train_ds, dev_ds, test_ds = loader.load_splits(**ds_kwargs)

    print(len(dev_ds))

    detected = 0
    total = 10
    for idx, ex in enumerate(dev_ds):
        print(ex.metadata.transcription)
        audio_data = ex.audio_data.to(device)
        engine.reset()
        seq_present = engine.infer(audio_data)
        print(idx, seq_present)
        if seq_present:
            detected += 1
        if idx == total:
            break

    print(f"stitched dataset: {detected}/{total}")

    # client = HowlClient(engine, ctx, device=device, logger=logger)
    # client.start().join()


if __name__ == "__main__":
    main()
