from pathlib import Path

import torch
from howl.client import HowlClient
from howl.context import InferenceContext
from howl.data.transform import ZmuvTransform
from howl.model import RegisteredModel, Workspace
from howl.model.inference import FrameInferenceEngine, SequenceInferenceEngine

from .args import ArgumentParserBuilder, opt


def main():
    apb = ArgumentParserBuilder()
    apb.add_options(
        opt("--model", type=str, choices=RegisteredModel.registered_names(), default="las"),
        opt("--workspace", type=str, default=str(Path("workspaces") / "default")),
    )
    args = apb.parser.parse_args()
    ws = Workspace(Path(args.workspace), delete_existing=False)
    settings = ws.load_settings()

    use_frame = settings.training.objective == "frame"
    ctx = InferenceContext(settings.training.vocab, token_type=settings.training.token_type, use_blank=not use_frame)

    device = torch.device(settings.training.device)
    zmuv_transform = ZmuvTransform().to(device)
    model = RegisteredModel.find_registered_class(args.model)(ctx.num_labels).to(device).eval()
    zmuv_transform.load_state_dict(torch.load(str(ws.path / "zmuv.pt.bin"), map_location=device))

    ws.load_model(model, best=True)
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
        engine = SequenceInferenceEngine(model, zmuv_transform, ctx)
    
    client = HowlClient(engine, ctx)
    client.start().join()


if __name__ == "__main__":
    main()
