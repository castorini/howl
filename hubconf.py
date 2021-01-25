"""
Configuration file for loading pretrained models using PyTorch hub

Usage example: torch.hub.load("castorini/howl", "hey_fire_fox")
"""
dependencies = ['howl', 'torch']

from pathlib import Path
from typing import Tuple

import torch

from howl.context import InferenceContext
from howl.data.transform import ZmuvTransform
from howl.model import RegisteredModel, Workspace
from howl.model.inference import (FrameInferenceEngine, InferenceEngine,
                                  SequenceInferenceEngine)
from howl.settings import SETTINGS


def hey_fire_fox(pretrained=True, **kwargs) -> Tuple[InferenceEngine, InferenceContext]:
    """
    Pretrained model for Firefox Voice
    """
    # TODO: Refactor into pretrained model settings class
    path = "howl-models/hey-fire-fox"
    model = "res8"
    vocab = ["hey", "fire", "fox"]
    inference_sequence = [0, 1, 2]
    max_window_size_seconds = 0.5
    token_type = "word"
    use_frame = True

    # Set up context and workspace
    ctx = InferenceContext(vocab, token_type=token_type,
                           use_blank=not use_frame)
    ws = Workspace(Path(path), delete_existing=False)

    # Load models
    zmuv_transform = ZmuvTransform()
    model = RegisteredModel.find_registered_class(model)(ctx.num_labels).eval()

    if pretrained:
        zmuv_transform.load_state_dict(
            torch.load(str(ws.path / 'zmuv.pt.bin')))
        ws.load_model(model, best=True)

    model.streaming()
    if use_frame:
        engine = FrameInferenceEngine(int(max_window_size_seconds * 1000),
                                      int(SETTINGS.training.eval_stride_size_seconds * 1000),
                                      SETTINGS.audio.sample_rate,
                                      model,
                                      zmuv_transform,
                                      negative_label=ctx.negative_label,
                                      coloring=ctx.coloring)
    else:
        engine = SequenceInferenceEngine(SETTINGS.audio.sample_rate,
                                         model,
                                         zmuv_transform,
                                         negative_label=ctx.negative_label,
                                         coloring=ctx.coloring)

    engine.sequence = inference_sequence
    return engine, ctx
