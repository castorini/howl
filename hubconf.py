"""
Configuration file for loading pretrained models using PyTorch hub
Usage example: torch.hub.load("castorini/howl", "hey_fire_fox")
"""
dependencies = ['howl', 'torch']

import pathlib
import typing

import torch

import howl.context as context
import howl.data.transform as transform
import howl.model as howl_model
import howl.model.inference as inference
from howl.settings import SETTINGS as _SETTINGS


def hey_fire_fox(pretrained=True, **kwargs) -> typing.Tuple[inference.InferenceEngine, context.InferenceContext]:
    """
    Pretrained model for Firefox Voice
    """
    # TODO: Refactor into pretrained model settings class
    path = "howl-models/hey-fire-fox"
    model_name = "res8"
    vocab = ["hey", "fire", "fox"]
    inference_sequence = [0, 1, 2]
    max_window_size_seconds = 0.5
    token_type = "word"
    use_frame = True

    # Set up context and workspace
    ctx = context.InferenceContext(
        vocab, token_type=token_type, use_blank=not use_frame)
    ws = howl_model.Workspace(pathlib.Path(path), delete_existing=False)

    # Load models
    zmuv_transform = transform.ZmuvTransform()
    model = howl_model.RegisteredModel.find_registered_class(
        model_name)(ctx.num_labels).eval()
    if pretrained:
        zmuv_transform.load_state_dict(
            torch.load(str(ws.path / 'zmuv.pt.bin')))
        ws.load_model(model, best=True)

    model.streaming()
    if use_frame:
        engine = inference.FrameInferenceEngine(int(max_window_size_seconds * 1000),
                                                int(_SETTINGS.training.eval_stride_size_seconds * 1000),
                                                _SETTINGS.audio.sample_rate,
                                                model,
                                                zmuv_transform,
                                                negative_label=ctx.negative_label,
                                                coloring=ctx.coloring)
    else:
        engine = inference.FrameInferenceEngine(_SETTINGS.audio.sample_rate,
                                                model,
                                                zmuv_transform,
                                                negative_label=ctx.negative_label,
                                                coloring=ctx.coloring)
    engine.sequence = inference_sequence
    return engine, ctx
