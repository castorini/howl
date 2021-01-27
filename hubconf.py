"""
Configuration file for loading pretrained models using PyTorch hub
Usage example: torch.hub.load("castorini/howl", "hey_fire_fox")
"""
dependencies = ['howl', 'torch']

import dataclasses
import os
import pathlib
import shutil
import typing
import zipfile

import torch

import howl.context as context
import howl.data.transform as transform
import howl.model as howl_model
import howl.model.inference as inference
from howl.settings import SETTINGS as _SETTINGS

_MODEL_URL = "https://github.com/castorini/howl-models/archive/v1.0.0.zip"
_MODEL_CACHE_FOLDER = "howl-models"

@dataclasses.dataclass
class _ModelSettings:
    path: str # Relative to the howl-models root folder
    model_name: str
    vocab: typing.List[str]
    inference_sequence: typing.List[int]
    num_mels: int
    inference_threshold: float
    max_window_size_seconds: int
    token_type: str
    use_frame: bool


def hey_fire_fox(pretrained=True, **kwargs) \
        -> typing.Tuple[inference.InferenceEngine, context.InferenceContext]:
    """
    Pretrained model for Firefox Voice
    """

    # Set cache directory
    cache_dir = pathlib.Path.home() / '.cache/howl'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Separate flag needed since PyTorch will pop the 'force_reload' flag first
    reload_models = kwargs.pop('reload_models', False)
    _download_howl_models(cache_dir, reload_models)

    # Set conf and env variables
    conf = _ModelSettings(path="howl/hey-fire-fox",
                          model_name="res8",
                          vocab=["hey", "fire", "fox"],
                          inference_sequence=[0, 1, 2],
                          num_mels=40,
                          inference_threshold=0,
                          max_window_size_seconds=0.5,
                          token_type="word",
                          use_frame=True)

    os.environ["NUM_MELS"] = str(conf.num_mels)
    os.environ["INFERENCE_THRESHOLD"] = str(conf.inference_threshold)
    os.environ["MAX_WINDOW_SIZE_SECONDS"] = str(conf.max_window_size_seconds)

    # Set up context and workspace
    ws_path: pathlib.Path = cache_dir / _MODEL_CACHE_FOLDER / conf.path
    ctx = context.InferenceContext(
        conf.vocab, token_type=conf.token_type, use_blank=not conf.use_frame)
    ws = howl_model.Workspace(ws_path, delete_existing=False)

    # Load models
    zmuv_transform = transform.ZmuvTransform()
    model = howl_model.RegisteredModel.find_registered_class(
        conf.model_name)(ctx.num_labels).eval()
    if pretrained:
        zmuv_transform.load_state_dict(
            torch.load(str(ws_path / 'zmuv.pt.bin')))
        ws.load_model(model, best=True)

    model.streaming()
    if conf.use_frame:
        engine = inference.FrameInferenceEngine(int(conf.max_window_size_seconds * 1000),
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
    engine.sequence = conf.inference_sequence
    return engine, ctx


def _download_howl_models(base_dir: str, reload_models: bool):
    # Download Howl models from Github release: https://github.com/castorini/howl-models
    cached_folder = os.path.join(base_dir, _MODEL_CACHE_FOLDER)

    use_cache = (not reload_models) and os.path.exists(cached_folder)
    if use_cache:
        return

    zip_path = os.path.join(base_dir, _MODEL_CACHE_FOLDER + '.zip')
    _remove_files(cached_folder)
    _remove_files(zip_path)
    torch.hub.download_url_to_file(_MODEL_URL, zip_path)

    # Extract files into folder
    with zipfile.ZipFile(zip_path) as model_zipfile:
        # Find name of extracted folder
        extraced_name = model_zipfile.infolist()[0].filename
        extracted_path = os.path.join(base_dir, extraced_name)
        _remove_files(extracted_path)
        model_zipfile.extractall(base_dir)

        # Rename folder
        shutil.move(extracted_path, cached_folder)


def _remove_files(path):
    # Remove file or folder if it exists
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
