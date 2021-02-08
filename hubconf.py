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
import howl.settings as settings

from howl.settings import SETTINGS as _SETTINGS

_MODEL_URL = "https://github.com/castorini/howl-models/archive/v1.0.0.zip"
_MODEL_CACHE_FOLDER = "howl-models"

@dataclasses.dataclass
class _ModelSettings:
    path: str # Relative to the howl-models root folder
    model_name: str
    vocab: typing.List[str]
    inference_sequence: typing.List[int]
    token_type: str
    use_frame: bool


def hey_fire_fox(pretrained=True, **kwargs) \
        -> typing.Tuple[inference.InferenceEngine, context.InferenceContext]:
    """
    Pretrained model for Firefox Voice
    """
    # Audio inference settings
    conf = _ModelSettings(path="howl/hey-fire-fox",
                          model_name="res8",
                          vocab=["hey", "fire", "fox"],
                          inference_sequence=[0, 1, 2],
                          token_type="word",
                          use_frame=False)
    ats = transform.augment.AudioTransformSettings(num_mels=40)
    _SETTINGS._training = settings.TrainingSettings(max_window_size_seconds=0.5)

    # Separate `reload_models` flag since PyTorch will pop the 'force_reload' flag
    reload_models = kwargs.pop('reload_models', False)
    cached_folder =_download_howl_models(reload_models)

    # Set up context and workspace
    ws_path: pathlib.Path = cached_folder / conf.path
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
        engine = inference.FrameInferenceEngine(int(_SETTINGS.training.max_window_size_seconds * 1000),
                                                int(_SETTINGS.training.eval_stride_size_seconds * 1000),
                                                _SETTINGS.audio.sample_rate,
                                                model,
                                                zmuv_transform,
                                                negative_label=ctx.negative_label,
                                                coloring=ctx.coloring,
                                                audio_transform_settings=ats)
    else:
        engine = inference.FrameInferenceEngine(_SETTINGS.audio.sample_rate,
                                                model,
                                                zmuv_transform,
                                                negative_label=ctx.negative_label,
                                                coloring=ctx.coloring,
                                                audio_transform_settings=ats)
    engine.sequence = conf.inference_sequence
    return engine, ctx


def _download_howl_models(reload_models: bool) -> str:
    """
    Download Howl models from Github release: https://github.com/castorini/howl-models
    
    Returns the cached folder location
    """

    # Create base cache directory
    base_dir = pathlib.Path.home() / '.cache/howl'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    cached_folder = os.path.join(base_dir, _MODEL_CACHE_FOLDER)

    # Check if existing cache should be used
    use_cache = (not reload_models) and os.path.exists(cached_folder)
    if use_cache:
        return
    
    # Clear cache
    zip_path = os.path.join(base_dir, _MODEL_CACHE_FOLDER + '.zip')
    _remove_files(cached_folder)
    _remove_files(zip_path)
    torch.hub.download_url_to_file(_MODEL_URL, zip_path, progress=False)

    # Extract files into folder
    with zipfile.ZipFile(zip_path) as model_zipfile:
        # Find name of extracted folder
        extracted_name = model_zipfile.infolist()[0].filename
        extracted_path = os.path.join(base_dir, extracted_name)
        _remove_files(extracted_path)
        model_zipfile.extractall(base_dir)

        # Rename folder
        shutil.move(extracted_path, cached_folder)

    return cached_folder

def _remove_files(path):
    # Remove file or folder if it exists
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
