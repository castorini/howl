"""
Configuration file for loading pretrained models using PyTorch hub

Usage example: torch.hub.load("castorini/howl", "hey_fire_fox")
"""
import os
import pathlib
import shutil
import typing
import zipfile

import torch

import howl.data.transform as transform
import howl.model as howl_model
from howl.context import InferenceContext
from howl.model.inference import (
    FrameInferenceEngine,
    InferenceEngine,
    SequenceInferenceEngine,
)

dependencies = ["howl", "torch"]


_MODEL_URL = "https://github.com/castorini/howl-models/archive/v1.1.0.zip"
_MODEL_CACHE_FOLDER = "howl-models"


def hey_fire_fox(pretrained=True, **kwargs):
    """Pretrained model for Firefox Voice"""
    engine, ctx = _load_model(pretrained, "res8", "howl/hey-fire-fox", **kwargs)
    return engine, ctx


def _load_model(
    pretrained: bool, model_name: str, workspace_path: str, device: str, **kwargs
) -> typing.Tuple[InferenceEngine, InferenceContext]:
    """
    Loads howl model from a workspace

    Arguments:
        pretrained (bool): load pretrained model weights
        model_name (str): name of the model to use
        workspace_path (str): relative path to workspace from root of howl-models

    Returns the inference engine and context
    """

    # Separate `reload_models` flag since PyTorch will pop the 'force_reload' flag
    reload_models = kwargs.pop("reload_models", False)
    cached_folder = _download_howl_models(reload_models)
    workspace_path = pathlib.Path(cached_folder) / workspace_path
    ws = howl_model.Workspace(workspace_path, delete_existing=False)

    # Load model settings
    settings = ws.load_settings()

    # Set up context
    use_frame = settings.training.objective == "frame"
    ctx = InferenceContext(settings.training.vocab, token_type=settings.training.token_type, use_blank=not use_frame)

    # Load models
    zmuv_transform = transform.ZmuvTransform()
    model = howl_model.RegisteredModel.find_registered_class(model_name)(ctx.num_labels).eval()

    # Load pretrained weights
    if pretrained:
        zmuv_transform.load_state_dict(torch.load(str(ws.path / "zmuv.pt.bin"), map_location=torch.device(device)))
        ws.load_model(model, best=True)

    # Load engine
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

    return engine, ctx


def _download_howl_models(reload_models: bool) -> str:
    """
    Download Howl models from Github release: https://github.com/castorini/howl-models

    Arguments:
        reload_models (bool): force reload if models are already cached

    Returns the cached howl models path
    """

    # Create base cache directory
    base_dir = pathlib.Path.home() / ".cache/howl"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    cached_folder = os.path.join(base_dir, _MODEL_CACHE_FOLDER)

    # Check if existing cache should be used
    use_cache = (not reload_models) and os.path.exists(cached_folder)
    if not use_cache:
        # Clear cache
        zip_path = os.path.join(base_dir, _MODEL_CACHE_FOLDER + ".zip")
        _remove_files(cached_folder)
        _remove_files(zip_path)

        print("Downloading howl-models...")
        torch.hub.download_url_to_file(_MODEL_URL, zip_path, progress=True)

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
