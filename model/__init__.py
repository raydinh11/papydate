import torch

from .final import Model
from .base import ModelBase

MODELS = {
    "model": Model,
}


def build_model(cfg_model):
    m = MODELS[cfg_model["global"]["model_name"]](cfg_model)
    m.init_weights()
    return m

def load_checkpoint(cfg_model, ckpt_path, device="cuda"):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    has_proj = any("timeline.proj" in k for k in sd)
    cfg_model["encoder_sequence"]["timeline_proj"] = has_proj
    m = build_model(cfg_model)
    m.load_state_dict(sd, strict=True)
    return m.to(device)
