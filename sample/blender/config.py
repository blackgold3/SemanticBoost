import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
import os

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def parse_args(phase="train"):
    parser = ArgumentParser()

    group = parser.add_argument_group("Training options")

    if phase == "render":
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/render.yaml",
            help="config file",
        )
        # group.add_argument("--motion_transfer", action='store_true', help="Motion Distribution Transfer")
        group.add_argument("--npy",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion files")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion folder")
        group.add_argument(
            "--mode",
            type=str,
            required=False,
            default="sequence",
            help="render target: video, sequence, frame",
        )
        group.add_argument(
            "--joint_type",
            type=str,
            required=False,
            default=None,
            help="mmm or vertices for skeleton",
        )
        group.add_argument(
            "--fps",
            type=float,
            required=False,
            default=20,
            help="rendered fps",
        )

        group.add_argument(
            "--keep_path",
            type=str,
            required=False,
            default=None,
            help="gt and pred mixture",
        )

    # remove None params, and create a dictionnary

    params = parser.parse_args()
    # params = {key: val for key, val in vars(opt).items() if val is not None}
    cfg_base = OmegaConf.load('sample/blender/configs/base.yaml')
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    cfg = OmegaConf.merge(cfg_exp)

    if phase == "render":
        if params.npy:
            cfg.RENDER.NPY = params.npy
            cfg.RENDER.INPUT_MODE = "npy"
        if params.dir:
            cfg.RENDER.DIR = params.dir
            cfg.RENDER.INPUT_MODE = "dir"
        cfg.RENDER.JOINT_TYPE = params.joint_type
        cfg.RENDER.MODE = params.mode
        cfg.RENDER.FPS = params.fps
        cfg.RENDER.KEEP_PATH = params.keep_path

    return cfg