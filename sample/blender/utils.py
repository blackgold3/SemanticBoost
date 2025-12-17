from pathlib import Path
import os
import hydra
import logging
import shortuuid  # type: ignore
logger = logging.getLogger(__name__)

def get_split_keyids(path: str, split: str):
    filepath = Path(path) / split
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")
    
def get_samples_folder(path, *, jointstype):
    output_dir = Path(hydra.utils.to_absolute_path(path))
    candidates = [x for x in os.listdir(output_dir) if "samples" in x]
    if not candidates:
        raise ValueError("There is no samples for this model.")

    amass = False
    for candidate in candidates:
        amass = amass or ("amass" in candidate)

    if amass:
        samples_path = output_dir / f"amass_samples_{jointstype}"
        if not samples_path.exists():
            jointstype = "mmm"
            samples_path = output_dir / f"amass_samples_mmm"
            if not samples_path.exists():
                raise ValueError("You must specify a correct jointstype.")
            logger.info(f"Samples from {jointstype} not found, take mmm instead.")
    else:
        samples_path = output_dir / "samples"
    return samples_path, amass, jointstype

def cfg_mean_nsamples_resolution(cfg):
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    return cfg.number_of_samples == 1


def get_path(sample_path: Path, is_amass: bool, gender: str, split: str, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    gender_str = gender + "_" if is_amass else ""
    path = sample_path / f"{fact_str}{gender_str}{split}{extra_str}"
    return path

def generate_id() -> str:
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(8)