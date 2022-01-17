import json
from pathlib import Path

import howl
from howl.config import TrainingConfig
from howl.trainer import Trainer
from training.run.args import ArgumentParserBuilder, opt

# WIP; please use train.py


def main(
    training_config_path: Path, num_gpus: int,
):
    """Train howl model

    Args:
        training_config_path (Path): location of the training yaml config
        num_gpus (int): number of gpus to use
    """
    training_cfg = TrainingConfig.parse_file(training_config_path)
    print(json.dumps(training_cfg.dict(), indent=4), num_gpus)

    Trainer(training_cfg)


def setup():
    """Parse the arguments"""
    training_config_path = str(howl.configs_path() / "wip_hey_fire_fox_res8.json")
    num_gpus = 1

    apb = ArgumentParserBuilder()
    apb.add_options(
        opt(
            "--training-config-path",
            type=str,
            default=training_config_path,
            help="location of the training yaml config",
        ),
        opt("--num-gpus", type=int, default=num_gpus, help="number of gpus to use"),
    )
    raw_args = apb.parser.parse_args()

    return raw_args


if __name__ == "__main__":
    args = setup()

    main(
        Path(args.training_config_path), args.num_gpus,
    )
