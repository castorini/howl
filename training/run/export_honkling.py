import json
import logging

import torch

from howl.utils.args_utils import ArgOption, ArgumentParserBuilder


def main():
    """Load model trained with Howl and generate a model for Honkling"""
    apb = ArgumentParserBuilder()
    apb.add_options(
        ArgOption("--input-file", "-i", type=str, required=True),
        ArgOption("--output-file", "-o", type=str, required=True),
        ArgOption("--name", type=str, required=True),
    )
    args = apb.parser.parse_args()

    state_dict = torch.load(args.input_file)
    json_dict = dict()
    if args.name == "RES8":
        state_dict["scale1.scale"] = torch.ones(45)
        state_dict["scale3.scale"] = torch.ones(45)
        state_dict["scale5.scale"] = torch.ones(45)
    for key, tensor in state_dict.items():
        logging.info(f"Converting {key}")
        json_dict[key] = tensor.tolist()
    with open(args.output_file, "w") as file:
        file.write(f"weights['{args.name}'] = ")
        json.dump(json_dict, file)


if __name__ == "__main__":
    main()
