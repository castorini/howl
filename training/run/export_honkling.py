import json
import logging

import torch

from .args import ArgumentParserBuilder, opt


def main():
    apb = ArgumentParserBuilder()
    apb.add_options(opt('--input-file', '-i', type=str, required=True),
                    opt('--output-file', '-o', type=str, required=True),
                    opt('--name', type=str, required=True))
    args = apb.parser.parse_args()

    sd = torch.load(args.input_file)
    json_dict = dict()
    if args.name == 'RES8':
        sd['scale1.scale'] = torch.ones(45)
        sd['scale3.scale'] = torch.ones(45)
        sd['scale5.scale'] = torch.ones(45)
    for key, tensor in sd.items():
        logging.info(f'Converting {key}')
        json_dict[key] = tensor.tolist()
    with open(args.output_file, 'w') as f:
        f.write(f'weights[\'{args.name}\'] = ')
        json.dump(json_dict, f)


if __name__ == '__main__':
    main()
