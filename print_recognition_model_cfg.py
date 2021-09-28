#!/home/kimkk/miniconda3/envs/lomin/bin/python

import inspect
import os
import argparse

import torch
import torch.onnx

from maskrcnn_benchmark.config import cfg


# detection model conf and weight file names
recognition_model = {
        "v1":
        {
            "config_file": "config_rec_v1_200627_001_100k.yaml",
            "weight_file": "model_rec_v1_200627_001_100k.pth"

        },
        "v2":
        {
            "config_file": "config_rec_v2_200828_001_1.2M.yaml",
            "weight_file": "model_rec_v2_200828_001_1.2M.pth"
        },
        "v3":
        {
            "config_file": "config_rec_v3_200924_001_038k.yaml",
            "weight_file": "model_rec_v3_200924_001_038k.pth"
        }
}


if __name__ == '__main__':

    # command line arguments definition
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Lomin OCR recognition model cfg printer")

    parser.add_argument('--version', choices=['v1', 'v2', 'v3'], default='v3',
                        help='set recognition model version')

    parser.add_argument('--output', required=True, help='file path for output')

    # command line argument parsing
    args = parser.parse_args()
    version = args.version

    # set model conf file path and mode weight file path
    # prefixed by ./model/[detection|recognition]
    config_file = os.path.join('model/recognition', recognition_model[version]["config_file"])
    weight_file = os.path.join('model/recognition', recognition_model[version]["weight_file"])


    # clone project level config and merge with experiment config
    cfg = cfg.clone()
    cfg.merge_from_file(config_file)

    with open(args.output, "w") as f:
        print(f"# {'-'*50}", file=f )

        # script file name
        print(f"# generated by '// defined in {inspect.// defined in {inspect.getfile(inspect.currentframe())}'", file=f )
        print(f"# {'-'*50}", file=f )

        print(f"# Recognition Model {version}", file=f )
        print(f"# - config file path:\n#   {config_file}", file=f)
        print(f"# - weight file path:\n#   {weight_file}", file=f)
        print(f"# {'-'*50}\n", file=f )

        print(f"{cfg}", file=f)

