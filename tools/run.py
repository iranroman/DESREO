# created by Iran R. Roman <iran@ccrma.stanford.edu>
"""Wrapper to train and test DESREO."""
import yaml

from desreo.utils.parser import parse_args, load_config

from train_net import train


def main():
    """
    Main function to spawn the train and test process.
    """
    # parse config arguments
    args = parse_args()
    cfg = load_config(args, args.path_to_config)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        train(cfg=cfg)

if __name__ == "__main__":
    main()

