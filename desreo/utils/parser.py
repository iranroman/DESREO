# created by Iran R. Roman <iran@ccrma.stanford.edu>
"""Argument parser functions."""

import argparse
import sys
import yaml


def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        cfg (str): path to the config file.
    """
    parser = argparse.ArgumentParser(
        description="Provide DESSEO training and/or evaluation pipeline."
    )
    parser.add_argument(
        "--config",
        dest="path_to_config",
        help="Path to the config files",
        default="configs/Golumbic_data.yaml",
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): 
            `args`
            `path_to_config`
    """
    with open(path_to_config, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print('\nUsing parameters:')
    print(yaml.dump(dict(cfg), allow_unicode=True, default_flow_style=False))
    return DotDictify(cfg)

class DotDictify(dict):
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDictify):
            value = DotDictify(value)
        super(DotDictify, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, DotDictify.MARKER)
        if found is DotDictify.MARKER:
            found = DotDictify()
            super(DotDictify, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__

