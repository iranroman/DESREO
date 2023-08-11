from desreo.utils.parser import parse_args, load_config
from desreo.datasets import Snoop_Dogg

def main():

    # parse the yaml
    args = parse_args()
    cfg = load_config(args, args.path_to_config)

    dataset = Snoop_Dogg(cfg)

if __name__ == '__main__':
    main()
