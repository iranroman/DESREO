from desreo.utils.parser import parse_args, load_config

def main():

    # parse the yaml
    args = parse_args()
    cfg = load_config(args, args.path_to_config)

    print(cfg)

if __name__ == '__main__':
    main()
