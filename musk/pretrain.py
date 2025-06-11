import argparse
from timm.models import create_model
from . import modeling  # required for timm model registration  # noqa: F401

def get_parser():
    parser = argparse.ArgumentParser(description="MUSK pretrain placeholder")
    parser.add_argument("--json-data", type=str, required=True)
    parser.add_argument("--model", type=str, default="musk_large_patch16_384")
    parser.add_argument("--device", type=str, default="cpu")
    return parser

def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    model = create_model(args.model)
    print(f"Model {args.model} created")
    return model

if __name__ == "__main__":
    main()
