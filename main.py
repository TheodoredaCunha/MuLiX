from preprocessing.curriculum.classify_text import (
    generate_caption_labels,
    generate_reports_from_saved_labels,
)
from train import build_parser as build_train_parser
from train import train


def info():
    print('info')


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["labels", "label-reports", "train", "info"])
    parser.add_argument("--data", default="data/MusicBench_train.json")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", help="Embedding device, for example cuda or cpu")
    parser.add_argument("--no-reports", action="store_true")
    args, extra_args = parser.parse_known_args()
    if args.mode != 'train' and extra_args:
        parser.error(f"unrecognized arguments: {' '.join(extra_args)}")

    if args.mode == 'labels':
        try:
            generate_caption_labels(
                data_path=args.data,
                model_name=args.model,
                save_paths={
                    "main_caption": "data/main_caption_classes.json",
                    "alt_caption": "data/alt_caption_classes.json"
                },
                device=args.device,
                make_reports=not args.no_reports,
            )
        except ImportError as exc:
            parser.exit(status=1, message=f"{exc}\n")
    elif args.mode == 'label-reports':
        try:
            generate_reports_from_saved_labels(data_path=args.data)
        except ImportError as exc:
            parser.exit(
                status=1,
                message=(
                    f"{exc}\nInstall matplotlib locally to generate the diagrams.\n"
                ),
            )
    elif args.mode == 'train':
        # Train flags (e.g. --epochs, --data, --device) are parsed by the train parser.
        train(build_train_parser().parse_args(sys.argv[2:]))
    else:
        info()


if __name__ == '__main__':
    main()
