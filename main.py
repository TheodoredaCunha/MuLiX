from preprocessing.curriculum.classify_text import generate_caption_labels


def info():
    print('info')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["labels", "info"])
    parser.add_argument("--data", default="data/MusicBench_train.json")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", help="Embedding device, for example cuda or cpu")
    parser.add_argument("--no-reports", action="store_true")
    args = parser.parse_args()

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
    else:
        info()


if __name__ == '__main__':
    main()
