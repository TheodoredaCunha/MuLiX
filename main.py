import sys
from preprocessing.curriculum.classify_text import generate_caption_labels


def info():
    print('info')

if __name__ == '__main__':
    mode = sys.argv[1]

    if mode == 'labels':
        generate_caption_labels(
            save_paths={
                "main_caption": "data/main_caption_classes.json",
                "alt_caption": "data/alt_caption_classes.json"
            }
        )
