import sys
from preprocessing.curriculum.classify_text import generate_caption_labels


def info():
    print('info')

if __name__ == '__main__':
    mode = sys.argv[1]

    if mode == 'labels': # train model
        generate_caption_labels()