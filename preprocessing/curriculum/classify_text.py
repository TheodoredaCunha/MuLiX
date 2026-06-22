from pathlib import Path
from typing import Dict, Optional, Union

from .embedding_classifier import build_prototype_embeddings, classify_caption_sentences
from .io_utils import load_caption_columns, load_json, save_json
from .prototypes import PROTOTYPES
from .reports import save_distribution_plots, save_label_examples
from .text_utils import split_caption_sentences

PathLike = Union[str, Path]

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_DATA_PATH = "data/MusicBench_train.json"
DEFAULT_SAVE_PATHS = {
    "main_caption": "preprocessing/curriculum/main_caption_classes_berttopic.json",
    "alt_caption": "preprocessing/curriculum/alt_caption_classes_berttopic.json",
}
DEFAULT_LABEL_PATHS = {
    "main_caption": "data/main_caption_classes.json",
    "alt_caption": "data/alt_caption_classes.json",
}


def generate_caption_labels(
    save_paths: Optional[Dict[str, str]] = None,
    data_path: PathLike = DEFAULT_DATA_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    report_dir: PathLike = "labels",
    make_reports: bool = True,
) -> Dict[str, object]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The labels workflow needs sentence-transformers. "
            "Install the project requirements before running `python main.py labels`."
        ) from exc

    save_paths = save_paths or DEFAULT_SAVE_PATHS

    captions = load_caption_columns(data_path)
    split_main_captions = split_caption_sentences(captions["main_caption"])
    split_alt_captions = split_caption_sentences(captions["alt_caption"])

    model_kwargs = {"device": device} if device else {}
    embedding_model = SentenceTransformer(model_name, **model_kwargs)
    prototype_embeddings = build_prototype_embeddings(embedding_model, PROTOTYPES)

    main_caption_classes = classify_caption_sentences(
        split_main_captions,
        embedding_model,
        prototype_embeddings,
    )
    alt_caption_classes = classify_caption_sentences(
        split_alt_captions,
        embedding_model,
        prototype_embeddings,
    )

    written_paths = {
        "main_caption": save_json(main_caption_classes, save_paths["main_caption"]),
        "alt_caption": save_json(alt_caption_classes, save_paths["alt_caption"]),
    }

    report_paths = []
    if make_reports:
        report_paths.extend(
            save_distribution_plots(main_caption_classes, alt_caption_classes, report_dir)
        )
        report_paths.append(
            save_label_examples(
                split_main_captions,
                main_caption_classes,
                split_alt_captions,
                alt_caption_classes,
                report_dir,
            )
        )

    print("Saved caption labels:")
    for path in written_paths.values():
        print(f"- {path}")

    if report_paths:
        print("Saved curriculum reports:")
        for path in report_paths:
            print(f"- {path}")

    return {
        "main_caption_classes": main_caption_classes,
        "alt_caption_classes": alt_caption_classes,
        "written_paths": written_paths,
        "report_paths": report_paths,
    }


def generate_reports_from_saved_labels(
    label_paths: Optional[Dict[str, str]] = None,
    data_path: PathLike = DEFAULT_DATA_PATH,
    report_dir: PathLike = "labels",
) -> Dict[str, object]:
    label_paths = label_paths or DEFAULT_LABEL_PATHS

    main_caption_classes = load_json(label_paths["main_caption"])
    alt_caption_classes = load_json(label_paths["alt_caption"])

    report_paths = save_distribution_plots(
        main_caption_classes,
        alt_caption_classes,
        report_dir,
    )

    captions = load_caption_columns(data_path)
    split_main_captions = split_caption_sentences(captions["main_caption"])
    split_alt_captions = split_caption_sentences(captions["alt_caption"])
    report_paths.append(
        save_label_examples(
            split_main_captions,
            main_caption_classes,
            split_alt_captions,
            alt_caption_classes,
            report_dir,
        )
    )

    print("Saved curriculum reports:")
    for path in report_paths:
        print(f"- {path}")

    return {
        "main_caption_classes": main_caption_classes,
        "alt_caption_classes": alt_caption_classes,
        "report_paths": report_paths,
    }
