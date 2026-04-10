from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import pandas as pd


UNIFIED_COLUMNS = ["dataset", "split", "pair_id", "text_1", "text_2", "label"]


def resolve_project_root(start_path: Optional[Path] = None) -> Path:
    base_path = (start_path or Path(__file__)).resolve()
    for candidate in [base_path, *base_path.parents]:
        if (candidate / "collections").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with `collections` directory.")


def _normalize_pairs_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    normalized["text_1"] = normalized["text_1"].fillna("").astype(str).str.strip()
    normalized["text_2"] = normalized["text_2"].fillna("").astype(str).str.strip()
    normalized["label"] = pd.to_numeric(normalized["label"], errors="coerce")

    normalized = normalized.dropna(subset=["label"])
    normalized["label"] = normalized["label"].astype(int)

    normalized = normalized[
        (normalized["text_1"].str.len() > 0) & (
            normalized["text_2"].str.len() > 0)
    ]

    return normalized[UNIFIED_COLUMNS].reset_index(drop=True)


def load_quora_pairs(
    collections_dir: Optional[Path] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if collections_dir is None:
        collections_dir = resolve_project_root() / "collections"

    quora_path = collections_dir / "Question_Pairs_Dataset" / "questions.csv"
    if not quora_path.exists():
        raise FileNotFoundError(f"Quora dataset file not found: {quora_path}")

    try:
        quora_df = pd.read_csv(quora_path)
    except pd.errors.ParserError:
        warnings.warn(
            "Detected malformed rows in questions.csv. "
            "Falling back to python parser and skipping invalid lines.",
            RuntimeWarning,
        )
        quora_df = pd.read_csv(
            quora_path,
            engine="python",
            on_bad_lines="skip",
        )

    required_cols = {"id", "question1", "question2", "is_duplicate"}
    missing = required_cols - set(quora_df.columns)
    if missing:
        raise ValueError(
            f"Quora dataset missing required columns: {sorted(missing)}")

    if sample_size is not None and sample_size > 0 and len(quora_df) > sample_size:
        quora_df = quora_df.sample(n=sample_size, random_state=random_state)

    unified = pd.DataFrame(
        {
            "dataset": "quora",
            "split": "full",
            "pair_id": quora_df["id"].astype(str),
            "text_1": quora_df["question1"],
            "text_2": quora_df["question2"],
            "label": quora_df["is_duplicate"],
        }
    )
    return _normalize_pairs_frame(unified)


def load_mrpc_pairs(
    collections_dir: Optional[Path] = None,
    include_train: bool = True,
    include_test: bool = True,
) -> pd.DataFrame:
    if not include_train and not include_test:
        raise ValueError("At least one split must be included for MRPC.")

    if collections_dir is None:
        collections_dir = resolve_project_root() / "collections"

    mrpc_dir = collections_dir / "MRPC"
    split_files = []

    if include_train:
        split_files.append(("train", mrpc_dir / "msr_paraphrase_train.txt"))
    if include_test:
        split_files.append(("test", mrpc_dir / "msr_paraphrase_test.txt"))

    frames = []

    for split_name, file_path in split_files:
        if not file_path.exists():
            raise FileNotFoundError(f"MRPC split file not found: {file_path}")

        try:
            mrpc_df = pd.read_csv(file_path, sep="\t")
        except pd.errors.ParserError:
            warnings.warn(
                f"Detected malformed rows in {file_path.name}. "
                f"Falling back to python parser and skipping invalid lines.",
                RuntimeWarning,
            )
            mrpc_df = pd.read_csv(
                file_path,
                sep="\t",
                engine="python",
                on_bad_lines="skip",
            )

        required_cols = {"Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"}
        missing = required_cols - set(mrpc_df.columns)
        if missing:
            raise ValueError(
                f"MRPC file {file_path.name} missing required columns: {sorted(missing)}"
            )

        pair_id = (
            mrpc_df["#1 ID"].astype(str).str.strip()
            + "_"
            + mrpc_df["#2 ID"].astype(str).str.strip()
        )

        frame = pd.DataFrame(
            {
                "dataset": "mrpc",
                "split": split_name,
                "pair_id": pair_id,
                "text_1": mrpc_df["#1 String"],
                "text_2": mrpc_df["#2 String"],
                "label": mrpc_df["Quality"],
            }
        )
        frames.append(frame)

    unified = pd.concat(frames, ignore_index=True)
    return _normalize_pairs_frame(unified)


def load_stage1_datasets(
    collections_dir: Optional[Path] = None,
    quora_sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if collections_dir is None:
        collections_dir = resolve_project_root() / "collections"

    quora = load_quora_pairs(
        collections_dir=collections_dir,
        sample_size=quora_sample_size,
        random_state=random_state,
    )
    mrpc = load_mrpc_pairs(collections_dir=collections_dir)

    combined = pd.concat([quora, mrpc], ignore_index=True)
    combined = combined.drop_duplicates(subset=["dataset", "pair_id"])
    return combined.reset_index(drop=True)


if __name__ == "__main__":
    collections_dir = resolve_project_root() / "collections"

    def _read_for_report(file_path: Path, separator: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, sep=separator)
        except pd.errors.ParserError:
            warnings.warn(
                f"Detected malformed rows in {file_path.name}. Retrying with python parser.",
                RuntimeWarning,
            )
            try:
                return pd.read_csv(file_path, sep=separator, engine="python")
            except pd.errors.ParserError:
                warnings.warn(
                    f"Still malformed rows in {file_path.name}. Skipping invalid lines.",
                    RuntimeWarning,
                )
                return pd.read_csv(
                    file_path,
                    sep=separator,
                    engine="python",
                    on_bad_lines="skip",
                )

    def _print_separator(char: str = "=") -> None:
        print(char * 72)

    def _print_dataset_report(file_path: Path, separator: str, columns: list[str]) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        df = _read_for_report(file_path, separator)

        _print_separator("=")
        print(f"PLIK: {file_path.name}")
        _print_separator("-")
        print(f"Liczba wierszy: {len(df)}")
        print("Niepuste wartości w kolumnach:")

        for col in columns:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {file_path.name}")

            non_empty = df[col].fillna("").astype(str).str.strip().ne("").sum()
            missing = len(df) - non_empty
            print(
                f"  - {col:<12} -> {non_empty:>7} niepustych | {missing:>7} pustych")

        print()

    files_to_report = [
        (
            collections_dir / "MRPC" / "msr_paraphrase_test.txt",
            "\t",
            ["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"],
        ),
        (
            collections_dir / "MRPC" / "msr_paraphrase_train.txt",
            "\t",
            ["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"],
        ),
        (
            collections_dir / "Question_Pairs_Dataset" / "questions.csv",
            ",",
            ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"],
        ),
    ]

    _print_separator("=")
    print("RAPORT KONTROLNY DANYCH WEJŚCIOWYCH")
    _print_separator("=")
    print()

    for file_path, separator, columns in files_to_report:
        _print_dataset_report(file_path, separator, columns)
