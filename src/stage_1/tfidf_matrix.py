from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from data_loading import load_stage1_datasets, resolve_project_root
from preprocessing import preprocess_pairs_dataframe


TFIDF_REQUIRED_COLUMNS = [
    "dataset",
    "split",
    "pair_id",
    "text_1_clean",
    "text_2_clean",
    "label",
]


@dataclass
class TfidfPairMatrices:
    """Kontener na wynik wektoryzacji TF-IDF dla par tekstow."""

    metadata: pd.DataFrame
    text_1_matrix: csr_matrix
    text_2_matrix: csr_matrix
    vectorizer: TfidfVectorizer

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    @property
    def n_pairs(self) -> int:
        return self.metadata.shape[0]


def validate_tfidf_input(df: pd.DataFrame) -> None:
    missing = set(TFIDF_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje wymaganych kolumn do budowy TF-IDF: {sorted(missing)}")

    if df.empty:
        raise ValueError("DataFrame wejsciowy jest pusty.")

    empty_text_1 = df["text_1_clean"].fillna("").astype(str).str.strip().eq("").any()
    empty_text_2 = df["text_2_clean"].fillna("").astype(str).str.strip().eq("").any()

    if empty_text_1 or empty_text_2:
        raise ValueError("W kolumnach text_1_clean / text_2_clean znajduja sie puste teksty.")


def build_joint_corpus(df: pd.DataFrame) -> pd.Series:
    """Buduje wspolny korpus do fitowania wektoryzatora TF-IDF."""
    validate_tfidf_input(df)
    return pd.concat([df["text_1_clean"], df["text_2_clean"]], axis=0, ignore_index=True)


def build_tfidf_vectorizer(
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int | float = 2,
    max_df: int | float = 0.95,
    sublinear_tf: bool = True,
    norm: str = "l2",
    use_idf: bool = True,
    smooth_idf: bool = True,
    lowercase: bool = False,
    max_features: Optional[int] = None,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        lowercase=lowercase,
        max_features=max_features,
    )


def _split_with_optional_stratify(
    indices: pd.Index,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.Index, pd.Index]:
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return pd.Index(idx_train), pd.Index(idx_test)


def assign_model_splits(
    df: pd.DataFrame,
    *,
    random_state: int = 42,
    quora_test_size: float = 0.2,
    quora_val_size: float = 0.1,
    mrpc_val_size: float = 0.1,
) -> pd.DataFrame:
    """
    Tworzy kolumne model_split z podzialem train/validation/test.

    Zasady:
    - Quora: stratyfikowany podzial train/validation/test.
    - MRPC: zachowuje test z plikow i dodatkowo wydziela validation z train.
    """
    required = {"dataset", "split", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn do podzialu modelowego: {sorted(missing)}")

    out = df.copy()
    out["model_split"] = "train"

    quora_mask = out["dataset"] == "quora"
    quora_df = out[quora_mask]
    if not quora_df.empty:
        q_train_idx, q_test_idx = _split_with_optional_stratify(
            quora_df.index,
            quora_df["label"],
            test_size=quora_test_size,
            random_state=random_state,
        )

        val_share_within_train = quora_val_size / max(1e-9, (1.0 - quora_test_size))
        q_train_df = out.loc[q_train_idx]
        q_train_idx_final, q_val_idx = _split_with_optional_stratify(
            q_train_df.index,
            q_train_df["label"],
            test_size=val_share_within_train,
            random_state=random_state,
        )

        out.loc[q_train_idx_final, "model_split"] = "train"
        out.loc[q_val_idx, "model_split"] = "validation"
        out.loc[q_test_idx, "model_split"] = "test"

    mrpc_mask = out["dataset"] == "mrpc"
    mrpc_df = out[mrpc_mask]
    if not mrpc_df.empty:
        out.loc[mrpc_df[mrpc_df["split"] == "test"].index, "model_split"] = "test"

        mrpc_train_idx = mrpc_df[mrpc_df["split"] == "train"].index
        if len(mrpc_train_idx) > 5:
            mrpc_train_df = out.loc[mrpc_train_idx]
            m_train_idx, m_val_idx = _split_with_optional_stratify(
                mrpc_train_df.index,
                mrpc_train_df["label"],
                test_size=mrpc_val_size,
                random_state=random_state,
            )
            out.loc[m_train_idx, "model_split"] = "train"
            out.loc[m_val_idx, "model_split"] = "validation"

    return out


def compute_tfidf_pair_matrices(
    df: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
    *,
    fit_split: str = "train",
    split_column: str = "model_split",
) -> TfidfPairMatrices:
    """
    Oblicza macierze TF-IDF dla text_1_clean i text_2_clean.

    Wektoryzator fitowany jest tylko na podzbiorze fit_split,
    a transform wykonywany jest na calej przekazanej ramce.
    """
    validate_tfidf_input(df)

    if split_column not in df.columns:
        raise ValueError(f"Brakuje kolumny splitu do fitowania: {split_column}")

    working_df = df.copy().reset_index(drop=True)

    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()

    fit_df = working_df[working_df[split_column] == fit_split]
    if fit_df.empty:
        raise ValueError(f"Brak rekordow do fitowania dla {split_column}={fit_split}")

    joint_corpus = build_joint_corpus(fit_df)
    vectorizer.fit(joint_corpus)

    text_1_matrix = vectorizer.transform(working_df["text_1_clean"]).tocsr()
    text_2_matrix = vectorizer.transform(working_df["text_2_clean"]).tocsr()

    metadata_cols = ["dataset", "split", split_column, "pair_id", "label"]
    metadata = working_df[metadata_cols].copy()

    return TfidfPairMatrices(
        metadata=metadata,
        text_1_matrix=text_1_matrix,
        text_2_matrix=text_2_matrix,
        vectorizer=vectorizer,
    )


def tfidf_summary(result: TfidfPairMatrices) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("RAPORT TF-IDF")
    lines.append("=" * 72)
    lines.append(f"Liczba par tekstow       : {result.n_pairs}")
    lines.append(f"Liczba cech slownika     : {result.vocabulary_size}")
    lines.append(f"Ksztalt macierzy text_1  : {result.text_1_matrix.shape}")
    lines.append(f"Ksztalt macierzy text_2  : {result.text_2_matrix.shape}")
    lines.append(f"Liczba niezerowych text_1: {result.text_1_matrix.nnz}")
    lines.append(f"Liczba niezerowych text_2: {result.text_2_matrix.nnz}")
    lines.append("")

    label_counts = result.metadata["label"].value_counts().sort_index()
    lines.append("Rozklad klas:")
    for label, count in label_counts.items():
        lines.append(f"  - label={label}: {count}")

    if "model_split" in result.metadata.columns:
        lines.append("")
        lines.append("Rozklad model_split:")
        split_counts = result.metadata["model_split"].value_counts()
        for split_name, count in split_counts.items():
            lines.append(f"  - {split_name}: {count}")

    lines.append("")
    feature_names = result.vectorizer.get_feature_names_out()
    preview_size = min(20, len(feature_names))
    lines.append(f"Pierwsze {preview_size} cech slownika:")
    for feature in feature_names[:preview_size]:
        lines.append(f"  - {feature}")

    return "\n".join(lines)


def save_tfidf_artifacts(
    result: TfidfPairMatrices,
    output_dir: Path,
    prefix: str = "tfidf",
    summary_text: str | None = None,
) -> dict[str, Path]:
    """Zapisuje artefakty TF-IDF do CSV/TXT (bez plikow NPZ)."""
    from reports import save_tfidf_artifacts as save_tfidf_artifacts_files

    return save_tfidf_artifacts_files(
        metadata=result.metadata,
        text_1_matrix=result.text_1_matrix,
        text_2_matrix=result.text_2_matrix,
        feature_names=result.vectorizer.get_feature_names_out(),
        output_dir=output_dir,
        prefix=prefix,
        summary_text=summary_text,
    )


def run_tfidf_pipeline(
    collections_dir: Optional[Path] = None,
    quora_sample_size: Optional[int] = None,
    random_state: int = 42,
    output_subdir: str = "results/stage_1/raports",
) -> TfidfPairMatrices:
    if collections_dir is None:
        collections_dir = resolve_project_root() / "collections"

    df_raw = load_stage1_datasets(
        collections_dir=collections_dir,
        quora_sample_size=quora_sample_size,
        random_state=random_state,
    )

    df_clean = preprocess_pairs_dataframe(df_raw, drop_empty_after_cleaning=True)
    df_split = assign_model_splits(df_clean, random_state=random_state)

    result = compute_tfidf_pair_matrices(df_split, fit_split="train", split_column="model_split")

    summary_text = tfidf_summary(result)
    output_dir = resolve_project_root() / output_subdir
    save_tfidf_artifacts(result, output_dir=output_dir, prefix="pairs", summary_text=summary_text)

    print(summary_text)
    print()
    print(f"Artefakty TF-IDF zapisano w: {output_dir}")

    return result


if __name__ == "__main__":
    run_tfidf_pipeline()
