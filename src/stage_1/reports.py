from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from scipy.sparse import csr_matrix


def save_text_report(report_text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def save_dataframe_csv(
    df: pd.DataFrame,
    output_path: Path,
    *,
    index: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index, encoding="utf-8")
    return output_path


def save_eda_sections_csv(sections: dict[str, str], output_path: Path) -> Path:
    """Zapisuje sekcje raportu EDA do CSV jako kolumny: section, report_text."""
    rows = [{"section": key, "report_text": value} for key, value in sections.items()]
    return save_dataframe_csv(pd.DataFrame(rows), output_path, index=False)


def _sparse_matrix_to_triplets(matrix: csr_matrix) -> pd.DataFrame:
    coo = matrix.tocoo()
    return pd.DataFrame(
        {
            "row": coo.row,
            "col": coo.col,
            "value": coo.data,
        }
    )


def save_tfidf_artifacts(
    metadata: pd.DataFrame,
    text_1_matrix: csr_matrix,
    text_2_matrix: csr_matrix,
    feature_names: Sequence[str],
    output_dir: Path,
    summary_text: str | None = None,
    *,
    prefix: str = "tfidf",
) -> dict[str, Path]:
    """
    Zapisuje:
    - metadane par do CSV
    - macierz text_1 do sparse CSV (row, col, value)
    - macierz text_2 do sparse CSV (row, col, value)
    - slownik cech do CSV
    - opcjonalny raport tekstowy
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / f"{prefix}_metadata.csv"
    text_1_path = output_dir / f"{prefix}_text_1_sparse.csv"
    text_2_path = output_dir / f"{prefix}_text_2_sparse.csv"
    vocab_path = output_dir / f"{prefix}_vocabulary.csv"
    summary_path = output_dir / f"{prefix}_summary.txt"

    save_dataframe_csv(metadata, metadata_path, index=False)
    save_dataframe_csv(_sparse_matrix_to_triplets(text_1_matrix), text_1_path, index=False)
    save_dataframe_csv(_sparse_matrix_to_triplets(text_2_matrix), text_2_path, index=False)

    vocabulary_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "index": range(len(feature_names)),
        }
    )
    save_dataframe_csv(vocabulary_df, vocab_path, index=False)

    if summary_text is not None:
        save_text_report(summary_text, summary_path)

    return {
        "metadata": metadata_path,
        "text_1": text_1_path,
        "text_2": text_2_path,
        "vocabulary": vocab_path,
        "summary": summary_path,
    }
