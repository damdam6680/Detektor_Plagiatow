from __future__ import annotations

from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loading import load_stage1_datasets, resolve_project_root
from plots import plot_label_distribution, plot_word_length_histogram
from preprocessing import preprocess_pairs_dataframe
from reports import save_text_report


def _safe_pct(count: int, total: int) -> float:
    return (100.0 * count / total) if total else 0.0


def add_text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje statystyki długości dla surowych i wyczyszczonych tekstów."""
    result = df.copy()

    for col in ["text_1", "text_2", "text_1_clean", "text_2_clean"]:
        if col in result.columns:
            result[f"{col}_char_len"] = result[col].fillna("").astype(str).str.len()
            result[f"{col}_word_len"] = result[col].fillna("").astype(str).str.split().apply(len)

    if {"text_1_clean_word_len", "text_2_clean_word_len"}.issubset(result.columns):
        result["word_len_diff"] = (
            result["text_1_clean_word_len"] - result["text_2_clean_word_len"]
        ).abs()

    if {"text_1_clean_char_len", "text_2_clean_char_len"}.issubset(result.columns):
        result["char_len_diff"] = (
            result["text_1_clean_char_len"] - result["text_2_clean_char_len"]
        ).abs()

    return result


def _token_stream(df: pd.DataFrame) -> list[str]:
    tokens: list[str] = []
    for col in ["text_1_clean", "text_2_clean"]:
        if col in df.columns:
            for text in df[col].fillna("").astype(str):
                tokens.extend(text.split())
    return tokens


def _top_tokens_and_bigrams(df: pd.DataFrame, top_n: int = 15) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    token_counter: Counter[str] = Counter()
    bigram_counter: Counter[str] = Counter()

    for col in ["text_1_clean", "text_2_clean"]:
        for text in df[col].fillna("").astype(str):
            toks = text.split()
            token_counter.update(toks)
            bigram_counter.update(f"{a} {b}" for a, b in zip(toks, islice(toks, 1, None)))

    return token_counter.most_common(top_n), bigram_counter.most_common(top_n)


def _reversed_pairs_count(df: pd.DataFrame) -> tuple[int, int]:
    ordered_pairs = {
        (r.text_1_clean, r.text_2_clean)
        for r in df[["text_1_clean", "text_2_clean"]].itertuples(index=False)
    }

    unordered_reversed: set[tuple[str, str]] = set()
    for left, right in ordered_pairs:
        if left != right and (right, left) in ordered_pairs:
            unordered_reversed.add(tuple(sorted((left, right))))

    records_in_reversed = df.apply(
        lambda row: tuple(sorted((row["text_1_clean"], row["text_2_clean"]))) in unordered_reversed
        and row["text_1_clean"] != row["text_2_clean"],
        axis=1,
    ).sum()

    return len(unordered_reversed), int(records_in_reversed)


def _cosine_confusion_examples(df: pd.DataFrame, top_n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.99, lowercase=False)
    joint_corpus = pd.concat([df["text_1_clean"], df["text_2_clean"]], ignore_index=True)
    vectorizer.fit(joint_corpus)

    x1 = vectorizer.transform(df["text_1_clean"])
    x2 = vectorizer.transform(df["text_2_clean"])

    # Przy normie L2 cosinus to iloczyn skalarny rzadkich wektorów.
    cosine_scores = np.asarray(x1.multiply(x2).sum(axis=1)).ravel()

    scored = df[["dataset", "split", "pair_id", "label", "text_1", "text_2"]].copy()
    scored["cosine_tfidf"] = cosine_scores

    false_similar = (
        scored[scored["label"] == 0]
        .sort_values("cosine_tfidf", ascending=False)
        .head(top_n)
    )
    false_dissimilar = (
        scored[scored["label"] == 1]
        .sort_values("cosine_tfidf", ascending=True)
        .head(top_n)
    )

    return false_similar, false_dissimilar


def dataset_overview(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EDA - PRZEGLAD DATASETOW")
    lines.append("=" * 72)

    lines.append("Liczebnosc wedlug dataset:")
    dataset_counts = df["dataset"].value_counts(dropna=False)
    for name, count in dataset_counts.items():
        lines.append(f"  - {name}: {count}")

    lines.append("")
    lines.append("Liczebnosc wedlug split:")
    split_counts = df["split"].value_counts(dropna=False)
    for name, count in split_counts.items():
        lines.append(f"  - {name}: {count}")

    lines.append("")
    return "\n".join(lines)


def class_distribution_by_dataset(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EDA - ROZKLAD KLAS PER DATASET")
    lines.append("=" * 72)

    grouped = df.groupby(["dataset", "label"]).size().unstack(fill_value=0).sort_index()

    for dataset_name, row in grouped.iterrows():
        total = int(row.sum())
        lines.append(f"\nDataset: {dataset_name} (n={total})")
        for label, count in row.items():
            lines.append(f"  - label={label}: {count} ({_safe_pct(int(count), total):.2f}%)")

    lines.append("")
    return "\n".join(lines)


def text_length_report(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EDA - DLUGOSCI TEKSTOW PER DATASET")
    lines.append("=" * 72)

    cols_to_report = [
        "text_1_clean_word_len",
        "text_2_clean_word_len",
        "word_len_diff",
        "text_1_clean_char_len",
        "text_2_clean_char_len",
        "char_len_diff",
    ]

    for dataset_name, subset in df.groupby("dataset"):
        lines.append(f"\nDataset: {dataset_name} (n={len(subset)})")
        for col in cols_to_report:
            if col not in subset.columns:
                continue
            desc = subset[col].describe()
            lines.append(f"  Kolumna: {col}")
            lines.append(f"    - mean  : {desc['mean']:.2f}")
            lines.append(f"    - std   : {desc['std']:.2f}")
            lines.append(f"    - p50   : {desc['50%']:.2f}")
            lines.append(f"    - p95   : {desc['95%']:.2f}" if "95%" in desc.index else f"    - max   : {desc['max']:.2f}")
            lines.append(f"    - max   : {desc['max']:.2f}")

    lines.append("")
    return "\n".join(lines)


def suspicious_examples_report(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EDA - IDENTYCZNE, KROTKIE I NIEROWNE PARY PER DATASET")
    lines.append("=" * 72)

    for dataset_name, subset in df.groupby("dataset"):
        total = len(subset)
        identical_clean = int((subset["text_1_clean"] == subset["text_2_clean"]).sum())
        very_short_pairs = int(((subset["text_1_clean_word_len"] <= 2) | (subset["text_2_clean_word_len"] <= 2)).sum())
        large_length_gap = int((subset["word_len_diff"] >= 10).sum())

        lines.append(f"\nDataset: {dataset_name} (n={total})")
        lines.append(f"  - Identyczne po preprocessingu: {identical_clean} ({_safe_pct(identical_clean, total):.2f}%)")
        lines.append(f"  - Krotkie pary (<=2 slowa): {very_short_pairs} ({_safe_pct(very_short_pairs, total):.2f}%)")
        lines.append(f"  - Nierowne dlugoscia (diff >=10): {large_length_gap} ({_safe_pct(large_length_gap, total):.2f}%)")

    lines.append("")
    return "\n".join(lines)


def data_quality_report(df_raw: pd.DataFrame, df_clean_all: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EDA - JAKOSC DANYCH PER DATASET")
    lines.append("=" * 72)

    for dataset_name in sorted(df_raw["dataset"].dropna().unique()):
        raw_ds = df_raw[df_raw["dataset"] == dataset_name].copy()
        clean_ds = df_clean_all[df_clean_all["dataset"] == dataset_name].copy()

        raw_total = len(raw_ds)
        clean_total = len(clean_ds)

        null_or_empty_before = int(
            raw_ds[["text_1", "text_2"]]
            .fillna("")
            .astype(str)
            .apply(lambda col: col.str.strip().eq(""))
            .any(axis=1)
            .sum()
        )
        empty_after = int(
            ((clean_ds["text_1_clean"].fillna("").str.strip() == "") | (clean_ds["text_2_clean"].fillna("").str.strip() == ""))
            .sum()
        )

        duplicated_pairs = int(clean_ds.duplicated(subset=["text_1_clean", "text_2_clean"], keep=False).sum())
        reversed_unique, reversed_records = _reversed_pairs_count(clean_ds)

        lines.append(f"\nDataset: {dataset_name}")
        lines.append(f"  - Rekordy surowe: {raw_total}")
        lines.append(f"  - Rekordy po preprocessingu (przed filtracja pustych): {clean_total}")
        lines.append(f"  - Null/puste przed czyszczeniem: {null_or_empty_before} ({_safe_pct(null_or_empty_before, raw_total):.2f}%)")
        lines.append(f"  - Puste po czyszczeniu: {empty_after} ({_safe_pct(empty_after, clean_total):.2f}%)")
        lines.append(f"  - Duplikaty par (identyczne text_1_clean/text_2_clean): {duplicated_pairs}")
        lines.append(f"  - Pary odwrocone (unikalne): {reversed_unique}")
        lines.append(f"  - Rekordy nalezace do par odwroconych: {reversed_records}")

    lines.append("")
    return "\n".join(lines)


def lexical_report(df_clean_nonempty: pd.DataFrame, top_n: int = 15) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EDA - LEKSYKA PER DATASET")
    lines.append("=" * 72)

    for dataset_name, subset in df_clean_nonempty.groupby("dataset"):
        tokens = _token_stream(subset)
        token_counter = Counter(tokens)
        vocab_size = len(token_counter)
        hapax_count = sum(1 for c in token_counter.values() if c == 1)
        hapax_pct = _safe_pct(hapax_count, vocab_size)
        top_tokens, top_bigrams = _top_tokens_and_bigrams(subset, top_n=top_n)

        lines.append(f"\nDataset: {dataset_name} (n={len(subset)})")
        lines.append(f"  - Rozmiar slownika po preprocessingu: {vocab_size}")
        lines.append(f"  - Slowa wystepujace tylko raz: {hapax_count} ({hapax_pct:.2f}% slownika)")

        lines.append(f"  - Top {top_n} tokenow:")
        for token, count in top_tokens:
            lines.append(f"      * {token}: {count}")

        lines.append(f"  - Top {top_n} bigramow:")
        for bigram, count in top_bigrams:
            lines.append(f"      * {bigram}: {count}")

    lines.append("")
    return "\n".join(lines)


def cosine_edge_cases_report(df_clean_nonempty: pd.DataFrame, top_n: int = 3) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EDA - PRZYKLADY TF-IDF COSINE (POTENCJALNIE MYLACE)")
    lines.append("=" * 72)

    for dataset_name, subset in df_clean_nonempty.groupby("dataset"):
        lines.append(f"\nDataset: {dataset_name}")
        false_similar, false_dissimilar = _cosine_confusion_examples(subset, top_n=top_n)

        lines.append("  Falszywie podobne (label=0, wysoki cosine):")
        if false_similar.empty:
            lines.append("    - brak")
        else:
            for _, row in false_similar.iterrows():
                lines.append("    " + "-" * 60)
                lines.append(
                    f"    pair_id={row['pair_id']} split={row['split']} cosine={row['cosine_tfidf']:.4f}"
                )
                lines.append(f"    text_1: {row['text_1']}")
                lines.append(f"    text_2: {row['text_2']}")

        lines.append("  Falszywie niepodobne (label=1, niski cosine):")
        if false_dissimilar.empty:
            lines.append("    - brak")
        else:
            for _, row in false_dissimilar.iterrows():
                lines.append("    " + "-" * 60)
                lines.append(
                    f"    pair_id={row['pair_id']} split={row['split']} cosine={row['cosine_tfidf']:.4f}"
                )
                lines.append(f"    text_1: {row['text_1']}")
                lines.append(f"    text_2: {row['text_2']}")

    lines.append("")
    return "\n".join(lines)


def sample_examples_report(df: pd.DataFrame, n: int = 3) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EDA - PRZYKLADOWE REKORDY PER DATASET")
    lines.append("=" * 72)

    for dataset_name, ds_subset in df.groupby("dataset"):
        lines.append(f"\nDataset: {dataset_name}")
        for label_value in sorted(ds_subset["label"].dropna().unique()):
            subset = ds_subset[ds_subset["label"] == label_value].head(n)
            lines.append(f"  Przkladowe pary dla label={label_value}:")
            if subset.empty:
                lines.append("    - brak")
                continue

            for _, row in subset.iterrows():
                lines.append("    " + "-" * 60)
                lines.append(f"    split={row['split']} pair_id={row['pair_id']}")
                lines.append(f"    text_1      : {row['text_1']}")
                lines.append(f"    text_2      : {row['text_2']}")
                lines.append(f"    text_1_clean: {row['text_1_clean']}")
                lines.append(f"    text_2_clean: {row['text_2_clean']}")

    lines.append("")
    return "\n".join(lines)


def run_eda(
    collections_dir: Optional[Path] = None,
    quora_sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if collections_dir is None:
        collections_dir = resolve_project_root() / "collections"

    df_raw = load_stage1_datasets(
        collections_dir=collections_dir,
        quora_sample_size=quora_sample_size,
        random_state=random_state,
    )
    df_clean_all = preprocess_pairs_dataframe(df_raw, drop_empty_after_cleaning=False)
    df_clean = preprocess_pairs_dataframe(df_raw, drop_empty_after_cleaning=True)
    df_eda = add_text_statistics(df_clean)

    root = resolve_project_root()
    eda_dir = root / "artifacts" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    full_report = "\n\n".join(
        [
            dataset_overview(df_eda),
            class_distribution_by_dataset(df_eda),
            text_length_report(df_eda),
            suspicious_examples_report(df_eda),
            data_quality_report(df_raw, df_clean_all),
            lexical_report(df_clean, top_n=15),
            cosine_edge_cases_report(df_clean, top_n=3),
            sample_examples_report(df_eda, n=3),
        ]
    )

    save_text_report(full_report, eda_dir / "eda_report.txt")

    plot_label_distribution(df_eda, eda_dir / "label_distribution.png")
    plot_word_length_histogram(df_eda, "text_1_clean_word_len", eda_dir / "text_1_clean_word_len_hist.png")
    plot_word_length_histogram(df_eda, "text_2_clean_word_len", eda_dir / "text_2_clean_word_len_hist.png")

    print("=" * 72)
    print("EDA zakonczona.")
    print(f"Raport zapisano do: {eda_dir / 'eda_report.txt'}")
    print(f"Wykresy zapisano do katalogu: {eda_dir}")
    print("=" * 72)

    return df_eda


if __name__ == "__main__":
    run_eda()
