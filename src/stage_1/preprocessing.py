from __future__ import annotations

import re
import html
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

from reports import save_dataframe_csv


PREPROCESSED_COLUMNS = [
    "dataset",
    "split",
    "pair_id",
    "text_1",
    "text_2",
    "text_1_clean",
    "text_2_clean",
    "label",
]


def _strip_accents(text: str) -> str:
    """
    Usuwa znaki diakrytyczne po normalizacji Unicode.
    Dla anglojęzycznych danych zwykle nie zmienia wiele,
    ale pomaga w ujednoliceniu tekstu.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_quotes_and_dashes(text: str) -> str:
    """
    Ujednolica różne typy apostrofów i myślników.
    """
    replacements = {
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201C": '"',   # left double quote
        "\u201D": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u00A0": " ",   # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _expand_basic_contractions(text: str) -> str:
    """
    Lekka ekspansja najczęstszych angielskich skrótów.
    Nie robimy pełnego słownika, żeby nie przeinaczać znaczenia.
    """
    contraction_map = {
        r"\bcan't\b": "can not",
        r"\bwon't\b": "will not",
        r"\bn't\b": " not",
        r"\bi'm\b": "i am",
        r"\bit's\b": "it is",
        r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",
        r"\bthat's\b": "that is",
        r"\bwhat's\b": "what is",
        r"\bthere's\b": "there is",
        r"\bthey're\b": "they are",
        r"\bwe're\b": "we are",
        r"\byou're\b": "you are",
        r"\bi've\b": "i have",
        r"\bwe've\b": "we have",
        r"\bthey've\b": "they have",
        r"\byou've\b": "you have",
        r"\bi'd\b": "i would",
        r"\bwe'd\b": "we would",
        r"\bthey'd\b": "they would",
        r"\byou'd\b": "you would",
        r"\bi'll\b": "i will",
        r"\bwe'll\b": "we will",
        r"\bthey'll\b": "they will",
        r"\byou'll\b": "you will",
    }

    for pattern, repl in contraction_map.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_accents: bool = False,
    normalize_numbers: bool = False,
) -> str:
    """
    Ostrożny preprocessing pod zadanie wykrywania parafraz/plagiatu.

    Założenia:
    - zachowujemy treść semantyczną,
    - nie usuwamy brutalnie stopwords,
    - nie stosujemy stemmingu/lemmatyzacji na siłę,
    - zostawiamy negacje, bo zmieniają sens zdania.
    """
    if text is None:
        return ""

    text = str(text)

    text = html.unescape(text)
    text = _normalize_quotes_and_dashes(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    if lowercase:
        text = text.lower()

    text = _expand_basic_contractions(text)

    # usuń URL-e i maile
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

    # usuń znaczniki HTML
    text = re.sub(r"<[^>]+>", " ", text)

    if remove_accents:
        text = _strip_accents(text)

    # opcjonalne ujednolicenie liczb
    if normalize_numbers:
        text = re.sub(r"\b\d+(?:[\.,]\d+)?\b", " <NUM> ", text)

    # zostawiamy litery, cyfry, apostrof i myślnik wewnątrz tekstu
    # resztę zamieniamy na spacje
    text = re.sub(r"[^a-zA-Z0-9\s'\-]", " ", text)

    # wielokrotne myślniki/apostrofy upraszczamy
    text = re.sub(r"[-]{2,}", "-", text)
    text = re.sub(r"[']{2,}", "'", text)

    # redukcja wielokrotnych spacji
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_pairs_dataframe(
    df: pd.DataFrame,
    lowercase: bool = True,
    remove_accents: bool = False,
    normalize_numbers: bool = False,
    drop_identical_after_cleaning: bool = False,
    drop_empty_after_cleaning: bool = True,
) -> pd.DataFrame:
    """
    Przyjmuje ramkę po etapie 1:
    ['dataset', 'split', 'pair_id', 'text_1', 'text_2', 'label']

    Zwraca ramkę z dodatkowymi kolumnami:
    - text_1_clean
    - text_2_clean
    """
    required_cols = {"dataset", "split", "pair_id", "text_1", "text_2", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje wymaganych kolumn: {sorted(missing)}")

    processed = df.copy()

    processed["text_1"] = processed["text_1"].fillna("").astype(str)
    processed["text_2"] = processed["text_2"].fillna("").astype(str)

    processed["text_1_clean"] = processed["text_1"].apply(
        lambda x: clean_text(
            x,
            lowercase=lowercase,
            remove_accents=remove_accents,
            normalize_numbers=normalize_numbers,
        )
    )
    processed["text_2_clean"] = processed["text_2"].apply(
        lambda x: clean_text(
            x,
            lowercase=lowercase,
            remove_accents=remove_accents,
            normalize_numbers=normalize_numbers,
        )
    )

    if drop_empty_after_cleaning:
        # opcjonalnie usuń rekordy, które po czyszczeniu stały się puste
        processed = processed[
            (processed["text_1_clean"].str.len() > 0) &
            (processed["text_2_clean"].str.len() > 0)
        ].copy()

    if drop_identical_after_cleaning:
        processed = processed[
            processed["text_1_clean"] != processed["text_2_clean"]
        ].copy()

    processed["label"] = pd.to_numeric(processed["label"], errors="coerce")
    processed = processed.dropna(subset=["label"]).copy()
    processed["label"] = processed["label"].astype(int)

    return processed[PREPROCESSED_COLUMNS].reset_index(drop=True)


def build_tfidf_corpus(df: pd.DataFrame) -> pd.Series:
    """
    Buduje wspólny korpus do fitowania TF-IDF.
    Ważne: fit robimy na wszystkich tekstach treningowych, a nie osobno
    dla text_1 i text_2.
    """
    required_cols = {"text_1_clean", "text_2_clean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje wymaganych kolumn: {sorted(missing)}")

    return pd.concat(
        [df["text_1_clean"], df["text_2_clean"]],
        axis=0,
        ignore_index=True
    )


def preprocessing_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    """
    Krótki raport kontrolny po preprocessingu.
    """
    print("=" * 72)
    print("RAPORT PREPROCESSINGU")
    print("=" * 72)
    print(f"Liczba wierszy przed: {len(df_before)}")
    print(f"Liczba wierszy po   : {len(df_after)}")
    print(f"Usunięte wiersze    : {len(df_before) - len(df_after)}")
    print()

    print("Przykładowe rekordy po czyszczeniu:")
    preview = df_after[["dataset", "split", "label", "text_1_clean", "text_2_clean"]].head(5)
    for idx, row in preview.iterrows():
        print("-" * 72)
        print(f"[{idx}] dataset={row['dataset']} split={row['split']} label={row['label']}")
        print(f"text_1_clean: {row['text_1_clean']}")
        print(f"text_2_clean: {row['text_2_clean']}")
    print("-" * 72)


if __name__ == "__main__":
    # przykład integracji z Twoim etapem 1
    from data_loading import load_stage1_datasets, resolve_project_root

    root = resolve_project_root()
    collections_dir = root / "collections"

    df_raw = load_stage1_datasets(collections_dir=collections_dir)

    df_clean = preprocess_pairs_dataframe(
        df_raw,
        lowercase=True,
        remove_accents=False,
        normalize_numbers=False,
        drop_identical_after_cleaning=False,
    )

    preprocessing_report(df_raw, df_clean)

    output_dir = root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "pairs_preprocessed.csv"
    save_dataframe_csv(df_clean, output_path, index=False)

    print()
    print(f"Zapisano wynik preprocessingu do: {output_path}")