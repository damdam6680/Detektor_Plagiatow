from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def plot_label_distribution(df: pd.DataFrame, output_path: Path) -> Path:
	counts = df["label"].value_counts().sort_index()

	plt.figure(figsize=(8, 5))
	counts.plot(kind="bar")
	plt.title("Rozkład klas label")
	plt.xlabel("label")
	plt.ylabel("Liczba rekordów")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150)
	plt.close()
	return output_path


def plot_word_length_histogram(df: pd.DataFrame, column: str, output_path: Path) -> Path:
	plt.figure(figsize=(8, 5))
	df[column].dropna().plot(kind="hist", bins=30)
	plt.title(f"Histogram długości: {column}")
	plt.xlabel("Liczba słów")
	plt.ylabel("Liczba rekordów")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150)
	plt.close()
	return output_path


def plot_text_length_boxplot(df: pd.DataFrame, output_path: Path) -> Path:
	plt.figure(figsize=(8, 5))
	df[["text_1_clean_word_len", "text_2_clean_word_len"]].dropna().boxplot()
	plt.title("Boxplot długości tekstów")
	plt.ylabel("Liczba słów")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150)
	plt.close()
	return output_path


def plot_word_len_diff_histogram(df: pd.DataFrame, output_path: Path) -> Path:
	plt.figure(figsize=(8, 5))
	df["word_len_diff"].dropna().plot(kind="hist", bins=30)
	plt.title("Histogram różnicy długości (word_len_diff)")
	plt.xlabel("|len(text_1) - len(text_2)|")
	plt.ylabel("Liczba rekordów")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150)
	plt.close()
	return output_path


def plot_cosine_tfidf_distribution_by_label(df: pd.DataFrame, output_path: Path) -> Path:
	if df.empty:
		raise ValueError("Nie mozna narysowac rozkladu cosine TF-IDF dla pustego DataFrame.")

	vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.99, lowercase=False)
	joint_corpus = pd.concat([df["text_1_clean"], df["text_2_clean"]], ignore_index=True)
	vectorizer.fit(joint_corpus)

	x1 = vectorizer.transform(df["text_1_clean"])
	x2 = vectorizer.transform(df["text_2_clean"])
	cosine_scores = np.asarray(x1.multiply(x2).sum(axis=1)).ravel()

	plot_df = pd.DataFrame({"label": df["label"].astype(int), "cosine_tfidf": cosine_scores})

	plt.figure(figsize=(9, 5))
	for label_value, color in [(0, "tab:orange"), (1, "tab:blue")]:
		series = plot_df.loc[plot_df["label"] == label_value, "cosine_tfidf"]
		if series.empty:
			continue
		plt.hist(series, bins=40, alpha=0.5, density=True, label=f"label={label_value}", color=color)

	plt.title("Rozklad cosine TF-IDF dla label=0 i label=1")
	plt.xlabel("cosine TF-IDF")
	plt.ylabel("Gestosc")
	plt.legend()
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150)
	plt.close()
	return output_path
