from __future__ import annotations

import argparse
from pathlib import Path

from data_loading import load_stage1_datasets, resolve_project_root
from eda import (
	add_text_statistics,
	class_distribution_by_dataset,
	cosine_edge_cases_report,
	data_quality_report,
	dataset_overview,
	lexical_report,
	sample_examples_report,
	suspicious_examples_report,
	text_length_report,
)
from plots import (
	plot_cosine_tfidf_distribution_by_label,
	plot_label_distribution,
	plot_text_length_boxplot,
	plot_word_len_diff_histogram,
	plot_word_length_histogram,
)
from preprocessing import preprocess_pairs_dataframe, preprocessing_report
from reports import save_dataframe_csv, save_eda_sections_csv
from tfidf_matrix import (
	assign_model_splits,
	compute_tfidf_pair_matrices,
	save_tfidf_artifacts,
	tfidf_summary,
)


def run_stage_1(
	quora_sample_size: int | None = None,
	random_state: int = 42,
) -> dict[str, Path]:
	root = resolve_project_root()
	collections_dir = root / "collections"
	results_dir = root / "results" / "stage_1"
	raports_dir = results_dir / "raports"
	plots_dir = results_dir / "plots"

	print("=" * 72)
	print("STAGE 1 - START")
	print("=" * 72)

	df_raw = load_stage1_datasets(
		collections_dir=collections_dir,
		quora_sample_size=quora_sample_size,
		random_state=random_state,
	)
	df_clean_all = preprocess_pairs_dataframe(df_raw, drop_empty_after_cleaning=False)
	df_clean = preprocess_pairs_dataframe(df_raw, drop_empty_after_cleaning=True)

	preprocessing_report(df_raw, df_clean)
	preprocessed_path = raports_dir / "pairs_preprocessed.csv"
	save_dataframe_csv(df_clean, preprocessed_path, index=False)

	df_eda = add_text_statistics(df_clean)
	eda_sections = {
		"dataset_overview": dataset_overview(df_eda),
		"class_distribution_by_dataset": class_distribution_by_dataset(df_eda),
		"text_length_report": text_length_report(df_eda),
		"suspicious_examples_report": suspicious_examples_report(df_eda),
		"data_quality_report": data_quality_report(df_raw, df_clean_all),
		"lexical_report": lexical_report(df_clean, top_n=15),
		"cosine_edge_cases_report": cosine_edge_cases_report(df_clean, top_n=3),
		"sample_examples_report": sample_examples_report(df_eda, n=3),
	}

	eda_report_path = raports_dir / "eda_report_sections.csv"
	save_eda_sections_csv(eda_sections, eda_report_path)
	plot_outputs: dict[str, Path] = {}
	dataset_plot_inputs = [("combined", df_eda)]
	dataset_plot_inputs.extend((name, subset.copy()) for name, subset in df_eda.groupby("dataset"))

	for dataset_name, subset in dataset_plot_inputs:
		if subset.empty:
			continue

		label_plot_path = plot_label_distribution(
			subset,
			plots_dir / f"{dataset_name}_label_distribution.png",
		)
		text1_plot_path = plot_word_length_histogram(
			subset,
			"text_1_clean_word_len",
			plots_dir / f"{dataset_name}_text_1_clean_word_len_hist.png",
		)
		text2_plot_path = plot_word_length_histogram(
			subset,
			"text_2_clean_word_len",
			plots_dir / f"{dataset_name}_text_2_clean_word_len_hist.png",
		)
		boxplot_path = plot_text_length_boxplot(
			subset,
			plots_dir / f"{dataset_name}_text_length_boxplot.png",
		)
		word_diff_hist_path = plot_word_len_diff_histogram(
			subset,
			plots_dir / f"{dataset_name}_word_len_diff_hist.png",
		)
		cosine_plot_path = plot_cosine_tfidf_distribution_by_label(
			subset,
			plots_dir / f"{dataset_name}_cosine_tfidf_by_label.png",
		)

		plot_outputs[f"{dataset_name}_label"] = label_plot_path
		plot_outputs[f"{dataset_name}_text1"] = text1_plot_path
		plot_outputs[f"{dataset_name}_text2"] = text2_plot_path
		plot_outputs[f"{dataset_name}_boxplot"] = boxplot_path
		plot_outputs[f"{dataset_name}_word_diff"] = word_diff_hist_path
		plot_outputs[f"{dataset_name}_cosine"] = cosine_plot_path

	df_model = assign_model_splits(df_clean, random_state=random_state)
	save_dataframe_csv(df_model, raports_dir / "pairs_preprocessed_with_model_split.csv", index=False)

	tfidf_result_joint = compute_tfidf_pair_matrices(
		df_model,
		fit_split="train",
		split_column="model_split",
	)
	joint_summary = tfidf_summary(tfidf_result_joint)
	save_tfidf_artifacts(
		tfidf_result_joint,
		output_dir=raports_dir,
		prefix="pairs_joint",
		summary_text=joint_summary,
	)
	print(joint_summary)

	for dataset_name in ["quora", "mrpc"]:
		df_dataset = df_model[df_model["dataset"] == dataset_name].reset_index(drop=True)
		if df_dataset.empty:
			continue

		tfidf_result_dataset = compute_tfidf_pair_matrices(
			df_dataset,
			fit_split="train",
			split_column="model_split",
		)
		dataset_summary = tfidf_summary(tfidf_result_dataset)
		save_tfidf_artifacts(
			tfidf_result_dataset,
			output_dir=raports_dir,
			prefix=f"pairs_{dataset_name}",
			summary_text=dataset_summary,
		)
		print(dataset_summary)

	print("=" * 72)
	print("STAGE 1 - KONIEC")
	print(f"Preprocessing CSV: {preprocessed_path}")
	print(f"EDA report       : {eda_report_path}")
	for plot_key, plot_path in sorted(plot_outputs.items()):
		print(f"EDA plot ({plot_key}): {plot_path}")
	print(f"Raports dir      : {raports_dir}")
	print(f"Plots dir        : {plots_dir}")
	print("=" * 72)

	result_paths = {
		"preprocessed_csv": preprocessed_path,
		"eda_report": eda_report_path,
		"raports_dir": raports_dir,
		"plots_dir": plots_dir,
	}
	result_paths.update(plot_outputs)
	return result_paths


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Uruchamia kompletny Stage 1.")
	parser.add_argument(
		"--quora-sample-size",
		type=int,
		default=None,
		help="Opcjonalna liczba próbek z Quora (domyślnie: pełny zbiór).",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Seed losowania używany przy próbkowaniu Quora.",
	)
	return parser


if __name__ == "__main__":
	args = _build_parser().parse_args()
	run_stage_1(
		quora_sample_size=getattr(args, "quora_sample_size", None),
		random_state=args.random_state,
	)
