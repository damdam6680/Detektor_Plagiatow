import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import jaccard_score
from tqdm import tqdm

INPUT_JSON = "wyniki/wyniki.json"
OUTPUT_FOLDER = "wyniki/wykresy"
DETAILED_FOLDER = os.path.join(OUTPUT_FOLDER, "detailed")

def calculate_jaccard_similarity(preds1, preds2):
    return jaccard_score(preds1, preds2, average='micro')

def generate_plots():
    if not os.path.exists(INPUT_JSON):
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for folder in [OUTPUT_FOLDER, DETAILED_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    metrics_all = ['accuracy', 'bal_acc', 'f1_score', 'precision', 'recall']
    metrics_class = ['accuracy', 'bal_acc', 'f1_score', 'precision', 'recall']
    algs = list(next(iter(data.values()))['metrics'].keys())

    for metric in tqdm(metrics_all, desc="Generowanie wykresów (boxplots)"):
        records = []
        for ds_name, ds_data in data.items():
            for alg_name, metrics_data in ds_data['metrics'].items():
                if metric in metrics_data:
                    for val in metrics_data[metric]:
                        records.append({"Algorithm": alg_name, metric: val})
        
        if not records:
            continue
            
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='Algorithm', y=metric, data=pd.DataFrame(records), hue='Algorithm', palette='viridis', legend=False)
        plt.title(f"Metric: {metric.replace('_', ' ').capitalize()} (All Datasets)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"boxplot_{metric}.png"))
        plt.close()
        
    # Sredni czas na jednym barplocie i rozbicie na zbiory
    time_records = []
    for ds_name, ds_data in data.items():
        for alg_name, metrics_data in ds_data['metrics'].items():
            if 'time' in metrics_data:
                time_records.append({
                    "Algorithm": alg_name, 
                    "Dataset": ds_name.replace('.csv', ''),
                    "Time": np.mean(metrics_data['time'])
                })
    
    if time_records:
        df_time = pd.DataFrame(time_records)
        
        # Wykres nr 1: Średni czas wszystkich (z błędem)
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Algorithm', y='Time', data=df_time, errorbar='sd', capsize=.1, palette='mako', hue='Algorithm', legend=False)
        ax.set_ylim(bottom=0)
        plt.title("Average Execution Time (Train + Predict) - All Datasets")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "bar_time_avg.png"))
        plt.close()
        
        # Wykres nr 2: Czas w podziale na zbior danych
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Dataset', y='Time', hue='Algorithm', data=df_time, palette='viridis')
        plt.title("Execution Time per Dataset")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "bar_time_datasets.png"))
        plt.close()

    for ds_name, ds_data in tqdm(data.items(), desc="Generowanie wykresów (dla każdego zbioru)"):
        records = []
        for alg_name, metrics_data in ds_data['metrics'].items():
            for i in range(len(metrics_data['accuracy'])):
                row = {"Algorithm": alg_name}
                for m in metrics_class:
                    if m in metrics_data:
                        row[m] = metrics_data[m][i]
                records.append(row)
        
        df_ds = pd.DataFrame(records)
        df_melt = df_ds.melt(id_vars='Algorithm', value_vars=metrics_class, var_name='Metric', value_name='Value')
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Metric', y='Value', hue='Algorithm', data=df_melt)
        plt.title(f"Results for dataset: {ds_name}")
        min_v = max(0, df_melt['Value'].min() - 0.1)
        max_v = min(1, df_melt['Value'].max() + 0.1)
        plt.ylim(min_v, max_v)
        plt.tight_layout()
        plt.savefig(os.path.join(DETAILED_FOLDER, f"bar_{ds_name.replace('.csv', '')}.png"))
        plt.close()

    robust_records = []
    for ds_name, ds_data in data.items():
        for alg_name, rob_data in ds_data['robustness'].items():
            for rate, vals in rob_data.items():
                for v in vals:
                    robust_records.append({"Algorithm": alg_name, "MissingRate": f"{int(float(rate)*100)}%", "BA": v})
    
    df_rob = pd.DataFrame(robust_records)
    plt.figure(figsize=(12, 7))
    sns.lineplot(x='MissingRate', y='BA', hue='Algorithm', data=df_rob, marker='o')
    plt.title("Robustness to Missing Data (Balanced Accuracy)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "robustness_combined.png"))
    plt.close()

    for alg in tqdm(algs, desc="Generowanie wykresów robustness (dla każdego algorytmu)"):
        df_alg = df_rob[df_rob['Algorithm'] == alg]
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='MissingRate', y='BA', data=df_alg, marker='o', errorbar='sd')
        plt.title(f"Robustness: {alg}")
        plt.ylabel("Mean BA (+/- SD)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"robustness_{alg}.png"))
        plt.close()

    alg_full_preds = {alg: [] for alg in algs}
    for ds_data in data.values():
        for alg in algs:
            alg_full_preds[alg].extend(ds_data['predictions'][alg])
    
    jaccard_matrix = pd.DataFrame(index=algs, columns=algs, dtype=float)
    for a1 in algs:
        for a2 in algs:
            jaccard_matrix.loc[a1, a2] = calculate_jaccard_similarity(alg_full_preds[a1], alg_full_preds[a2])
    
    plt.figure(figsize=(11, 9))
    sns.heatmap(jaccard_matrix, annot=True, cmap='YlGnBu', fmt=".3f")
    plt.title("Jaccard Similarity Heatmap (Agreement Ratio)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "jaccard_similarity_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    generate_plots()