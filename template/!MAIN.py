import os, json, warnings, time
import projekt.Detektor_Plagiatow.template.plots as plots
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import kruskal, wilcoxon

warnings.filterwarnings("ignore")
##################################
##################################
##################################
# Tutaj po przecinku, możesz dodać / usuwać algorytmy, które porównujesz
##################################
ALGS = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB()
}
MISSING = [0.0, 0.1, 0.2, 0.3] # Ilość missingów, używane do robustness, żeby pokazać jak algorytm sobie radzi z brakującymi danymi
# wyniki na wykresie i w raporcie
##################################
##################################
##################################
##################################
ALPHA = 0.005

def table(names, data, metrics, border_char='-', wall_char='|'):
    col_widths = []
    for m in metrics:
        max_val_len = 0
        for n in names:
            if m in data[n]:
                val = f"{np.mean(data[n][m]):.3f}"
                max_val_len = max(max_val_len, len(val))
        col_widths.append(max(len(str(m)), max_val_len, 8))
    
    header = f"{wall_char} {'Algorytm':<14} {wall_char} " + f" {wall_char} ".join([f"{str(m):<{w}}" for m, w in zip(metrics, col_widths)]) + f" {wall_char}"
    divider = border_char * len(header)
    res = [divider, header, divider]
    for n in names:
        vals = []
        for m, w in zip(metrics, col_widths):
            if m in data[n]:
                v = f"{np.mean(data[n][m]):.3f}"
            else:
                v = "N/A"
            vals.append(f"{v:<{w}}")
        row = f"{wall_char} {n:<14} {wall_char} " + f" {wall_char} ".join(vals) + f" {wall_char}"
        res.append(row)
    res.append(divider)
    return "\n".join(res)

def p_table(names, samples, border_char='-', wall_char='|'):
    comparisons = (len(names) * (len(names) - 1)) / 2
    col_widths = [max(len(n), 10) for n in names]
    header = f"{wall_char} {'P-Value (Bonf)':<14} {wall_char} " + f" {wall_char} ".join([f"{n:<{w}}" for n, w in zip(names, col_widths)]) + f" {wall_char}"
    divider = border_char * len(header)
    res = [divider, header, divider]
    
    # Kruskal-Wallisa
    all_scores = [samples[n]['bal_acc'] for n in names]
    stat, p_omnibus = kruskal(*all_scores)
    omnibus_res = f"Kruskal-Wallis Test: stat={stat:.3f}, p={p_omnibus:.4f}"
    
    for n1 in names:
        row_vals = []
        for n2 in names:
            if n1 == n2:
                row_vals.append("-" * 10)
            else:
                _, p = wilcoxon(samples[n1]['bal_acc'], samples[n2]['bal_acc'])
                adj_p = min(1.0, p * comparisons)  # Poprawka Bonferroniego - pamiętaj o tym w artykule/pracy, żeby wspomnieć, że użyto!!!!!!!!
                sym = "*" if adj_p < ALPHA else " "
                row_vals.append(f"{adj_p:.3f}{sym}")
        
        row_str = f"{wall_char} {n1:<14} {wall_char} " + f" {wall_char} ".join([f"{v:<{w}}" for v, w in zip(row_vals, col_widths)]) + f" {wall_char}"
        res.append(row_str)
    res.append(divider)
    return omnibus_res + "\n" + "\n".join(res)

def main():
    os.makedirs("wyniki/wykresy/detailed", exist_ok=True)
    os.makedirs("wyniki/csv_eksport", exist_ok=True)
    datasets = [f for f in os.listdir("zbiory") if f.endswith(".csv")]
    all_res = {}
    m_keys = ['accuracy', 'bal_acc', 'f1_score', 'precision', 'recall', 'time']
    glob_metrics = {n: {m: [] for m in m_keys} for n in ALGS}
    glob_rob = {n: {str(r): [] for r in MISSING} for n in ALGS}
    glob_preds = {n: [] for n in ALGS}
    per_ds = []

    for ds in tqdm(datasets, desc="Analiza zbiorów danych"):
        df = pd.read_csv(f"zbiory/{ds}")
        X = pd.get_dummies(df.iloc[:, :-1]).astype(float)
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        ds_met = {n: {m: [] for m in m_keys} for n in ALGS}
        ds_rob = {n: {str(r): [] for r in MISSING} for n in ALGS}
        ds_preds = {n: [] for n in ALGS}

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(X, y):
            imp, sc = SimpleImputer(), StandardScaler()
            X_tr, X_te = sc.fit_transform(imp.fit_transform(X.iloc[tr])), sc.transform(imp.transform(X.iloc[te]))
            y_tr, y_te = y[tr], y[te]
            
            for n, m in ALGS.items():
                start_time = time.time()
                m.fit(X_tr, y_tr)
                y_p = m.predict(X_te)
                elapsed_time = time.time() - start_time
                ds_preds[n].extend(y_p.tolist())
                glob_preds[n].extend(y_p.tolist())
                
                vals = {
                    'accuracy': accuracy_score(y_te, y_p),
                    'bal_acc': balanced_accuracy_score(y_te, y_p),
                    'f1_score': f1_score(y_te, y_p, average='weighted', zero_division=0),
                    'precision': precision_score(y_te, y_p, average='weighted', zero_division=0),
                    'recall': recall_score(y_te, y_p, average='weighted', zero_division=0),
                    'time': elapsed_time
                }
                for k, v in vals.items():
                    ds_met[n][k].append(v)
                    glob_metrics[n][k].append(v)
                
                for r in MISSING:
                    X_m = X.iloc[te].astype(float).copy()
                    mask = np.random.RandomState(42).rand(*X_m.shape) < r
                    X_m[mask] = np.nan
                    y_pm = m.predict(sc.transform(imp.transform(X_m)))
                    v = balanced_accuracy_score(y_te, y_pm)
                    ds_rob[n][str(r)].append(v)
                    glob_rob[n][str(r)].append(v)

        imb = ", ".join([f"{v:.2f}" for v in pd.Series(y).value_counts(normalize=True).values])
        header_text = (
            "\n\n"
            "//////////////////////////\n"
            f"DATASET: {ds.upper()}\n"
            f"Classes: {len(np.unique(y))}\n"
            f"Imbalance: {imb}"
        )
        per_ds.append(header_text)
        per_ds.append(table(ALGS.keys(), ds_met, m_keys))
        per_ds.append("\n\n")
        all_res[ds] = {"metrics": ds_met, "robustness": ds_rob, "predictions": ds_preds}

    names = list(ALGS.keys())
    rep = ["SUMMARY REPORT\n", "GLOBAL METRICS (AVERAGE)\n"]
    rep.append(table(names, glob_metrics, m_keys))
    rep.append("\n\n")
    
    rep.append("="*80)
    rep.append("STATISTICAL SIGNIFICANCE ANALYSIS (BONFERRONI CORRECTION APPLIED)")
    rep.append("="*80 + "\n")
    rep.append(p_table(names, glob_metrics, border_char='~', wall_char='!'))
    rep.append("\n\n")
    
    rep.append("GLOBAL ROBUSTNESS TO MISSING DATA\n")
    rob_avg = {n: {str(r): [np.mean(glob_rob[n][str(r)])] for r in MISSING} for n in names}
    rep.append(table(names, rob_avg, [str(r) for r in MISSING], border_char='=', wall_char='+'))
    rep.append("\n\n")
    
    rep.append("JACCARD SIMILARITY (TOTAL AGREEMENT)\n")
    j_mat = {n: {n2: [jaccard_score(glob_preds[n], glob_preds[n2], average='micro')] for n2 in names} for n in names}
    rep.append(table(names, j_mat, names, border_char='*', wall_char='>'))
    rep.append("\n\n")
    
    thick_line = "#" * 100
    rep.append(f"\n\n\n{thick_line}\n{thick_line}\n{thick_line}\nDETAILED DATASET RESULTS\n{thick_line}\n{thick_line}\n{thick_line}\n\n\n")
    rep.extend(per_ds)

    with open("wyniki/raport.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    with open("wyniki/wyniki.json", "w", encoding="utf-8") as f:
        json.dump(all_res, f)

    # EKSPORT DO CSV
    df_glob_metrics = pd.DataFrame({n: {m: np.mean(glob_metrics[n][m]) for m in m_keys} for n in names}).T
    df_glob_metrics.to_csv("wyniki/csv_eksport/1_global_metrics.csv", index_label="Algorithm")
    
    comparisons = (len(names) * (len(names) - 1)) / 2
    p_val_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for n1 in names:
        for n2 in names:
            if n1 == n2:
                p_val_mat.loc[n1, n2] = np.nan
            else:
                _, p = wilcoxon(glob_metrics[n1]['bal_acc'], glob_metrics[n2]['bal_acc'])
                p_val_mat.loc[n1, n2] = min(1.0, p * comparisons)
    p_val_mat.to_csv("wyniki/csv_eksport/2_statistical_significance_bonferroni.csv", index_label="Algorithm")
    
    df_rob = pd.DataFrame({n: {str(r): np.mean(glob_rob[n][str(r)]) for r in MISSING} for n in names}).T
    df_rob.to_csv("wyniki/csv_eksport/3_robustness_missing_data.csv", index_label="Algorithm")
    
    jaccard_df = pd.DataFrame(j_mat).T
    jaccard_df.to_csv("wyniki/csv_eksport/4_jaccard_similarity.csv", index_label="Algorithm")
    
    records_csv = []
    for ds_name, ds_data in all_res.items():
        for alg_name, metrics_data in ds_data['metrics'].items():
            row = {"Dataset": ds_name.replace(".csv", ""), "Algorithm": alg_name}
            for m in m_keys:
                if m in metrics_data:
                    row[m] = np.mean(metrics_data[m])
            records_csv.append(row)
    pd.DataFrame(records_csv).to_csv("wyniki/csv_eksport/5_detailed_all_datasets.csv", index=False)

    print("Raport w folderze 'wyniki/raport.txt' oraz pliki .csv w 'wyniki/csv_eksport/'.")
    plots.generate_plots()
    print("Wykresy w folderze 'wyniki/wykresy/'.")

if __name__ == "__main__":
    main()
