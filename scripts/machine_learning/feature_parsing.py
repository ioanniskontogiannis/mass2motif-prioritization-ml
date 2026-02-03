
import pickle
import numpy as np
import pandas as pd
import tomotopy as tp
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import load_motifDB, motifDB2motifs
from motif_utils import (
    process_motifs,
    build_motif_to_docs,
    check_motif_spectrally,
    compute_mcs_support,
)

# 1. Load MS2LDA run

RUN_PATH = "/home/ioannis/thesis_data/negative_test_results_filtered"

motifDB_1, motifDB_2 = load_motifDB(f"{RUN_PATH}/motifset_optimized.json")
motifs = motifDB2motifs(motifDB_2)

with open(f"{RUN_PATH}/doc2spec_map.pkl", "rb") as f:
    doc2spec_map = pickle.load(f)

lda_model = tp.LDAModel.load(f"{RUN_PATH}/ms2lda.bin")



# 2. Build motif_to_docs

prob_threshold = 0.5
motif_to_docs = build_motif_to_docs(lda_model, prob_threshold)

# 3. Run process_motifs and save the results

results_sos = process_motifs(
    motifs,
    lda_model,
    doc2spec_map,
    motif_to_docs,
    prob_threshold=prob_threshold,
    sim_threshold=0.5,
    sos_cal=True
)

import pickle

cache = {
    "results_sos": results_sos,
    "motif_to_docs": motif_to_docs
}

with open("/home/ioannis/thesis_data/negative_test_results_filtered/motif_cache.pkl", "wb") as f:
    pickle.dump(cache, f)


# 4. Extract features
rows = []
for motif_id in results_sos["motif_ids"]:

    # Structural metrics
    idx = results_sos["motif_ids"].index(motif_id)
    intra = results_sos["intra_sims"][idx]
    inter = results_sos["inter_sims"][idx]
    mcs_size = results_sos["num_atoms"][idx]
    n_peaks = results_sos["len_frag_loss"][idx]
    mcs_support = compute_mcs_support(results_sos, motif_id)

    # Spectral metrics
    info = check_motif_spectrally(
        motif_id,
        motifs,
        motif_to_docs,
        doc2spec_map,
        lda_model,
        tol=0.01
    )

    if info is None:
        continue

    pf = np.array(info["peak_fractions"])

    rows.append({
        "motif_id": motif_id,
        "intra": intra,
        "inter": inter,
        "mcs_size": mcs_size,
        "mcs_support": mcs_support,
        "n_peaks": n_peaks,
        "n_spectra": info["n_spectra"],
        "peak_mean": pf.mean(),
        "peak_std": pf.std(),
        "peak_min": pf.min(),
        "peak_max": pf.max(),
    })

df = pd.DataFrame(rows)

# Load the trained model
with open("motif_priority_model_2.pkl", "rb") as f:
    model = pickle.load(f)

# To ensure that the order of the features is exactly the same with the one used in training
feature_order = [
    "intra",
    "inter",
    "mcs_size",
    "mcs_support",
    "n_peaks",
    "n_spectra",
    "peak_mean",
    "peak_std",
    "peak_min",
    "peak_max"
]

X_new = df[feature_order]
df["predicted_score"] = model.predict(X_new)
df_sorted = df.sort_values("predicted_score", ascending=False)

output_path = "/home/ioannis/thesis_data/negative_test_results_filtered/neg_prioritized_motifs_filtered_2.xlsx"
df_sorted.to_excel(output_path, index=False)



print("Motif prioritization complete.")

