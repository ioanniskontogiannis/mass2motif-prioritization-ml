
import sys
sys.path.insert(0, '../../')
import MS2LDA


preprocessing_parameters = {
    "min_mz": 0,
    "max_mz": 1000,
    "max_frags": 1000,
    "min_frags": 3,
    "min_intensity": 0.01,
    "max_intensity": 1
}

convergence_parameters = {
    "step_size": 50,
    "window_size": 10,
    "threshold": 0.001,
    "type": "perplexity_history"
}

annotation_parameters = {
    "criterium": "best", # return cluster with most compounds in it after optimization ("best" also an option)
    "cosine_similarity": 0.70, #0.8 how similar are the spectra compared to motifs in the optimization
    "n_mols_retrieved": 10, # 10 molecules retrieved from database by Spec2Vec
    "s2v_model_path": "../../MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_Spec2Vec_pos_CleanedLibraries.model",
    "s2v_library_embeddings": "../../MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_CleanedLibraries_Spec2Vec_pos_embeddings.npy",
    "s2v_library_db": "../../MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_CombLibraries_spectra.db",
}

n_motifs = 10 
n_iterations = 500

import random
random.seed(42)
model_parameters = {
    "rm_top": 0, 
    "min_cf": 0,
    "min_df": 3,
    "alpha": 0.6, #A higher alpha makes the document preferences "smoother" over topics
    "eta": 0.1, #and a higher eta makes the topic preferences "smoother" over words.
    "seed": 42,
}

train_parameters = {
    "parallel": 3,
    "workers": 0, 
}

dataset_parameters = {
    "acquisition_type": "DDA",
    "significant_digits": 2,
    "charge": 1,
    "name": "DDA-Suspectlist",
    "output_folder": "/home/ioannis/thesis_data/small_test_100motifs_output", 
}

fingerprint_parameters = {
    "fp_type": "maccs",
    "threshold": 0.8,
}

motif_parameter = 20

from matchms.importing import load_from_mgf
dataset = list(load_from_mgf("/home/ioannis/thesis_data/Case_Study_Fungal_dataset.mgf"))

motif_spectra, optimized_motifs, motif_fps = MS2LDA.run(dataset, n_motifs=n_motifs, n_iterations=n_iterations,
        dataset_parameters=dataset_parameters,
        train_parameters=train_parameters,
        model_parameters=model_parameters,
        convergence_parameters=convergence_parameters,
        annotation_parameters=annotation_parameters,
        motif_parameter=motif_parameter,
        preprocessing_parameters=preprocessing_parameters,
        fingerprint_parameters=fingerprint_parameters)