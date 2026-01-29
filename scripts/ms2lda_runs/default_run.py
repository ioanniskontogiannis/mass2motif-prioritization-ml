import sys
sys.path.insert(0, '../../')
import MS2LDA
from matchms.importing import load_from_mgf
import time


dataset_parameters = {
        "acquisition_type": "DDA",
        "charge": 1,
        "significant_digits": 2,
        "name": "cleaned_pos_results",
        "output_folder": "/home/ioannis/thesis_data/positive_test_results_filtered"
    }
train_parameters =  {
        "parallel": 3,
        "workers": 0
    }
model_parameters = {
        "rm_top": 0,
        "min_cf": 0,
        "min_df": 3,
        "alpha": 0.6,
        "eta": 0.01,
        "seed": 42
    }
convergence_parameters = {
        "step_size": 50,
        "window_size": 10,
        "threshold": 0.005,
        "type": "perplexity_history"
    }
annotation_parameters = {
        "criterium": "biggest",
        "cosine_similarity": 0.90,
        "n_mols_retrieved": 5,
        "s2v_model_path": "/home/ioannis/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_Spec2Vec_pos_CleanedLibraries.model",
        "s2v_library_embeddings": "/home/ioannis/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_CleanedLibraries_Spec2Vec_pos_embeddings.npy",
        "s2v_library_db": "/home/ioannis/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_CombLibraries_spectra.db"
    }
preprocessing_parameters = {
        "min_mz": 0,
        "max_mz": 2000,
        "max_frags": 1000,
        "min_frags": 5,
        "min_intensity": 0.01,
        "max_intensity": 1.0
    }
motif_parameter = 50
fingerprint_parameters = {
        "fp_type": "maccs",
        "threshold": 0.8
    }

dataset = list(load_from_mgf("/home/ioannis/thesis_data/testing_positive_cleaned_filtered.mgf"))
start= time.time()
n_motifs = 200
n_iterations = 5000
motif_spectra, optimized_motifs, motif_fps = MS2LDA.run(dataset, n_motifs=n_motifs, n_iterations=n_iterations,
        dataset_parameters=dataset_parameters,
        train_parameters=train_parameters,
        model_parameters=model_parameters,
        convergence_parameters=convergence_parameters,
        annotation_parameters=annotation_parameters,
        motif_parameter=motif_parameter,
        preprocessing_parameters=preprocessing_parameters,
        fingerprint_parameters=fingerprint_parameters)
print('It is working')

end =time.time()

elapsed = end - start
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"MS2LDA run completed in {minutes} min {seconds} sec")
