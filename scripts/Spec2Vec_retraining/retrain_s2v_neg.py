from matchms.importing import load_from_mgf
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model

# Paths
spectra_path_neg = "/lustre/BIF/nobackup/konto008/thesis_data/s2v_filtered_2_neg.mgf"
model_file_neg = "/lustre/BIF/nobackup/konto008/thesis_data/311025_s2v_filtered_neg.model"

def main():
    print("Loading spectra...")
    spectrums_neg = load_from_mgf(spectra_path_neg)

    print("Converting to SpectrumDocuments...")
    spectrum_documents_neg = [SpectrumDocument(s, n_decimals=2) for s in spectrums_neg]

    print(f"Training Spec2Vec model on {len(spectrum_documents_neg)} spectra...")
    model_neg = train_new_word2vec_model(
        spectrum_documents_neg,
        iterations=[13],   
        filename=model_file_neg,
        workers=8,
        progress_logger=True
    )
    print("Training complete. Model saved at:", model_file_neg)

if __name__ == "__main__":
    main()

