
import sys
sys.path.insert(0, "/lustre/BIF/nobackup/konto008/MS2LDA")

import numpy as np
import sqlite3
import pickle
from tqdm import tqdm
from rdkit import Chem
import matchms.filtering as msfilters
from MS2LDA.Preprocessing.load_and_clean import load_mgf
from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, load_s2v_and_library


spectra_path_neg = "/lustre/BIF/nobackup/konto008/thesis_data/s2v_filtered_2_neg.mgf"
model_file_neg = "/lustre/BIF/nobackup/konto008/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_negative_mode/311025_s2v_filtered_neg.model"


dummy_object = {}

# create a dummy pickle file in order to bypass load_s2v_and_library
with open("dummy.pkl", "wb") as f:
    pickle.dump(dummy_object, f)

dummy_pkl_file = "dummy.pkl"

s2v_similarity, _ = load_s2v_and_library(model_file_neg, dummy_pkl_file)

spectra_neg = load_mgf(spectra_path_neg)

cleaned_spectra_neg = []
embeddings_neg = []


for spectrum in tqdm(spectra_neg):
    # metadata filters
    spectrum = msfilters.default_filters(spectrum)
    spectrum = msfilters.add_retention_index(spectrum)
    spectrum = msfilters.add_retention_time(spectrum)
    spectrum = msfilters.require_precursor_mz(spectrum)

    # normalize and filter peaks
    spectrum = msfilters.normalize_intensities(spectrum)
    spectrum = msfilters.select_by_relative_intensity(spectrum, 0.001, 1)
    spectrum = msfilters.select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)
    spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=500)
    spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=3)

    if spectrum: 
        
        smi = spectrum.get("smiles")
        mol = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(mol)
        smi2D = Chem.MolToSmiles(mol)
        
        spectrum.set("smiles", smi2D)
        cleaned_spectra_neg.append(spectrum)
        
        embedding = calc_embeddings(s2v_similarity, [spectrum])
        embeddings_neg.append(embedding)


x,y,z = np.array(embeddings_neg).shape

np.save("/lustre/BIF/nobackup/konto008/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_negative_mode/311025_filtered_Spec2Vec_neg_embeddings.npy", np.array(embeddings_neg).reshape(x,z))

def create_spectra_db_from_list(spectra_list, db_path):
    """Create an SQLite database to store spectra from a list of spectra objects."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS spectra (
            id TEXT PRIMARY KEY,
            smiles TEXT,
            spectrum BLOB
        )
    ''')
    conn.commit()

    for i, spectrum in enumerate(spectra_list):
        spectrum_id = i
        smiles = spectrum.get('smiles')
        spectrum_blob = pickle.dumps(spectrum)
        cursor.execute('''
            INSERT OR REPLACE INTO spectra (id, smiles, spectrum)
            VALUES (?, ?, ?)
        ''', (spectrum_id, smiles, spectrum_blob))

    conn.commit()
    conn.close()

create_spectra_db_from_list(cleaned_spectra_neg, "/lustre/BIF/nobackup/konto008/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_negative_mode/311025_filtered_spectra.db")

