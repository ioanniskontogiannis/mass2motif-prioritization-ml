import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.DataStructs import TanimotoSimilarity
from MS2LDA.utils import retrieve_spec4doc
from MS2LDA.Add_On.Fingerprints.FP_annotation import annotate_motifs as calc_fingerprints
from rdkit.Chem import rdFMCS
from matchms import Spectrum
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from typing import Optional, List

def safe_mol(smiles: Optional[str]):
    """
    Safely convert a SMILES string into an RDKit Mol object.

    Parameters
    ----------
    smiles : str or None
        A SMILES representation of a molecule. May be None or invalid.

    Returns
    -------
    Mol or None
        An RDKit Mol object if parsing succeeds, otherwise None.

    Notes
    -----
    This wrapper prevents RDKit parsing errors from interrupting
    downstream processing when annotations or candidate molecules
    contain invalid SMILES strings.
    """
    if smiles is None:
        return None
    try:
        return MolFromSmiles(smiles)
    except:
        return None


def calculate_sos(fp1, fp2):
    """
    Compute the Subset Overlap Score (SOS) between two binary fingerprints.

    Parameters
    ----------
    fp1, fp2 : list[int] or iterable of {0,1}
        Two molecular fingerprints represented as binary vectors.

    Returns
    -------
    float
        The fraction of 'on' bits in the smaller fingerprint that are
        also present in the larger fingerprint. Ranges from 0 to 1.

    Notes
    -----
    SOS is an asymmetric similarity measure. It quantifies whether the
    structural features of the smaller fingerprint are contained within
    the larger one, making it useful for motif–molecule consistency
    checks where subset relationships matter.
    """
    if sum(fp1) < sum(fp2):
        smaller_fp = fp1
        bigger_fp = fp2
    else:
        smaller_fp = fp2
        bigger_fp = fp1
    
    smaller_fp_sum = sum(smaller_fp)
    fp_intersection = 0
    for bit1, bit2 in zip(smaller_fp, bigger_fp):
        if bit1 == 1 and bit2 == 1:
            fp_intersection += 1

    if fp_intersection == 0:
        return 0
    else:
        return fp_intersection / smaller_fp_sum


def compute_pairwise_similarity(mols: List, threshold: float = 0.5) -> float:
    """
    Compute the fraction of molecule pairs with Tanimoto similarity above a threshold.

    Parameters
    ----------
    mols : list of RDKit Mol
        A list of molecules for which pairwise similarity is computed.
    threshold : float, optional
        Minimum Tanimoto similarity required for a pair to be counted
        as a match. Default is 0.5.

    Returns
    -------
    float
        The proportion of molecule pairs whose Tanimoto similarity
        meets or exceeds the threshold. Returns 0.0 if fewer than two
        molecules are provided.

    Notes
    -----
    This metric is used to quantify intra‑motif molecular homogeneity,
    i.e., whether molecules associated with a Mass2Motif tend to be
    structurally similar.
    """
    if len(mols) < 2:
        return 0.0
    fps = [RDKFingerprint(m) for m in mols]
    total = 0
    matches = 0
    n = len(fps)
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if TanimotoSimilarity(fps[i], fps[j]) >= threshold:
                matches += 1
    return matches / total if total > 0 else 0.0



def compute_mcs_num_atoms(mols):
    """
    Compute the Maximum Common Substructure (MCS) across a set of molecules.

    Parameters
    ----------
    mols : list of RDKit Mol
        Molecules for which the MCS is computed.

    Returns
    -------
    tuple
        (num_atoms, smarts_string)
        num_atoms : int
            Number of atoms in the MCS. Returns 0 if no MCS is found.
        smarts_string : str
            SMARTS representation of the MCS. Empty string if MCS fails.

    Notes
    -----
    Uses RDKit's FMCS algorithm with constraints that enforce chemically
    meaningful matches (e.g., ring‑only matching). This is used to assess
    structural coherence of motif annotations.
    """
    if not mols:
        return 0, ""
    try:
        mcs = rdFMCS.FindMCS(
            mols,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            completeRingsOnly=True,
            ringMatchesRingOnly=True,
            timeout=30,
        )
        return int(mcs.numAtoms), mcs.smartsString
    except:
        return 0, ""
    
def build_motif_to_docs(lda_model, prob_threshold: float = 0.5):
    """
    Construct a mapping from motif IDs to the spectra (documents) in which
    they appear above a given probability threshold.

    Parameters
    ----------
    lda_model : tomotopy.LDAModel
        The MS2LDA model containing per-document motif probabilities.
        Each document corresponds to one MS/MS spectrum.
    prob_threshold : float, optional
        Minimum motif probability required for a motif to be considered
        present in a document. Default is 0.5.

    Returns
    -------
    dict
        A dictionary mapping:
            motif_id (int) → list of document IDs (ints)
        where each document ID corresponds to a spectrum in which the
        motif is active above the specified threshold.

    Notes
    -----
    This mapping is fundamental for downstream motif evaluation. It
    determines which spectra contribute to:
        - spectral reproducibility checks
        - MCS support calculations
        - intra/inter molecular similarity
        - motif prevalence (degree)
    """
    motif_to_docs = defaultdict(list)
    for doc_id, doc in enumerate(lda_model.docs):
        for motif_id, prob in doc.get_topics():
            if prob >= prob_threshold:
                motif_to_docs[motif_id].append(doc_id)
    return motif_to_docs


def process_motifs(
    motifs,
    lda_model,
    doc2spec_map,
    motif_to_docs,
    prob_threshold: float = 0.5,
    sim_threshold: float = 0.5,
    sos_cal: bool = True
) -> dict:
    """
    Compute structural and similarity-based quality metrics for Mass2Motifs.

    Parameters
    ----------
    motifs : list
        Motif objects from a motifDB, each containing peaks and auto-annotations.
    lda_model : tomotopy.LDAModel
        MS2LDA model used to determine motif usage across documents.
    doc2spec_map : dict
        Mapping from document IDs to spectrum metadata (including SMILES).
    motif_to_docs : dict
        Mapping of motif_id → list of documents where the motif probability
        exceeds `prob_threshold`.
    prob_threshold : float, optional
        Minimum motif probability for including a document.
    sim_threshold : float, optional
        Tanimoto cutoff for candidate–candidate similarity.
    sos_cal : bool, optional
        If True, compute inter-similarity using SOS; otherwise use Tanimoto.

    Returns
    -------
    dict
        {
            "motif_ids": list of processed motif IDs,
            "num_atoms": MCS atom counts from annotation molecules,
            "len_frag_loss": number of MS2 features per motif,
            "mcs_smarts": SMARTS strings of MCS results,
            "molecules_by_motif": candidate RDKit molecules per motif,
            "intra_sims": similarity among candidate molecules,
            "inter_sims": similarity between representative fingerprint and candidates
        }

    Notes
    -----
    Motifs with ≤1 annotation or no qualifying documents are skipped.
    The function evaluates motif coherence by combining annotation chemistry,
    MS2LDA usage patterns, and molecular similarity metrics.
    """

    motif_ids = []
    num_atoms = []
    len_frag_loss = []
    mcs_smarts = []
    intra_sims = []
    inter_sims = []
    molecules_by_motif = {}

    for motif in tqdm(motifs):
        annotation = motif.get("auto_annotation", [])
        if len(annotation) <= 1:
            continue

        try:
            motif_id = int(motif.get("id").split("_")[1])
        except Exception:
            continue

        # if motif never appears in any doc above prob_threshold, skip
        if motif_id not in motif_to_docs:
            continue

        motif_ids.append(motif_id)
        len_frag_loss.append(len(getattr(motif.peaks, "mz", [])))

        # ---- MCS on annotation molecules ----
        ann_mols = [safe_mol(s) for s in annotation]
        ann_mols = [m for m in ann_mols if m is not None]
        n_atoms, smarts = compute_mcs_num_atoms(ann_mols)
        num_atoms.append(n_atoms)
        mcs_smarts.append(smarts)

        # ---- representative fingerprint from annotations ----
        rep_fp_ = calc_fingerprints([annotation], fp_type="maccs", threshold=0.9)[0]

        rep_fp = ExplicitBitVect(len(rep_fp_))
        for i, bit in enumerate(rep_fp_):
            if bit:
                rep_fp.SetBit(i)

        # ---- Collect candidate molecules using motif_to_docs ----
        candidate_mols = []
        for doc_id in motif_to_docs[motif_id]:
            spec = retrieve_spec4doc(doc2spec_map, lda_model, doc_id)
            mol = safe_mol(spec.get("smiles"))
            if mol is not None:
                candidate_mols.append(mol)

        molecules_by_motif[motif_id] = candidate_mols

        # ---- Intra similarity (candidates vs candidates) ----
        if len(candidate_mols) < 2:
            intra_sims.append(0.0)
        else:
            fps = [RDKFingerprint(m) for m in candidate_mols]
            total = 0
            matches = 0
            n = len(fps)
            for i in range(n):
                for j in range(i + 1, n):
                    total += 1
                    if TanimotoSimilarity(fps[i], fps[j]) >= sim_threshold:
                        matches += 1
            intra_sims.append(matches / total if total > 0 else 0.0)

        # ---- Inter similarity (motif representative vs candidates) ----
        if not candidate_mols:
            inter_sims.append(0.0)
        elif sos_cal:
            fps = [RDKFingerprint(m) for m in candidate_mols]
            rep_fp_list = list(rep_fp)

            sims = []
            for fp in fps:
                fp_list = list(fp)
                sim = calculate_sos(rep_fp_list, fp_list)
                sims.append(sim)

            inter_sims.append(float(np.mean(sims)) if sims else 0.0)
        else:
            fps = [RDKFingerprint(m) for m in candidate_mols]
            sims = [TanimotoSimilarity(rep_fp, fp) for fp in fps]
            inter_sims.append(float(np.mean(sims)) if sims else 0.0)

    return {
        "motif_ids": motif_ids,
        "num_atoms": num_atoms,
        "len_frag_loss": len_frag_loss,
        "mcs_smarts": mcs_smarts,
        "molecules_by_motif": molecules_by_motif,
        "intra_sims": intra_sims,
        "inter_sims": inter_sims
    }

def compute_mcs_support(results, motif_id):
    """
    Compute the fraction of candidate molecules that contain the
    Maximum Common Substructure (MCS) identified for a given motif.

    Parameters
    ----------
    results : dict
        Output dictionary from `process_motifs`, containing:
            - "motif_ids": list of motif IDs
            - "mcs_smarts": list of SMARTS strings for each motif's MCS
            - "molecules_by_motif": dict mapping motif_id → list of RDKit Mol objects
    motif_id : int
        The numeric ID of the motif whose MCS support is being evaluated.

    Returns
    -------
    float
        A value between 0 and 1 representing the proportion of molecules
        associated with the motif that contain the MCS as a substructure.
        Returns 0.0 if no molecules are available.

    Notes
    -----
    MCS support quantifies how consistently the computed MCS actually
    appears across the molecules linked to the motif. High support
    indicates strong structural coherence, while low support suggests
    that the MCS may be spurious or only present in a subset of molecules.
    """    
    # Get the SMARTS for this motif
    idx = results["motif_ids"].index(motif_id)
    smarts = results["mcs_smarts"][idx]
    mcs_query = Chem.MolFromSmarts(smarts)
    
    # Get candidate molecules
    mols = results["molecules_by_motif"][motif_id]
    total = len(mols)
    
    # Count how many contain the MCS
    support = 0
    for mol in mols:
        if mol.HasSubstructMatch(mcs_query):
            support += 1
    
    return support / total if total > 0 else 0.0

def get_spectrum(doc_id, doc2spec_map, lda_model):
    """
    Return (mzs, intensities) for a given document ID (matchms Spectrum).

    Parameters
    ----------
    doc_id : int
        Document ID from the MS2LDA model.
    doc2spec_map : dict
        Mapping from doc_id → spectrum metadata.
    lda_model : tomotopy.LDAModel
        The MS2LDA model used to retrieve the spectrum.

    Returns
    -------
    tuple of np.ndarray
        (mzs, intensities) arrays for the spectrum.
    """
    spec = retrieve_spec4doc(doc2spec_map, lda_model, doc_id)
    return np.array(spec.peaks.mz), np.array(spec.peaks.intensities)



def peak_present(mz_array, target, tol=0.01):
    """Check if a target m/z is present in a spectrum within tolerance."""
    return np.any(np.abs(mz_array - target) <= tol)


def get_motif_peaks(motifs, motif_id):
    """
    Return the list of m/z peaks for a given motif ID.

    Parameters
    ----------
    motifs : list
        List of motif objects from motifDB2motifs().
    motif_id : int
        The numeric motif ID (e.g., 73).

    Returns
    -------
    list of float
        The m/z values of the peaks defining the motif.
    """
    for m in motifs:
        try:
            mid = int(m.get("id").split("_")[1])
        except Exception:
            continue

        if mid == motif_id:
            return list(m.peaks.mz)

    raise ValueError(f"Motif {motif_id} not found in provided motifs.")


def check_motif_spectrally(motif_id, motifs, motif_to_docs, doc2spec_map, lda_model, tol=0.01):
    """
    Compute peak‑level spectral consistency statistics for a Mass2Motif.

    Parameters
    ----------
    motif_id : int
        The numeric ID of the motif (e.g., 73).
    motifs : list
        List of motif objects from motifDB2motifs().
    motif_to_docs : dict
        Mapping motif_id → list of document IDs where the motif
        appears above the probability threshold.
    doc2spec_map : dict
        Mapping from document IDs to spectrum metadata.
    lda_model : tomotopy.LDAModel
        The MS2LDA model used to retrieve spectra.
    tol : float, optional
        m/z tolerance for peak matching (default = 0.01 Da).

    Returns
    -------
    dict or None
        A dictionary containing:
            - motif_id : int
            - peaks : list of float
            - n_spectra : int
            - peak_counts : list of int
            - both_count : int
            - peak_fractions : list of float
            - both_fraction : float

        Returns None if the motif does not appear in any spectra.
    """

    # Get motif peaks
    peaks = get_motif_peaks(motifs, motif_id)

    # Get documents/spectra where motif is active
    if motif_id not in motif_to_docs:
        return None

    docs = motif_to_docs[motif_id]

    # Check peak presence across spectra
    presence_matrix = []
    for doc_id in docs:
        mzs, ints = get_spectrum(doc_id, doc2spec_map, lda_model)
        presence = [peak_present(mzs, p, tol) for p in peaks]
        presence_matrix.append(presence)

    # Summaries
    n = len(docs)
    peak_counts = [sum(row[i] for row in presence_matrix) for i in range(len(peaks))]
    both_count = sum(all(row) for row in presence_matrix)

    return {
        "motif_id": motif_id,
        "peaks": peaks,
        "n_spectra": n,
        "peak_counts": peak_counts,
        "both_count": both_count,
        "peak_fractions": [c / n for c in peak_counts],
        "both_fraction": both_count / n if n > 0 else 0.0,
    }


def print_peak_summary(info):
    """
    Print summary statistics for peak consistency.
    Expects the dictionary returned by check_motif().

    Returns
    -------
    Dict
    The summary statistics for peak consistency
    """
    peak_fractions = info["peak_fractions"]
    arr = np.array(peak_fractions)

    peak_mean = arr.mean()
    peak_std = arr.std()
    peak_min = arr.min()
    peak_max = arr.max()

    print("\nSummary statistics for peak consistency:")
    print(f"  Mean consistency     : {peak_mean:.3f}")
    print(f"  Std deviation        : {peak_std:.3f}")
    print(f"  Minimum consistency  : {peak_min:.3f}")
    print(f"  Maximum consistency  : {peak_max:.3f}")

    peak_statistics = {
        "peak_mean": peak_mean,
        "peak_std": peak_std,
        "peak_min": peak_min,
        "peak_max": peak_max,}

    return peak_statistics