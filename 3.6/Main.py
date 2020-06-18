import sys
sys.path.insert(0, './Modules/')

import numpy as np

from file_reader import read_file
from mol_utils import get_fragments
from build_encoding import get_encodings, encode_molecule, decode_molecule, encode_list, save_decodings, save_encodings, read_decodings, read_encodings
from models import build_models
from training import train
from rewards import clean_good
from rdkit import rdBase
import logging
import pickle as pkl
logging.getLogger().setLevel(logging.INFO)
rdBase.DisableLog('rdApp.error')


def main(fragment_file, lead_file):
    fragment_mols = read_file(fragment_file)
    lead_mols = read_file(lead_file)
    fragment_mols += lead_mols

    logging.info("Read %s molecules for fragmentation library", len(fragment_mols))
    logging.info("Read %s lead moleculs", len(lead_mols))

    fragments, used_mols = get_fragments(fragment_mols)
    logging.info("Num fragments: %s", len(fragments))
    logging.info("Total molecules used: %s", len(used_mols))
    assert len(fragments)
    assert len(used_mols)
    encodings, decodings = get_encodings(fragments)
    save_encodings(encodings)
    save_decodings(decodings)
    logging.info("Saved encodings and decodings")
    lead_mols = np.asarray(fragment_mols[-len(lead_mols):])[used_mols[-len(lead_mols):]]
# =============================================================================
#     decodings = read_decodings()
#     encodings = read_encodings()
# =============================================================================
    X = encode_list(lead_mols, encodings)
    print(X.shape)
    if X.shape[0] == 0:
        return -1
    logging.info("Building models")
    actor, critic = build_models(X.shape[1:])

    #X = clean_good(X, decodings)
    print(X.shape)
    if X.shape[0] == 0:
        return -1
    logging.info("Training")
    history = train(X, actor, critic, decodings)
    logging.info("Saving")
    np.save("History/history.npy", history)




if __name__ == "__main__":

    fragment_file = "Data/molecules.smi"
    lead_file = "Data/AKT_pchembl.csv"


    if len(sys.argv) > 1:
        fragment_file = sys.argv[1]

    if len(sys.argv) > 2:
        lead_file = sys.argv[2]

    main(fragment_file, lead_file)

