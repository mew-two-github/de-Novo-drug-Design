import sys
sys.path.insert(0,'./Modules/')
import numpy as np
from file_reader import read_file
import pandas as pd
from rdkit import Chem
from mol_utils import get_fragments
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import argparse
import Show_Epoch
import logging
from keras.utils.generic_utils import get_custom_objects
import keras
sys.path.insert(0,'./Modules/')
from models import maximization
from rewards import get_padel, clean_folder, modify_fragment
from build_encoding import get_encodings, encode_molecule, decode_molecule, encode_list, save_decodings, save_encodings, read_decodings, read_encodings
from global_parameters import MAX_SWAP, MAX_FRAGMENTS, GAMMA, BATCH_SIZE, EPOCHS, TIMES, FEATURES
#similar to bunch_Eval, except that it is without the rewards
def get_pIC50(mols):
    folder_path =  "./generated_molecules/"
    file_path = "./descriptors.csv"
    #Cleaning up the older files
    clean_folder(folder_path)
    
    i = 0
    for mol in mols:
        print(Chem.MolToMolBlock(mol),file=open(str(folder_path)+str(i)+'.mol','w'))
        i += 1
    get_padel(folder_path,file_path)
      #Reading the descriptors
    X = pd.read_csv(file_path)
    #Filling Null Values
    X.fillna(value=0,inplace=True)
    X.Name = pd.to_numeric(X.Name, errors='coerce')
    X.sort_values(by='Name',inplace=True)
    #X.to_csv('./try.csv',index=False)
    #Removing the columns with zero variance in original data
    with open('./saved_models/drop.txt','rb') as fp:
        bad_cols = pickle.load(fp)
    X_step1 = X.drop(columns=bad_cols,inplace=False)
    X_step2 = X_step1
    
    
    
    #Doing StandardScaler() as applied to original data
    with open('./saved_models/new_scaler.pkl','rb') as fp:
        scaler = pickle.load(fp)
    X2 = scaler.transform(X_step2.astype('float64'))
    X_step3 = pd.DataFrame(data=X2,columns=X_step2.columns)
    
    #X.head()
    #Dropping columns with low correlation with pIC50
    
    # =============================================================================
    #     X.to_csv('./X.csv',index=False)
    #     X_step1.to_csv('./X_step1.csv')
    #     X_step2.to_csv('./X_step2.csv')
    #     X_step3.to_csv('./X_step3.csv')
    # =============================================================================
    
    
    
    #Using the Random forest Predictor
    with open('./saved_models/new_RFR.pkl','rb') as fp:
        pp = pickle.load(fp)
    predictions = pp.predict(X_step3)
    
    print('Properties predicted for {} molecules'.format(len(predictions)))
    return predictions

def modify_mols(X,decodings):
    batch_mol = X.copy()
    org_mols = batch_mol.copy() #saving a copy of the original molecules
    BATCH_SIZE = len(X)
    n_actions = MAX_FRAGMENTS * MAX_SWAP + 1
    stopped = np.zeros(BATCH_SIZE) != 0
    #loss = maximization()
    #get_custom_objects().update({"maximization": loss.computeloss})
    actor = keras.models.load_model('./saved_models/generation', custom_objects={'maximization': maximization})
    TIMES = 1
    for t in range(TIMES):
        #for each mol, a no. between 0-1 indicating the time-step
        tm = (np.ones((BATCH_SIZE,1)) * t) / TIMES
        #predictions for all the 512 molecules: 512*n_actions
        probs = actor.predict([batch_mol, tm])
        actions = np.zeros((BATCH_SIZE))
        old_batch = batch_mol.copy()
        rewards = np.zeros((BATCH_SIZE,1))
        # Find probabilities for modification actions
        for i in range(BATCH_SIZE):#for every molecules in the batch
            #choose the action which has maximum probability
            actions[i]=np.argmax(probs[i])
    
        # Select actions
        for i in range(BATCH_SIZE):

            a = int(actions[i])
            if stopped[i] or a == n_actions - 1:
                stopped[i] = True
                if t == 0:
                    rewards[i] += -1. #Why?
                continue

            #Converting the n_actions*1 position to the actual position    
            a = int(a // MAX_SWAP)#Integer Division, to get the location of the fragment

            s = a % MAX_SWAP# it is the location where the swap happens in that fragment
            if batch_mol[i,a,0] == 1:#Checking whether the fragment is non-empty?
                
                #In ith molecule, in its ath fragment, the sth position is flipped

                batch_mol[i,a] = modify_fragment(batch_mol[i,a], s)#changes 0 to 1 and 1 to 0
            #Evaluating multiple molecules at the same time
            np.save("History/in-1250.npy", org_mols)
            np.save("History/out-1250.npy", batch_mol)

def main(epoch,gen):
    if gen == 1:
        lead_file = "Data/AKT_pchembl.csv"
        fragment_file = "Data/molecules.smi"
        fragment_mols = read_file(fragment_file)
        lead_mols = read_file(lead_file)
        fragment_mols += lead_mols

        logging.info("Read %s molecules for fragmentation library", len(fragment_mols))
        logging.info("Read %s lead molecules", len(lead_mols))

        fragments, used_mols = get_fragments(fragment_mols)
        logging.info("Num fragments: %s", len(fragments))
        logging.info("Total molecules used: %s", len(used_mols))
        assert len(fragments)
        assert len(used_mols)
        lead_mols = np.asarray(fragment_mols[-len(lead_mols):])[used_mols[-len(lead_mols):]]

        decodings = read_decodings()
        encodings = read_encodings()
        logging.info("Loaded encodings and decodings")

        X = encode_list(lead_mols, encodings)
        modify_mols(X,decodings)
        epoch=1250
    file_name = './past outputs/out'+str(epoch)+'.csv'
    logging.info("Collecting and storing all molecules in {}".format(file_name))
    Show_Epoch.main(epoch,file_name)

    df = pd.read_csv('./past outputs/out'+str(epoch)+'.csv',sep=";")
        
    moli = []
    molm = []

    for i in range(len(df)):
        if (Chem.MolFromSmiles(df.iloc[i,1])) is not None:
            moli.append(Chem.MolFromSmiles(df.iloc[i,0]))
            molm.append(Chem.MolFromSmiles(df.iloc[i,1]))
    logging.info("Predicting pIC50 values of the initial molecules")
    ini = get_pIC50(moli)
    logging.info("Predicting pIC50 values of the predicted molecules")
    mod = get_pIC50(molm)
    ini = np.asarray(ini)
    mod = np.asarray(mod)
    
    changes = pd.DataFrame(np.transpose(np.asarray([ini,mod])),columns=['Modified','Initial'])
    changes['Delta'] = changes['Modified'] - changes['Initial']
    changes.sort_values(by='Delta',ascending=False,inplace=True)
    inact_to_act = changes.loc[(changes['Modified']>7) & (changes['Initial']<7),['Modified','Initial','Delta']].sort_values(by='Delta',ascending=False)
    
    changes.to_csv('./past outputs/out_pIC'+str(epoch)+'.csv',index=False)
    inact_to_act.to_csv('./past outputs/act_pIC'+str(epoch)+'.csv',index=False)
    
    print(inact_to_act.head())
    print(changes.head())

    bins = np.linspace(4,10,14)
    #changes = changes.loc[changes.Delta>0]
    plt.hist(changes['Initial'], bins, alpha=0.5, label='initial',color='blue')
    plt.hist(changes['Modified'], bins, alpha=0.5, label='modified',color='green')
    plt.legend(loc='upper right')
    plt.show()

    sp = changes.loc[changes['Delta']>0].sum()['Delta']
    sn = changes.loc[changes['Delta']<0].sum()['Delta']
    cp = changes.loc[changes['Delta']>0].count()['Delta']
    cn = changes.loc[changes['Delta']<0].count()['Delta']
    print('Sum of positive changes = {}\tNo. of +ves = {}\nSum of negative changes = {}\tNo. of -ves = {}'.format(sp,cp,sn,cn))
    return 0


parser = argparse.ArgumentParser()
parser.add_argument("-epoch", dest="epoch", help="Epoch for which the results are to be viewed", default=0)
parser.add_argument("-gen",dest="gen",help="Pass as 1 if you want to generate new molecules",default=0)
if __name__ == "__main__":

    args = parser.parse_args()
    epoch = int(args.epoch)
    gen = int(args.gen)
    main(int(args.epoch),int(gen))


