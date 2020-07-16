import sys
sys.path.insert(0,'./Modules/')
import numpy as np
from file_reader import read_file
import pandas as pd
from rdkit import Chem
from mol_utils import get_fragments
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import pickle
import argparse
import xgboost as xgb
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
    get_padel(folder_path,file_path,'-1')
      #Reading the descriptors
    xg_all = pd.read_csv(file_path)

    names = xg_all['Name']

    bad = []
    with open('./saved_models/good_columns','rb') as f:
        cols = pickle.load(f)
    for col in xg_all.columns:
        if col not in cols:
            bad.append(col)
    xg_all.drop(columns=bad,inplace=True)
    #Verifying that all the required columns are there
    assert len(xg_all.columns) == len(cols)
    xg_all['Name'] = names

    files = xg_all[pd.isnull(xg_all).any(axis=1)]['Name']
    xg_all.dropna(inplace=True)
    mol= []
    if len(files) !=0:
        uneval_folder = "C:\\Users\\HP\\AZC_Internship\\DeepFMPO\\3.6\\unevalmol\\"
        clean_folder(uneval_folder)
        for f in files:
            m = Chem.MolFromMolFile(folder_path+str(f)+'.mol')
            print(Chem.MolToMolBlock((m)),file=open(str(uneval_folder)+str(f)+'.mol','w'))

        get_padel(uneval_folder,'./uneval_desc.csv','-1')
        unevalmol = pd.read_csv('./uneval_desc.csv')

        unevalmol.drop(columns=bad,inplace=True)
        print(unevalmol.isna().sum(axis=1))
        xg_all = pd.concat([xg_all,unevalmol])
    #xg_all.to_csv('./xgall.csv')
    xg_all.fillna(value=0,inplace=True)
    regressor = xgb.XGBRegressor()
    regressor.load_model('./saved_models/best_from_gs38.model')

    xg_all.sort_values(by='Name',inplace=True)
    xg_all.drop(columns='Name',inplace=True)
    predictions = regressor.predict(xg_all)
    
    print('Properties predicted for {} molecules'.format(len(predictions)))
    return predictions

def modify_mols(X,decodings,stoch=1):
    batch_mol = X.copy()
    org_mols = batch_mol.copy() #saving a copy of the original molecules
    BATCH_SIZE = len(X)
    n_actions = MAX_FRAGMENTS * MAX_SWAP + 1
    stopped = np.zeros(BATCH_SIZE) != 0
    #loss = maximization()
    #get_custom_objects().update({"maximization": loss.computeloss})
    actor = keras.models.load_model('./saved_models/generation', custom_objects={'maximization': maximization})
    TIMES = 8
    rand_select = np.random.rand(BATCH_SIZE)
    for t in range(TIMES):
        #for each mol, a no. between 0-1 indicating the time-step
        tm = (np.ones((BATCH_SIZE,1)) * t) / TIMES
        #predictions for all the 512 molecules: 512*n_actions
        probs = actor.predict([batch_mol, tm])
        actions = np.zeros((BATCH_SIZE))
        if stoch == 1:
        # Find probabilities for modification actions
            for i in range(BATCH_SIZE):#for every molecules in the batch

                    a = 0
                    while True:
                        rand_select[i] -= probs[i,a]
                        if rand_select[i] < 0 or a + 1 == n_actions:
                            break
                        a += 1
                    #choosing a random action
                    actions[i] = a
        else:
            for i in range(BATCH_SIZE):#for every molecules in the batch
                #choose the action which has maximum probability
                actions[i]=np.argmax(probs[i])
    
        # Select actions
        for i in range(BATCH_SIZE):

            a = int(actions[i])
            if stopped[i] or a == n_actions - 1:
                stopped[i] = True
                continue

            #Converting the n_actions*1 position to the actual position    
            a = int(a // MAX_SWAP)#Integer Division, to get the location of the fragment

            s = a % MAX_SWAP# it is the location where the swap happens in that fragment
            if batch_mol[i,a,0] == 1:#Checking whether the fragment is non-empty?
                
                #In ith molecule, in its ath fragment, the sth position is flipped

                batch_mol[i,a] = modify_fragment(batch_mol[i,a], s)#changes 0 to 1 and 1 to 0
            #Evaluating multiple molecules at the same time
            e = 1000
            np.save("./History/in-{}.npy".format(e), org_mols)
            np.save("./History/out-{}.npy".format(e), batch_mol)

def main(epoch,gen):
    if gen == 1:
        lead_file = "Data/trial.csv"
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
        epoch=1000
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
    
    changes =  pd.DataFrame(np.transpose(np.asarray([ini,mod])),columns=['Initial_pIC','Modified_pIC'])
    changes['Initial_mol'] = df.iloc[:,0]
    changes['Modified_mol'] = df.iloc[:,1]
    changes['Delta'] = changes['Modified_pIC'] - changes['Initial_pIC']
    changes.sort_values(by='Delta',ascending=False,inplace=True)

    inact_to_act = changes.loc[(changes['Modified_pIC']>7) & (changes['Initial_pIC']<7),['Modified_pIC','Initial_pIC','Delta']].sort_values(by='Delta',ascending=False)
    
    changes.to_csv('./past outputs/out_pIC'+str(epoch)+'.csv',index=False)
    inact_to_act.to_csv('./past outputs/act_pIC'+str(epoch)+'.csv',index=False)
    
    print(inact_to_act.head())
    print(changes.head())
    from rdkit.Chem import Draw
    moli = []
    molm = []
    for i in range(5):
        moli.append(Chem.MolFromSmiles(changes.iloc[i,2]))
        moli.append(Chem.MolFromSmiles(changes.iloc[i,3]))
    plot = Draw.MolsToGridImage(moli, molsPerRow=2)
    plot.show()
    #plot.save('/past outputs/epoch.png')
    bins = np.linspace(4,10,14)
    #changes = changes.loc[changes.Delta>0]
    plt.hist(changes['Initial_pIC'], bins, alpha=0.5, label='initial',color='blue')
    plt.hist(changes['Modified_pIC'], bins, alpha=0.5, label='modified',color='green')
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
parser.add_argument("-stoch",dest="stoch",help="Pass as 0 if you do not want to generate new molecules by sampling actions from a probability distribution",default=1)
if __name__ == "__main__":

    args = parser.parse_args()
    epoch = int(args.epoch)
    gen = int(args.gen)
    stoch = int(args.stoch)
    start_time = time.time()
    main(int(args.epoch),int(gen))
    print("---Time taken = {} seconds ---".format(time.time() - start_time))


