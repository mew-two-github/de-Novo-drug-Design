import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
import numpy as np
from build_encoding import decode
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.rdMolDescriptors as MolDescriptors
import subprocess
import pickle
import pandas as pd





# Cache evaluated molecules (rewards are only calculated once)
evaluated_mols = {}




def modify_fragment(f, swap):
    f[-(1+swap)] = (f[-(1+swap)] + 1) % 2
    return f





def get_key(fs):
    return tuple([np.sum([(int(x)* 2 ** (len(a) - y))
                    for x,y in zip(a, range(len(a)))]) if a[0] == 1 \
                     else 0 for a in fs])


#function to get Padel descriptors and store in a csv file
def get_padel(mol_folder_path,file_path):
    cmd_list = ['java','-jar','C:\\Users\\HP\\PaDEL-Descriptor\\PaDEL-Descriptor.jar','-dir',mol_folder_path,'-2d','-file',file_path,'-usefilenameasmolname']
    out = subprocess.Popen(cmd_list, 
           stdout=subprocess.PIPE, 
           stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()
    print(stdout)
    print(stderr)

#Function that processes the padel descriptors and predicts the value
def get_pIC(mol):
    mol_folder_path = "./generated_molecules/"
    print(Chem.MolToMolBlock(mol),file=open(str(mol_folder_path)+'generated.mol','w+'))
    file_path = "./generated_molecules/descriptors.csv"
    get_padel(mol_folder_path,file_path)
    X = pd.read_csv(file_path)

    #Removing the columns with zero variance in original data
    with open('./saved_models/drop1.txt','rb') as fp:
        bad_cols = pickle.load(fp)
    X.drop(columns=bad_cols,inplace=True)
    X.drop(columns='Name',inplace=True)

    #Doing StandardScaler() as applied to original data
    with open('./saved_models/scaler.pkl','rb') as fp:
        scaler = pickle.load(fp)
    X2 = scaler.transform(X)
    X = pd.DataFrame(data=X2,columns=X.columns)
    #X.head()
    #Dropping columns with low correlation with pIC50
    with open('./saved_models/drop2.txt','rb') as fp:
        bad_cols = pickle.load(fp)
    X.drop(columns=bad_cols,inplace=True)
    with open('./saved_models/pca.pkl','rb') as fp:
        pca = pickle.load(fp)
    
    #Applying PCA
    cols = []
    for i in range(pca.n_components):
        cols.append('comp'+str(i+1))
    principalComponents= pca.transform(X)
    X_red = pd.DataFrame(data=principalComponents, columns=cols)
    X_red.head()

    #Using the Random forest Predictor
    with open('./saved_models/predictor.pkl','rb') as fp:
        pp = pickle.load(fp)
    prediction = pp.predict(X_red)
    return prediction

# Main function for the evaluation of molecules.
def evaluate_chem_mol(mol):
    try:
        Chem.GetSSSR(mol)
        pIC = get_pIC(mol)

        ret_val = [
            True,
            pIC-7
        ]
    except:
        ret_val = [False] * 2

    return ret_val



# Same as above but decodes and check if a cached value could be used.
def evaluate_mol(fs, epoch, decodings):

    global evaluated_mols

    key = get_key(fs)

    if key in evaluated_mols:
        return evaluated_mols[key][0]

    try:
        mol = decode(fs, decodings)
        ret_val = evaluate_chem_mol(mol)
    except:
        ret_val = [False] * 2

    evaluated_mols[key] = (np.array(ret_val), epoch)

    return np.array(ret_val)



# Calculate rewards and give penalty if a locked/empty fragment is changed.
def get_reward(fs,epoch,dist):

    if fs[fs[:,0] == 0].sum() < 0:
        return -0.1

    return (dist * evaluate_mol(fs, epoch)).sum()



# Get initial distribution of rewards among lead molecules
def get_init_dist(X, decodings):

    arr = np.asarray([evaluate_mol(X[i], -1, decodings) for i in range(X.shape[0])])
    dist = arr.shape[0] / (1.0 + arr.sum(0))
    return dist


# Discard molecules which fulfills all targets (used to remove to good lead molecules).
def clean_good(X, decodings):
    X = [X[i] for i in range(X.shape[0]) if not
        evaluate_mol(X[i], -1, decodings).all()]
    return np.asarray(X)

