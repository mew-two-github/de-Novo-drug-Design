import rdkit.Chem as Chem
#from rdkit.Chem import Descriptors
import numpy as np
from build_encoding import decode
# =============================================================================
# import rdkit.Chem.Crippen as Crippen
# import rdkit.Chem.rdMolDescriptors as MolDescriptors
# =============================================================================
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
    Padel_path = 'C:\\Users\\HP\\PaDEL-Descriptor\\PaDEL-Descriptor.jar'
    max_time = '3000' #in milliseconds
    cmd_list = ['java','-jar',Padel_path, '-dir', mol_folder_path, '-2d','-file', file_path,'-maxruntime', max_time]
    out = subprocess.Popen(cmd_list, 
           stdout=subprocess.PIPE, 
           stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()
    stdout = stdout.decode('utf-8')
    with open('./Padel.txt','a') as f:
        f.write(stdout)


#Function that processes the padel descriptors and predicts the value
def get_pIC(mol):
    mol_folder_path = "./generated_molecules/"
    #Generating PaDEL descriptors using get_padel
    
    print(Chem.MolToMolBlock((mol)),file=open(str(mol_folder_path)+'generated.mol','w'))
    file_path = "C:\\Users\\HP\\AZC_Internship\\DeepFMPO\\3.6\\descriptors.csv"
    get_padel(mol_folder_path,file_path)
    #Reading the descriptors
    X = pd.read_csv(file_path)
    #Filling Null Values
    X.fillna(value=0,inplace=True)
    #Removing the columns with zero variance in original data
    with open('./saved_models/drop1.txt','rb') as fp:
        bad_cols = pickle.load(fp)
    X_step1 = X.drop(columns=bad_cols,inplace=False)
    X_step2 = X_step1.drop(columns='Name',inplace=False)
    

    
    #Doing StandardScaler() as applied to original data
    with open('./saved_models/scaler.pkl','rb') as fp:
        scaler = pickle.load(fp)
    X2 = scaler.transform(X_step2.astype('float64'))
    X_step3 = pd.DataFrame(data=X2,columns=X_step2.columns)
    
    #X.head()
    #Dropping columns with low correlation with pIC50
    with open('./saved_models/drop2.txt','rb') as fp:
        bad_cols = pickle.load(fp)
    X_step3.drop(columns=bad_cols,inplace=True)
    
# =============================================================================
#     X.to_csv('./X.csv',index=False)
#     X_step1.to_csv('./X_step1.csv')
#     X_step2.to_csv('./X_step2.csv')
#     X_step3.to_csv('./X_step3.csv')
# =============================================================================
    
    #X.fillna(value=0)
    #Applying PCA
    #np.where(x.values >= np.finfo(np.float64).max)
    with open('./saved_models/pca.pkl','rb') as fp:
        pca = pickle.load(fp)
    cols = []
    for i in range(pca.n_components):
        cols.append('comp'+str(i+1))    
    principalComponents= pca.transform(X_step3)
    X_red = pd.DataFrame(data=principalComponents, columns=cols)
    

    #Using the Random forest Predictor
    with open('./saved_models/predictor.pkl','rb') as fp:
        pp = pickle.load(fp)
    prediction = pp.predict(X_red)
    #print(prediction.shape)
    return prediction.item(0)

# Main function for the evaluation of molecules.
def evaluate_chem_mol(mol):
    try:
        Chem.GetSSSR(mol)
        pIC = get_pIC(mol)

        ret_val = [
            True,
            (pIC-7)/3
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



# =============================================================================
# df = pd.read_csv('./out.csv',engine="python")
# for i in range(len(df)):
#     print("Molecule number {}".format(i+1))
#     mol1 = Chem.MolFromSmiles(df.iloc[i,0])
#     mol2 = Chem.MolFromSmiles(df.iloc[i,1])
#     print(get_pIC(mol1),get_pIC(mol2))
# =============================================================================

