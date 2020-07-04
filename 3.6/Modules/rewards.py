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
from collections import OrderedDict
from time import sleep
import os, shutil
import glob
import math
import xgboost as xgb

#A scaling factor for reward
const = math.exp(3)



# Cache evaluated molecules (rewards are only calculated once)
evaluated_mols = {}




def modify_fragment(f, swap):
    f[-(1+swap)] = (f[-(1+swap)] + 1) % 2
    return f

# Discard molecules which fulfills all targets (used to remove to good lead molecules).
def clean_good(X, decodings):
    X = [X[i] for i in range(X.shape[0]) if not
        evaluate_mol(X[i], -1, decodings).all()]
    return np.asarray(X)



def get_key(fs):
    return tuple([np.sum([(int(x)* 2 ** (len(a) - y))
                    for x,y in zip(a, range(len(a)))]) if a[0] == 1 \
                     else 0 for a in fs])
# Get initial distribution of rewards among lead molecules
def get_init_dist(X, decodings):

    #arr = np.asarray([evaluate_mol(X[i], -1, decodings) for i in range(X.shape[0])])
    arr = np.asarray(bunch_eval(X,-1,decodings))
    dist = arr.shape[0] / (1.0 + arr.sum(0)) #sum(0) => sum over all rows for each col
    return dist

#function to get Padel descriptors and store in a csv file
def get_padel(mol_folder_path,file_path,max_time='1500'):
    Padel_path = 'C:\\Users\\HP\\PaDEL-Descriptor\\PaDEL-Descriptor.jar'
    cmd_list = ['java','-jar',Padel_path, '-dir', mol_folder_path, '-2d','-file', file_path,'-maxruntime', max_time,"-descriptortypes", 'xg_desc3.xml','-usefilenameasmolname']
    out = subprocess.Popen(cmd_list, 
           stdout=subprocess.PIPE, 
           stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()
    stdout = stdout.decode('utf-8')
    with open('./Padel.txt','a') as f:
        f.write(stdout)


def clean_folder(folder_path):
    folder = folder_path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as ex:
            print('Failed to delete %s. Reason: %s' % (file_path, ex))
#Bunch evaluation
def bunch_evaluation(mols):
    folder_path =  "./generated_molecules/"
    file_path = "./descriptors.csv"
    
    #Cleaning up the older files
    clean_folder(folder_path)
    
    i = 0
    SSSR =[]
    for mol in mols:
         try:
             Chem.GetSSSR(mol)
             print(Chem.MolToMolBlock((mol)),file=open(str(folder_path)+str(i)+'.mol','w'))
             SSSR.append(True)
         except:
             SSSR.append(False)
         i = i +1

    get_padel(folder_path,file_path)

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
        for f in files:
            m = Chem.MolFromMolFile(folder_path+str(f)+'.mol')
            mol.append(m)

        i = 0
        for m in mol:
            print(Chem.MolToMolBlock((m)),file=open(str(folder_path)+str(files[i])+'.mol','w'))
            i = i + 1
        get_padel(folder_path,'./uneval_desc.csv','-1')
        unevalmol = pd.read_csv('./uneval_desc.csv')


        unevalmol.drop(columns=bad,inplace=True)
        print(unevalmol.isna().sum(axis=1))
        xg_all = pd.concat([xg_all,unevalmol])

    regressor = xgb.XGBRegressor()
    regressor.load_model('./saved_models/best_from_gs38.model')

    xg_all.sort_values(by='Name',inplace=True)
    xg_all.drop(columns='Name',inplace=True)
    preds = regressor.predict(xg_all_all)
    
    print('Properties predicted for {} molecules'.format(len(preds)))
    
    Evaluations = []
    j  = 0
    for i in range(len(SSSR)):

        if SSSR[i] == True:    
            pIC = predictions[j]
            val = pIC

            Evaluations.append([SSSR[i],val])
            j = j + 1
        else:
            Evaluations.append([False,-10])
    print(' Evaluations completed')
    return Evaluations
    
    #print(prediction.shape)
    
    
         
def bunch_eval(fs, epoch, decodings):

    global evaluated_mols
    keys = []
    od = OrderedDict()
    #total_molecules = len(fs)
    #print("Evaluating totally {} molecules".format(total_molecules))
    for f in fs:
        key = get_key(f)
        keys.append(key)
        od[key] = ([False,False])
    #print(len(od))
    to_evaluate = []
    i = 0
    unused = keys.copy()
    for key in keys:
        if key in evaluated_mols:
            od[key] = evaluated_mols[key][0]
            while key in unused:
                unused.remove(key)
        else:
            try:
                mol = decode(fs[i], decodings)
                od[key] = len(to_evaluate)
                to_evaluate.append(mol)
               # evaluated_mols[key] = (np.array(ret_val), epoch)
            except:
                od[key] = [False,-10]
                evaluated_mols[key] = (np.asarray([False,-10]),epoch)
        i = i + 1
    print('New molecules for evaluation: {}'.format(len(to_evaluate)))
    if len(to_evaluate)!=0:
        Evaluations = bunch_evaluation(to_evaluate)
        #print("Length of Evaluations {}".format(len(Evaluations)))
        assert len(Evaluations) == len(to_evaluate)
        for i in range(len(Evaluations)):
            for key in unused:
                if od[key] == i:
                    value = Evaluations[i]
                    od[key] = value
                    evaluated_mols[key] = (np.array(value),epoch)
    ret_vals = []
    with open('./ret_vals.pkl','wb') as f:
        pickle.dump(ret_vals,f)
    for key in keys:
        ret_vals.append(np.asarray(od[key]))
    ret_vals = np.asarray(ret_vals)
    print('Shape of return values {}'.format(ret_vals.shape))
    return (ret_vals)

# =============================================================================
# df = pd.read_csv('./out.csv',engine="python")
# for i in range(len(df)):
#     print("Molecule number {}".format(i+1))
#     mol1 = Chem.MolFromSmiles(df.iloc[i,0])
#     mol2 = Chem.MolFromSmiles(df.iloc[i,1])
#     print(get_pIC(mol1),get_pIC(mol2))
# =============================================================================