# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 20:55:14 2020

@author: HP
"""
'''
import sys
sys.path.insert(0, './Modules/')
from build_encoding import read_encodings
encodings = read_encodings()
print(len(encodings.popitem()[1]))
'''
'''
import numpy as np
dist = np.load('dist.npy')
print(dist)
arr = [[3,2],[5,6]]
arr = np.asarray(arr)
print(arr.sum(0),arr.sum(1))
c = 0
val = c==0
print(val)
print(np.load('r_tot.npy'))
'''
'''
#get_padel(folder_path,file_path)
import numpy as np
import matplotlib.pyplot as plt
hist = np.load('history/history.npy')
print(np.argmax(hist))
plt.plot(range(len(hist)),hist)
plt.show()
'''
'''
import pandas as pd
import numpy as np
import pickle
import rdkit.Chem as Chem
import sys
import os
sys.path.insert(0,'./Modules')
from rewards import clean_folder, get_padel
file_path = "./descriptors.csv"
xg_all = pd.read_csv(file_path)
folder_path =  "./generated_molecules/"
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
print(files)
uneval_folder = "./unevalmol/"
for i in files:
    clean_folder(uneval_folder)
    m = Chem.MolFromMolFile(folder_path+str(i)+'.mol')
    print(Chem.MolToMolBlock((m)),file=open(str(i)+'.mol','w'))
    os.move

get_padel(uneval_folder,'./uneval_desc.csv','-1')
unevalmol = pd.read_csv('./uneval_desc.csv')
'''
'''
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000
winsound.Beep(frequency,duration)
'''
'''
#Function that processes the padel descriptors and predicts the value
def get_pIC(mol):
    mol_folder_path = "./generated_molecules/"
    #Cleaning up the older files
    files = glob.glob(mol_folder_path)
    for f in files:
        os.remove(f)
    #Generating PaDEL descriptors using get_padel
    
    print(Chem.MolToMolBlock((mol)),file=open(str(mol_folder_path)+'generated.mol','w'))
    file_path = "C:\\Users\\HP\\AZC_Internship\\DeepFMPO\\3.6\\descriptors.csv"
    get_padel(mol_folder_path,file_path)
    #Reading the descriptors
    X = pd.read_csv(file_path)
    #Filling Null Values
    X.fillna(value=0,inplace=True)
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
    prediction = pp.predict(X_step3)
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







'''