import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import argparse
sys.path.insert(0,'./Modules/')
from rewards import get_padel, clean_folder
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
    X.to_csv('./try.csv',index=False)
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

parser = argparse.ArgumentParser()
parser.add_argument("-SMILES", dest="SMILEFile", help="File containing smiles string separated by ;", default=None)
parser.add_argument("-image", dest="image", help="File to save image in", default=None)

df = pd.read_csv('./past outputs/out159.csv',engine="python")
from rewards import bunch_evaluation
moli = []
molm = []
for i in range(len(df)):
    moli.append(ch.MolFromSmiles(df.iloc[i,0]))
    molm.append(ch.MolFromSmiles(df.iloc[i,1]))
