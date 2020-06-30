import numpy as np
import pandas as pd
from rdkit import Chem
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import argparse
import Show_Epoch
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
    print(X.isna().count())
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

def main(epoch):
    file_name = './past outputs/out'+str(epoch)+'.csv'
    Show_Epoch.main(epoch,file_name)

    df = pd.read_csv('./past outputs/out'+str(epoch)+'.csv',sep=";")
    df.to_csv('./past outputs/out'+str(epoch)+'.csv',sep=";",index=False)
    moli = []
    molm = []
    for i in range(len(df)):
        if (Chem.MolFromSmiles(df.iloc[i,1])) is not None:
            moli.append(Chem.MolFromSmiles(df.iloc[i,0]))
            molm.append(Chem.MolFromSmiles(df.iloc[i,1]))
    ini = get_pIC50(moli)
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
parser.add_argument("-epoch", dest="epoch", help="Epoch for which the results are to be viewed", default=None)
if __name__ == "__main__":

    args = parser.parse_args()
    epoch = int(args.epoch)
    main(int(args.epoch))


