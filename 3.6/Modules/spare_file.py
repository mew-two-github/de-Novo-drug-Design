
a = ['abc',2]
b = ['defg',3]
c = []
c.append(a)
c.append(b)
c.append(['hij',4])
for l in c:
    print(l)
# =============================================================================
# d = []
# for i in range(len(c)):
#     d.append(c[i][0])
# print(d)
# 
# =============================================================================
import pandas as pd
from rdkit import Chem as ch
from rewards import bunch_evaluation
# =============================================================================
# df = pd.read_csv('C:\\Users\\HP\\AZC_Internship\\DeepFMPO\\3.6\\Data\\AKT_pchembl.csv')
# print(df.iloc[0:6,1])
# mol = []
# i = 1
# for smile in df['Smiles']:
#     if i>=6:
#         break
#     mol.append(ch.MolFromSmiles(smile))
#     i += 1
# out = bunch_evaluation(mol)
# for o in out:
#     print(o[1]*3 + 7)
# 
# =============================================================================
