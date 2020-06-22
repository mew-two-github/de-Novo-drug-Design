""" from collections import OrderedDict
od = OrderedDict()
od.update({'a':1,'b':2})
print([key for key in od.keys()]) """
# =============================================================================
# a = ['abc',2]
# b = ['defg',3]
# c = []
# c.append(a)
# c.append(b)
# c.append(['hij',4])
# for l in c:
#     print(l)
# =============================================================================
# =============================================================================
# d = []
# for i in range(len(c)):
#     d.append(c[i][0])
# print(d)
# 
# =============================================================================
# =============================================================================
# import pandas as pd
# from rdkit import Chem as ch
# from rewards import bunch_evaluation
# import os
# import glob
# 
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
# =============================================================================

# =============================================================================
# folder_path =  "C:/Users/HP/AZC_Internship/DeepFMPO/3.6/generated_molecules"
# file_path = "./descriptors.csv"    
# 
# 
# import os, shutil
# folder = folder_path
# for filename in os.listdir(folder):
#     file_path = os.path.join(folder, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))
# 
# =============================================================================
'''from build_encoding import read_encodings
encodings = read_encodings()
print(encodings)'''

'''import numpy as np
BATCH_SIZE = 10
TIMES = 4

stopped = np.zeros(5) != 0
for t in range(TIMES):
    tm = (np.ones((BATCH_SIZE,1)) * t) / TIMES
    print(tm)'''
MAX_F = 12
for a in range(MAX_F*(5)+1):
    b = int(a // 5)
    print(b,b% 5)