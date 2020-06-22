# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 20:55:14 2020

@author: HP
"""
import sys
sys.path.insert(0, './Modules/')
from build_encoding import read_encodings
encodings = read_encodings()
print(len(encodings.popitem()[1]))