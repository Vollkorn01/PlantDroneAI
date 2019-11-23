#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:49:22 2019

@author: alexander
"""

import os
import pandas as pd

L  = []

directory = "/Users/alexander/Downloads/data_224x224_small/train/bitter_dock"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('gs://evident-minutia-180203-vcm/data_224x224_small/train/bitter_dock/' + filename + ',bitter_dock')
        print('gs://evident-minutia-180203-vcm/data_224x224_small/train/bitter_dock/' + filename + ',bitter_dock')
        continue
    else:
        continue

directory = "/Users/alexander/Downloads/data_224x224_small/train/other"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('gs://evident-minutia-180203-vcm/data_224x224_small/train/other/' + filename + ',other')
        print('gs://evident-minutia-180203-vcm/data_224x224_small/train/other/' + filename + ',other')
        continue
    else:
        continue

directory = "/Users/alexander/Downloads/data_224x224_small/test/bitter_dock"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('gs://evident-minutia-180203-vcm/data_224x224_small/test/bitter_dock/' + filename + ',bitter_dock')
        print('gs://evident-minutia-180203-vcm/data_224x224_small/test/bitter_dock/' + filename + ',bitter_dock')
        continue
    else:
        continue

directory = "/Users/alexander/Downloads/data_224x224_small/test/other"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('gs://evident-minutia-180203-vcm/data_224x224_small/test/other/' + filename + ',other')
        print('gs://evident-minutia-180203-vcm/data_224x224_small/test/other/' + filename + ',other')
        continue
    else:
        continue

df = pd.DataFrame(L, columns=['[set,]image_path[,label]'])

df.to_csv(r'/Users/alexander/Downloads/data_224x224_small/labels.csv', index = False, doublequote=False)
