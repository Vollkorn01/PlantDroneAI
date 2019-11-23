#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:49:22 2019

@author: alexander
"""

import os
import pandas as pd

L  = []

directory = "/Users/alexander/Documents/DevProjects/PlantDroneAI/data_224x224_split/train/bitter_dock"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('TRAIN,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/train/bitter_dock/' + filename + ',bitter_dock')
        print('TRAIN,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/train/bitter_dock/' + filename + ',bitter_dock')
        continue
    else:
        continue

directory = "/Users/alexander/Documents/DevProjects/PlantDroneAI/data_224x224_split/train/other"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('TRAIN,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/train/other/' + filename + ',other')
        print('TRAIN,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/train/other/' + filename + ',other')
        continue
    else:
        continue

directory = "/Users/alexander/Documents/DevProjects/PlantDroneAI/data_224x224_split/test/bitter_dock"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('TEST,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/test/bitter_dock/' + filename + ',bitter_dock')
        print('TEST,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/test/bitter_dock/' + filename + ',bitter_dock')
        continue
    else:
        continue

directory = "/Users/alexander/Documents/DevProjects/PlantDroneAI/data_224x224_split/test/other"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('TEST,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/test/other/' + filename + ',other')
        print('TEST,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/test/other/' + filename + ',other')
        continue
    else:
        continue
    
directory = "/Users/alexander/Documents/DevProjects/PlantDroneAI/data_224x224_split/validation/bitter_dock"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('VALIDATION,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/validation/bitter_dock/' + filename + ',bitter_dock')
        print('VALIDATION,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/validation/bitter_dock/' + filename + ',bitter_dock')
        continue
    else:
        continue
    
directory = "/Users/alexander/Documents/DevProjects/PlantDroneAI/data_224x224_split/validation/other"
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".jpg"):
        L.append('VALIDATION,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/validation/other/' + filename + ',other')
        print('VALIDATION,' + 'gs://evident-minutia-180203-vcm/data_224x224_split/validation/other/' + filename + ',other')
        continue
    else:
        continue

df = pd.DataFrame(L, columns=['[set,]image_path[,label]'])

df.to_csv(r'/Users/alexander/Documents/DevProjects/PlantDroneAI/data_224x224_split/labels.csv', index = False, doublequote=False)
