#!/usr/bin/env python

import pathlib
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from link_bot_data.new_classifier_dataset import NewClassifierDataset
from link_bot_data.image_classifier_dataset import ImageClassifierDataset

print("Repeating")
c = NewClassifierDataset([pathlib.Path('./classifier_data/fs-2/')])
dataset = c.get_datasets(mode='train', shuffle=False, seed=0, batch_size=1)

for i in dataset:
    if i['label'].numpy()[0] == 0:
        x0 = i
        s0 = x0['state'].numpy()
        break

count = 0
for i, e in enumerate(dataset):
    s = e['state'].numpy()
    if np.all(s == s0):
        print(i)
        count += 1
print("COUNT", count)
        
        
print("Augmentation")
aug = ImageClassifierDataset([pathlib.Path('./classifier_data/augmentation/')])
dataset = aug.get_datasets(mode='train', shuffle=False, seed=0, batch_size=1)
for i in dataset:
    if i['label'].numpy()[0] == 0:
        x0 = i
        s0 = x0['image'].numpy()
        break
        
count = 0
for i, e in enumerate(dataset):
    s = e['image'].numpy()
    if np.all(s == s0):
        print(i)
        count += 1
print("COUNT", count)
