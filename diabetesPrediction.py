# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g15bMMOFdl0biD0_2j5TJmom5ywO1g2P
"""

from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
import numpy as np
#pandas- librărie pentru lucrul cu fișierele
import pandas as pd
import torch
import torch.nn as nn

df=pd.read_csv("/content/diabetes.csv",header=None)

df.values

#Dataset - o clasă din PyTorch foarte utilă gestionării seturilor de date
class DiabetesDataset(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, download, etc.
    def __init__(self):
        #Citim setul de date
        df=pd.read_csv("/content/diabetes.csv",header=None, dtype=np.float32)
        xy = torch.from_numpy(df.values)
        self.len = xy.shape[0]
        #Vom folosi ca input toate valorile mai puțin ultima coloană
        self.x_data = xy[:, 0:-1]
        #Vom folosi ca output ultima coloană
        self.y_data = xy[:, [-1]]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset()
#DataLoader - un utilitar ce ne ajută să împărțim setul de date pe batch-uri și astfel să facem antrenare în mod Mini-Batch
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=1)

dataset[0]

class Model(nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

model=Model()

criterion = nn.BCELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(200):
  for i, data in enumerate(train_loader, 0):
      # get the inputs
      inputs, labels = data

      # Forward pass: Compute predicted y by passing x to the model
      y_pred = model(inputs)

      # Compute and print loss
      loss = criterion(y_pred, labels)
      print(f'Epoch {epoch + 1} | Batch: {i+1} | Loss: {loss.item():.4f}')

      # Zero gradients, perform a backward pass, and update the weights.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

