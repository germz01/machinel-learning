import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import pandas as pd

import nn as NN
import utils as u
import validation as valid
import metrics

import time
import json
import os.path
import imp




from pprint import pprint
###########################################################

early = pd.read_csv('../data/CUP/results/early_stop/df_early_stop.csv')

early.describe()

early.columns
plt.ion()
early.describe()
plt.scatter(x=early['MEE_min'], y=early['MEE_GL'])
plt.show()

sns.scatterplot()
plt.scatter(x=early['stop_min'], y=early['stop_GL'], label='GL')
plt.scatter(x=early['stop_min'], y=early['stop_PQ'], label='PQ')
plt.xlabel('Minimum MEE epochs')
plt.ylabel('Early stopping epochs')

plt.close()
plt.scatter(x=early['hidden_sizes'], y=early['stop_GL']-early['stop_min'], label='GL')
plt.scatter(x=early['hidden_sizes'], y=early['stop_PQ']-early['stop_min'], label='PQ')
plt.legend()
plt.axhline(0)
plt.axvline(100)

plt.grid()

x = np.arange(0,5000,1)

plt.plot(x,x)

plt.legend()


plt.close()
plt.show()

plt.scatter(x=early['hidden_sizes'], y=early['MEE_final'])
sns.lineplot(x=early['hidden_sizes'], y=early['MEE_final'])
plt.scatter(x=early['hidden_sizes'], y=early['MEE_min'])
sns.lineplot(x=early['hidden_sizes'], y=early['MEE_min'])
sns.lineplot(x=early['hidden_sizes'], y=early['MEE_PQ'])
plt.scatter(x=early['hidden_sizes'], y=early['MEE_PQ'])
plt.legend()
plt.grid()
