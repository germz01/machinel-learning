import numpy as np
import utils
import neural_network
import imp


import pandas as pd

df = pd.read_csv('iris.data.txt')
df.columns = (['x'+str(i) for i in range(4)]+['y_raw'])

y_labels = df['y_raw'].unique()

d_map = {}
i=0
for k in y_labels:
    d_map[k] = i
    i+=1

df['y'] = df['y_raw'].map(d_map)
df.drop('y_raw', axis = 1, inplace = True)
df = df.values
###########################################################
import copy

df_shuffled = copy.copy(df)
np.random.shuffle(df_shuffled)

def split(df, perc = 0.7):
    i_split = int(np.floor(df.shape[0]*perc))

    return df[:i_split,:] , df[(i_split+1):,:]
    

df_train, df_test = split(df_shuffled, perc = 0.7)

X_train, X_test = df_train[:,:4], df_test[:,:4]
Y_train, Y_test = df_train[:,4], df_test[:,4]

imp.reload(utils)
imp.reload(neural_network)
nn = neural_network.NeuralNetwork(hidden_sizes = (10,), max_epochs = 1000)
nn.train(X_train, Y_train, eta = 0.01)

#nn.topology = [4,100, 2]

Y_train
np.round(nn.y_pred,0)

utils.plot_learning_curve(nn.empirical_risk, nn.max_epochs,'iris_learning.pdf')


(np.round(nn.predict(X_test),0)-Y_test.reshape((1,44)))

