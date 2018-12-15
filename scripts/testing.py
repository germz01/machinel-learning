import nn as NN
import numpy as np
import matplotlib.pyplot as plt
import imp
import utils as u

X = np.concatenate((np.random.normal(2., 1., (3, 2)),
                    np.random.normal(5., 1., (2, 2))),
                   axis=0)
y = np.array([1, 1, 1, 0, 0]).reshape(5, 1)




imp.reload(NN)
imp.reload(u)

nn = NN.NeuralNetwork(hidden_sizes = [10 ] )

eta = .1
alpha = .9
nn.train(X, y, eta, alpha, [0.01, 'l2'], 1000)

nn.topology


epochs = 500
# nota: loss online e epochs sono su range diversi
# da sistemare successivamente a batch/minibatch/online implementation
fig, ax = plt.subplots(figsize = (15,7))
plt.plot(range(len(nn.loss_online[:epochs])), nn.loss_online[:epochs])
plt.plot(range(len(nn.loss_epochs[:epochs])), nn.loss_epochs[:epochs])

plt.savefig('../images/temp_loss.pdf')
plt.close()
