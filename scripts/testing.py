import nn as NN
import numpy as np
import matplotlib.pyplot as plt
import imp
import utils as u
import losses as lss


p_class1 = 700 # number of patterns
p_class2 = 300
n = 10 # attributes/features

X = np.vstack((np.random.normal(2., 1., (p_class1, n)),
                    np.random.normal(6., 1., (p_class2, n))),
)

y = np.vstack((
    np.hstack( (np.ones(p_class1), np.zeros(p_class2))),
    np.hstack( (np.zeros(p_class1), np.ones(p_class2)))
)).T


imp.reload(NN)
imp.reload(u)
nn = NN.NeuralNetwork(hidden_sizes = [100 ] )

nn.train_mb(X,y,
            eta = 0.2,
            epochs = 1000,
            batch_size = 100,
            w_par = 6)

y_pred = nn.predict(X)
np.round(y_pred,2)
y.T
np.mean(np.sqrt(np.einsum('kp->p', (y_pred-y.T)**2)))


plt.plot(range(len(nn.loss_epochs)), nn.loss_epochs)
plt.savefig('learning_curve.pdf')
plt.close()

plt.plot(range(len(nn.loss_batch)), nn.loss_batch)
plt.savefig('learning_curve_batch.pdf')
plt.close()

plt.close()

g.sum(axis = 1)/(g.shape[1])

def lapply(l, f):
    ''' 
    Apply function f to each element of the list l 
    (R style)
    '''
    return [f(el) for el in l]

lapply(nn.a, len)    



nn.topology


epochs = 500
# nota: loss online e epochs sono su range diversi
# da sistemare successivamente a batch/minibatch/online implementation
fig, ax = plt.subplots(figsize = (15,7))
plt.plot(range(len(nn.loss_online[:epochs])), nn.loss_online[:epochs])
plt.plot(range(len(nn.loss_epochs[:epochs])), nn.loss_epochs[:epochs])

plt.savefig('../images/temp_loss.pdf')
plt.close()
