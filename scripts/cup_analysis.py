import matplotlib as mpl
# mpl.use('TkAgg')
# mpl.use('TkAgg')

import validation as validation
import imp
import numpy as np
import json
import utils as u

import pandas as pd
import matplotlib.pyplot as plt
import plotnine as gg

import seaborn as sns

from pprint import pprint

imp.reload(validation)

###########################################################


'''
Exp 0:
Esperimento iniziale con hidden sizes tra 10-100
con 1000 valori della griglia ed epochs ridotti a 300
../exp0/
'''
nexp = 3

# read experiment parameters
with open('../data/CUP/results/exp0/cup_experiment_{}_parameters.json'.format(nexp)) as f:
    experiment_params = json.load(f)

pprint(experiment_params)

# load results
selection = validation.ModelSelectionCV(
    grid=None,
    fname='../data/CUP/results/exp0/' +
    'cup_experiment_{}_results.json.gz'.format(nexp))

df = selection.load_results_pandas(flat=False)
df.shape

df.memory_usage(deep=True).sum()/1000/1000  # MBytes
df.columns
df.shape
###########################################################
df[['id_grid', 'id_fold', 'id_trial']].head(20)

df_filtered = df.query('mee_va<1.8')
# df_filtered = df_filtered.query('hidden_sizes>40')

data = df_filtered
data.head()

plt.close()
sns.lmplot(
    data=data,
    x='hidden_sizes',
    y='mee_va',
    hue='id_fold',
    robust=False,
    scatter=False
)
plt.scatter(x=data['hidden_sizes'], y=data['mee_va'],
            color='gray', alpha=0.3)
plt.grid()


sns.scatterplot(
    data=data,
    x='hidden_sizes',
    # x='eta',
    y='mee_va',
    markers=True, alpha=0.7,
    legend=False
)
plt.grid()


sns.regplot(
    data=data,
    x='hidden_sizes',
    y='mee_va',
    marker=True,
    scatter=False
)
plt.grid()

plt.close()
sns.lmplot(
    data=data,
    x='hidden_sizes',
    y='mee_va',
    scatter=False
)
plt.grid()


sns.lineplot(data=data,
             x='hidden_sizes',
             # x='eta',
             y='mse_va',
             hue='id_fold',
             err_style='band')



plt.axvline(10)
plt.axvline(5)
plt.grid()
plt.close()

sns.lineplot(data=df,
             x='hidden_sizes',
             y='mse_va', style='id_fold',
             err_style='band')


###########################################################

'''
experiment in exp1/
griglia con topologie fissate con scala logaritmica, per osservare 
comportamento per alto numero di hidden sizes

'''
# leggo dati experiment 1
exp1_params = []
exp1_res = []

for nexp in range(2,13+1):

    # read experiment parameters
    with open('../data/CUP/results/exp1/cup_experiment_{}_parameters.json'.format(nexp)) as f:
        exp1_params.append(json.load(f))

    # load results
    selection = validation.ModelSelectionCV(
        grid=None,
        fname='../data/CUP/results/exp1/' +
        'cup_experiment_{}_results.json.gz'.format(nexp))

    temp = selection.load_results_pandas(flat=False)
    exp1_res.append(temp)

# dataframe risultati esperimento 1 concatenati    
exp1 = pd.concat(exp1_res)

plt.close()
sns.scatterplot(
    data=exp1,
    x='hidden_sizes',
    # x='eta',
    y='mee_va',
    hue='id_fold',
    style='id_fold',
    legend=False,
    markers=True, alpha=1.)
plt.grid()


exp1_1000 = exp1.query('hidden_sizes>900 and hidden_sizes<1200')
exp1_1000.head()
len(exp1.iloc[0]['error_per_epochs'])

plt.close()
sns.scatterplot(
    data=exp1_1000,
    x='eta',
    # x='eta',
    y='mee_va',
    hue='id_fold',
    markers=True, alpha=1.)

exp1_1000.describe()
exp1_1000.sort_values(['eta'], inplace=True)

exp1.sort_values(['mee_va'], inplace=True)

exp1.head()[['mee_va', 'hidden_sizes', 'eta']]

# plot learning curves
imp.reload(u)
min_epochs = 50
max_epochs = len(exp1.iloc[0]['error_per_epochs'])
# max_epochs = 500
for i in range(20):
    row = exp1.iloc[i]
    u.plot_learning_curve_info(
        error_per_epochs=row['mee_per_epochs'][min_epochs:max_epochs],
        error_per_epochs_va=row['mee_per_epochs_va'][min_epochs:max_epochs],
        task='validation',
        hyperparams=row['hyperparams'],
        fname='../data/CUP/results/exp1_img/exp_{}_curve_{:02d}.png'.format(
            1, i)
    )
a
# cerco di capire l'intervallo migliore per eta

exp1_500_1700 = exp1.query('hidden_sizes>500 and hidden_sizes<1700')


plt.close()
sns.scatterplot(
    data=exp1_500_1700,
    x='eta',
    # x='eta',
    y='mee_va',
    style='id_fold',
    markers=True, alpha=1.)
a
###########################################################
''' osservazioni:

con hidden sizes > di qualche centinaio si hanno mediamente risultati
migliori, serve approfondire il comportamento per valori 
compresi tra 300-1500 circa

'''
###########################################################

###########################################################

'''
Exp 2/
Esperimento con griglia su valori di hidden sizes alti

'''

nexp = 1
# read experiment parameters
with open('../data/CUP/results/exp2/cup_experiment_{}_parameters.json'.format(nexp)) as f:
    experiment_params = json.load(f)

pprint(experiment_params)

# load results
selection = validation.ModelSelectionCV(
    grid=None,
    fname='../data/CUP/results/exp2/' +
    'cup_experiment_{}_results.json.gz'.format(nexp))

exp2 = selection.load_results_pandas(flat=False)

exp2.describe()['mee_va']

exp2_filtered = exp2.query('mee_va<1.4')

plt.close()
sns.lmplot(
    data=exp2_filtered,
    x='hidden_sizes',
    y='mee_va',
    hue='id_fold',
    robust=False,
    scatter=False
)
plt.scatter(x=exp2['hidden_sizes'], y=exp2['mee_va'],
            color='gray', alpha=0.3)
plt.grid()

'''
osservazioni: 
valori medi abbastanza costanti, vediamo le learning curves

'''

# plotting learning curve
exp2.sort_values(['mee_va'], inplace=True)
exp2_filtered.sort_values(['mee_va'], inplace=True)

exp2.describe()['mee_va']
exp3.describe()['mee_va']


exp2.shape

imp.reload(u)
max_epochs = len(exp2.iloc[0]['error_per_epochs'])
min_epochs = 50
for i in range(100):
    row = exp2_filtered.iloc[i]
    u.plot_learning_curve_info(
        error_per_epochs=row['mee_per_epochs'][min_epochs:max_epochs],
        error_per_epochs_va=row['mee_per_epochs_va'][min_epochs:max_epochs],
        task='validation',
        hyperparams=row['hyperparams'],
        fname='../data/CUP/results/exp2_img/exp_{}_curve_{:02d}.png'.format(
            experiment, i)
    )

###########################################################

###########################################################

'''
Exp 3/
hidden sizes con valori intermedi, compresi tra 100 e 300

'''

nexp = 1
# read experiment parameters
with open('../data/CUP/results/exp3/cup_experiment_{}_parameters.json'.format(nexp)) as f:
    experiment_params = json.load(f)

pprint(experiment_params)

# load results
selection = validation.ModelSelectionCV(
    grid=None,
    fname='../data/CUP/results/exp3/' +
    'cup_experiment_{}_results.json.gz'.format(nexp))

exp3 = selection.load_results_pandas(flat=False)

exp3.head(20).describe()['mee_va']

exp3_filtered = exp3.query('mee_va<1.4')

plt.close()
sns.lmplot(
    data=exp3_filtered,
    x='hidden_sizes',
    y='mee_va',
    hue='id_fold',
    robust=False,
    scatter=False
)
plt.scatter(x=exp3['hidden_sizes'], y=exp3['mee_va'],
            color='gray', alpha=0.3)
plt.grid()

'''
osservazioni: 
valori medi abbastanza costanti, vediamo le learning curves

'''

# plotting learning curve
exp3.sort_values(['mee_va'], inplace=True)
exp3_filtered.sort_values(['mee_va'], inplace=True)

exp3.head()['mee_va']

exp3.shape

imp.reload(u)
max_epochs = len(exp3.iloc[0]['error_per_epochs'])
min_epochs = 50
for i in range(300):
    row = exp3_filtered.iloc[i]
    u.plot_learning_curve_info(
        error_per_epochs=row['mee_per_epochs'][min_epochs:max_epochs],
        error_per_epochs_va=row['mee_per_epochs_va'][min_epochs:max_epochs],
        task='validation',
        hyperparams=row['hyperparams'],
        fname='../data/CUP/results/exp3_img/exp_{}_curve_{:02d}.png'.format(
            experiment, i)
    )

###########################################################

'''
exp2 +exp3 together
'''
exp123 = pd.concat([exp1, exp2, exp3])

exp123_filtered = exp123.query('mee_va<1.4')


plt.close()
sns.lineplot(
    data=exp123,
    x='hidden_sizes',
    y='mee_va',
    # robust=False,
    # scatter=False
)

plt.scatter(x=exp123['hidden_sizes'], y=exp123['mee_va'],
            color='gray', alpha=0.3)
plt.grid()

###########################################################
###########################################################

'''
exp5/ 
Experiment using batch approach over large number of hidden sizes

'''


nexp = 1
# read experiment parameters
with open('../data/CUP/results/exp5/cup_experiment_{}_parameters.json'.format(nexp)) as f:
    experiment_params = json.load(f)

pprint(experiment_params)

# load results
selection = validation.ModelSelectionCV(
    grid=None,
    fname='../data/CUP/results/exp5/' +
    'cup_experiment_{}_results.json.gz'.format(nexp))

exp5 = selection.load_results_pandas(flat=False)



exp5.sort_values(['mee_va'], inplace=True)
exp5.head(20).describe()['mee_va']

# binninng hidden sizes
hbin = 100
exp5['h_cat'] = (np.array(exp5['hidden_sizes'], dtype=int)/hbin)*hbin
exp5['h_cat']


# exp5_filtered = exp5.query('mee_va<1.4')
exp5_filtered = exp5.query('hidden_sizes>1000')
exp5_filtered = exp5

exp5.describe()['mee_va']

plt.close()
sns.lmplot(
    data=exp5_filtered,
    x='h_cat',
    y='mee_va',
    hue='id_fold',
    robust=False,
    scatter=False
)
plt.scatter(x=exp5_filtered['h_cat'], y=exp5_filtered['mee_va'],
            color='gray', alpha=0.3)
plt.grid()

plt.close()
sns.lineplot(
    data=exp5_filtered,
    x='hidden_sizes',
    y='mee_va',
    hue='id_fold',
)





# plotting learning curve
exp5.sort_values(['mee_va'], inplace=True)
exp5_filtered.sort_values(['mee_va'], inplace=True)

exp5.head()['mee_va']

exp5.shape

imp.reload(u)
max_epochs = len(exp5.iloc[0]['error_per_epochs'])
min_epochs = 50
for i in range(200):
    row = exp5.iloc[i]
    u.plot_learning_curve_info(
        error_per_epochs=row['mee_per_epochs'][min_epochs:max_epochs],
        error_per_epochs_va=row['mee_per_epochs_va'][min_epochs:max_epochs],
        task='validation',
        hyperparams=row['hyperparams'],
        fname='../data/CUP/results/exp5_img/exp_{}_curve_{:02d}.png'.format(
            nexp, i)
    )
