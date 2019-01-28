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

experiment = 3

# read experiment parameters
with open('../data/CUP/results/cup_experiment_{}_parameters.json'.format(experiment)) as f:
    experiment_params = json.load(f)

pprint(experiment_params)

# load results
selection = validation.ModelSelectionCV(
    grid=None,
    fname='../data/CUP/results/' +
    'cup_experiment_{}_results.json.gz'.format(experiment))

df = selection.load_results_pandas(flat=False)
df.shape

df.memory_usage(deep=True).sum()/1000/1000  # MBytes
df.columns
df.shape
###########################################################
df[['id_grid', 'id_fold', 'id_trial']].head(20)

df_filtered = df.query('mse_va<4.5')
df_filtered.describe()['eta']

data = df_filtered

plt.close()
sns.scatterplot(
    data=data,
    x='hidden_sizes',
    # x='eta',
    y='mse_va',
    hue='eta',
    markers=True, alpha=0.6)


plt.close()
sns.lmplot(
    data=data,
    x='hidden_sizes',
    y='mse_va',
    hue='id_fold'
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
             y='mse_va', hue='id_fold',
             err_style='band')


###########################################################

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
    markers=True, alpha=1.)

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
a

exp1_1000.describe().
exp1_1000.sort_values(['eta'], inplace=True)

# plot learning curves
imp.reload(u)
min_epochs = 50
max_epochs = len(exp1.iloc[0]['error_per_epochs'])
# max_epochs = 500
for i in range(60):
    row = exp1_1000.iloc[i]
    u.plot_learning_curve_info(
        error_per_epochs=row['mee_per_epochs'][min_epochs:max_epochs],
        error_per_epochs_va=row['mee_per_epochs_va'][min_epochs:max_epochs],
        task='validation',
        hyperparams=row['hyperparams'],
        fname='../data/CUP/results/exp1_img/exp_{}_curve_{:02d}.png'.format(
            experiment, i)
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
# 
df.sort_values(['accuracy', 'mse_va'], ascending=[False, True], inplace=True)
df[['accuracy', 'mse_va', 'eta']].head(200).describe()

df.shape

# seleziona la migliore per ciascun valore della griglia, ovvero prendendo il minimo al variare di fold e trial

df_min = df.groupby(['id_grid']).agg({
    'mse_va': [np.median, np.std],
    'accuracy': [np.median, np.std]
})
df_min.sort_values(('mse_va', 'median'), inplace=True)
df_min.head()
df_min.sort_values(('accuracy', 'median'), ascending=False, inplace=True)
df_min.head(30)
df_min.shape
df_min

df_min.columns = ['_'.join(x) for x in df_min.columns.ravel()]
# index_min = df.groupby(['id_grid'])['mse_va'].idxmin()
df_min.head()

df_min['threshold']=df_min['mse_va_median'].iloc[0]+2*df_min['mse_va_std']

# migliori risultati per accuracy
df_best = df_min.query('accuracy_median==1')

# faccio la join
df_parameters = df.drop_duplicates('id_grid')

df_join = pd.merge(df_best, df_parameters, on='id_grid', how='inner')
df_join.shape

df_join.describe()[['eta','alpha', 'hidden_sizes']]





df_min.query('mse_va_median<threshold')

df.query('id_grid==80').iloc[0][['id_grid', 'eta', 'alpha', 'hidden_sizes']]





df_agg.query('accuracy==1.00').sort_values(['mse_va_std']).head()
df_agg['threshold']=df_agg['mse_va_median'].iloc[0]+3*df_agg['mse_va_std']



df_agg['mse_va_median'].iloc[0]

df_agg.head(37)

df_agg[['mse_va_std', 'mse_va_median', 'threshold',
        'eta']]
df_agg.head(100).describe()['eta']





len(index_min)

df_min = df.iloc[index_min]


grouped = df.groupby(['id_grid']).agg({
    'mse_va': [np.median, np.std],
    'accuracy': [np.median, np.std]
})



plt.close()
###########################################################


df.sort_values(['mee_va'])


imp.reload(u)
max_epochs = len(df.iloc[0]['error_per_epochs'])
# max_epochs = 500
for i in range(100):
    row = df.iloc[i]
    u.plot_learning_curve_info(
        error_per_epochs=row['error_per_epochs'][:max_epochs],
        error_per_epochs_va=row['error_per_epochs_va'][:max_epochs],
        task='validation',
        hyperparams=row['hyperparams'],
        fname='../data/CUP/results/img/exp_{}_curve_{:02d}.png'.format(
            experiment, i)
    )


###########################################################

df_sorted = df.sort_values('accuracy', ascending=False)
df_sorted[['mse_va', 'accuracy', 'mse_tr', 'eta', 'alpha']].head(10)

df_sorted['accuracy'].iloc[0]

grouped = df.groupby(['id_grid']).agg({
    'mse_va': [np.median, np.std],
    'accuracy': [np.median, np.std]
})

grouped['id_grid']=grouped.index
grouped.columns = ['_'.join(x) for x in grouped.columns.ravel()]

(grouped.head())

grouped.shape

grouped.columns
df_agg = pd.merge(grouped, df, on='id_grid', how='inner').drop_duplicates('id_grid')
df_agg.shape

df_agg.columns

df_agg.sort_values(['accuracy_median', 'mse_va_median'],
                   ascending=[False, True], inplace=True)

df_agg[['accuracy_median', 'mse_va_median', 'eta']].query('eta<1').head()



df_agg.sort_values(['accuracy_median', 'mse_va_median'], ascending=[False,True]).head(10)[['id_grid','eta']]


grouped.sort_values('mse_median')[['mse_median', 'mse_std']].head(10)

df.query('id_grid==2829')[['hidden_sizes','id_fold','id_trial']]




imp.reload(validation)

df_agg_best = validation.df2df_flat(df_agg.sort_values('mse').iloc[:10])
df_agg_best


df_grid = df.groupby('id_grid').agg('mean').sort_values('mse', ascending=True)


df_group_hidden = df.groupby('hidden_sizes').agg(['median','std']).sort_values(('mse','median'))
df_group_hidden.columns

df_group_hidden.head()

a
# ho individuato l'hidden size con un minimo, voglio individuare
# gli altri parametri migliori, qual'è la combinazione migliore?

# faccio la media/median sui vari fold

df_h = df.query('hidden_sizes==6')
df_h['alpha'].describe()

plt.close()
sns.lineplot(data=df_h,
             x='eta',
             y='mse',
)
plt.close()
sns.lineplot(data=df_h,
             x='eta',
             y='mse',
             hue='id_fold'
)


plt.close()
sns.scatterplot(data = df_h,
                x='alpha',
                y='mse',
                hue='eta'
)
plt.axhline(df_h['mse'].median())

pd.set_option('display.expand_frame_repr', True)
df_h.query('eta>0.4 and eta<0.6').sort_values('mse')['id_grid']

id_grid = 791

id_grid = 7

## voglio andare a plottare la learning curve corrispondente all'id_grid trovato, per ciascun fold

df_selected = df.query('id_grid =={}'.format(id_grid))


for i in range(5):
    plt.plot(np.arange(500),
             df_selected['error_per_epochs'].iloc[i], label='train')
    plt.plot(np.arange(500),
             df_selected['error_per_epochs_va'].iloc[i], label='va')

    plt.legend()
    plt.savefig('../data/monks/results/learning_fold_{}.png'.format(i))
    plt.close()


###########################################################


###########################################################


sns.scatterplot(x='hidden_sizes', y='mse', data = df,
                hue = 'eta')


plt.scatter(df['hidden_sizes'], df['mse'], alpha=0.5)
plt.plot(
    x=df_group_hidden.index,
    y=df_group_hidden['mse']['median'])
plt.close()



df_grid.query('hidden_sizes <= 20 & mse <= 0.1').describe()



p = gg.ggplot(df_grid, gg.aes(y='mse',x='hidden_sizes', color='alpha'))
p+gg.geom_point()

p = gg.ggplot(df, gg.aes(y='f1_score',x='hidden_sizes', color='hidden_sizes'))
p + gg.geom_point()+gg.ggtitle('')+gg.facet_wrap('~id_fold')


df_best_flat = valid.df2df_flat(df_best)

df_best = df_grid.sort_values('mse', ascending=True).iloc[:50]

df_best['alpha'].describe()
df_best['eta'].describe()

p = gg.ggplot(df_best, gg.aes(y='accuracy', x='eta', color = 'alpha'))
p+gg.geom_point()


df_heat = df[['alpha', 'eta', 'mse', 'hidden_sizes']]
df_heat['hidden_sizes_bin'] = np.round(df['hidden_sizes']/5)*5
df_heat['alpha_bin'] = np.round(df['alpha'], 1)
df_heat['eta_bin'] = np.round(df['eta'], 1)
df_heat['eta_bin'] = np.round(df['eta'], 1)

df_heat

df_pivot = df_heat.pivot_table(
    index='eta_bin',
    columns='alpha_bin',
    values='mse',
    aggfunc='median')
df_pivot

sns.heatmap(df_pivot)

plt.close()
df.columns

