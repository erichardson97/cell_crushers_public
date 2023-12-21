import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import os
import shutil
import pickle
import sklearn
from data_utils import *
from utils import *
from scipy.stats import linregress
from autogluon.tabular import TabularPredictor


data_directory = '/mnt/bioadhoc/Users/erichard/cell_crushers/data/'
data = pd.read_csv(os.path.join(data_directory, 'IntegratedData_Normalized.tsv'), sep = '\t', index_col = 0)
target = 'Day14_IgG_Titre'
data = data.rename(columns = {target: 'Target'})
data = data[(data['Target'].notna())&(data['Titre_IgG_PT'].notna())]
data['Target'] = data['Target'].map(float).map(np.log2)
data['Titre_IgG_PT'] = data['Titre_IgG_PT'].map(float).map(np.log2)

github_directory = '/content/drive/MyDrive/Github'

features_sans_target = pd.read_csv(os.path.join(data_directory, 'features_sans_target.txt'))['feature name'].values
filtered_genes = pd.read_csv(os.path.join(data_directory, 'FilteredGenes.txt'))['feature name'].unique()
demographic = ['age', 'biological_sex', 'infancy_vac', 'Titre_IgG_PT']
cell_freqs = ['Cellfrequency_TcmCD4', 'Cellfrequency_TemCD4', 'Cellfrequency_Bcells','Cellfrequency_ASCs (Plasmablasts)']
genes = [p for p in features_sans_target if 'GEX' in p]
cytokines = [p for p in features_sans_target if 'Cytokine' in p]
cytokines.remove('Cytokine_IFNG')
cytokines.remove('Cytokine_TNF')
candidate_features = list(filtered_genes) + cytokines + demographic + cell_freqs


output_path = '/mnt/bioadhoc/Users/erichard/CMIPB_Autogluon'
cv_results = []
for repeat in range(10):
  splits = pd.read_pickle(os.path.join(data_directory, 'cv_folds', f'Repeat{repeat}_CV_Idx.p'))
  mini_results = {'Fold':[],'Repeat':[],'TopModel':[],'Score':[], 'Baseline':[]}
  for fold in range(5):
    train = data.iloc[splits[fold]['Train']][candidate_features+['Target']]
    train[train.columns] = StandardScaler().fit_transform(train)
    slope, intercept, _, _, _ = linregress(train['Titre_IgG_PT'], train['Target'])
    train["Target"] = train['Target'] - (train['Titre_IgG_PT']*slope+intercept)
    test = data.iloc[splits[fold]['Test']][candidate_features+['Target']]
    test[test.columns] = StandardScaler().fit_transform(test)
    #test['Target'] = test['Target'] - (test['Titre_IgG_PT']*slope+intercept)
    path = os.path.join(output_path, f'Repeat{repeat}/Fold{fold}')
    predictor = TabularPredictor(label = 'Target', path = path)
    predictor.fit(train, presets='best_quality')
    summary = predictor.fit_summary()
    with open(os.path.join(path, 'Summary.p'), 'wb') as k:
      pickle.dump(summary, k)
    score = corr_coeff_report(predictor.predict(test)+test['Titre_IgG_PT']*slope+intercept, test['Target'])
    mini_results['Fold'].append(fold)
    mini_results['Repeat'].append(repeat)
    mini_results['TopModel'].append(max(summary['model_performance'].items(),key=lambda x:x[1])[0])
    mini_results['Baseline'].append(corr_coeff_report(test['Titre_IgG_PT'], test['Target']))
    mini_results['Score'].append(score)
  mini_results = pd.DataFrame(mini_results)
  mini_results.to_csv(os.path.join(path, f'Repeat{repeat}/Performance.csv'))
  cv_results.append(mini_results)
cv_results = pd.concat(cv_results)
cv_results.to_csv(os.path.join(path, 'PerformanceTotal.csv'))
