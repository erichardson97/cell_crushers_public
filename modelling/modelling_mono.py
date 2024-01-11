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
from scipy.stats import spearmanr, linregress
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', dest = 'data_directory')
args = parser.parse_args()

use_baseline = True
use_olink = True
use_cellfreq = True
use_genes = True

def calc_residuals_for_prediction(baseline, y):
  slope, intercept, r, p, se = linregress(baseline, y)
  residuals = [y_val - (p*slope + intercept) for p, y_val in zip(baseline, y)]
  return slope, intercept, np.array(residuals)

def calc_residuals_for_prediction_Rank(baseline, y):
  slope, intercept, r, p, se = linregress(baseline, y)
  predictions = baseline*slope + intercept
  prediction_rank = np.argsort(predictions)
  true_rank = np.argsort(y)
  residuals = (true_rank - prediction_rank) / prediction_rank.shape[0]
  return slope, intercept, np.array(residuals)


def residuals_model(base_class: sklearn.base.BaseEstimator):
  class ResidualModel(base_class):

    def __init__(self, **kwargs):
      super().__init__(**kwargs)

    def fit(self, X, y):
      baseline = X[:, -1]
      if not use_baseline:
        X = X[:, :-1]
      slope, intercept, residuals = calc_residuals_for_prediction(baseline, y)
      self.slope = slope
      self.intercept = intercept
      super().fit(X, residuals)

    def predict(self, X):
      baseline = X[:, -1]
      if not use_baseline:
        X = X[:, :-1]
      residuals = super().predict(X)
      return self.slope * baseline + self.intercept + residuals
  return ResidualModel

def repeat_cv(data, candidate_features, args_for_cv, output_path, cv_type = 'RegularCV', cv_path: str = ''):
  cvobj = CV(data[candidate_features+['Target', 'dataset']])
  total = []
  coefs_ = []
  n_repeats = 10 if cv_type == 'RegularCV' else 1
  for f in range(n_repeats):
    args_for_cv['plot_dir'] = os.path.join(output_path, f'Repeat{f}')
    if cv_type == 'RegularCV':
      args_for_cv['precomputed_split'] = os.path.join(cv_path, f'Repeat{f}_CV_Idx.p')
    if os.path.exists(args_for_cv['plot_dir']) is False:
      os.mkdir(args_for_cv['plot_dir'])
    output, _, coefs = cvobj.RunCV(cv_type=cv_type, cv_args = args_for_cv)
    output['repeat'] = f
    coefs['repeat'] = f
    total.append(output)
    coefs_.append(coefs)
    output.to_csv(os.path.join(args_for_cv['plot_dir'], f'Performance{f}.csv'))
    coefs.to_csv(os.path.join(args_for_cv['plot_dir'], f'FeatureImportance{f}.csv'))
  total = pd.concat(total)
  total.to_csv(os.path.join(output_path, f'PerformanceTotal.csv'))
  return total
  

def train_test_split(X, y, n_splits = 5, repeats = 10):
  split_indexes = {}
  for repeat in range(repeats):
    fold = 0
    split_indexes[repeat] = {}
    for train_idx, test_idx in KFold(n_splits = n_splits, shuffle = True).split(X, y):
        split_indexes[repeat][fold] = {'Train':train_idx, 'Test':test_idx}
        fold += 1  
  return split_indexes

def load_data(path = '/content/drive/MyDrive/CMIPB_Files/IntegratedData.tsv', target = 'Day14_IgG_Titre', transform = True, keep = True):
  data = pd.read_csv(path, sep = '\t', index_col = 0)
  if ('Target' in data) & (target != 'Target'):
    del data['Target']
  data = data.rename(columns = {target: 'Target'})
  data = data[(data['Target'].notna())]
  ds = Dataset(data)
  return ds
  
data_directory = args.data_directory
results_directory = '/'.join(data_directory.split('/')[:-1]) + '/results'
if os.path.exists(results_directory) is False:
  os.mkdir(results_directory)
  
features = pd.read_pickle(os.path.join(data_directory, 'AllFeatures.p'))
for p in features:
  features[p] = list(features[p])
  
model_params = {}
model_classes = {}
return_coef = {}
for alpha in [.01, 0.05, 0.1, 1]:
  model_params[f'Lasso_{alpha}'] = {'alpha':alpha}
  model_classes[f'Lasso_{alpha}'] = Lasso
  model_params[f'Ridge_{alpha}'] = {'alpha':alpha}
  model_classes[f'Ridge_{alpha}'] = Ridge
  model_params[f'Lasso_Residuals_{alpha}'] = {'alpha':alpha}
  model_classes[f'Lasso_Residuals_{alpha}'] = residuals_model(Lasso)
  model_params[f'Ridge_Residuals_{alpha}'] = {'alpha':alpha}
  model_classes[f'Ridge_Residuals_{alpha}'] = residuals_model(Ridge)
model_params[f'Linear'] = {}
model_classes[f'Linear'] = residuals_model(LinearRegression)
for x in model_classes:
  return_coef[x] = 'coef_'
for params in ParameterGrid({'max_features':[None, 'sqrt', 'log2'], 'n_estimators':[10,50,100,150,500]}):
  max_feat = params['max_features']
  n_estimators = params['n_estimators']
  model_params[f'RandomForest_{max_feat}_{n_estimators}'] = params
  model_classes[f'RandomForest_{max_feat}_{n_estimators}'] = params
  return_coef[f'RandomForest_{max_feat}_{n_estimators}'] = 'feature_importances_'
  model_params[f'RandomForest_Residuals_{max_feat}_{n_estimators}'] = params
  model_classes[f'RandomForest_{max_feat}_{n_estimators}'] = RandomForestRegressor
  model_classes[f'RandomForest_Residuals_{max_feat}_{n_estimators}'] = residuals_model(RandomForestRegressor)
  return_coef[f'RandomForest_Residuals_{max_feat}_{n_estimators}'] = 'feature_importances_'
  return_coef[f'RandomForest_{max_feat}_{n_estimators}'] = 'feature_importances_'
  
for params in ParameterGrid({'loss':['squared_error','absolute_error'], 'n_estimators':[100, 1000, 1000], 'subsample':[0.8,0.9,1],
                             'max_features':[None, 'sqrt', 'log2']}):
    max_feat = params['max_features']
    n_estimators = params['n_estimators']
    loss = params['loss']
    subsample = params['subsample']    
    model_classes[f'GradientBoost_{max_feat}_{n_estimators}_{loss}_{subsample}'] = GradientBoostingRegressor
    model_params[f'GradientBoost_{max_feat}_{n_estimators}_{loss}_{subsample}'] = params
    return_coef[f'GradientBoost_{max_feat}_{n_estimators}_{loss}_{subsample}'] = 'feature_importances_'                           
    model_classes[f'GradientBoost_Residuals_{max_feat}_{n_estimators}_{loss}_{subsample}'] = residuals_model(GradientBoostingRegressor)
    model_params[f'GradientBoost_Residuals_{max_feat}_{n_estimators}_{loss}_{subsample}'] = params
    return_coef[f'GradientBoost_Residuals_{max_feat}_{n_estimators}_{loss}_{subsample}'] = 'feature_importances_'


cv_split = '/mnt/bioadhoc/Users/erichard/cell_crushers/data/cv_folds'

for cv_type in ['RegularCV','CrossDataset']:
  for target in ['Target', 'Target_FC']:
    for file in glob(os.path.join(data_directory, 'correlation_filtered', '*tsv')):
      threshold = file.split('/')[-1][19:].split('.tsv')[0]
      ds = load_data(file, target = target)
      ds.filter(['Titre_IgG_PT','Target'])
      genes = [p for p in ds.data if 'GEX' in p]
      feature_list = genes 
      if use_olink:
        feature_list += features['cytokine']
      if use_cellfreq:
        feature_list +=  features['cell_freq']
      feature_list = [p for p in feature_list if p not in features['demographic']]
      feature_list += features['demographic']
      ds.filter(feature_list, nan_policy = 'drop')
      output_directory = os.path.join(results_directory, f'Model_NoncorrelatedGenes_{threshold}_{cv_type}_{target}')
      if os.path.exists(output_directory) is False:
        os.mkdir(output_directory)
      args_for_cv = {'target':'Target', 'n_splits':5, 'score_function':corr_coeff_report, 'features':feature_list,
                   'transformation':False, 'plot_dir':output_directory, 'transformation_args':{}, 'model_params': model_params,
                   'model_classes':model_classes, 'return_coef':return_coef, 'plot' : False, 'baseline':feature_list[-1]}
      repeat_cv(ds.data, feature_list, args_for_cv, output_directory, cv_type = cv_type, cv_path = '/mnt/bioadhoc/Users/erichard/cell_crushers/ig_task/data/cv_split')
      ds.data[feature_list].to_csv(os.path.join(output_directory,'dataset.tsv'),sep='\t')
        
    for gene_type in ['genes', 'filtered_genes']:
      ds = load_data(os.path.join(data_directory, "IntegratedData_Normalized.tsv"), target = target)
      ds.filter(['Titre_IgG_PT','Target'])
      feature_list = features[gene_type]
      if use_olink:
        feature_list += features['cytokine']
      if use_cellfreq:
        feature_list +=  features['cell_freq']
      feature_list = [p for p in feature_list if p not in features['demographic']]
      feature_list += features['demographic']
      ds.filter(feature_list, nan_policy = 'keep')
      output_directory = os.path.join(results_directory, f'Model_{gene_type}_{cv_type}_{target}')
      if os.path.exists(output_directory) is False:
        os.mkdir(output_directory)
      args_for_cv = {'target':'Target', 'n_splits':5, 'score_function':corr_coeff_report, 'features':feature_list,
                   'transformation':False, 'plot_dir':output_directory, 'transformation_args':{}, 'model_params': model_params,
                   'model_classes':model_classes, 'return_coef':return_coef, 'plot' : False, 'baseline':feature_list[-1]}
      repeat_cv(ds.data, feature_list, args_for_cv, output_directory, cv_type = cv_type, cv_path = '/mnt/bioadhoc/Users/erichard/cell_crushers/ig_task/data/cv_split')
      ds.data[feature_list].to_csv(os.path.join(output_directory,'dataset.tsv'),sep='\t')
      for n_components in [10, 15, 30, len(feature_list)]:
        if cv_type != "CrossDataset":
          if ((n_components >= len(features[gene_type])) | (n_components >= int(ds.data.shape[0]*0.8))):
            continue
        else:
          if ((n_components >= len(features[gene_type])) | (n_components >= ds.data['dataset'].value_counts().min())):
            continue
        output_directory = os.path.join(results_directory, f'Model_{gene_type}_PCGenes_{n_components}_{cv_type}_{target}')
        if os.path.exists(output_directory) is False:
          os.mkdir(output_directory)
        args_for_cv['transformation'] = reduce_dimensions
        args_for_cv['transformation_args'] = {'features':np.array(feature_list),'features_to_change' : np.array(features[gene_type]),
                'reducer':PCA, 'n_components':n_components}
        repeat_cv(ds.data, feature_list, args_for_cv, output_directory, cv_type = cv_type, cv_path = '/mnt/bioadhoc/Users/erichard/cell_crushers/ig_task/data/cv_split')
        ds.data[feature_list].to_csv(os.path.join(output_directory,'dataset.tsv'),sep='\t')
        if cv_type != 'CrossDataset':
          if ((n_components >= len(feature_list)) & (n_components >= int(ds.data.shape[0]*0.8))): 
            continue
        else:
          if ((n_components >= len(feature_list)) | (n_components >= ds.data['dataset'].value_counts().min())):
            continue
        output_directory = os.path.join(results_directory, f'Model_{gene_type}_PCTotal_{n_components}_{cv_type}_{target}')
        if os.path.exists(output_directory) is False:
          os.mkdir(output_directory)
        args_for_cv['transformation'] = reduce_dimensions
        args_for_cv['transformation_args'] = {'features':np.array(feature_list),'features_to_change' : np.array(feature_list),
                'reducer':PCA, 'n_components':n_components}
        
        repeat_cv(ds.data, feature_list, args_for_cv, output_directory, cv_type = cv_type, cv_path = '/mnt/bioadhoc/Users/erichard/cell_crushers/ig_task/data/cv_split')
        ds.data[feature_list].to_csv(os.path.join(output_directory,'dataset.tsv'),sep='\t')
      # for n_components in [10, 15, 30, 50, len(features[gene_type])]:
      #     if n_components >= len(features[gene_type]):
      #       continue
      #     if cv_type != 'CrossDataset':
      #       if len(features[gene_type]) >= int(ds.data.shape[0]*0.8):
      #         continue
      #     else:
      #       if len(features[gene_type]) >= ds.data['dataset'].value_counts().min():
      #         continue
      #     output_directory = os.path.join(results_directory, f'Model_{gene_type}_ReGain_{n_components}_{cv_type}_{target}')
      #     if os.path.exists(output_directory) is False:
      #       os.mkdir(output_directory)
      #     args_for_cv['transformation'] = reduce_dimensions
      #     args_for_cv['transformation_args'] = {'features':np.array(feature_list),'features_to_change' : np.array(features[gene_type]),
      #             'reducer':ReGainBootleg, 'n_components':n_components}
      #     repeat_cv(ds.data, feature_list, args_for_cv, output_directory, cv_type = cv_type, cv_path = '/mnt/bioadhoc/Users/erichard/cell_crushers/ig_task/data/cv_split')
      #     ds.data[feature_list].to_csv(os.path.join(output_directory,'dataset.tsv'),sep='\t')
