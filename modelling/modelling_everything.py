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
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from glob import glob


def calc_residuals_for_prediction(baseline, y):
  slope, intercept, r, p, se = linregress(baseline, y)
  residuals = [y_val - (p*slope + intercept) for p, y_val in zip(baseline, y)]
  return slope, intercept, np.array(residuals)

def residuals_model(base_class: sklearn.base.BaseEstimator):
  class ResidualModel(base_class):

    def __init__(self, **kwargs):
      super().__init__(**kwargs)

    def fit(self, X, y):
      baseline = X[:, -1]
      X = X[:, :-1]
      slope, intercept, residuals = calc_residuals_for_prediction(baseline, y)
      self.slope = slope
      self.intercept = intercept
      super().fit(X, residuals)

    def predict(self, X):
      baseline = X[:, -1]
      X = X[:, :-1]
      residuals = super().predict(X)
      return self.slope * baseline + self.intercept + residuals
  return ResidualModel

def repeat_cv(data, candidate_features, args_for_cv, output_path, cv_type = 'RegularCV'):
  cvobj = CV(data[candidate_features+['Target', 'dataset']])
  total = []
  coefs_ = []
  for f in range(10):
    args_for_cv['plot_dir'] = os.path.join(output_path, f'Repeat{f}')
    if cv_type == 'RegularCV':
      args_for_cv['precomputed_split'] = f'/mnt/bioadhoc/Users/erichard/cell_crushers/data/cv_folds/Repeat{f}_CV_Idx.p'
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
  

def load_data(path = '/content/drive/MyDrive/CMIPB_Files/IntegratedData.tsv', target = 'Day14_IgG_Titre', transform = True):
  data = pd.read_csv(path, sep = '\t', index_col = 0)
  data = data.rename(columns = {target: 'Target'})
  data = data[data['Target'].notna()]
  data['Target'] = data['Target'].map(np.log2)
  data['Titre_IgG_PT'] = data['Titre_IgG_PT'].map(np.log2)
  ds = Dataset(data)
  return ds
  
data_directory = '/mnt/bioadhoc/Users/erichard/cell_crushers/data/'
results_directory = '/mnt/bioadhoc/Users/erichard/cell_crushers/results'

features = pd.read_pickle(os.path.join(data_directory, 'AllFeatures.p'))

use_demographic = True
use_cells = True
use_olink = True
gene_type = 'all_genes' #'filtered', 'uncorrelated'

model_params = {}
model_classes = {}
return_coef = {}

for alpha in [.01, 0.05, 0.1, 1]:
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
  model_params[f'RandomForest_Residuals_{max_feat}_{n_estimators}'] = params
  model_classes[f'RandomForest_{max_feat}_{n_estimators}'] = RandomForestRegressor
  model_classes[f'RandomForest_Residuals_{max_feat}_{n_estimators}'] = residuals_model(RandomForestRegressor)
  return_coef[f'RandomForest_Residuals_{max_feat}_{n_estimators}'] = 'feature_importances_'
  return_coef[f'RandomForest_{max_feat}_{n_estimators}'] = 'feature_importances_'




# for file in glob(os.path.join(data_directory, 'correlation_filtered', '*tsv')):
#   target = 'Day14_IgG_Titre'
#   threshold = file.split('/')[-1][19:].split('.tsv')[0]
#   ds = load_data(file)
#   ds.filter(['Titre_IgG_PT','Target'])
#   genes = [p for p in ds.data if 'GEX' in p]
#   feature_list =  genes + features['cell_freq'] + features['cytokine'] + features['demographic']
#   ds.filter(feature_list)
#   assert feature_list[-1] == 'Titre_IgG_PT'
#   output_directory = os.path.join(results_directory, f'Model_NoncorrelatedGenes_{threshold}')
#   if os.path.exists(output_directory) is False:
#     os.mkdir(output_directory)
#   args_for_cv = {'target':'Target', 'n_splits':5, 'score_function':corr_coeff_report, 'features':feature_list,
#                'transformation':False, 'plot_dir':output_directory, 'transformation_args':{}, 'model_params': model_params,
#                'model_classes':model_classes, 'return_coef':return_coef, 'plot' : False}
#   repeat_cv(ds.data, feature_list, args_for_cv, output_directory)
#   for n_components in [10, 15, 30, 50, len(genes)]:
#     if n_components >= len(genes):
#       continue
#     if len(genes) >= int(ds.data.shape[0]*0.8):
#       continue
#     output_directory = os.path.join(results_directory, f'Model_NoncorrelatedGenes{threshold}_ReGain_{n_components}')
#     if os.path.exists(output_directory) is False:
#       os.mkdir(output_directory)
#     args_for_cv['transformation'] = reduce_dimensions
#     args_for_cv['transformation_args'] = {'features':np.array(feature_list),'features_to_change' : np.array(genes),
#             'reducer':ReGainBootleg, 'n_components':n_components}
#     repeat_cv(ds.data, feature_list, args_for_cv, output_directory)
    
for gene_type in ['GO_Genes']: #['all_genes', 'filtered_genes', 'literature_genes','literature_genes>1', 'GO_Genes']:
  target = 'Day14_IgG_Titre'
  ds = load_data(os.path.join(data_directory, "IntegratedData_Normalized.tsv"))
  ds.filter(['Titre_IgG_PT','Target'])
  feature_list = features[gene_type] + features['cell_freq'] + features['cytokine'] + features['demographic']
  assert feature_list[-1] == 'Titre_IgG_PT'
  ds.filter(feature_list)
  output_directory = os.path.join(results_directory, f'Model_{gene_type}')
  if os.path.exists(output_directory) is False:
    os.mkdir(output_directory)
  args_for_cv = {'target':'Target', 'n_splits':5, 'score_function':corr_coeff_report, 'features':feature_list,
               'transformation':False, 'plot_dir':output_directory, 'transformation_args':{}, 'model_params': model_params,
               'model_classes':model_classes, 'return_coef':return_coef, 'plot' : False}
  repeat_cv(ds.data, feature_list, args_for_cv, output_directory)
  for n_components in [10, 15, 30, len(feature_list)]:
    if ((n_components < len(features[gene_type])) & (n_components < int(ds.data.shape[0]*0.8))):
      output_directory = os.path.join(results_directory, f'Model_{gene_type}_PCGenes_{n_components}')
      if os.path.exists(output_directory) is False:
        os.mkdir(output_directory)
      args_for_cv['transformation'] = reduce_dimensions
      args_for_cv['transformation_args'] = {'features':np.array(feature_list),'features_to_change' : np.array(features[gene_type]),
              'reducer':PCA, 'n_components':n_components}
      repeat_cv(ds.data, feature_list, args_for_cv, output_directory)
    if ((n_components < len(feature_list)) & (n_components < int(ds.data.shape[0]*0.8))): 
      output_directory = os.path.join(results_directory, f'Model_{gene_type}_PCTotal_{n_components}')
      if os.path.exists(output_directory) is False:
        os.mkdir(output_directory)
      args_for_cv['transformation'] = reduce_dimensions
      args_for_cv['transformation_args'] = {'features':np.array(feature_list),'features_to_change' : np.array(feature_list),
              'reducer':PCA, 'n_components':n_components}
      
      repeat_cv(ds.data, feature_list, args_for_cv, output_directory)
  for n_components in [10, 15, 30, 50, len(features[gene_type])]:
      if n_components >= len(features[gene_type]):
        continue
      if len(features[gene_type]) >= int(ds.data.shape[0]*0.8):
        continue
      output_directory = os.path.join(results_directory, f'Model_{gene_type}_ReGain_{n_components}')
      if os.path.exists(output_directory) is False:
        os.mkdir(output_directory)
      args_for_cv['transformation'] = reduce_dimensions
      args_for_cv['transformation_args'] = {'features':np.array(feature_list),'features_to_change' : np.array(features[gene_type]),
              'reducer':ReGainBootleg, 'n_components':n_components}
      repeat_cv(ds.data, feature_list, args_for_cv, output_directory)



