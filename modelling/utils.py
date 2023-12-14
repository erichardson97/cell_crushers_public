import pandas as pd
from typing import Union, Callable
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold, ParameterGrid



def corr_coeff_report(y_pred: list, y_true: list) -> float:
  '''
  Function that returns Spearman's correlation coefficient between predictions
  and true target values.
  In contrast to corr_coeff, this is not a loss function.
  '''
  return spearmanr(y_pred, y_true)[0]

def corr_coeff(y_pred: list, y_true: list) -> float:
  '''
  Function that turns Spearman's correlation coefficient between predictions
  and true target values, into a loss (i.e. the smaller the coefficient/if coefficient is negative,
  the larger the loss).
  In contrast to corr_coeff_report, this IS to be used as a loss function (NB
  non-differentiable).
  '''
  return float(1 - spearmanr(y_pred, y_true)[0])

def return_property(model, string):
  return getattr(model, string)

class HyperparamSearch():
  def __init__(self):
      pass
    
  def run_search(self, params: dict, cv_func: Callable, cv_args: dict):
      parameter_indices = {}
      cv_results_total = []
      for f, g in enumerate(ParameterGrid(params)):
          parameter_indices[f] = g
          cv_args['model_params'] = g
          cv_results = cv_func(**cv_args)
          cv_results_total.append(cv_results)
      return cv_results_total, parameter_indices

class CV():
  
  def __init__(self, data: pd.DataFrame):
    self.data = data

  def RunCV(self, cv_type: str, cv_args: dict):
    if cv_type == 'LOOCV':
      return self.loocv(**cv_args)
    elif cv_type == 'RegularCV':
      return self.regular_ol_cv(**cv_args)
    elif cv_type == 'CrossDataset_Nested':
      return self.cross_dataset_CV_Nested(**cv_args)
    elif cv_type == 'CrossDataset':
      return self.cross_dataset_CV(**cv_args)
    

  def loocv(self, features: list, target: str, model_class):
    '''
    Function for Leave-one out cross-validation.
    Score is MSE, because there is only one sample.
    So cannot calculate a correlation coefficient.
    '''
    X = self.data[features].values
    y = self.data[target].values
    scores = {'Test':[], 'MSE':[]}
    trained_models = {}
    for train_idx, test_idx in KFold(n_splits = len(X)).split(X, y):
      train_X, train_y = X[train_idx], y[train_idx]
      test_X, test_y = X[test_idx], y[test_idx]
      model = model_class().fit(train_X, train_y)
      val = model.predict(test_X)
      mse = ((test_y - val)**2)[0]
      scores['Test'].append(test_idx[0])
      scores['MSE'].append(mse)
      trained_models[test_idx[0]] = model
    scores = pd.DataFrame(scores)
    return scores, trained_models


  def regular_ol_cv(self, features: list, target: str, n_splits: int, score_function: Callable, model_class, model_params: dict = {}, return_coef: str = 'coef_'):
    '''
    Regular CV with no stratification by year.
    '''
    X = self.data[features].values
    baseline = self.data['Titre_IgG_PT'].values
    y = self.data[target].values
    fold = 0
    scores = {'Fold':[], 'Score':[], 'MSE':[], 'Baseline':[]}
    trained_models = {}
    for train_idx, test_idx in KFold(n_splits = n_splits, shuffle = True).split(X, y):
      train_X, train_y = X[train_idx], y[train_idx]
      test_X, test_y = X[test_idx], y[test_idx]
      model = model_class(**model_params)
      model.fit(train_X, train_y)
      val = model.predict(test_X)
      score = score_function(test_y, val)
      scores['Fold'].append(fold)
      scores['Score'].append(score)
      scores['MSE'].append(mean_squared_error(test_y, val))
      scores['Baseline'].append(score_function(test_y, baseline[test_idx]))
      trained_models[fold] = model
      fold += 1
    scores = pd.DataFrame(scores)
    if return_coef:
      coefficient_df = pd.DataFrame(dict((p, return_property(trained_models[p], return_coef)) for p in trained_models)).T
      coefficient_df.columns = features
    else:
      coefficient_df = None
    return scores, trained_models, coefficient_df

  
  def cross_dataset_CV_Nested(self, features: list, target: str, n_splits: int, score_function: Callable, model_class, model_params: dict = {}, return_coef = 'coef_'):
    '''
    Nest CV i.e. train on 80% of 2020, test on 20% of 2021, etc.
    '''
  
    X_1 = self.data[self.data['dataset']=='2020_dataset'][features].values
    y_1 = self.data[self.data['dataset']=='2020_dataset'][target].values
    X_1_baseline = self.data[self.data['dataset']=='2020_dataset']['Titre_IgG_PT'].values
    X_2 = self.data[self.data['dataset']=='2021_dataset'][features].values
    y_2 = self.data[self.data['dataset']=='2021_dataset'][target].values
    X_2_baseline = self.data[self.data['dataset']=='2021_dataset']['Titre_IgG_PT'].values
    outer = 0
    scores = {'Score':[], 'Outer':[], 'Inner':[], 'Train_Year':[], 'Baseline':[]}
    trained_models = {}
    for train_idx, _ in KFold(n_splits = 5, shuffle=True).split(X_1, y_1):
        train_X, train_y = X_1[train_idx], y_1[train_idx]
        inner = 0
        for _, test_idx in KFold(n_splits = 5).split(X_2, y_2):
          test_X, test_y = X_2[test_idx], y_2[test_idx]
          model = model_class(**model_params).fit(train_X, train_y)
          val = model.predict(test_X)
          score = score_function(test_y, val)
          scores['Outer'].append(outer)
          scores['Inner'].append(inner)
          scores['Score'].append(score)
          scores['Train_Year'].append(2020)
          scores['Baseline'].append(score_function(X_2_baseline[test_idx], test_y))
          trained_models[f'{outer}_{inner}_2020'] = model
          inner += 1
        outer += 1
    for train_idx, _ in KFold(n_splits = 5).split(X_2, y_2):
        train_X, train_y = X_2[train_idx], y_2[train_idx]
        inner = 0
        for _, test_idx in KFold(n_splits = 5).split(X_1, y_1):
          test_X, test_y = X_1[test_idx], y_1[test_idx]
          model = model_class().fit(train_X, train_y)
          val = model.predict(test_X)
          score = score_function(test_y, val)
          scores['Outer'].append(outer)
          scores['Inner'].append(inner)
          scores['Score'].append(score)
          scores['Train_Year'].append(2021)
          scores['Baseline'].append(score_function(X_1_baseline[test_idx], test_y))
          trained_models[f'{outer}_{inner}_2021'] = model
          inner += 1
        outer += 1
    if return_coef:
      coefficient_df = pd.DataFrame(dict((p, return_property(trained_models[p], return_coef)) for p in trained_models)).T
      coefficient_df.columns = features
    else:
      coefficient_df = None
    scores = pd.DataFrame(scores)
    return scores, trained_models, coefficient_df
  
  
  def cross_dataset_CV(self, features: list, target: str, n_splits: int, score_function: Callable, model_class, model_params: dict = {}, return_coef = 'coef_'):
    '''
    Train on 2020 and test on 2021 and vice versa.
    '''
    X_1 = self.data[self.data['dataset']=='2020_dataset'][features].values
    baseline_X1 = self.data[self.data['dataset']=='2020_dataset']['Titre_IgG_PT']
    y_1 = self.data[self.data['dataset']=='2020_dataset'][target].values
    X_2 = self.data[self.data['dataset']=='2021_dataset'][features].values
    baseline_X2 = self.data[self.data['dataset']=='2021_dataset']['Titre_IgG_PT']
    y_2 = self.data[self.data['dataset']=='2021_dataset'][target].values
    outer = 0
    scores = {'Score':[], 'Train_Year':[], 'Test_Year':[], 'Baseline':[]}
    trained_models = {}
    train_X, train_y = X_1, y_1
    test_X, test_y = X_2, y_2
    model = model_class(**model_params).fit(train_X, train_y)
    trained_models['Train2020_Test2021'] = model
    val = model.predict(test_X)
    score = score_function(val, test_y)
    scores['Score'].append(score)
    scores['Train_Year'].append(2020)
    scores['Test_Year'].append(2021)
    scores['Baseline'].append(score_function(baseline_X2, test_y))
    model = model_class(**model_params).fit(test_X, test_y)
    trained_models['Train2021_Test2020'] = model
    val = model.predict(train_X)
    score = score_function(val, train_y)
    scores['Score'].append(score)
    scores['Train_Year'].append(2021)
    scores['Test_Year'].append(2020)
    scores['Baseline'].append(score_function(baseline_X1, train_y))
    scores = pd.DataFrame(scores)
    if return_coef:
      coefficient_df = pd.DataFrame(dict((p, return_property(trained_models[p], return_coef)) for p in trained_models)).T
      coefficient_df.columns = features
    else:
      coefficient_df = None
    return scores, trained_models, coefficient_df




