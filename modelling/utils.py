import pandas as pd
from typing import Union, Callable, Protocol
import numpy as np
from scipy.stats import spearmanr, linregress
from sklearn.model_selection import train_test_split, KFold, ParameterGrid, StratifiedKFold
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
from itertools import combinations
from statsmodels.formula import api as smf
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import os
import pickle

class ScikitClass(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...

def plot_total(total):
  order = total.groupby('Model').apply(lambda x:x["Score"].mean()).sort_values(ascending=False).index
  df = total.melt(id_vars=['Model'],value_vars=['Baseline',"Score"])
  ax = sns.barplot(data = df, x = 'Model', y = 'value', hue = 'variable', order=order)
  for tick in ax.get_xticklabels():
    tick.set_rotation(90)


class ReGainBootleg():
  def __init__(self, n_components= 100):
    self.n_genes = n_components
      
  def fit(self, X, y, fill_missing = True):
    if len(y.shape) == 1:
      y = y.reshape(-1, 1)
    self.data = pd.DataFrame(np.hstack([X, y]))
    self.data.columns = [f'Feat{p}' for p in range(X.shape[1])]+['Target']
    if fill_missing:
      self.fill_missing(self.data, self.data.columns[:-1])
    self.ReGainMatrix(self.data,  n_features = self.n_genes)
    return self

  def fit_transform(self, X, y, fill_missing = True):
    self.fit(X, y, fill_missing = fill_missing)
    return self.transform(X)

  def fill_missing(self, data, features):
    for feat in features:
      median_val = data[feat].median()
      data[feat] = data[feat].fillna(median_val)
    return data

  def ReGainMatrix(self, data, n_features = 100):
    assert data.shape[1]-1 < data.shape[0]
    features = self.data.columns[:-1]
    total_features = set(features)
    pairs = combinations(features, 2)
    total_mat = dict()
    for pair in pairs:
      a1 = pair[0]
      a2 = pair[1]
      other = total_features.difference(set([a1, a2]))
      model = f'Target ~ {a1} + {a2} + {a1}*{a2} +' + ' + '.join(other)
      fit = smf.ols(data = data, formula = model).fit()
      params = fit.pvalues.loc[f'{a1}:{a2}']
      total_mat[f'{a1}:{a2}'] = params
    self.features = [p[0] for p in sorted(total_mat.items(),key=lambda x:x[1])[:n_features]]

  def transform(self, data):
    data = pd.DataFrame(data)
    data.columns = [f'Feat{p}' for p in range(data.shape[1])]
    for pair in self.features:
      data[pair] = data.apply(lambda x:x[pair.split(':')[0]]/x[pair.split(':')[1]] if x[pair.split(':')[1]] != 0 else x[pair.split(':')[0]], axis =1 )
    return data[self.features]


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

def calc_residuals_for_prediction(baseline, y):
  slope, intercept, r, p, se = linregress(baseline, y)
  residuals = [y_val - (p*slope + intercept) for p, y_val in zip(baseline, y)]
  return slope, intercept, np.array(residuals)


def make_plots(plot_dir, model_name, fold_idx, X, y, train_idx, test_idx, test_preds, test_y, baseline_y, score, baseline_score):
  fname = os.path.join(plot_dir, f'{model_name}_{fold_idx}Test.png')
  df = pd.DataFrame([test_preds, test_y, baseline_y]).T
  fig, axs = plt.subplots(1,2, figsize=(10,5))
  ax = axs[0]
  sns.scatterplot(data = df, x = 0, y = 1, alpha = 0.2, ax = ax)
  ax.set_xlabel('Prediction')
  ax.set_ylabel('Target')
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], linestyle='--')
  ax = axs[1]
  sns.scatterplot(data = df, x = 2, y = 1, alpha = 0.2, ax = ax)
  ax.set_xlabel('Baseline')
  ax.set_ylabel('Target')
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], linestyle='--')
  axs[0].set_title(r'$\rho$'+f' {score:.2f}', fontweight = 'bold')
  axs[1].set_title(r'$\rho$'+f' {baseline_score:.2f}', fontweight = 'bold')
  plt.savefig(fname, dpi = 300)
  plt.close()

def reduce_dimensions(X: np.array, y: np.array, features: np.array, features_to_change: np.array, n_components: int, reducer, trained: bool = False, supervised: bool = True, interpretable: bool = False):
    feature_idxs = np.where(np.isin(features, features_to_change))[0]
    features_to_keep = np.where(~np.isin(features, features_to_change))[0]
    if not trained:
      reduction = reducer(n_components = n_components)
      if supervised:
        X_trans = reduction.fit(X[:, feature_idxs], y).transform(X[:, feature_idxs])
      else:
        X_trans = reduction.fit(X[:, feature_idxs]).transform(X[:, feature_idxs])
    else:
      reduction = reducer
      X_trans = reduction.transform(X[:, feature_idxs])
    X_new = np.hstack([X_trans.values, X[:, features_to_keep]])
    if interpretable:
        old_feature_names = dict((f'Feat{k}', p) for k, p in enumerate(features))
        new_feature_order = [old_feature_names[p.split(':')[0]]+':'+old_feature_names[p.split(':')[1]] for p in X_trans.columns]
        new_feature_order = new_feature_order + list(features[features_to_keep])
    new_feature_order = [f'NewFeat{p}' for p in range(n_components)] + list(features[features_to_keep])
    return X_new, reduction, new_feature_order

def residuals_model(base_class: ScikitClass):
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


  def regular_ol_cv(self, features: list, target: str, n_splits: int, plot_dir: str, score_function: Callable, model_classes: dict = {}, model_params: dict = {}, return_coef: bool | dict = False, normalize = True,
                     plot: bool = True, transformation: bool | Callable = False, transformation_args: dict = {}, precomputed_split: bool = False):
    '''
    Regular CV with no stratification by year.
    '''
    X = self.data[features].values
    baseline = self.data['Titre_IgG_PT'].values
    y = self.data[target].values

    scores = {'Fold':[], 'Score':[], 'MSE':[], 'Baseline':[], 'Model':[]}
    trained_models = defaultdict(dict)
    feature_order = defaultdict(dict)
    if precomputed_split == False:
        fold = 0
        split_indexes = {}
        for train_idx, test_idx in KFold(n_splits = n_splits, shuffle = True).split(X, y):
            split_indexes[fold] = {'Train':train_idx, 'Test':test_idx}
            fold += 1  
        with open(os.path.join(plot_dir, 'CV_Idx.p'), 'wb') as k:
            pickle.dump(split_indexes, k)
    else:
        split_indexes = pd.read_pickle(precomputed_split)
    for fold in split_indexes:
        train_idx = split_indexes[fold]['Train']
        test_idx = split_indexes[fold]['Test']
        train_X, train_y = X[train_idx], y[train_idx]
        test_X, test_y = X[test_idx], y[test_idx]
        if normalize:
            train_X = StandardScaler().fit_transform(train_X)
            train_y = StandardScaler().fit_transform(train_y.reshape(-1,1)).ravel()
            test_X = StandardScaler().fit_transform(test_X)
            test_y = StandardScaler().fit_transform(test_y.reshape(-1,1)).ravel()
        if transformation:
            assert train_X.shape[1] == test_X.shape[1]
            train_X, transformer, new_feature_order =  transformation(train_X, train_y, **transformation_args)
            test_X, _, _ = transformation(test_X, test_y, reducer = transformer, n_components = transformation_args['n_components'],
                               features = transformation_args['features'], features_to_change = transformation_args['features_to_change'], trained = True)
            feature_order[fold] = new_feature_order
        for model_name in model_classes:
            model_class = model_classes[model_name]
            model = model_class(**model_params[model_name])
            model.fit(train_X, train_y)
            assert test_X.shape[1] == train_X.shape[1]
            val = model.predict(test_X)
            score = score_function(test_y, val)
            baseline_score = score_function(test_y, baseline[test_idx])
            if plot:
              make_plots(plot_dir, model_name, fold, X, y, train_idx, test_idx, val, test_y, baseline[test_idx], score=score, baseline_score=baseline_score)
            scores['Fold'].append(fold)
            scores['Score'].append(score)
            scores['MSE'].append(mean_squared_error(test_y, val))
            scores['Baseline'].append(baseline_score)
            scores['Model'].append(model_name)
            trained_models[fold][model_name] = model
    scores = pd.DataFrame(scores)
    if return_coef:
        if not transformation:
            coefficient_df = pd.concat([pd.DataFrame(dict((p, dict((features[m], y) for m,y in enumerate(return_property(trained_models[x][p], return_coef[p])))) for p in trained_models[x] if p in return_coef)).T.assign(Fold=x) for x in trained_models])
        else:
            coefficient_df = pd.concat([pd.DataFrame(dict((p, dict((feature_order[x][m], y) for m,y in enumerate(return_property(trained_models[x][p], return_coef[p])))) for p in trained_models[x] if p in return_coef)).T.assign(Fold=x) for x in trained_models])
    else:
        coefficient_df = None
    return scores, trained_models, coefficient_df

  
  def cross_dataset_CV_Nested(self, features: list, target: str, n_splits: int, score_function: Callable, model_classes, model_params: dict = {}, return_coef = 'coef_'):
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
  
  
  def cross_dataset_CV(self, features: list, target: str, n_splits: int, plot_dir: str, score_function: Callable, model_classes: dict, model_params: dict = {}, return_coef = 'coef_',
                      normalize: bool = True, plot: bool = True, transformation: bool | Callable = False, transformation_args: dict = {}):
    '''
    Train on 2020 and test on 2021 and vice versa.
    '''
    X_1 = self.data[self.data['dataset']=='2020_dataset'][features].values
    baseline_X1 = self.data[self.data['dataset']=='2020_dataset']['Titre_IgG_PT']
    y_1 = self.data[self.data['dataset']=='2020_dataset'][target].values
    X_2 = self.data[self.data['dataset']=='2021_dataset'][features].values
    baseline_X2 = self.data[self.data['dataset']=='2021_dataset']['Titre_IgG_PT']
    y_2 = self.data[self.data['dataset']=='2021_dataset'][target].values
    scores = {'Score':[], 'Train_Year':[], 'Test_Year':[], 'Baseline':[], 'Model':[]}
    trained_models = defaultdict(dict)
    feature_order = defaultdict(dict)
    train_X, train_y = X_1, y_1
    test_X, test_y = X_2, y_2
    if normalize:
        train_X = StandardScaler().fit_transform(train_X)
        train_y = StandardScaler().fit_transform(train_y.reshape(-1,1)).ravel()
        test_X = StandardScaler().fit_transform(test_X)
        test_y = StandardScaler().fit_transform(test_y.reshape(-1, 1)).ravel()
    if transformation:
        assert train_X.shape[1] == test_X.shape[1]
        train_X, transformer, new_feature_order =  transformation(train_X, train_y, **transformation_args)
        test_X, _, _ = transformation(test_X, test_y, reducer = transformer, n_components = transformation_args['n_components'],
                               features = transformation_args['features'], features_to_change = transformation_args['features_to_change'], trained = True)
        feature_order['Train2020_Test2021'] = new_feature_order
    for model_name in model_classes:
        model_class = model_classes[model_name]
        model = model_class(**model_params[model_name])
        model.fit(train_X, train_y)
        trained_models['Train2020_Test2021'][model_name] = model
        val = model.predict(test_X)
        score = score_function(val, test_y)
        baseline_score = score_function(baseline_X2, test_y)
        scores['Score'].append(score)
        scores['Train_Year'].append(2020)
        scores['Test_Year'].append(2021)
        scores['Baseline'].append(baseline_score)
        scores['Model'].append(model_name)
    train_X, train_y = X_2, y_2
    test_X, test_y = X_1, y_1
    if transformation:
        assert train_X.shape[1] == test_X.shape[1]
        train_X, transformer, new_feature_order =  transformation(train_X, train_y, **transformation_args)
        test_X, _, _ = transformation(test_X, test_y, reducer = transformer, n_components = transformation_args['n_components'],
                               features = transformation_args['features'], features_to_change = transformation_args['features_to_change'], trained = True)
        feature_order['Train2020_Test2021'] = new_feature_order
    for model_name in model_classes:
        model_class = model_classes[model_name]
        model = model_class(**model_params[model_name])
        model.fit(train_X, train_y)
        trained_models['Train2021_Test2020'][model_name] = model
        val = model.predict(test_X)
        score = score_function(val, test_y)
        baseline_score = score_function(baseline_X1, test_y)
        scores['Score'].append(score)
        scores['Train_Year'].append(2021)
        scores['Test_Year'].append(2020)
        scores['Baseline'].append(baseline_score)
        scores['Model'].append(model_name)                      
    scores = pd.DataFrame(scores)
    if return_coef:
        if not transformation:
            coefficient_df = pd.concat([pd.DataFrame(dict((p, dict((features[m], y) for m,y in enumerate(return_property(trained_models[x][p], return_coef[p])))) for p in trained_models[x] if p in return_coef)).T.assign(Fold=x) for x in trained_models])
        else:
            coefficient_df = pd.concat([pd.DataFrame(dict((p, dict((feature_order[x][m], y) for m,y in enumerate(return_property(trained_models[x][p], return_coef[p])))) for p in trained_models[x] if p in return_coef)).T.assign(Fold=x) for x in trained_models])
    else:
        coefficient_df = None
    return scores, trained_models, coefficient_df


class CV_GMM():
  
  def __init__(self, data: pd.DataFrame):
    self.data = data

  def RunCV(self, cv_type: str, cv_args: dict):
    if cv_type == 'RegularCV':
      return self.regular_ol_cv(**cv_args)
    elif cv_type == 'CrossDataset':
      return self.cross_dataset_CV(**cv_args)
    

  def regular_ol_cv(self, features: list, target: str, n_splits: int, plot_dir: str, score_function: Callable, model_classes: dict = {}, model_params: dict = {}, return_coef: bool | dict = False, normalize = True,
                     plot: bool = True, transformation: bool | Callable = False, transformation_args: dict = {}, precomputed_split: bool = False):
    '''
    Regular CV with no stratification by year.
    '''
    self.data['Cluster'] = GaussianMixture(n_components=2).fit_predict(self.data['Titre_IgG_PT'].values.reshape(-1,1))
    self.data = self.data.sort_values('Cluster')
    X = self.data[features].values
    baseline = self.data['Titre_IgG_PT'].values
    gmm_clusters = self.data['Cluster'].values                
    y = self.data[target].values

    scores = {'Fold':[], 'Score':[], 'MSE':[], 'Baseline':[], 'Model':[]}
    trained_models = defaultdict(dict)
    feature_order = defaultdict(dict)
    if precomputed_split == False:
        fold = 0
        split_indexes = {}
        for train_idx, test_idx in StratifiedKFold(n_splits = n_splits, shuffle = True).split(X, gmm_clusters):
            split_indexes[fold] = {'Train':train_idx, 'Test':test_idx}
            fold += 1  
        with open(os.path.join(plot_dir, 'CV_Idx.p'), 'wb') as k:
            pickle.dump(split_indexes, k)
    else:
        split_indexes = pd.read_pickle(precomputed_split)
    for fold in split_indexes:
        train_idx = split_indexes[fold]['Train']
        test_idx = split_indexes[fold]['Test']
        train_X, train_y = X[train_idx], y[train_idx]
        test_X, test_y = X[test_idx], y[test_idx]
        train_label, test_label = gmm_clusters[train_idx], gmm_clusters[test_idx]
        if normalize:
            train_X = StandardScaler().fit_transform(train_X)
            train_y = StandardScaler().fit_transform(train_y.reshape(-1,1)).ravel()
            test_X = StandardScaler().fit_transform(test_X)
            test_y = StandardScaler().fit_transform(test_y.reshape(-1,1)).ravel()
        if transformation:
            assert train_X.shape[1] == test_X.shape[1]
            train_X, transformer, new_feature_order =  transformation(train_X, train_y, **transformation_args)
            test_X, _, _ = transformation(test_X, test_y, reducer = transformer, n_components = transformation_args['n_components'],
                               features = transformation_args['features'], features_to_change = transformation_args['features_to_change'], trained = True)
            feature_order[fold] = new_feature_order
        for model_name in model_classes:
            model_class = model_classes[model_name]
            model1 = model_class(**model_params[model_name])
            model2 = model_class(**model_params[model_name])
            model1.fit(train_X[:, np.where(train_label==0)[0]], train_y[np.where(train_label==0)[0]])
            model2.fit(train_X[:, np.where(train_label==1)[0]], train_y[np.where(train_label==1)[0]])
            assert test_X.shape[1] == train_X.shape[1]
            order = test_X[np.argsort(test_label)]
            test_y = test_Y[np.argsort(test_label)]
            baseline_test = baseline[np.argsort(test_label)]
            test_labels = test_label[np.argsort(test_label)]
            val1 = model1.predict(test_X[:, np.where(test_labels==0)[0]])
            val2 = model2.predict(test_X[:, np.where(test_labels==1)[0]])
            val = np.hstack([val1, val2])
            score = score_function(test_y, val)
            baseline_score = score_function(test_y, baseline_test)
            scores['Fold'].append(fold)
            scores['Score'].append(score)
            scores['MSE'].append(mean_squared_error(test_y, val))
            scores['Baseline'].append(baseline_score)
            scores['Model'].append(model_name)
            trained_models[fold][model_name] = model
    scores = pd.DataFrame(scores)
    if return_coef:
        if not transformation:
            coefficient_df = pd.concat([pd.DataFrame(dict((p, dict((features[m], y) for m,y in enumerate(return_property(trained_models[x][p], return_coef[p])))) for p in trained_models[x] if p in return_coef)).T.assign(Fold=x) for x in trained_models])
        else:
            coefficient_df = pd.concat([pd.DataFrame(dict((p, dict((feature_order[x][m], y) for m,y in enumerate(return_property(trained_models[x][p], return_coef[p])))) for p in trained_models[x] if p in return_coef)).T.assign(Fold=x) for x in trained_models])
    else:
        coefficient_df = None
    return scores, trained_models, coefficient_df

