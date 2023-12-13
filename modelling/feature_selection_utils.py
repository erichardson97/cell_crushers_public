from scipy.stats import spearmanr, levene, mannwhitneyu
from scipy.stats.mstats import gmean
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from typing import Union, Callable
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

class StatisticsTable():
  def __init__(self, target = 'Target'):
    self.target = target
    
  def run(self, train_data, test_data, features, outf):
    self.load_train_data(train_data)
    self.load_test_data(test_data)
    self.combine_datasets()
    self.load_features_for_analysis(features)
    self.means_and_variances()
    self.statistical_tests()
    self.statistics_table.to_csv(outf, sep = '\t')
  
  def load_train_data(self, data):
    self.train_data = data
    
  def load_test_data(self, data):
    self.test_data = data
    
  def combine_datasets(self):
    self.data = pd.concat([self.train_data, self.test_data])
    self.data['Type'] = self.data['dataset'].map(lambda x:'Test' if x == '2022_dataset' else 'Train')
    
  def load_features_for_analysis(self, features: list):
    self.features = features
    for feature in self.features:
      self.data[feature] = self.data[feature].map(float)
      
  def means_and_variances(self):
    self.means = dict(self.data.groupby('Type').apply(lambda x:dict((p, np.mean(x[p])) for p in self.features)))
    self.variances = dict(self.data.groupby('Type').apply(lambda x:dict((p, np.var(x[p])) for p in self.features)))
    
  def statistical_tests(self):
    statistics = {}
    for feature in features:
      stat, p = mannwhitneyu(*self.data.groupby('Type').apply(lambda x:x[feature].values))
      statistics[feature] = {'MannWhitneyU_stat':stat,'MannWhitneyU_p':p}
      stat, p = levene(*self.data.groupby('Type').apply(lambda x:x[feature].values))
      statistics[feature]['Levene_stat'] = stat
      statistics[feature]['Levene_p'] = p
    statistics = pd.DataFrame(statistics).T
    reject, pvals, _, _ = multipletests(statistics['Levene_p'], method='fdr_bh')
    statistics['Levene_p_corrected'] = pvals
    reject, pvals, _, _ = multipletests(statistics['MannWhitneyU_p'], method='fdr_bh')
    statistics['MannWhitneyU_p_corrected'] = pvals
    statistics['Variance_Test'] = statistics.index.map(lambda x:self.variances['Test'][x])
    statistics['Variance_Train'] = statistics.index.map(lambda x:self.variances['Train'][x])
    statistics['Mean_Test'] = statistics.index.map(lambda x:self.means['Test'][x])
    statistics['Mean_Train'] = statistics.index.map(lambda x:self.means['Train'][x])
    self.statistics_table = statistics



def calc_autocorrel_matrix(dataset: pd.DataFrame, feature_list: list) -> pd.DataFrame:
  """
  Just uses pd.DataFrame.corr() on subset of dataset.
  """
  return dataset[feature_list].corr()

def greedy_algorithm(auto: pd.DataFrame, threshold: float = 0.7) -> dict:
  """
  -Accepts a correlation matrix.
  -Performs greedy clustering:
    1. Identify neighbours at threshold.
    2. Select feature with most neighbors. This is cluster 1. Its friends are its neighbors.
    3. Iterate through list. Find the next feature with the most neighbours. He is the next cluster,
       and his friends are his neighbours that haven't already been clustered.
    4. Et cetera.
  NB this is only reliably greedy in the first clustering step.
  -Returns a dictionary where cluster ids are the central cluster and members are his friends.
  """
  auto_dict = auto.to_dict()
  neighbours = {}
  for x in auto_dict:
    neighbours[x] = set()
    for p in auto_dict[x]:
      if auto_dict[x][p] >= threshold:
        neighbours[x].add(p)
  clusters = {}
  seen = set()
  cluster_count = 0
  neighbors_sorted = sorted(dict((p, len(neighbours[p])) for p in neighbours).items(), key = lambda x:x[1], reverse=True)
  for k,p in neighbors_sorted:
    if k not in seen:
      clusters[k] = set(neighbours[k]).difference(seen)
      seen.add(k)
      seen.update(set(neighbours[k]))
      # cluster_count += 1
  return clusters

def cluster_features(dataset_normalized: pd.DataFrame, dataset_filtered: pd.DataFrame, feature_set: list, threshold: float = 0.8) -> pd.DataFrame:
  """
  Iteratively clusters features, recalculates as geomean of consituent, renormalize, etc, until
  no feature has > threshold correlation with any other feature.
  Accepts:
    1. Input normalized dataset.
    2. Input filtered dataset.
    3. Feature list to cluster.
    4. Max identity allowed.
  Returns:
    New normalized features.
  """
  corr_mat = calc_autocorrel_matrix(dataset_normalized, feature_set)
  neighbors = greedy_algorithm(corr_mat, threshold = threshold)
  neighbors = pd.DataFrame({'Neighbors':neighbors})
  neighbors['n'] = neighbors['Neighbors'].map(len)
  data_frame = dataset_filtered
  n = 0
  while neighbors[neighbors['n']>1].shape[0] > 0:
    print(f'Iteration {n}')
    total = dict()
    feature_groups = neighbors['Neighbors']
    for feature_group in feature_groups:
      if len(feature_group)>1:
        total[','.join(feature_group)] = data_frame[list(feature_group)].apply(gmean,axis=1).to_dict()
      else:
        total[','.join(feature_group)] = data_frame[list(feature_group)[0]].to_dict()
    total = pd.DataFrame(total)
    data_frame = total.copy()
    total[total.columns] = StandardScaler().fit_transform(total.values)
    corr_mat = calc_autocorrel_matrix(total, total.columns)
    neighbors = greedy_algorithm(corr_mat, threshold = threshold)
    neighbors = pd.DataFrame({'Neighbors':neighbors})
    neighbors['n'] = neighbors['Neighbors'].map(len)
    n += 1
  return total

def calc_ratio(val1, val2):
  return max([val1, val2])/min([val1, val2]) if min([val1, val2]) != 0 else 1e-6

def consistency_between_years(dataset: pd.DataFrame, features: list, correlation_function: Callable = spearmanr, target: str = 'Target') -> pd.DataFrame:
  year2020 = dataset[dataset['dataset']=='2020_dataset'][features + [target]].values
  year2021 = dataset[dataset['dataset']=='2021_dataset'][features + [target]].values
  spearman_values = {}
  for index, feature in enumerate(features):
    values_2020 = spearmanr(year2020[:, index], year2020[:, -1], nan_policy = 'omit')
    values_2021 = spearmanr(year2021[:, index], year2021[:, -1], nan_policy = 'omit')
    spearman_values[feature] = {'2020_rho': values_2020[0], '2020_p': values_2020[1],
                                '2021_rho': values_2021[0], '2021_p': values_2021[1]}
  spearman_values = pd.DataFrame(spearman_values).T
  spearman_values['Ratio'] = spearman_values.apply(lambda x:calc_ratio(x['2020_rho'], x['2021_rho']), axis = 1)
  std = spearman_values[spearman_values["Ratio"]!=1e-6]['Ratio'].std()
  spearman_values['Z_score_absolute'] = spearman_values['Ratio'].map(lambda x:(abs(x) - 1)/std)
  return spearman_values
  
  







