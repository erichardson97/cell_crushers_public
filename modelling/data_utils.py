import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Callable

class Dataset():
  """
  Dataset class which accepts pd dataframe.
  Filter to subjects for which all have a value.
  Z-scale.
  """
  def __init__(self, data: pd.DataFrame, target: str = 'Target'):
    self.data = data
    self.demographic_features = ['Titre_IgG_PT', 'age', 'biological_sex', 'infancy_vac', 'dataset']
    
  def filter(self, feature_list: list = []):
    '''
    Pass a feature list and will filter to the subset of rows which have a value
    for this feature
    '''
    if feature_list != []:
      self.feature_list = list(set(feature_list).union(set(self.demographic_features))) + ['Target']
    else:
      self.feature_list = self.data.columns
    self.data_filtered = self.data.dropna(subset = self.feature_list)
    
  def make_float(self, feature_list: list = []):
    '''
    Pass a feature list and will filter to the subset of rows which have a value
    for this feature
    '''
    for feat in feature_list:
      self.data_filtered[feat] = self.data_filtered[feat].map(float)
      
  def normalize(self, features_to_normalize: list):
    '''
    Pass a list of continuous features for use by StandardScaler
    '''
    self.data_normalized = self.data_filtered.copy()
    self.data_normalized[features_to_normalize] = StandardScaler().fit_transform(self.data_filtered[features_to_normalize])

  def transform_predictors(self, features: list, function: Callable, function_args: dict = {}):
    self.transformed = pd.DataFrame(function(**function_args).fit_transform(self.data_normalized[features]))
