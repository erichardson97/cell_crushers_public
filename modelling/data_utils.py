import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Callable
from collections import defaultdict

class Dataset():
  """
  Dataset class which accepts pd dataframe.
  Methods:
  1.  Filter to subjects for which all have a value.
  2.  Z-scale.
  3.  Ambiguous transform (it looks like I was doing something there but I can't remember why)...
  4.  Record transform: keep list of transforms applied to each features.
  5.  Record datatypes: keep dict of datatypes of features at each transform.
  
  """
  def __init__(self, data: pd.DataFrame, target: str = 'Target'):
    self.data = data
    self.demographic_features = ['Titre_IgG_PT', 'age', 'biological_sex', 'infancy_vac', 'dataset', 'subject_id']
    self.transform_record = defaultdict(list)
    self.transform_id = 0
    self.dtypes = dict()

  def record_transform(self, feature_list, transform_type):
    for feat in feature_list:
      self.transform_record[feat].append(f'{transform_type} {self.transform_id}')
    self.dtypes[self.transform_id] = dict(self.data.dtypes)
    self.transform_id += 1

  def filter(self, feature_list: list = []):
    '''
    Pass a feature list and will filter to the subset of rows which have a value
    for this feature
    '''
    if feature_list != []:
      self.feature_list = list(set(feature_list).union(set(self.demographic_features))) + ['Target']
    else:
      self.feature_list = self.data.columns
    self.data = self.data.dropna(subset = self.feature_list)
    self.record_transform(self.feature_list, 'Remove NaN')
    
  def make_float(self, feature_list: list = []):
    '''
    Pass a feature list and will filter to the subset of rows which have a value
    for this feature
    '''
    for feat in feature_list:
      self.data[feat] = self.data[feat].map(float)
    self.record_transform(feature_list, 'Assert float')
      
  def normalize(self, features_to_normalize: list):
    '''
    Pass a list of continuous features for use by StandardScaler
    '''
    self.data = self.data.copy()
    features_to_normalize.append('Target')
    features_to_normalize = list(set(features_to_normalize))
    self.data[features_to_normalize] = StandardScaler().fit_transform(self.data[features_to_normalize])
    self.record_transform(features_to_normalize, 'Z-scale')

  def transform_predictors(self, features: list, function: Callable, function_args: dict = {}):
    self.data = pd.DataFrame(function(**function_args).fit_transform(self.data[features]))
    self.record_transform(features_to_normalize, 'Custom transform')
    
