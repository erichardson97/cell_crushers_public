from utils import *
from data_utils import *
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import sys
from copy import deepcopy
from glob import glob


  
model_dictionary = {'RandomForest':RandomForestRegressor, 'GradientBoost':GradientBoostingRegressor,
                    'Linear':LinearRegression, 'Ridge':Ridge, 'Lasso':Lasso, 'ElasticNet':ElasticNet, 'XGBoost':XGBRegressor}
for r in ['RandomForest', 'Linear', 'GradientBoost', 'Ridge', 'Lasso', 'ElasticNet','XGBoost']:
  model_dictionary[f'{r}_Residual'] = residuals_model(model_dictionary[r])
model_dictionary['PCA'] = PCA

def load_data(path = '/content/drive/MyDrive/CMIPB_Files/IntegratedData.tsv', target = 'Target', transformation: dict = {'Target':np.log2, 'Titre_IgG_PT':np.log2}):
  data = pd.read_csv(path, sep = '\t', index_col = 0)
  if ('Target' in data) & (target != 'Target'):
    del data['Target']
  data = data.rename(columns = {target: 'Target'})
  data = data[(data['Target'].notna())]
  for key in transformation:
    data[key] = data[key].map(transformation[key])
  ds = Dataset(data)
  return ds

def populate_cv_args(cv_name: str, model: str, baseline: str, cv_params: dict, model_params: dict, outpath: str):
  cv_args = {}
  if 'precomputed_split' in cv_params:
    cv_args['precomputed_split'] = cv_params[cv_name]['precomputed_split']
  else:
    if cv_params[cv_name]['cv_type'] != 'CrossDataset':
      cv_args['precomputed_split'] = False
  cv_args['n_splits'] = cv_params[cv_name]['n_folds'] if 'n_folds' in cv_params[cv_name] else 5
  cv_args['score_function'] = corr_coeff_report
  cv_args['model_classes'] =  {model:model_dictionary[model]}
  cv_args['model_params'] = {model:model_params['fit']}
  cv_args['return_coef'] = {model:model_params['return']}
  cv_args['baseline'] = baseline
  cv_args['target'] = 'Target'
  cv_args['plot'] = False
  cv_args['plot_dir'] = outpath
  return cv_args
  
def run_model(model_name: str, data_dir: str, data_params: dict, model_params: dict, cv_params: dict, 
          transformation_params: dict, feature_params: dict, outpath: str, target: str):
  if os.path.exists(outpath) is False:
    os.mkdir(outpath)
  model = model_params[model_name]['model_type']
  baseline = data_params['Baseline']
  filename = data_params['Filename']
  transformation = {p:eval(data_params['Transformation'][p]) for p in data_params['Transformation']}
  ds = load_data(os.path.join(data_dir, filename), transformation = transformation, target = target)
  for cv_name in cv_params:
    cv_args = populate_cv_args(cv_name, model, baseline, cv_params, model_params[model_name], outpath)
    cv_type = cv_params[cv_name]['cv_type']
    if 'precomputed_split' in cv_params[cv_name]:
      precomputed_split = cv_params[cv_name]['precomputed_split']
    else:
      precomputed_split = False
    for feature_name in feature_params:
      feature_list = feature_params[feature_name]
      ds = load_data(os.path.join(data_dir, filename), transformation = transformation, target = target)
      ds.filter(feature_list, data_params['nan_policy'])
      cv_args['features'] = feature_list
      for transformation_name in transformation_params:
        transformation_args = deepcopy(transformation_params[transformation_name]['transformation_args'])
        transformation_func = False if transformation_params[transformation_name]['transformation_func'] == False else eval(transformation_params[transformation_name]['transformation_func'])
        if 'reducer' in transformation_args:
          if type(transformation_args['reducer']) == str:
            reducer = model_dictionary[transformation_args['reducer']]
        features_to_change = transformation_args['features_to_change'] if 'features_to_change' in transformation_args else []
        cv_args['transformation_args'] = transformation_args
        cv_args['transformation_args']['features'] = feature_list
        if len(features_to_change) == 0:
          cv_args['transformation_args']['features_to_change'] = feature_list
        if type(features_to_change) != list:
          cv_args['transformation_args']['features_to_change'] = eval(features_to_change)
        assert len(transformation_args['features_to_change']) <= len(feature_list)
        if 'reducer' in transformation_args:
          cv_args['transformation_args']['reducer'] = reducer
        cv_args['transformation'] = transformation_func
        run_name = '_'.join([model_name, feature_name, transformation_name, cv_type])
        output_dir = os.path.join(outpath, run_name)
        if transformation_func != False:
          if transformation_args['n_components'] >= int(ds.data.shape[0]*0.8):
            sys.stderr.write(f'N components > n subjects in training data. (Skipping {cv_name}...)\n')
            continue
          elif transformation_args['n_components'] >= len(cv_args['transformation_args']['features_to_change']):
            sys.stderr.write(f'N components > n input features. (Skipping {cv_name}...)\n')
            continue
        if os.path.exists(output_dir) is False:
          os.mkdir(output_dir)
        for n in range(cv_params[cv_name]['n_repeats']):
          if precomputed_split != False:
            cv_args['precomputed_split'] = os.path.join(precomputed_split,f'Repeat{n}_CV_Idx.p')
          cv = CV(ds.data)
          scores, models, coefficients = cv.RunCV(cv_type = cv_type, cv_args = cv_args)
          if type(scores) == type(None):
            continue
          scores['Repeat'] = n
          coefficients['Repeat'] = n
          for file in glob('*warnings.txt'):
            os.rename(file, os.path.join(output_dir, f'Repeat{n}_{file}'))
          scores.to_csv(os.path.join(output_dir, f'Repeat{n}.csv'))
          coefficients.to_csv(os.path.join(output_dir,f'FeatImportance{n}.csv'))
          # with open(os.path.join(output_dir, "TrainedModels.p"), 'wb') as k:
          #   pickle.dump(models, k)
          
      
