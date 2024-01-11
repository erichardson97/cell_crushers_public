from feature_selection_utils import *
from processing_utils import *
from utils import *
import argparse
import yaml
import os
import sys

def check_model_file(model_file_path):
  file = yaml.safe_load(open(model_file_path, 'r'))
  error_message = 'Misspecified model yaml file.'
  if 'data_path' not in file:
    sys.stderr.write(error_message)
    sys.exit(2)
  if os.path.exists(file['data_path']) is False:
    sys.stderr.write('Data directory does not exist.')
    sys.exit(2)
  elif os.path.exists(file['data_path']) is False:
    sys.stderr.write('Data directory does not exist.')
    sys.exit(2)
  if os.path.exists(file['data_params']) is False:
    sys.stderr.write('Data directory does not exist.')
    sys.exit(2)
  elif os.path.exists(file['data_params']) is False:
    sys.stderr.write('Data directory does not exist.')
    sys.exit(2)
  if 'cv_spec' not in file:
    sys.stderr.write(error_message)
    sys.exit(2)
  if os.path.exists(file['cv_spec']) is False:
    sys.stderr.write('CV spec file does not exist.')
    sys.exit(2)
  if 'model_spec' not in file:
    sys.stderr.write(error_message)
    sys.exit(2)
  elif os.path.exists(file['model_spec']) is False:
    sys.stderr.write('Missing model spec file.')
    sys.exit(2)
  if 'feature_spec' not in file:
    sys.stderr.write(error_message)
    sys.exit(2)
  elif os.path.exists(file['feature_spec']) is False:
    sys.stderr.write('Missing feature spec file.')
    sys.exit(2) 
  if 'transformation' not in file:
    sys.stderr.write(error_message)
    sys.exit(2)
  elif os.path.exists(file['transformation']) is False:
    sys.stderr.write('Missing transformation spec file.')
    sys.exit(2) 
  return file
  
parser = argparse.ArgumentParser()
parser.add_arguments('--model_args', dest = 'model_args', description = 'YAML file containign model arguments.')
args = parser.parse_args()
if os.path.exists(args.model_args) is False:
  sys.stderr.write('Please supply the model file.')
  parser.print_help()
  sys.exit(2)

model_file = args.model_args
file = check_model_file(model_file)
model_spec = yaml.safe_load(open(file['model_spec'], 'r'))
data_params = yaml.safe_load(open(file['data_params'], 'r'))
feature_spec = yaml.safe_load(open(file['feature_spec'], 'r'))
cv_spec = yaml.safe_load(open(file['cv_spec'], 'r'))
transformation_params = yaml.safe_load(open(file['transformation'], 'r'))

for model in model_spec:
    run_model(model, data_path = data_path, data_params = data_params,
              model_params = model_spec[model], features = feature_spec, cv_spec = cv_spec, 
              transformation = transformation_params, outpath = file['outpath'])
  

