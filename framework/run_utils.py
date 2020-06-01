import os
import json
import datetime
import numpy as np
import glob

import framework.configbase


def gen_common_pathcfg(path_cfg_file, is_train=False):
  path_cfg = framework.configbase.PathCfg()
  path_cfg.load(json.load(open(path_cfg_file)))

  output_dir = path_cfg.output_dir

  path_cfg.log_dir = os.path.join(output_dir, 'log')
  path_cfg.model_dir = os.path.join(output_dir, 'model')
  path_cfg.pred_dir = os.path.join(output_dir, 'pred')
  if not os.path.exists(path_cfg.log_dir):
    os.makedirs(path_cfg.log_dir)
  if not os.path.exists(path_cfg.model_dir):
    os.makedirs(path_cfg.model_dir)
  if not os.path.exists(path_cfg.pred_dir):
    os.makedirs(path_cfg.pred_dir)

  if is_train:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path_cfg.log_file = os.path.join(path_cfg.log_dir, 'log-' + timestamp)
  else:
    path_cfg.log_file = None

  return path_cfg


def find_best_val_models(log_dir, model_dir):
  step_jsons = glob.glob(os.path.join(log_dir, 'val.step.*.json'))
  epoch_jsons = glob.glob(os.path.join(log_dir, 'val.epoch.*.json'))

  val_names, val_scores = [], []
  for i, json_file in enumerate(step_jsons + epoch_jsons):
    json_name = os.path.basename(json_file)
    scores = json.load(open(json_file))
    val_names.append(json_name)
    val_scores.append(scores)
    
  measure_names = list(val_scores[0].keys())
  model_files = {}
  for measure_name in measure_names:
    # for metrics: the lower the better
    if 'loss' in measure_name or 'medr' in measure_name or 'meanr' in measure_name:
      idx = np.argmin([scores[measure_name] for scores in val_scores])
    # for metrics: the higher the better
    else:
      idx = np.argmax([scores[measure_name] for scores in val_scores])
    json_name = val_names[idx]
    model_file = os.path.join(model_dir, 
      'epoch.%s.th'%(json_name.split('.')[2]) if 'epoch' in json_name \
      else 'step.%s.th'%(json_name.split('.')[2]))
    model_files.setdefault(model_file, [])
    model_files[model_file].append(measure_name)

  name2file = {'-'.join(measure_name): model_file for model_file, measure_name in model_files.items()}

  return name2file
