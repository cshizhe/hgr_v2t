import os
import sys
import argparse
import json
import time

import t2vretrieval.models.globalmatch
import t2vretrieval.readers.mpdata

import torch.utils.data.dataloader as dataloader
import framework.run_utils
import framework.logbase

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', default=False, action='store_true')
  parser.add_argument('--load_video_first', action='store_true', default=False)
  parser.add_argument('--resume_file', default=None)
  parser.add_argument('--eval_set')
  parser.add_argument('--num_workers', type=int, default=0)
  opts = parser.parse_args()

  path_cfg = framework.run_utils.gen_common_pathcfg(
    opts.path_cfg_file, is_train=opts.is_train)
  if path_cfg.log_file is not None:
    _logger = framework.logbase.set_logger(path_cfg.log_file, 'trn_%f'%time.time())
  else:
    _logger = None
 
  model_cfg = t2vretrieval.models.globalmatch.GlobalMatchModelConfig()
  model_cfg.load(opts.model_cfg_file)
  _model = t2vretrieval.models.globalmatch.GlobalMatchModel(model_cfg, _logger=_logger)

  if opts.is_train:
    model_cfg.save(os.path.join(path_cfg.log_dir, 'model.cfg'))
    path_cfg.save(os.path.join(path_cfg.log_dir, 'path.cfg'))
    json.dump(vars(opts), open(os.path.join(path_cfg.log_dir, 'opts.cfg'), 'w'), indent=2)

    trn_dataset = t2vretrieval.readers.mpdata.MPDataset(path_cfg.name_file['trn'], 
      path_cfg.mp_ft_files['trn'], path_cfg.word2int_file, model_cfg.max_words_in_sent, 
      ref_caption_file=path_cfg.ref_caption_file['trn'], is_train=True, _logger=_logger)
    trn_reader = dataloader.DataLoader(trn_dataset, batch_size=model_cfg.trn_batch_size, 
      shuffle=True, collate_fn=t2vretrieval.readers.mpdata.collate_fn)
    val_dataset = t2vretrieval.readers.mpdata.MPDataset(path_cfg.name_file['val'], 
      path_cfg.mp_ft_files['val'], path_cfg.word2int_file, model_cfg.max_words_in_sent, 
      ref_caption_file=path_cfg.ref_caption_file['val'], is_train=False, _logger=_logger)
    val_reader = dataloader.DataLoader(val_dataset, batch_size=model_cfg.tst_batch_size, 
      shuffle=False, collate_fn=t2vretrieval.readers.mpdata.collate_fn)

    _model.train(trn_reader, val_reader, path_cfg.model_dir, path_cfg.log_dir,
      resume_file=opts.resume_file)

  else:
    tst_dataset = t2vretrieval.readers.mpdata.MPDataset(path_cfg.name_file[opts.eval_set], 
      path_cfg.mp_ft_files[opts.eval_set], path_cfg.word2int_file, model_cfg.max_words_in_sent, 
      ref_caption_file=path_cfg.ref_caption_file[opts.eval_set], is_train=False, _logger=_logger)
    tst_reader = dataloader.DataLoader(tst_dataset, batch_size=model_cfg.tst_batch_size, 
      shuffle=False, collate_fn=t2vretrieval.readers.mpdata.collate_fn)

    model_str_scores = []
    is_first_eval = True
    if opts.resume_file is None:
      model_files = framework.run_utils.find_best_val_models(path_cfg.log_dir, path_cfg.model_dir)
    else:
      model_files = {'predefined': opts.resume_file}

    for measure_name, model_file in model_files.items():
      if 'predefined' not in measure_name and 'rsum' not in measure_name:
        continue
      set_pred_dir = os.path.join(path_cfg.pred_dir, opts.eval_set)
      if not os.path.exists(set_pred_dir):
        os.makedirs(set_pred_dir)
      tst_pred_file = os.path.join(set_pred_dir, 
        os.path.splitext(os.path.basename(model_file))[0]+'.npy')
      
      scores = _model.test(tst_reader, tst_pred_file, tst_model_file=model_file)
      if scores is not None:
        if is_first_eval:
          score_names = scores.keys()
          model_str_scores.append(','.join(score_names))
          is_first_eval = False
          print(model_str_scores[-1])
        str_scores = [measure_name, os.path.basename(model_file)]
        for score_name in score_names:
          str_scores.append('%.2f'%(scores[score_name]))
        str_scores = ','.join(str_scores)
        print(str_scores)
        model_str_scores.append(str_scores)

    if len(model_str_scores) > 0:
      score_log_file = os.path.join(path_cfg.pred_dir, opts.eval_set, 'scores.csv')
      with open(score_log_file, 'w') as f:
        for str_scores in model_str_scores:
          print(str_scores, file=f)


if __name__ == '__main__':
  main()

