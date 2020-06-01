import os
import sys
import argparse
import numpy as np
import json

import t2vretrieval.models.globalmatch

from t2vretrieval.models.globalmatch import VISENC, TXTENC

def prepare_mp_globalmatch_model(root_dir):
  anno_dir = os.path.join(root_dir, 'annotation', 'RET')
  mp_ft_dir = os.path.join(root_dir, 'ordered_feature', 'MP')
  split_dir = os.path.join(root_dir, 'public_split')
  res_dir = os.path.join(root_dir, 'results', 'RET.released')
  
  mp_ft_names = ['resnet152.pth']
  dim_mp_fts = [np.load(os.path.join(mp_ft_dir, mp_ft_name, 'val_ft.npy')).shape[-1] \
    for mp_ft_name in mp_ft_names]
  num_words = len(np.load(os.path.join(anno_dir, 'int2word.npy')))

  model_cfg = t2vretrieval.models.globalmatch.GlobalMatchModelConfig()
  
  model_cfg.max_words_in_sent = 30
  model_cfg.margin = 0.2
  model_cfg.max_violation = True #False
  model_cfg.hard_topk = 1
  model_cfg.loss_direction = 'bi'
  model_cfg.trn_batch_size = 128 
  model_cfg.tst_batch_size = 1000
  model_cfg.monitor_iter = 1000
  model_cfg.summary_iter = 1000

  visenc_cfg = model_cfg.subcfgs[VISENC]
  visenc_cfg.dim_fts = dim_mp_fts
  visenc_cfg.dim_embed = 1024 
  visenc_cfg.dropout = 0.2

  txtenc_cfg = model_cfg.subcfgs[TXTENC]
  txtenc_cfg.num_words = num_words
  txtenc_cfg.dim_word = 300 
  txtenc_cfg.fix_word_embed = False
  txtenc_cfg.rnn_hidden_size = 1024 
  txtenc_cfg.num_layers = 1
  txtenc_cfg.rnn_type = 'gru' # lstm, gru
  txtenc_cfg.bidirectional = True
  txtenc_cfg.dropout = 0.2
  
  txtenc_name = '%s%s%s'%('bi' if txtenc_cfg.bidirectional else '', txtenc_cfg.rnn_type,
    '.fix' if txtenc_cfg.fix_word_embed else '')
  output_dir = os.path.join(res_dir, 'globalmatch', 
    'mp.vis.%s.txt.%s.%d.loss.%s%s.glove.init'%
    ('-'.join(mp_ft_names),
      txtenc_name,
      visenc_cfg.dim_embed, 
      model_cfg.loss_direction, 
      '.max.%d'%model_cfg.hard_topk if model_cfg.max_violation else '')
    )
  print(output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  model_cfg.save(os.path.join(output_dir, 'model.json'))

  path_cfg = {
    'output_dir': output_dir,
    'mp_ft_files': {},
    'name_file': {},
    'word2int_file': os.path.join(anno_dir, 'word2int.json'),
    'int2word_file': os.path.join(anno_dir, 'int2word.npy'),
    'ref_caption_file': {},
  }
  for setname in ['trn', 'val', 'tst']:
    path_cfg['mp_ft_files'][setname] = [
      os.path.join(mp_ft_dir, mp_ft_name, '%s_ft.npy'%setname) for mp_ft_name in mp_ft_names
    ]
    path_cfg['name_file'][setname] = os.path.join(split_dir, '%s_names.npy'%setname)
    path_cfg['ref_caption_file'][setname] = os.path.join(anno_dir, 'ref_captions.json')
    
  with open(os.path.join(output_dir, 'path.json'), 'w') as f:
    json.dump(path_cfg, f, indent=2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('root_dir')
  opts = parser.parse_args()

  prepare_mp_globalmatch_model(opts.root_dir)
  
