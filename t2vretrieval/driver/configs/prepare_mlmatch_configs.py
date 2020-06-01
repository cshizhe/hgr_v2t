import os
import sys
import argparse
import numpy as np
import json

import t2vretrieval.models.mlmatch

from t2vretrieval.models.mlmatch import VISENC, TXTENC
from t2vretrieval.readers.rolegraphs import ROLES

def prepare_match_model(root_dir):
  anno_dir = os.path.join(root_dir, 'annotation', 'RET')
  attn_ft_dir = os.path.join(root_dir, 'ordered_feature', 'SA')
  split_dir = os.path.join(root_dir, 'public_split')
  res_dir = os.path.join(root_dir, 'results', 'RET.released')
  
  attn_ft_names = ['resnet152.pth']
  num_words = len(np.load(os.path.join(anno_dir, 'int2word.npy')))

  model_cfg = t2vretrieval.models.mlmatch.RoleGraphMatchModelConfig()
  
  model_cfg.max_frames_in_video = 20 
  model_cfg.max_words_in_sent = 30
  model_cfg.num_verbs = 4
  model_cfg.num_nouns = 6

  model_cfg.attn_fusion = 'embed' # sim, embed
  model_cfg.simattn_sigma = 4
  model_cfg.margin = 0.2
  model_cfg.loss_direction = 'bi'

  model_cfg.num_epoch = 50
  model_cfg.max_violation = True #False
  model_cfg.hard_topk = 1 #3
  model_cfg.loss_weights = None #[1, 0.2, 0.2, 0.2]

  model_cfg.trn_batch_size = 64
  model_cfg.tst_batch_size = 300
  model_cfg.monitor_iter = 1000
  model_cfg.summary_iter = 1000

  visenc_cfg = model_cfg.subcfgs[VISENC]
  visenc_cfg.dim_fts = [2048]
  visenc_cfg.dim_embed = 1024 
  visenc_cfg.dropout = 0.2
  visenc_cfg.share_enc = False
  visenc_cfg.num_levels = 3 

  txtenc_cfg = model_cfg.subcfgs[TXTENC]
  txtenc_cfg.num_words = num_words
  txtenc_cfg.dim_word = 300 
  txtenc_cfg.fix_word_embed = False
  txtenc_cfg.rnn_hidden_size = 1024
  txtenc_cfg.num_layers = 1
  txtenc_cfg.rnn_type = 'gru' # lstm, gru
  txtenc_cfg.bidirectional = True
  txtenc_cfg.dropout = 0.2
  txtenc_cfg.num_roles = len(ROLES)

  txtenc_cfg.gcn_num_layers = 1
  txtenc_cfg.gcn_attention = True #False
  txtenc_cfg.gcn_dropout = 0.5
  
  txtenc_name = '%s%s%s.%drole.gcn.%dL%s'%(
    'bi' if txtenc_cfg.bidirectional else '', txtenc_cfg.rnn_type,
    '.fix' if txtenc_cfg.fix_word_embed else '',
    txtenc_cfg.num_roles, txtenc_cfg.gcn_num_layers, 
    '.attn' if txtenc_cfg.gcn_attention else '')

  output_dir = os.path.join(res_dir, 'mlmatch', 
    'vis.%s%s.txt.%s.%d.loss.%s.af.%s.%d%s.glove.init'%
    ('-'.join(attn_ft_names),
      '.shareenc' if visenc_cfg.share_enc else '',
      txtenc_name, 
      visenc_cfg.dim_embed, 
      model_cfg.loss_direction, 
      model_cfg.attn_fusion, model_cfg.simattn_sigma,
      '.4loss' if model_cfg.loss_weights is not None else '',
      )
    )
  print(output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  model_cfg.save(os.path.join(output_dir, 'model.json'))

  path_cfg = {
    'output_dir': output_dir,
    'attn_ft_files': {},
    'name_file': {},
    'word2int_file': os.path.join(anno_dir, 'word2int.json'),
    'int2word_file': os.path.join(anno_dir, 'int2word.npy'),
    'ref_caption_file': {},
    'ref_graph_file': {},
  }
  for setname in ['trn', 'val', 'tst']:
    path_cfg['attn_ft_files'][setname] = [
      os.path.join(attn_ft_dir, ft_name, '%s_ft.hdf5'%setname) for ft_name in attn_ft_names
    ]
    path_cfg['name_file'][setname] = os.path.join(split_dir, '%s_names.npy'%setname)
    path_cfg['ref_caption_file'][setname] = os.path.join(anno_dir, 'ref_captions.json')
    path_cfg['ref_graph_file'][setname] = os.path.join(anno_dir, 'sent2rolegraph.augment.json')
    
  with open(os.path.join(output_dir, 'path.json'), 'w') as f:
    json.dump(path_cfg, f, indent=2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('root_dir')
  opts = parser.parse_args()

  prepare_match_model(opts.root_dir)
  
