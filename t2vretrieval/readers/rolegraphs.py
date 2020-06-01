import os
import json
import numpy as np
import h5py
import collections
import torch

import t2vretrieval.readers.mpdata

ROLES = ['V', 'ARG1', 'ARG0', 'ARG2', 'ARG3', 'ARG4',
 'ARGM-LOC', 'ARGM-MNR', 'ARGM-TMP', 'ARGM-DIR', 'ARGM-ADV', 
 'ARGM-PRP', 'ARGM-PRD', 'ARGM-COM', 'ARGM-MOD', 'NOUN']

class RoleGraphDataset(t2vretrieval.readers.mpdata.MPDataset):
  def __init__(self, name_file, attn_ft_files, word2int_file,
    max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file, 
    max_attn_len=20, load_video_first=False, is_train=False, _logger=None):
    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train
    self.attn_ft_files = attn_ft_files
    self.max_attn_len = max_attn_len
    self.load_video_first = load_video_first

    self.names = np.load(name_file)
    self.word2int = json.load(open(word2int_file))

    self.num_videos = len(self.names)
    self.print_fn('num_videos %d' % (self.num_videos))

    if ref_caption_file is None:
      self.ref_captions = None
    else:
      self.ref_captions = json.load(open(ref_caption_file))
      self.captions = set()
      self.pair_idxs = []
      for i, name in enumerate(self.names):
        for j, sent in enumerate(self.ref_captions[name]):
          self.captions.add(sent)
          self.pair_idxs.append((i, j))
      self.captions = list(self.captions)
      self.num_pairs = len(self.pair_idxs)
      self.print_fn('captions size %d' % self.num_pairs)

    if self.load_video_first:
      self.all_attn_fts, self.all_attn_lens = [], []
      for name in self.names:
        attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
        attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')
        self.all_attn_fts.append(attn_fts)
        self.all_attn_lens.append(attn_len)
      self.all_attn_fts = np.array(self.all_attn_fts)
      self.all_attn_lens = np.array(self.all_attn_lens)

    self.num_verbs = num_verbs
    self.num_nouns = num_nouns
    
    self.role2int = {}
    for i, role in enumerate(ROLES):
      self.role2int[role] = i
      self.role2int['C-%s'%role] = i
      self.role2int['R-%s'%role] = i

    self.ref_graphs = json.load(open(ref_graph_file))

  def load_attn_ft_by_name(self, name, attn_ft_files):
    attn_fts = []
    for i, attn_ft_file in enumerate(attn_ft_files):
      with h5py.File(attn_ft_file, 'r') as f:
        key = name.replace('/', '_')
        attn_ft = f[key][...]
        attn_fts.append(attn_ft)
    attn_fts = np.concatenate([attn_ft for attn_ft in attn_fts], axis=-1)
    return attn_fts

  def pad_or_trim_feature(self, attn_ft, max_attn_len, trim_type='top'):
    seq_len, dim_ft = attn_ft.shape
    attn_len = min(seq_len, max_attn_len)

    # pad
    if seq_len < max_attn_len:
      new_ft = np.zeros((max_attn_len, dim_ft), np.float32)
      new_ft[:seq_len] = attn_ft
    # trim
    else:
      if trim_type == 'top':
        new_ft = attn_ft[:max_attn_len]
      elif trim_type == 'select':
        idxs = np.round(np.linspace(0, seq_len-1, max_attn_len)).astype(np.int32)
        new_ft = attn_ft[idxs]
    return new_ft, attn_len

  def get_caption_outs(self, out, sent, graph):
    graph_nodes, graph_edges = graph
    #print(graph)

    verb_node2idxs, noun_node2idxs = {}, {}
    edges = []
    out['node_roles'] = np.zeros((self.num_verbs + self.num_nouns, ), np.int32)

    # root node
    sent_ids, sent_len = self.process_sent(sent, self.max_words_in_sent)
    out['sent_ids'] = sent_ids
    out['sent_lens'] = sent_len

    # graph: add verb nodes
    node_idx = 1
    out['verb_masks'] = np.zeros((self.num_verbs, self.max_words_in_sent), np.bool)
    for knode, vnode in graph_nodes.items():
      k = node_idx - 1
      if k >= self.num_verbs:
        break
      if vnode['role'] == 'V' and np.min(vnode['spans']) < self.max_words_in_sent:
        verb_node2idxs[knode] = node_idx
        for widx in vnode['spans']:
          if widx < self.max_words_in_sent:
            out['verb_masks'][k][widx] = True
        out['node_roles'][node_idx - 1] = self.role2int['V']
        # add root to verb edge
        edges.append((0, node_idx))
        node_idx += 1
        
    # graph: add noun nodes
    node_idx = 1 + self.num_verbs
    out['noun_masks'] = np.zeros((self.num_nouns, self.max_words_in_sent), np.bool)
    for knode, vnode in graph_nodes.items():
      k = node_idx - self.num_verbs - 1
      if k >= self.num_nouns:
          break
      if vnode['role'] not in ['ROOT', 'V'] and np.min(vnode['spans']) < self.max_words_in_sent:
        noun_node2idxs[knode] = node_idx
        for widx in vnode['spans']:
          if widx < self.max_words_in_sent:
            out['noun_masks'][k][widx] = True
        out['node_roles'][node_idx - 1] = self.role2int.get(vnode['role'], self.role2int['NOUN'])
        node_idx += 1

    # graph: add verb_node to noun_node edges
    for e in graph_edges:
      if e[0] in verb_node2idxs and e[1] in noun_node2idxs:
        edges.append((verb_node2idxs[e[0]], noun_node2idxs[e[1]]))
        edges.append((noun_node2idxs[e[1]], verb_node2idxs[e[0]]))

    num_nodes = 1 + self.num_verbs + self.num_nouns
    rel_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src_nodeidx, tgt_nodeidx in edges:
      rel_matrix[tgt_nodeidx, src_nodeidx] = 1
    # row norm
    for i in range(num_nodes):
      s = np.sum(rel_matrix[i])
      if s > 0:
        rel_matrix[i] /= s

    out['rel_edges'] = rel_matrix
    return out

  def __getitem__(self, idx):
    out = {}
    if self.is_train:
      video_idx, cap_idx = self.pair_idxs[idx]
      name = self.names[video_idx]
      sent = self.ref_captions[name][cap_idx]
      out = self.get_caption_outs(out, sent, self.ref_graphs[sent])
    else:
      video_idx = idx
      name = self.names[idx]
    
    if self.load_video_first:
      attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
    else:
      attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
      attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')
    
    out['names'] = name
    out['attn_fts'] = attn_fts
    out['attn_lens'] = attn_len
    return out

  def iterate_over_captions(self, batch_size):
    # the sentence order is the same as self.captions
    for s in range(0, len(self.captions), batch_size):
      e = s + batch_size
      data = []
      for sent in self.captions[s: e]:
        out = self.get_caption_outs({}, sent, self.ref_graphs[sent])
        data.append(out)
      outs = collate_graph_fn(data)
      yield outs

def collate_graph_fn(data):
  outs = {}
  for key in ['names', 'attn_fts', 'attn_lens', 'sent_ids', 'sent_lens',
              'verb_masks', 'noun_masks', 'node_roles', 'rel_edges']:
    if key in data[0]:
      outs[key] = [x[key] for x in data]

  batch_size = len(data)

  # reduce attn_lens
  if 'attn_fts' in outs:
    max_len = np.max(outs['attn_lens'])
    outs['attn_fts'] = np.stack(outs['attn_fts'], 0)[:, :max_len]

  # reduce caption_ids lens
  if 'sent_lens' in outs:
    max_cap_len = np.max(outs['sent_lens'])
    outs['sent_ids'] = np.array(outs['sent_ids'])[:, :max_cap_len]
    outs['verb_masks'] = np.array(outs['verb_masks'])[:, :, :max_cap_len]
    outs['noun_masks'] = np.array(outs['noun_masks'])[:, :, :max_cap_len]
  return outs
