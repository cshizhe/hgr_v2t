import os
import json
import numpy as np

import torch.utils.data
BOS, EOS, UNK = 0, 1, 2

class MPDataset(torch.utils.data.Dataset):
  def __init__(self, name_file, mp_ft_files, word2int_file, max_words_in_sent, 
    ref_caption_file=None, is_train=False, _logger=None):
    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info
    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train

    self.names = np.load(name_file)
    self.word2int = json.load(open(word2int_file))
    
    self.mp_fts = []
    for mp_ft_file in mp_ft_files:
      self.mp_fts.append(np.load(mp_ft_file))
    self.mp_fts = np.concatenate(self.mp_fts, axis=-1)
    self.num_videos = len(self.mp_fts)
    self.print_fn('mp_fts size %s' % (str(self.mp_fts.shape)))

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

  def process_sent(self, sent, max_words):
    tokens = [self.word2int.get(w, UNK) for w in sent.split()]
    # # add BOS, EOS?
    # tokens = [BOS] + tokens + [EOS]
    tokens = tokens[:max_words]
    tokens_len = len(tokens)
    tokens = np.array(tokens + [EOS] * (max_words - tokens_len))
    return tokens, tokens_len

  def __len__(self):
    if self.is_train:
      return self.num_pairs
    else:
      return self.num_videos

  def __getitem__(self, idx):
    out = {}
    if self.is_train:
      video_idx, cap_idx = self.pair_idxs[idx]
      name = self.names[video_idx]
      mp_ft = self.mp_fts[video_idx]
      sent = self.ref_captions[name][cap_idx]
      cap_ids, cap_len = self.process_sent(sent, self.max_words_in_sent) 
      out['caption_ids'] = cap_ids
      out['caption_lens'] = cap_len
    else:
      name = self.names[idx]
      mp_ft = self.mp_fts[idx]
    
    out['names'] = name
    out['mp_fts'] = mp_ft
    return out

  def iterate_over_captions(self, batch_size):
    # the sentence order is the same as self.captions
    for s in range(0, len(self.captions), batch_size):
      e = s + batch_size
      cap_ids, cap_lens = [], []
      for sent in self.captions[s: e]:
        cap_id, cap_len = self.process_sent(sent, self.max_words_in_sent)
        cap_ids.append(cap_id)
        cap_lens.append(cap_len)
      yield {
        'caption_ids': np.array(cap_ids, np.int32),
        'caption_lens': np.array(cap_lens, np.int32),
      }


def collate_fn(data):
  outs = {}
  for key in ['names', 'mp_fts', 'caption_ids', 'caption_lens']:
    if key in data[0]:
      outs[key] = [x[key] for x in data]

  # reduce caption_ids lens
  if 'caption_lens' in outs:
    max_cap_len = np.max(outs['caption_lens'])
    outs['caption_ids'] = np.array(outs['caption_ids'])[:, :max_cap_len]

  return outs



