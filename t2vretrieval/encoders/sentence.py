import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase
from framework.modules.embeddings import Embedding
import framework.ops

class SentEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.num_words = 0
    self.dim_word = 300
    self.fix_word_embed = False
    self.rnn_type = 'gru' # gru, lstm
    self.bidirectional = True
    self.rnn_hidden_size = 1024
    self.num_layers = 1
    self.dropout = 0.5

  def _assert(self):
    assert self.rnn_type in ['gru', 'lstm'], 'invalid rnn_type'

class SentEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.embedding = Embedding(self.config.num_words, self.config.dim_word,
      fix_word_embed=self.config.fix_word_embed)
    dim_word = self.config.dim_word
    
    self.rnn = framework.ops.rnn_factory(self.config.rnn_type,
      input_size=dim_word, hidden_size=self.config.rnn_hidden_size, 
      num_layers=self.config.num_layers, dropout=self.config.dropout,
      bidirectional=self.config.bidirectional, bias=True, batch_first=True)
   
    self.dropout = nn.Dropout(self.config.dropout)
    self.init_weights()

  def init_weights(self):
    directions = ['']
    if self.config.bidirectional:
      directions.append('_reverse')
    for layer in range(self.config.num_layers):
      for direction in directions:
        for name in ['i', 'h']:
          weight = getattr(self.rnn, 'weight_%sh_l%d%s'%(name, layer, direction))
          nn.init.orthogonal_(weight.data)
          bias = getattr(self.rnn, 'bias_%sh_l%d%s'%(name, layer, direction))
          nn.init.constant_(bias, 0)
          if name == 'i' and self.config.rnn_type == 'lstm':
            bias.data.index_fill_(0, torch.arange(
              self.config.rnn_hidden_size, self.config.rnn_hidden_size*2).long(), 1)
          
  def forward_text_encoder(self, word_embeds, seq_lens, init_states):
     # outs.size = (batch, seq_len, num_directions * hidden_size)
    outs, states = framework.ops.calc_rnn_outs_with_sort(
      self.rnn, word_embeds, seq_lens, init_states)
    return outs

  def forward(self, cap_ids, cap_lens, init_states=None, return_dense=False):
    '''
    Args:
      cap_ids: LongTensor, (batch, seq_len)
      cap_lens: FloatTensor, (batch, )
    Returns:
      if return_dense:
        embeds: FloatTensor, (batch, seq_len, embed_size)
      else:
        embeds: FloatTensor, (batch, embed_size)
    '''
    word_embeds = self.embedding(cap_ids)
    
    hiddens = self.forward_text_encoder(
      self.dropout(word_embeds), cap_lens, init_states)
    batch_size, max_seq_len, hidden_size = hiddens.size()

    if self.config.bidirectional:
      splited_hiddens = torch.split(hiddens, self.config.rnn_hidden_size, dim=2) 
      hiddens = (splited_hiddens[0] + splited_hiddens[1]) / 2

    if return_dense:
      return hiddens
    else:
      sent_masks = framework.ops.sequence_mask(cap_lens, max_seq_len, inverse=False).float()
      sent_embeds = torch.sum(hiddens * sent_masks.unsqueeze(2), 1) / cap_lens.unsqueeze(1).float()
      return sent_embeds


class SentAttnEncoder(SentEncoder):
  def __init__(self, config):
    super().__init__(config)
    self.ft_attn = nn.Linear(self.config.rnn_hidden_size, 1)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, cap_ids, cap_lens, init_states=None, return_dense=False):
    hiddens = super().forward(cap_ids, cap_lens, init_states=init_states, return_dense=True)

    attn_scores = self.ft_attn(hiddens).squeeze(2)
    cap_masks = framework.ops.sequence_mask(cap_lens, max_len=attn_scores.size(1), inverse=False)
    attn_scores = attn_scores.masked_fill(cap_masks == 0, -1e18)
    attn_scores = self.softmax(attn_scores)

    if return_dense:
      return hiddens, attn_scores
    else:
      return torch.sum(hiddens * attn_scores.unsqueeze(2), 1)
  
