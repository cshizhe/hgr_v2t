""" Embeddings module """
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
  """
  Implements the sinusoidal positional encoding for
  non-recurrent neural networks.

  Implementation based on "Attention Is All You Need"

  Args:
    dim_embed (int): embedding size (even number)
  """

  def __init__(self, dim_embed, max_len=100):
    super(PositionalEncoding, self).__init__()

    pe = torch.zeros(max_len, dim_embed)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim_embed, 2, dtype=torch.float) *
               -(math.log(10000.0) / dim_embed)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    
    self.pe = pe # size=(max_len, dim_embed)
    self.dim_embed = dim_embed

  def forward(self, emb, step=None):
    if emb.device != self.pe.device:
      self.pe = self.pe.to(emb.device)
    if step is None:
      # emb.size = (batch, seq_len, dim_embed)
      emb = emb + self.pe[:emb.size(1)]
    else:
      # emb.size = (batch, dim_embed)
      emb = emb + self.pe[step]
    return emb


class Embedding(nn.Module):
  """Words embeddings for encoder/decoder.
  Args:
    word_vec_size (int): size of the dictionary of embeddings.
    word_vocab_size (int): size of dictionary of embeddings for words.
    position_encoding (bool): see :obj:`modules.PositionalEncoding`
  """
  def __init__(self, word_vocab_size, word_vec_size, 
    position_encoding=False, fix_word_embed=False, max_len=100):
    super(Embedding, self).__init__()

    self.word_vec_size = word_vec_size
    self.we = nn.Embedding(word_vocab_size, word_vec_size)
    if fix_word_embed:
      self.we.weight.requires_grad = False
    self.init_weight()

    self.position_encoding = position_encoding
    if self.position_encoding:
      self.pe = PositionalEncoding(word_vec_size, max_len=max_len)

  def init_weight(self):
    std = 1. / (self.word_vec_size**0.5)
    nn.init.uniform_(self.we.weight, -std, std)

  def forward(self, word_idxs, step=None):
    """Computes the embeddings for words.
    Args:
      word_idxs (`LongTensor`): index tensor 
        size = (batch, seq_len) or (batch, )
    Return:
      embeds: `FloatTensor`, 
        size = (batch, seq_len, dim_embed) or (batch, dim_embed)
    """
    embeds = self.we(word_idxs)
    if self.position_encoding:
      embeds = self.pe(embeds, step=step)
    return embeds
