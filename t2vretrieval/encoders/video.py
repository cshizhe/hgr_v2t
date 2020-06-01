import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase

class MPEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = [2048]
    self.dim_embed = 1024
    self.dropout = 0

class MPEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    input_size = sum(self.config.dim_fts)
    self.ft_embed = nn.Linear(input_size, self.config.dim_embed, bias=True)
    self.dropout = nn.Dropout(self.config.dropout)

  def forward(self, inputs):
    '''
    Args:
      inputs: (batch, dim_fts) or (batch, max_seq_len, dim_fts)
    Return:
      embeds: (batch, dim_embed) or (batch, max_seq_len, dim_fts)
    '''
    embeds = self.ft_embed(inputs)
    embeds = self.dropout(embeds)
    return embeds

