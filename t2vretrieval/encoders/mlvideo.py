import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase

class MultilevelEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = [2048]
    self.dim_embed = 1024
    self.dropout = 0

    self.num_levels = 3
    self.share_enc = False

class MultilevelEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    input_size = sum(self.config.dim_fts)
    self.dropout = nn.Dropout(self.config.dropout)

    num_levels = 1 if self.config.share_enc else self.config.num_levels
    self.level_embeds = nn.ModuleList([
      nn.Linear(input_size, self.config.dim_embed, bias=True) for k in range(num_levels)
    ])

    self.ft_attn = nn.Linear(self.config.dim_embed, 1, bias=True)
      
  def forward(self, inputs, input_lens):
    '''
    Args:
      inputs: (batch, max_seq_len, dim_fts)
    Return:
      sent_embeds: (batch, dim_embed)
      verb_embeds: (batch, max_seq_len, dim_embed)
      noun_embeds: (batch, max_seq_len, dim_embed)
    '''
    embeds = []
    for k in range(self.config.num_levels):
      if self.config.share_enc:
        k = 0
      embeds.append(self.dropout(self.level_embeds[k](inputs)))

    attn_scores = self.ft_attn(embeds[0]).squeeze(2)
    input_pad_masks = framework.ops.sequence_mask(input_lens, 
      max_len=attn_scores.size(1), inverse=True)
    attn_scores = attn_scores.masked_fill(input_pad_masks, -1e18)
    attn_scores = torch.softmax(attn_scores, dim=1)
    sent_embeds = torch.sum(embeds[0] * attn_scores.unsqueeze(2), 1)

    return sent_embeds, embeds[1], embeds[2]
