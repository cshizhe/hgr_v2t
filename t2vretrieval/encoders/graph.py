import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
  def __init__(self, embed_size, dropout=0.0):
    super().__init__()
    self.embed_size = embed_size
    self.ctx_layer = nn.Linear(self.embed_size, self.embed_size, bias=False)
    self.layernorm = nn.LayerNorm(embed_size)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, node_fts, rel_edges):
    '''Args:
      node_fts: (batch_size, num_nodes, embed_size)
      rel_edges: (batch_size, num_nodes, num_nodes)
    '''
    ctx_embeds = self.ctx_layer(torch.bmm(rel_edges, node_fts))
    node_embeds = node_fts + self.dropout(ctx_embeds)
    node_embeds = self.layernorm(node_embeds)
    return node_embeds

class AttnGCNLayer(GCNLayer):
  def __init__(self, embed_size, d_ff, dropout=0.0):
    super().__init__(embed_size, dropout=dropout)
    self.edge_attn_query = nn.Linear(embed_size, d_ff)
    self.edge_attn_key = nn.Linear(embed_size, d_ff)
    self.attn_denominator = math.sqrt(d_ff)
  
  def forward(self, node_fts, rel_edges):
    '''
    Args:
      node_fts: (batch_size, num_nodes, embed_size)
      rel_edges: (batch_size, num_nodes, num_nodes)
    '''
    # (batch_size, num_nodes, num_nodes)
    attn_scores = torch.einsum('bod,bid->boi', 
      self.edge_attn_query(node_fts), 
      self.edge_attn_key(node_fts)) / self.attn_denominator
    attn_scores = attn_scores.masked_fill(rel_edges == 0, -1e18)
    attn_scores = torch.softmax(attn_scores, dim=2)
    # some nodes do not connect with any edge
    attn_scores = attn_scores.masked_fill(rel_edges == 0, 0)

    ctx_embeds = self.ctx_layer(torch.bmm(attn_scores, node_fts))
    node_embeds = node_fts + self.dropout(ctx_embeds)
    node_embeds = self.layernorm(node_embeds)
    return node_embeds

class GCNEncoder(nn.Module):
  def __init__(self, dim_input, dim_hidden, num_hidden_layers, 
    embed_first=False, dropout=0, attention=False):
    super().__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.num_hidden_layers = num_hidden_layers
    self.embed_first = embed_first
    self.attention = attention

    if self.attention:
      gcn_fn = AttnGCNLayer
    else:
      gcn_fn = GCNLayer

    if self.embed_first:
      self.first_embedding = nn.Sequential(
        nn.Linear(self.dim_input, self.dim_hidden),
        nn.ReLU())

    self.layers = nn.ModuleList()
    for k in range(num_hidden_layers):
      if self.attention:
        h2h = gcn_fn(self.dim_hidden, self.dim_hidden // 2, dropout=dropout)
      else:
        h2h = gcn_fn(self.dim_hidden, dropout=dropout)
      self.layers.append(h2h)

  def forward(self, node_fts, rel_edges):
    if self.embed_first:
      node_fts = self.first_embedding(node_fts)

    for k in range(self.num_hidden_layers):
      layer = self.layers[k]
      node_fts = layer(node_fts, rel_edges)
      
    # (batch_size, num_nodes, dim_hidden)
    return node_fts

