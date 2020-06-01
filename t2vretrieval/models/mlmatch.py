import numpy as np
import torch

import framework.ops

import t2vretrieval.encoders.mlsent
import t2vretrieval.encoders.mlvideo

import t2vretrieval.models.globalmatch
from t2vretrieval.models.criterion import cosine_sim

from t2vretrieval.models.globalmatch import VISENC, TXTENC

class RoleGraphMatchModelConfig(t2vretrieval.models.globalmatch.GlobalMatchModelConfig):
  def __init__(self):
    super().__init__()
    self.num_verbs = 4
    self.num_nouns = 6
    
    self.attn_fusion = 'embed' # sim, embed
    self.simattn_sigma = 4

    self.hard_topk = 1
    self.max_violation = True

    self.loss_weights = None

    self.subcfgs[VISENC] = t2vretrieval.encoders.mlvideo.MultilevelEncoderConfig()
    self.subcfgs[TXTENC] = t2vretrieval.encoders.mlsent.RoleGraphEncoderConfig()


class RoleGraphMatchModel(t2vretrieval.models.globalmatch.GlobalMatchModel):
  def build_submods(self):
    return {
      VISENC: t2vretrieval.encoders.mlvideo.MultilevelEncoder(self.config.subcfgs[VISENC]),
      TXTENC: t2vretrieval.encoders.mlsent.RoleGraphEncoder(self.config.subcfgs[TXTENC])
    }

  def forward_video_embed(self, batch_data):
    vid_fts = torch.FloatTensor(batch_data['attn_fts']).to(self.device)
    vid_lens = torch.LongTensor(batch_data['attn_lens']).to(self.device)
    # (batch, max_vis_len, dim_embed)
    vid_sent_embeds, vid_verb_embeds, vid_noun_embeds = self.submods[VISENC](vid_fts, vid_lens)
    return {
      'vid_sent_embeds': vid_sent_embeds,
      'vid_verb_embeds': vid_verb_embeds, 
      'vid_noun_embeds': vid_noun_embeds,
      'vid_lens': vid_lens,
    }

  def forward_text_embed(self, batch_data):
    sent_ids = torch.LongTensor(batch_data['sent_ids']).to(self.device)
    sent_lens = torch.LongTensor(batch_data['sent_lens']).to(self.device)
    verb_masks = torch.BoolTensor(batch_data['verb_masks']).to(self.device)
    noun_masks = torch.BoolTensor(batch_data['noun_masks']).to(self.device)
    node_roles = torch.LongTensor(batch_data['node_roles']).to(self.device)
    rel_edges = torch.FloatTensor(batch_data['rel_edges']).to(self.device)
    verb_lens = torch.sum(verb_masks, 2)
    noun_lens = torch.sum(noun_masks, 2)
    # sent_embeds: (batch, dim_embed)
    # verb_embeds, noun_embeds: (batch, num_xxx, dim_embed)
    sent_embeds, verb_embeds, noun_embeds = self.submods[TXTENC](
      sent_ids, sent_lens, verb_masks, noun_masks, node_roles, rel_edges)
    return {
      'sent_embeds': sent_embeds, 'sent_lens': sent_lens, 
      'verb_embeds': verb_embeds, 'verb_lens': verb_lens, 
      'noun_embeds': noun_embeds, 'noun_lens': noun_lens,
      }

  def generate_phrase_scores(self, vid_embeds, vid_masks, phrase_embeds, phrase_masks):
    '''Args:
      - vid_embeds: (batch, num_frames, embed_size)
      - vid_masks: (batch, num_frames)
      - phrase_embeds: (batch, num_phrases, embed_size)
      - phrase_masks: (batch, num_phrases)
    '''
    batch_vids, num_frames, _ = vid_embeds.size()
    vid_pad_masks = (vid_masks == 0).unsqueeze(1).unsqueeze(3)
    batch_phrases, num_phrases, dim_embed = phrase_embeds.size()

    # compute component-wise similarity
    vid_2d_embeds = vid_embeds.view(-1, dim_embed)
    phrase_2d_embeds = phrase_embeds.view(-1, dim_embed)
    # size = (batch_vids, batch_phrases, num_frames, num_phrases)
    ground_sims = cosine_sim(vid_2d_embeds, phrase_2d_embeds).view(
      batch_vids, num_frames, batch_phrases, num_phrases).transpose(1, 2)

    vid_attn_per_word = ground_sims.masked_fill(vid_pad_masks, 0)
    vid_attn_per_word[vid_attn_per_word < 0] = 0
    vid_attn_per_word = framework.ops.l2norm(vid_attn_per_word, dim=2)
    vid_attn_per_word = vid_attn_per_word.masked_fill(vid_pad_masks, -1e18)
    vid_attn_per_word = torch.softmax(self.config.simattn_sigma * vid_attn_per_word, dim=2)
    
    if self.config.attn_fusion == 'embed':
      vid_attned_embeds = torch.einsum('abcd,ace->abde', vid_attn_per_word, vid_embeds)
      word_attn_sims = torch.einsum('abde,bde->abd',
        framework.ops.l2norm(vid_attned_embeds),
        framework.ops.l2norm(phrase_embeds))
    elif self.config.attn_fusion == 'sim':
      # (batch_vids, batch_phrases, num_phrases)
      word_attn_sims = torch.sum(ground_sims * vid_attn_per_word, dim=2) 

    # sum: (batch_vid, batch_phrases)
    phrase_scores = torch.sum(word_attn_sims * phrase_masks.float().unsqueeze(0), 2) \
                   / torch.sum(phrase_masks, 1).float().unsqueeze(0).clamp(min=1)
    return phrase_scores

  def generate_scores(self, **kwargs):
    ##### shared #####
    vid_lens = kwargs['vid_lens'] # (batch, )
    num_frames = kwargs['vid_verb_embeds'].size(1)
    vid_masks = framework.ops.sequence_mask(vid_lens, num_frames, inverse=False)

    ##### sentence-level scores #####
    sent_scores = cosine_sim(kwargs['vid_sent_embeds'], kwargs['sent_embeds'])

    ##### verb-level scores #####
    vid_verb_embeds = kwargs['vid_verb_embeds'] # (batch, num_frames, dim_embed)
    verb_embeds = kwargs['verb_embeds'] # (batch, num_verbs, dim_embed)
    verb_lens = kwargs['verb_lens'] # (batch, num_verbs)
    verb_masks = framework.ops.sequence_mask(torch.sum(verb_lens > 0, 1).long(), 
      self.config.num_verbs, inverse=False)
    # sum: (batch_vids, batch_sents)
    verb_scores = self.generate_phrase_scores(vid_verb_embeds, vid_masks, verb_embeds, verb_masks)

    ##### noun-level scores #####
    vid_noun_embeds = kwargs['vid_noun_embeds'] # (batch, num_frames, dim_embed)
    noun_embeds = kwargs['noun_embeds'] # (batch, num_nouns, dim_embed)
    noun_lens = kwargs['noun_lens'] # (batch, num_nouns)
    noun_masks = framework.ops.sequence_mask(torch.sum(noun_lens > 0, 1).long(), 
      self.config.num_nouns, inverse=False)
    # sum: (batch_vids, batch_sents)
    noun_scores = self.generate_phrase_scores(vid_noun_embeds, vid_masks, noun_embeds, noun_masks)

    return sent_scores, verb_scores, noun_scores

  def forward_loss(self, batch_data, step=None):
    enc_outs = self.forward_video_embed(batch_data)
    cap_enc_outs = self.forward_text_embed(batch_data)
    enc_outs.update(cap_enc_outs)

    sent_scores, verb_scores, noun_scores = self.generate_scores(**enc_outs)
    scores = (sent_scores + verb_scores + noun_scores) / 3
    
    sent_loss = self.criterion(sent_scores)
    verb_loss = self.criterion(verb_scores)
    noun_loss = self.criterion(noun_scores)
    fusion_loss = self.criterion(scores) 
    if self.config.loss_weights is None:
      loss = fusion_loss
    else:
      loss = self.config.loss_weights[0] * fusion_loss + \
             self.config.loss_weights[1] * sent_loss + \
             self.config.loss_weights[2] * verb_loss + \
             self.config.loss_weights[3] * noun_loss
    
    if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
      neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
      self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f'%(
        step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]), 
        torch.mean(torch.max(neg_scores, 0)[0])))
      self.print_fn('\tstep %d: sent_loss %.4f, verb_loss %.4f, noun_loss %.4f, fusion_loss %.4f'%(
        step, sent_loss.data.item(), verb_loss.data.item(), noun_loss.data.item(), fusion_loss.data.item()))

    return loss
      
  def evaluate_scores(self, tst_reader):
    K = self.config.subcfgs[VISENC].num_levels
    vid_names, all_scores = [], [[] for _ in range(K)]
    cap_names = tst_reader.dataset.captions
    for vid_data in tst_reader:
      vid_names.extend(vid_data['names'])
      vid_enc_outs = self.forward_video_embed(vid_data)
      for k in range(K):
        all_scores[k].append([])
      for cap_data in tst_reader.dataset.iterate_over_captions(self.config.tst_batch_size):
        cap_enc_outs = self.forward_text_embed(cap_data)
        cap_enc_outs.update(vid_enc_outs)
        indv_scores = self.generate_scores(**cap_enc_outs)
        for k in range(K):
          all_scores[k][-1].append(indv_scores[k].data.cpu().numpy())
      for k in range(K):
        all_scores[k][-1] = np.concatenate(all_scores[k][-1], axis=1)
    for k in range(K):
      all_scores[k] = np.concatenate(all_scores[k], axis=0) # (n_img, n_cap)
    all_scores = np.array(all_scores) # (3, n_img, n_cap)
    return vid_names, cap_names, all_scores

  def evaluate(self, tst_reader, return_outs=False):
    vid_names, cap_names, scores = self.evaluate_scores(tst_reader)

    i2t_gts = []
    for vid_name in vid_names:
      i2t_gts.append([])
      for i, cap_name in enumerate(cap_names):
        if cap_name in tst_reader.dataset.ref_captions[vid_name]:
          i2t_gts[-1].append(i)

    t2i_gts = {}
    for i, t_gts in enumerate(i2t_gts):
      for t_gt in t_gts:
        t2i_gts.setdefault(t_gt, [])
        t2i_gts[t_gt].append(i)

    fused_scores = np.mean(scores, 0)

    metrics = self.calculate_metrics(fused_scores, i2t_gts, t2i_gts)
    
    if return_outs:
      outs = {
        'vid_names': vid_names,
        'cap_names': cap_names,
        'scores': scores,
      }
      return metrics, outs
    else:
      return metrics




