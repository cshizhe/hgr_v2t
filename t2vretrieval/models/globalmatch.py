import os
import numpy as np
import collections
import json

import torch

import framework.ops
import framework.configbase
import framework.modelbase

import t2vretrieval.encoders.video
import t2vretrieval.encoders.sentence
import t2vretrieval.models.criterion
import t2vretrieval.models.evaluation
from t2vretrieval.models.criterion import cosine_sim

VISENC = 'video_encoder'
TXTENC = 'text_encoder'

class GlobalMatchModelConfig(framework.configbase.ModelConfig):
  def __init__(self):
    super().__init__()
    self.max_frames_in_video = None
    self.max_words_in_sent = 30
    self.margin = 0.2
    self.max_violation = False
    self.hard_topk = 1
    self.loss_direction = 'bi'

    self.subcfgs[VISENC] = t2vretrieval.encoders.video.MPEncoderConfig()
    self.subcfgs[TXTENC] = t2vretrieval.encoders.sentence.SentEncoderConfig()

class GlobalMatchModel(framework.modelbase.ModelBase):
  def build_submods(self):
    submods = {
      VISENC: t2vretrieval.encoders.video.MPEncoder(self.config.subcfgs[VISENC]),
      TXTENC: t2vretrieval.encoders.sentence.SentEncoder(self.config.subcfgs[TXTENC]),
    }
    return submods

  def build_loss(self):
    criterion = t2vretrieval.models.criterion.ContrastiveLoss(
      margin=self.config.margin,  
      max_violation=self.config.max_violation,
      topk=self.config.hard_topk,
      direction=self.config.loss_direction)
    return criterion

  def forward_video_embed(self, batch_data):
    vid_fts = torch.FloatTensor(batch_data['mp_fts']).to(self.device)
    vid_embeds = self.submods[VISENC](vid_fts)
    return {'vid_embeds': vid_embeds}

  def forward_text_embed(self, batch_data):
    cap_ids = torch.LongTensor(batch_data['caption_ids']).to(self.device)
    cap_lens = torch.LongTensor(batch_data['caption_lens']).to(self.device)
    cap_embeds = self.submods[TXTENC](cap_ids, cap_lens)
    return {'cap_embeds': cap_embeds}

  def generate_scores(self, **kwargs):
    # compute image-sentence similarity
    vid_embeds = kwargs['vid_embeds']
    cap_embeds = kwargs['cap_embeds']
    scores = cosine_sim(vid_embeds, cap_embeds) # s[i, j] i: im_idx, j: s_idx
    return scores

  def forward_loss(self, batch_data, step=None):
    vid_enc_outs = self.forward_video_embed(batch_data)
    cap_enc_outs = self.forward_text_embed(batch_data)
    cap_enc_outs.update(vid_enc_outs)
    scores = self.generate_scores(**cap_enc_outs)
    loss = self.criterion(scores)

    if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
      neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
      self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f'%(
        step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]), 
        torch.mean(torch.max(neg_scores, 0)[0])))

    return loss

  def evaluate_scores(self, tst_reader):
    vid_names, all_scores = [], []
    cap_names = tst_reader.dataset.captions
    for vid_data in tst_reader:
      vid_names.extend(vid_data['names'])
      vid_enc_outs = self.forward_video_embed(vid_data)
      all_scores.append([])
      for cap_data in tst_reader.dataset.iterate_over_captions(self.config.tst_batch_size):
        cap_enc_outs = self.forward_text_embed(cap_data)
        cap_enc_outs.update(vid_enc_outs)
        scores = self.generate_scores(**cap_enc_outs)
        all_scores[-1].append(scores.data.cpu().numpy())
      all_scores[-1] = np.concatenate(all_scores[-1], axis=1)
    all_scores = np.concatenate(all_scores, axis=0) # (n_img, n_cap)
    return vid_names, cap_names, all_scores

  def calculate_metrics(self, scores, i2t_gts, t2i_gts):
    # caption retrieval
    cr1, cr5, cr10, cmedr, cmeanr = t2vretrieval.models.evaluation.eval_q2m(scores, i2t_gts)
    # image retrieval
    ir1, ir5, ir10, imedr, imeanr = t2vretrieval.models.evaluation.eval_q2m(scores.T, t2i_gts)
    # sum of recalls to be used for early stopping
    rsum = cr1 + cr5 + cr10 + ir1 + ir5 + ir10

    metrics = collections.OrderedDict()
    metrics['ir1'] = ir1
    metrics['ir5'] = ir5
    metrics['ir10'] = ir10
    metrics['imedr'] = imedr
    metrics['imeanr'] = imeanr
    metrics['cr1'] = cr1
    metrics['cr5'] = cr5
    metrics['cr10'] = cr10
    metrics['cmedr'] = cmedr
    metrics['cmeanr'] = cmeanr
    metrics['rsum'] = rsum
    
    return metrics

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

    metrics = self.calculate_metrics(scores, i2t_gts, t2i_gts)
    if return_outs:
      outs = {
        'vid_names': vid_names,
        'cap_names': cap_names,
        'scores': scores,
      }
      return metrics, outs
    else:
      return metrics

  def validate(self, val_reader, step=None):
    self.eval_start()
    metrics = self.evaluate(val_reader)
    return metrics

  def test(self, tst_reader, tst_pred_file, tst_model_file=None):
    if tst_model_file is not None:
      self.load_checkpoint(tst_model_file)
    self.eval_start()

    if tst_reader.dataset.ref_captions is None:
      vid_names, cap_names, scores = self.evaluate_scores(tst_reader)
      outs = {
        'vid_names': vid_names,
        'cap_names': cap_names,
        'scores': scores,
      }
      metrics = None
    else:
      metrics, outs = self.evaluate(tst_reader, return_outs=True)
    
    with open(tst_pred_file, 'wb') as f:
     np.save(f, outs)

    return metrics


