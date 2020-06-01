import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import framework.logbase


class ModelBase(object):
  def __init__(self, config, _logger=None, gpu_id=0):
    '''initialize model 
    (support single GPU, otherwise need to be customized)
    '''
    self.device = torch.device("cuda:%d"%gpu_id if torch.cuda.is_available() else "cpu")
    self.config = config
    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.submods = self.build_submods()
    for submod in self.submods.values():
      submod.to(self.device)
    self.criterion = self.build_loss()
    self.params, self.optimizer, self.lr_scheduler = self.build_optimizer()

    num_params, num_weights = 0, 0
    for key, submod in self.submods.items():
      for varname, varvalue in submod.state_dict().items():
        self.print_fn('%s: %s, shape=%s, num:%d' % (
          key, varname, str(varvalue.size()), np.prod(varvalue.size())))
        num_params += 1
        num_weights += np.prod(varvalue.size())
    self.print_fn('num params %d, num weights %d'%(num_params, num_weights))
    self.print_fn('trainable: num params %d, num weights %d'%(
      len(self.params), sum([np.prod(param.size()) for param in self.params])))

  def build_submods(self):
    raise NotImplementedError('implement build_submods function: return submods')

  def build_loss(self):
    raise NotImplementedError('implement build_loss function: return criterion')

  def forward_loss(self, batch_data, step=None):
    raise NotImplementedError('implement forward_loss function: return loss and additional outs')
    
  def validate(self, val_reader, step=None):
    self.eval_start()
    # raise NotImplementedError('implement validate function: return metrics')

  def test(self, tst_reader, tst_pred_file, tst_model_file=None):
    if tst_model_file is not None:
      self.load_checkpoint(tst_model_file)
    self.eval_start()
    # raise NotImplementedError('implement test function')

  ########################## boilerpipe functions ########################
  def build_optimizer(self):
    trn_params = []
    trn_param_ids = set()
    per_param_opts = []
    for key, submod in self.submods.items():
      if self.config.subcfgs[key].freeze:
        for param in submod.parameters():
          param.requires_grad = False
      else:
        params = []
        for param in submod.parameters():
          # sometimes we share params in different submods
          if param.requires_grad and id(param) not in trn_param_ids:
            params.append(param)
            trn_param_ids.add(id(param))
        per_param_opts.append({
          'params': params, 
          'lr': self.config.base_lr * self.config.subcfgs[key].lr_mult,
          'weight_decay': self.config.subcfgs[key].weight_decay,
          })
        trn_params.extend(params)
    if len(trn_params) > 0:
      optimizer = optim.Adam(per_param_opts, lr=self.config.base_lr)
      lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
        milestones=self.config.decay_boundarys, gamma=self.config.decay_rate)
    else:
      optimizer, lr_scheduler = None, None
      print('no traiable parameters')
    return trn_params, optimizer, lr_scheduler

  def train_start(self):
    for key, submod in self.submods.items():
      submod.train()
    torch.set_grad_enabled(True)

  def eval_start(self):
    for key, submod in self.submods.items():
      submod.eval()
    torch.set_grad_enabled(False)

  def save_checkpoint(self, ckpt_file, submods=None):
    if submods is None:
      submods = self.submods
    state_dicts = {}
    for key, submod in submods.items():
      state_dicts[key] = {}
      for varname, varvalue in submod.state_dict().items():
        state_dicts[key][varname] = varvalue.cpu()
    torch.save(state_dicts, ckpt_file)

  def load_checkpoint(self, ckpt_file, submods=None):
    if submods is None:
      submods = self.submods
    state_dicts = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
    
    num_resumed_vars = 0
    for key, state_dict in state_dicts.items():
      if key in submods:
        own_state_dict = submods[key].state_dict()
        new_state_dict = {}
        for varname, varvalue in state_dict.items():
          if varname in own_state_dict:
            new_state_dict[varname] = varvalue
            num_resumed_vars += 1
        own_state_dict.update(new_state_dict)
        submods[key].load_state_dict(own_state_dict)
    self.print_fn('number of resumed variables: %d'%num_resumed_vars)
    
  def pretty_print_metrics(self, prefix, metrics):
    metric_str = []
    for measure, score in metrics.items():
      metric_str.append('%s %.4f'%(measure, score))
    metric_str = ' '.join(metric_str)
    self.print_fn('%s: %s' % (prefix, metric_str))

  def get_current_base_lr(self):
      return self.optimizer.param_groups[0]['lr']

  def train_one_batch(self, batch_data, step):
    self.optimizer.zero_grad()
    loss = self.forward_loss(batch_data, step=step)
    loss.backward()
    self.optimizer.step()

    loss_value = loss.data.item()
    if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
      self.print_fn('\ttrn step %d lr %.8f %s: %.4f' % (step, self.get_current_base_lr(), 'loss', loss_value))
    return {'loss': loss_value}

  def train_one_epoch(self, step, trn_reader, val_reader, model_dir, log_dir):
    self.train_start()

    avg_loss, n_batches = {}, {}
    for batch_data in trn_reader:
      loss = self.train_one_batch(batch_data, step)
      for loss_key, loss_value in loss.items():
        avg_loss.setdefault(loss_key, 0)
        n_batches.setdefault(loss_key, 0)
        avg_loss[loss_key] += loss_value
        n_batches[loss_key] += 1
      step += 1

      if self.config.save_iter > 0 and step % self.config.save_iter == 0:
        self.save_checkpoint(os.path.join(model_dir, 'step.%d.th'%step))
      
      if (self.config.save_iter > 0 and step % self.config.save_iter == 0) \
        or (self.config.val_iter > 0 and step % self.config.val_iter == 0):
        metrics = self.validate(val_reader, step=step)
        with open(os.path.join(log_dir, 'val.step.%d.json'%step), 'w') as f:
          json.dump(metrics, f, indent=2)
        self.pretty_print_metrics('\tval step %d'%step, metrics)
        self.train_start()

    for loss_key, loss_value in avg_loss.items():
      avg_loss[loss_key] = loss_value / n_batches[loss_key]
    return avg_loss, step

  def epoch_postprocess(self, epoch):
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

  def train(self, trn_reader, val_reader, model_dir, log_dir, resume_file=None):
    assert self.optimizer is not None

    if resume_file is not None:
      self.load_checkpoint(resume_file)

    # first validate
    metrics = self.validate(val_reader)
    self.pretty_print_metrics('init val', metrics)

    # training
    step = 0
    for epoch in range(self.config.num_epoch):
      avg_loss, step = self.train_one_epoch(
        step, trn_reader, val_reader, model_dir, log_dir)
      self.pretty_print_metrics('epoch (%d/%d) trn'%(epoch, self.config.num_epoch), avg_loss)
      self.epoch_postprocess(epoch)

      if self.config.save_per_epoch:
        self.save_checkpoint(os.path.join(model_dir, 'epoch.%d.th'%epoch))
      
      if self.config.val_per_epoch:
        metrics = self.validate(val_reader, step=step)
        with open(os.path.join(log_dir, 
          'val.epoch.%d.step.%d.json'%(epoch, step)), 'w') as f:
          json.dump(metrics, f, indent=2)
        self.pretty_print_metrics('epoch (%d/%d) val' % (epoch, self.config.num_epoch), metrics)
      
