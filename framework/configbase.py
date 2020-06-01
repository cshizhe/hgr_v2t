import json
import enum

import numpy as np

class ModuleConfig(object):
  """config of a module
  basic attributes:
    [freeze] boolean, whether to freeze the weights in this module in training.
    [lr_mult] float, the multiplier to the base learning rate for weights in this modules.
    [opt_alg] string, 'Adam|SGD|RMSProp', optimizer
  """
  def __init__(self):
    self.freeze = False
    self.lr_mult = 1.0
    self.opt_alg = 'Adam'
    self.weight_decay = 0

  def load_from_dict(self, cfg_dict):
    for key, value in cfg_dict.items():
      if key in self.__dict__:
        setattr(self, key, value)
    self._assert()

  def save_to_dict(self):
    out = {}
    for attr in self.__dict__:
      val = self.__dict__[attr]
      out[attr] = val
    return out

  def _assert(self):
    """check compatibility between configs
    """
    # raise NotImplementedError("""please customize %s._assert"""%(self.__class__.__name__))
    pass


class ModelConfig(object):
  def __init__(self):
    self.subcfgs = {} # save configure of submodules

    self.trn_batch_size = 128
    self.tst_batch_size = 128
    self.num_epoch = 100
    self.val_per_epoch = True
    self.save_per_epoch = True
    self.val_iter = -1
    self.save_iter = -1
    self.monitor_iter = -1
    self.summary_iter = -1 # tensorboard summary

    self.base_lr = 1e-4
    self.decay_schema = None #'MultiStepLR'
    self.decay_boundarys = []
    self.decay_rate = 1

  def load(self, cfg_file):
    with open(cfg_file) as f:
      data = json.load(f)
    for key, value in data.items():
      if key == 'subcfgs':
        for subname, subcfg in data[key].items():
          self.subcfgs[subname].load_from_dict(subcfg)
      else:
        setattr(self, key, value)

  def save(self, out_file):
    out = {}
    for key in self.__dict__:
      if key == 'subcfgs':
        out['subcfgs'] = {}
        for subname, subcfg in self.__dict__['subcfgs'].items():
          out['subcfgs'][subname] = subcfg.save_to_dict()
      else:
        out[key] = self.__dict__[key]
    with open(out_file, 'w') as f:
      json.dump(out, f, indent=2)


class PathCfg(object):
  def __init__(self):
    self.log_dir = ''
    self.model_dir = ''
    self.pred_dir = ''

    self.log_file = ''
    self.val_metric_file = ''
    self.model_file = ''
    self.predict_file = ''

  def load(self, config_dict):
    for key, value in config_dict.items():
      setattr(self, key, value)

  def save(self, output_path):
    data = {}
    for key in self.__dict__:
      data[key] = self.__getattribute__(key)
    with open(output_path, 'w') as f:
      json.dump(data, f, indent=2)
