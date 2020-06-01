import os
import logging

import numpy as np

def set_logger(log_path, log_name='training'):
  if log_path is None:
    print('log_path is empty')
    return None
    
  if os.path.exists(log_path):
    print('%s already exists'%log_path)
    return None

  logger = logging.getLogger(log_name)
  logger.setLevel(logging.DEBUG)

  logfile = logging.FileHandler(log_path)
  console = logging.StreamHandler()
  logfile.setLevel(logging.INFO)
  logfile.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
  console.setLevel(logging.DEBUG)
  console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
  logger.addHandler(logfile)
  logger.addHandler(console)

  return logger
