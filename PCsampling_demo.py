import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
##################################################################
import sampling as sampling
from sampling import ReverseDiffusionPredictor,LangevinCorrector,AnnealedLangevinDynamics ,EulerMaruyamaPredictor,AncestralSamplingPredictor
import aapm_sin_ncsnpp_3h as configs_3h  
import aapm_sin_ncsnpp_wavelet as configs_A
##################################################################

sys.path.append('..')
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np

from utils import restore_checkpoint

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
from sde_lib import VESDE, VPSDE, subVPSDE
import os.path as osp
if len(sys.argv) > 1:
  start = int(sys.argv[1])
  end = int(sys.argv[2])



checkpoint_num = [[23,24]]#25 3h,A
# print(checkpoint_num)
# assert False
def get_predict(num):
  if num == 0:
    return None
  elif num == 1:
    return EulerMaruyamaPredictor
  elif num == 2:
    return ReverseDiffusionPredictor

def get_correct(num):
  if num == 0:
    return None
  elif num == 1:
    return LangevinCorrector
  elif num == 2:
    return AnnealedLangevinDynamics


#checkpoint_num = [23,24,25,32,35,44]
# checkpoint_num = [6,8,10,12,14,16]
predicts = [2]
corrects = [1]
for predict in predicts:
  for correct in corrects:
    for check_num in checkpoint_num:
      sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
      if sde.lower() == 'vesde':
        ckpt_filename_3h = './exp_3h/checkpoints/checkpoint_{}.pth'.format(check_num[0])
        ckpt_filename_A = './exp_wavelet/checkpoints/checkpoint_{}.pth'.format(check_num[1])
        assert os.path.exists(ckpt_filename_3h)
        assert os.path.exists(ckpt_filename_A)
        config_3h = configs_3h.get_config()
        config_A = configs_A.get_config()
        sde_3h = VESDE(sigma_min=config_3h.model.sigma_min, sigma_max=config_3h.model.sigma_max, N=config_3h.model.num_scales)
        sde_A = VESDE(sigma_min=config_A.model.sigma_min, sigma_max=config_A.model.sigma_max, N=config_A.model.num_scales)
        sampling_eps = 1e-5

      # 3h model
      batch_size = 1 #@param {"type":"integer"}
      config_3h.training.batch_size = batch_size
      config_3h.eval.batch_size = batch_size

      random_seed = 0 #@param {"type": "integer"}

      sigmas = mutils.get_sigmas(config_3h)
      h3_model = mutils.create_model(config_3h)

      optimizer = get_optimizer(config_3h, h3_model.parameters())
      ema = ExponentialMovingAverage(h3_model.parameters(),
                                    decay=config_3h.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                  model=h3_model, ema=ema)

      state = restore_checkpoint(ckpt_filename_3h, state, config_3h.device)
      ema.copy_to(h3_model.parameters())

      # wavelet model
      batch_size = 1 #@param {"type":"integer"}
      config_A.training.batch_size = batch_size
      config_A.eval.batch_size = batch_size

      random_seed = 0 #@param {"type": "integer"}

      sigmas = mutils.get_sigmas(config_A)
      A_model = mutils.create_model(config_A)

      optimizer = get_optimizer(config_A, A_model.parameters())
      ema = ExponentialMovingAverage(A_model.parameters(),
                                    decay=config_A.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                  model=A_model, ema=ema)

      state = restore_checkpoint(ckpt_filename_A, state, config_A.device)
      ema.copy_to(A_model.parameters())

      #@title PC sampling
      # img_size = config_hh.data.image_size
      # channels = config_hh.data.num_channels
      # shape = (batch_size, channels, img_size, img_size)
      # predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
      # corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
      predictor = get_predict(predict) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
      corrector = get_correct(correct) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}

      snr = 0.16#0.16 #@param {"type": "number"}
      n_steps = 1#@param {"type": "integer"}
      probability_flow = False #@param {"type": "boolean"}
      sampling_fn = sampling.get_pc_sampler(sde_3h, sde_A, predictor, corrector,
                                            None, snr, n_steps=n_steps,
                                            probability_flow=probability_flow,
                                            continuous_3h=config_3h.training.continuous,
                                            continuous_A=config_A.training.continuous,
                                            eps=sampling_eps, device_3h=config_3h.device, device_A=config_A.device)

      sampling_fn(h3_model,A_model,check_num,predict,correct)

