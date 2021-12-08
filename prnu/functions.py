# -*- coding: UTF-8 -*-
r"""

PRNU related functions

Original authors:

:author: Luca Bondi (luca.bondi@polimi.it)
:author: Paolo Bestagini (paolo.bestagini@polimi.it)
:author: Nicolò Bonettini (nicolo.bonettini@polimi.it)

Politecnico di Milano 2018

Small subsequent changes for handling PyTorch module prediction as denoising method were included by:

:author: Simone Alghisi (simone.alghisi-1@studenti.unitn.it)
:author: Samuele Bortolotti (samuele.bortolotti@studenti.unitn.it)
:author: Massimo Rizzoli (massimo.rizzoli@studenti.unitn.it)

Università di Trento 2021
"""

import numpy as np
import pywt
from numpy.fft import fft2, ifft2
from scipy.ndimage import filters
from sklearn.metrics import roc_curve, auc
from skimage.restoration import estimate_sigma
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable

class ArgumentError(Exception):
  """
  Argument error exception
  """
  pass

"""
Extraction functions
"""

def extract_single(im: np.ndarray,
                  levels: int = 4,
                  sigma: float = 5,
                  model: nn.Module = None,
                  device: str = 'cuda',
                  wdft_sigma: float = 0) -> np.ndarray:
  """
  Extract noise residual from a single image
  :param im: grayscale or color image, np.uint8
  :param levels: number of wavelet decomposition levels
  :param sigma: estimated noise power
  :param wdft_sigma: estimated DFT noise power
  :return: noise residual
  """
  W = noise_extract(im, levels, sigma, model, device)
  W = rgb2gray(W)
  W = zero_mean_total(W)
  W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
  W = wiener_dft(W, W_std).astype(np.float32)

  return W

def noise_extract(im: np.ndarray, levels: int = 4, sigma: float = 5, 
                  model: nn.Module = None, device: str = 'cuda') -> np.ndarray:
  """
  NoiseExtract as from Binghamton toolbox.
  If model is not None, the wavelet denoising method is employed, otherwise the noise
  will be extracted using the neural network model performing a prediction on the image.

  :param im: grayscale or color image, np.uint8
  :param levels: number of wavelet decomposition levels
  :param sigma: estimated noise power
  :param model: PyTorch neural network model
  :param device: model device either cuda or cpu
  :return: noise residual
  """

  assert (im.dtype == np.uint8)
  assert (im.ndim in [2, 3])

  im = im.astype(np.float32)

  if model:
    model = model.to(device)
    model.eval()

    # Normalise image
    im /= 255

    if sigma is None:
      # estimate sigmas for each image
      sigma = estimate_sigma(im, average_sigmas=True, multichannel=True)
    else:
      sigma /= 255

    # HxWxC to CxHxW
    im = np.transpose(im, (2, 0, 1))

    # Increase the size to make it compatible with a batch
    im = np.expand_dims(im, 0)

    im = Variable(torch.as_tensor(im, dtype=torch.float).to(device))

    sigma = np.asarray([sigma])
    sigma = Variable(torch.as_tensor(sigma, dtype=torch.float).to(device))

    # Predict noise and scale to 255
    W = model(im, sigma) * 255
    W = W.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
  else:
    noise_var = sigma ** 2

    if im.ndim == 2:
      im.shape += (1,)

    W = np.zeros(im.shape, np.float32)

    for ch in range(im.shape[2]):

      wlet = None
      while wlet is None and levels > 0:
        try:
          wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
        except ValueError:
          levels -= 1
          wlet = None
      if wlet is None:
        raise ValueError('Impossible to compute Wavelet filtering for input size: {}'.format(im.shape))

      wlet_details = wlet[1:]

      wlet_details_filter = [None] * len(wlet_details)

      # Cycle over Wavelet levels 1:levels-1
      
      for wlet_level_idx, wlet_level in enumerate(wlet_details):
        # Cycle over H,V,D components
        level_coeff_filt = [None] * 3
        for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
          level_coeff_filt[wlet_coeff_idx] = wiener_adaptive(wlet_coeff, noise_var)
        wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

      # Set filtered detail coefficients for Levels > 0 ---
      wlet[1:] = wlet_details_filter

      # Set to 0 all Level 0 approximation coefficients ---
      wlet[0][...] = 0

      # Invert wavelet transform ---
      wrec = pywt.waverec2(wlet, 'db4')
      try:
        W[:, :, ch] = wrec
      except ValueError:
        W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
        W[:, :, ch] = wrec

    W = W[:im.shape[0], :im.shape[1]]

  return W

def noise_extract_compact(args):
  """
  Extract residual, multiplied by the image. Useful to save memory in multiprocessing operations

  :param args: (im, levels, sigma, model, device), see noise_extract for usage
  :return: residual, multiplied by the image
  """
  w = noise_extract(*args)
  im = args[0]
  return (w * im / 255.).astype(np.float32)

def extract_multiple_aligned(imgs: list, model: nn.Module = None, device: str = 'cuda', 
                            levels: int = 4, sigma: float = 5, tqdm_str: str = '') -> np.ndarray:
  """
  Extract PRNU from a list of images. Images are supposed to be the same size and properly oriented.
  If model is not None, the wavelet denoising method is employed, otherwise the noise
  will be extracted using the neural network model performing a prediction on the image.

  :param imgs: list of images of size (H,W,Ch) and type np.uint8
  :param model: PyTorch model
  :param device: model device either cuda or cpu
  :param levels: number of wavelet decomposition levels
  :param sigma: estimated noise power
  :param tqdm_str: tqdm description (see tqdm documentation)
  :return: PRNU
  """
  assert (isinstance(imgs[0], np.ndarray))
  assert (imgs[0].ndim == 3)
  assert (imgs[0].dtype == np.uint8)

  h, w, ch = imgs[0].shape

  if sigma is None:
    print('\ncomputing sigma for each image\n')

  RPsum = np.zeros((h, w, ch), np.float32)
  NN = np.zeros((h, w, ch), np.float32)
  # Single process
  for im in tqdm(imgs, disable=tqdm_str is '', desc=tqdm_str, dynamic_ncols=True):
    RPsum += noise_extract_compact((im, levels, sigma, model, device))
    NN += (inten_scale(im) * saturation(im)) ** 2

  K = RPsum / (NN + 1)
  K = rgb2gray(K)
  K = zero_mean_total(K)
  K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)

  return K

def cut_ctr(array: np.ndarray, sizes: tuple) -> np.ndarray:
  """
  Cut a multi-dimensional array at its center, according to sizes

  :param array: multidimensional array
  :param sizes: tuple of the same length as array.ndim
  :return: multidimensional array, center cut
  """
  array = array.copy()
  if not (array.ndim == len(sizes)):
    raise ArgumentError('array.ndim must be equal to len(sizes)')
  for axis in range(array.ndim):
    axis_target_size = sizes[axis]
    axis_original_size = array.shape[axis]
    if axis_target_size > axis_original_size:
      raise ValueError(
        'Can\'t have target size {} for axis {} with original size {}'.format(axis_target_size, axis,
                                            axis_original_size))
    elif axis_target_size < axis_original_size:
      axis_start_idx = (axis_original_size - axis_target_size) // 2
      axis_end_idx = axis_start_idx + axis_target_size
      array = np.take(array, np.arange(axis_start_idx, axis_end_idx), axis)
  return array


def wiener_dft(im: np.ndarray, sigma: float) -> np.ndarray:
  """
  Adaptive Wiener filter applied to the 2D FFT of the image

  :param im: multidimensional array
  :param sigma: estimated noise power
  :return: filtered version of input im
  """
  noise_var = sigma ** 2
  h, w = im.shape

  im_noise_fft = fft2(im)
  im_noise_fft_mag = np.abs(im_noise_fft / (h * w) ** .5)

  im_noise_fft_mag_noise = wiener_adaptive(im_noise_fft_mag, noise_var)

  zeros_y, zeros_x = np.nonzero(im_noise_fft_mag == 0)

  im_noise_fft_mag[zeros_y, zeros_x] = 1
  im_noise_fft_mag_noise[zeros_y, zeros_x] = 0

  im_noise_fft_filt = im_noise_fft * im_noise_fft_mag_noise / im_noise_fft_mag
  im_noise_filt = np.real(ifft2(im_noise_fft_filt))

  return im_noise_filt.astype(np.float32)


def zero_mean(im: np.ndarray) -> np.ndarray:
  """
  ZeroMean called with the 'both' argument, as from Binghamton toolbox.

  :param im: multidimensional array
  :return: zero mean version of input im
  """
  # Adapt the shape ---
  if im.ndim == 2:
    im.shape += (1,)

  h, w, ch = im.shape

  # Subtract the 2D mean from each color channel ---
  ch_mean = im.mean(axis=0).mean(axis=0)
  ch_mean.shape = (1, 1, ch)
  i_zm = im - ch_mean

  # Compute the 1D mean along each row and each column, then subtract ---
  row_mean = i_zm.mean(axis=1)
  col_mean = i_zm.mean(axis=0)

  row_mean.shape = (h, 1, ch)
  col_mean.shape = (1, w, ch)

  i_zm_r = i_zm - row_mean
  i_zm_rc = i_zm_r - col_mean

  # Restore the shape ---
  if im.shape[2] == 1:
    i_zm_rc.shape = im.shape[:2]

  return i_zm_rc


def zero_mean_total(im: np.ndarray) -> np.ndarray:
  """
  ZeroMeanTotal as from Binghamton toolbox.

  :param im: multidimensional array
  :return: zero mean version of input im
  """
  im[0::2, 0::2] = zero_mean(im[0::2, 0::2])
  im[1::2, 0::2] = zero_mean(im[1::2, 0::2])
  im[0::2, 1::2] = zero_mean(im[0::2, 1::2])
  im[1::2, 1::2] = zero_mean(im[1::2, 1::2])
  return im


def rgb2gray(im: np.ndarray, multichannel: bool = True) -> np.ndarray:
  """
  RGB to gray as from Binghamton toolbox.

  :param im: multidimensional array
  :param multichannel: if false, return green channel as gray
  :return: grayscale version of input image
  """
  rgb2gray_vector = np.asarray([0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
  rgb2gray_vector.shape = (3, 1)

  if im.ndim == 2:
    im_gray = np.copy(im)
  elif im.shape[2] == 1:
    im_gray = np.copy(im[:, :, 0])
  elif im.shape[2] == 3:
    if multichannel:
      w, h = im.shape[:2]
      im = np.reshape(im, (w * h, 3))
      im_gray = np.dot(im, rgb2gray_vector)
      im_gray.shape = (w, h)
    else:
      im_gray = im[:, :, 1]
  else:
    raise ValueError('Input image must have 1 or 3 channels')

  return im_gray.astype(np.float32)

def threshold(wlet_coeff_energy_avg: np.ndarray, noise_var: float) -> np.ndarray:
  """
  Noise variance theshold as from Binghamton toolbox.

  :param wlet_coeff_energy_avg:
  :param noise_var:
  :return: noise variance threshold
  """
  res = wlet_coeff_energy_avg - noise_var
  return (res + np.abs(res)) / 2


def wiener_adaptive(x: np.ndarray, noise_var: float, **kwargs) -> np.ndarray:
  """
  WaveNoise as from Binghamton toolbox.
  Wiener adaptive flter aimed at extracting the noise component
  For each input pixel the average variance over a neighborhoods of different window sizes is first computed.
  The smaller average variance is taken into account when filtering according to Wiener.
  
  :param x: 2D matrix
  :param noise_var: Power spectral density of the noise we wish to extract (S)
  :return: wiener filtered version of input x
  """
  window_size_list = list(kwargs.pop('window_size_list', [3, 5, 7, 9]))

  energy = x ** 2

  avg_win_energy = np.zeros(x.shape + (len(window_size_list),))
  for window_idx, window_size in enumerate(window_size_list):
    avg_win_energy[:, :, window_idx] = filters.uniform_filter(energy,
                                  window_size,
                                  mode='constant')

  coef_var = threshold(avg_win_energy, noise_var)
  coef_var_min = np.min(coef_var, axis=2)

  x = x * noise_var / (coef_var_min + noise_var)

  return x


def inten_scale(im: np.ndarray) -> np.ndarray:
  """
  IntenScale as from Binghamton toolbox

  :param im: type np.uint8
  :return: intensity scaled version of input x
  """

  assert (im.dtype == np.uint8)

  T = 252
  v = 6
  out = np.exp(-1 * (im - T) ** 2 / v)
  out[im < T] = im[im < T] / T

  return out


def saturation(im: np.ndarray) -> np.ndarray:
  """
  Saturation as from Binghamton toolbox

  :param im: type np.uint8
  :return: saturation map from input im
  """
  assert (im.dtype == np.uint8)

  if im.ndim == 2:
    im.shape += (1,)

  h, w, ch = im.shape

  if im.max() < 250:
    return np.ones((h, w, ch))

  im_h = im - np.roll(im, (0, 1), (0, 1))
  im_v = im - np.roll(im, (1, 0), (0, 1))
  satur_map = \
    np.bitwise_not(
      np.bitwise_and(
        np.bitwise_and(
          np.bitwise_and(
            im_h != 0, im_v != 0
          ), np.roll(im_h, (0, -1), (0, 1)) != 0
        ), np.roll(im_v, (-1, 0), (0, 1)) != 0
      )
    )

  max_ch = im.max(axis=0).max(axis=0)

  for ch_idx, max_c in enumerate(max_ch):
    if max_c > 250:
      satur_map[:, :, ch_idx] = \
        np.bitwise_not(
          np.bitwise_and(
            im[:, :, ch_idx] == max_c, satur_map[:, :, ch_idx]
          )
        )

  return satur_map


def inten_sat_compact(args):
  """
  Memory saving version of inten_scale followed by saturation

  :param args:
  :return: intensity scale and saturation of input
  """
  im = args[0]
  return ((inten_scale(im) * saturation(im)) ** 2).astype(np.float32)


"""
Cross-correlation functions
"""


def crosscorr_2d(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
  """
  PRNU 2D cross-correlation

  :param k1: 2D matrix of size (h1,w1)
  :param k2: 2D matrix of size (h2,w2)
  :return: 2D matrix of size (max(h1,h2),max(w1,w2))
  """
  assert (k1.ndim == 2)
  assert (k2.ndim == 2)

  max_height = max(k1.shape[0], k2.shape[0])
  max_width = max(k1.shape[1], k2.shape[1])

  k1 -= k1.flatten().mean()
  k2 -= k2.flatten().mean()

  k1 = np.pad(k1, [(0, max_height - k1.shape[0]), (0, max_width - k1.shape[1])], mode='constant', constant_values=0)
  k2 = np.pad(k2, [(0, max_height - k2.shape[0]), (0, max_width - k2.shape[1])], mode='constant', constant_values=0)

  k1_fft = fft2(k1, )
  k2_fft = fft2(np.rot90(k2, 2), )

  return np.real(ifft2(k1_fft * k2_fft)).astype(np.float32)


def aligned_cc(k1: np.ndarray, k2: np.ndarray) -> dict:
  """
  Aligned PRNU cross-correlation

  :param k1: (n1,nk) or (n1,nk1,nk2,...)
  :param k2: (n2,nk) or (n2,nk1,nk2,...)
  :return: {'cc':(n1,n2) cross-correlation matrix, 'ncc':(n1,n2) normalized cross-correlation matrix}
  """

  # Type cast
  k1 = np.array(k1).astype(np.float32)
  k2 = np.array(k2).astype(np.float32)

  ndim1 = k1.ndim
  ndim2 = k2.ndim
  assert (ndim1 == ndim2)

  k1 = np.ascontiguousarray(k1).reshape(k1.shape[0], -1)
  k2 = np.ascontiguousarray(k2).reshape(k2.shape[0], -1)

  assert (k1.shape[1] == k2.shape[1])

  k1_norm = np.linalg.norm(k1, ord=2, axis=1, keepdims=True)
  k2_norm = np.linalg.norm(k2, ord=2, axis=1, keepdims=True)

  k2t = np.ascontiguousarray(k2.transpose())

  cc = np.matmul(k1, k2t).astype(np.float32)
  ncc = (cc / (k1_norm * k2_norm.transpose())).astype(np.float32)

  return {'cc': cc, 'ncc': ncc}


def pce(cc: np.ndarray, neigh_radius: int = 2) -> dict:
  """
  PCE position and value

  :param cc: as from crosscorr2d
  :param neigh_radius: radius around the peak to be ignored while computing floor energy
  :return: {'peak':(y,x), 'pce': peak to floor ratio, 'cc': cross-correlation value at peak position
  """
  assert (cc.ndim == 2)
  assert (isinstance(neigh_radius, int))

  out = dict()

  max_idx = np.argmax(cc.flatten())
  max_y, max_x = np.unravel_index(max_idx, cc.shape)

  peak_height = cc[max_y, max_x]

  cc_nopeaks = cc.copy()
  cc_nopeaks[max_y - neigh_radius:max_y + neigh_radius, max_x - neigh_radius:max_x + neigh_radius] = 0

  pce_energy = np.mean(cc_nopeaks.flatten() ** 2)

  out['peak'] = (max_y, max_x)
  out['pce'] = (peak_height ** 2) / pce_energy * np.sign(peak_height)
  out['cc'] = peak_height

  return out


"""
Statistical functions
"""


def stats(cc: np.ndarray, gt: np.ndarray, ) -> dict:
  """
  Compute statistics

  :param cc: cross-correlation or normalized cross-correlation matrix
  :param gt: boolean multidimensional array representing groundtruth
  :return: statistics dictionary
  """
  assert (cc.shape == gt.shape)
  assert (gt.dtype == np.bool)

  assert (cc.shape == gt.shape)
  assert (gt.dtype == np.bool)

  fpr, tpr, th = roc_curve(gt.flatten(), cc.flatten())
  auc_score = auc(fpr, tpr)

  # EER
  eer_idx = np.argmin((fpr - (1 - tpr)) ** 2, axis=0)
  eer = float(fpr[eer_idx])

  outdict = {
    'tpr': tpr,
    'fpr': fpr,
    'th': th,
    'auc': auc_score,
    'eer': eer,
  }

  return outdict


def gt(l1: list or np.ndarray, l2: list or np.ndarray) -> np.ndarray:
  """
  Determine the Ground Truth matrix given the labels

  :param l1: fingerprints labels
  :param l2: residuals labels
  :return: groundtruth matrix
  """
  l1 = np.array(l1)
  l2 = np.array(l2)

  assert (l1.ndim == 1)
  assert (l2.ndim == 1)

  gt_arr = np.zeros((len(l1), len(l2)), np.bool)

  for l1idx, l1sample in enumerate(l1):
    gt_arr[l1idx, l2 == l1sample] = True

  return gt_arr
