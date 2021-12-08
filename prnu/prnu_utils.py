# -*- coding: UTF-8 -*-
"""

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

import os
from glob import glob
from torch import nn
import numpy as np
from PIL import Image
from skimage.restoration import estimate_sigma
from tqdm import tqdm
from typing import Dict, Any
import prnu

def prnu_extract(flat_image_dataset: str, model: nn.Module = None, device: str = 'cuda', \
                 gray: bool = False, cut_dim: tuple = (512, 512, 3), \
                 mean_sigma: bool = False, sigma: float = None) -> Dict[str, np.ndarray]:
  """
  Extract the PRNU from a flat image dataset either with a PyTorch module for noise extraction
  or wavelet. If model is specified the denoising neural network will be employed, otherwise the PRNU will
  be estimated using wavelet denoising.

  :param flat_image_dataset: flat grayscale or color images dataset path
  :param model: PyTorch denoising neural network model
  :param device: execute on cpu ('cpu') or on gpu ('cuda')
  :param gray: grayscale image
  :param cut_dim: size of the image patches for estimating the camera PRNU
  :param mean_sigma: use the flat_image_dataset mean sigma for each noise extraction
  :param sigma: noise sigma level
  :return: dictionary which associate each camera model with the curresponding estimated PRNU
  """

  assert (len(cut_dim) == 3), 'cut_dim should have 3 dimensions (H,W,Ch)'

  flat_image_dataset = flat_image_dataset[:-1] if flat_image_dataset.endswith('\\') else flat_image_dataset

  ff_dirlist = np.array(sorted(glob(''.join([flat_image_dataset, '/*.jpg']))))
  ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])

  assert (ff_device.size > 0), 'Empty flat image dataset'

  print('Computing fingerprints')
  fingerprint_device = sorted(np.unique(ff_device))

  height, width = cut_dim[:2]
  k = np.zeros((len(fingerprint_device), height, width), dtype=np.float32)
  for i, camera_device in enumerate(fingerprint_device):
    if mean_sigma:
      sigma = 0
    imgs = []
    for img_path in tqdm(ff_dirlist[ff_device == camera_device], desc='Cutting images {}'.format(i)):
      im = Image.open(img_path)
      im_arr = np.asarray(im)
      if im_arr.dtype != np.uint8:
        print('Error while reading image: {}'.format(img_path))
        continue
      if im_arr.ndim != 3:
        print('Image is not RGB: {}'.format(img_path))
        continue
      im_cut = prnu.cut_ctr(im_arr, cut_dim)
      if gray:
        im_cut = prnu.rgb2gray(im_cut, multichannel=False).astype(np.uint8)
        im_cut = np.expand_dims(im_cut, -1)
      if mean_sigma:
        sigma += estimate_sigma(im_cut, average_sigmas=True, multichannel=True)
      imgs.extend([im_cut])
    if mean_sigma:
      sigma /= len(imgs)
    k[i, :, :] = prnu.extract_multiple_aligned(imgs=imgs, model=model, device=device,
                                              tqdm_str='Calculating Noiseprints {}'.format(i), \
                                              sigma=sigma)
  # create a dictionary associating each fingerprint id with the corresponding estimated prnu
  return dict(zip(fingerprint_device, k))

def prnu_test(nat_image_dataset: str, k: np.array, fingerprint_device: np.array, \
              model: nn.Module = None, device: str = 'cuda', gray: bool = False, \
              cut_dim: tuple = (512, 512, 3), mean_sigma: bool = False, \
              sigma: float = None) -> Dict[str, Dict[str, Any]]:
  """
  Testing the PRNU using a natural image dataset and using the ROC Curve and the cross correlation to estimate the performance.
  If model is specified the denoising neural network will be employed, otherwise the noise of the images will
  be estimated using wavelet denoising.

  :param nat_image_dataset: natural grayscale or color images dataset path
  :param k: cameras PRNU
  :param fingerprint_device: fingerprint devices (must match the corresponding PRNU in the k parameter array)
  :param model: PyTorch denoising neural network model
  :param device: model device, either cuda or cpu
  :param gray: grayscale images
  :param cut_dim: size of the image patches for estimating the camera PRNU
  :param mean_sigma: use the flat_image_dataset mean sigma for each noise extraction
  :param sigma: noise sigma level
  :return: dictionary containing the ROC curve and Cross correlation statistics
  """

  assert (len(cut_dim) == 3), 'cut_dim should have 3 dimensions (H,W,Ch)'

  nat_dirlist = np.array(sorted(glob(''.join([nat_image_dataset, '/*.jpg']))))
  nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])

  nat_image_dataset = nat_image_dataset[:-1] if nat_image_dataset.endswith('\\') else nat_image_dataset

  assert (nat_device.size > 0), 'Empty nat image dataset'

  print('Computing residuals')

  if mean_sigma:
    sigma = 0

  imgs = []
  for img in tqdm(nat_dirlist, desc='Cutting images'):
    im_cut = prnu.cut_ctr(np.asarray(Image.open(img)), cut_dim)
    if gray:
      im_cut = prnu.rgb2gray(im_cut, multichannel=False).astype(np.uint8)
      im_cut = np.expand_dims(im_cut, -1)
    if mean_sigma:
      sigma += estimate_sigma(im_cut, average_sigmas=True, multichannel=True)
    imgs += [im_cut]

  if mean_sigma:
    sigma /= len(imgs)

  height, width = cut_dim[:2]
  w = np.zeros((len(imgs), height, width), dtype=np.float32)
  if sigma is None:
    print('\ncomputing sigma for each image\n')
  for i, im in enumerate(tqdm(imgs, desc='Extracting noise from nat')):
    w[i, :, :] = prnu.extract_single(im, model=model, device=device, sigma=sigma)

  # Computing Ground Truth
  gt = prnu.gt(fingerprint_device, nat_device)

  print('Computing cross correlation')
  cc_aligned_rot = prnu.aligned_cc(k, w)['cc']

  print('Computing statistics cross correlation')
  stats_cc = prnu.stats(cc_aligned_rot, gt)

  print('Computing PCE')
  pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

  for fingerprint_idx, fingerprint_k in enumerate(k):
    for natural_idx, natural_w in enumerate(w):
      cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
      pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']

  print('Computing statistics on PCE')
  stats_pce = prnu.stats(pce_rot, gt)

  print('AUC on CC {:.2f}'.format(stats_cc['auc']))
  print('AUC on PCE {:.2f}'.format(stats_pce['auc']))

  return { 'cc': stats_cc, 'pce': stats_pce }
