import healpy as hp
import numpy as np
import os
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
import time
from tqdm import tqdm

NSIDE=2048
LMIN=2
LMAX=3000

def generate_map(cls, sigma, lmax=LMAX, nside=NSIDE): #need to generate many of these and get their estimator for errorbars
    # Draw a Gaussian CMB sky from S (Cl)
    alm_sig = hp.synalm(cls[:lmax+1], lmax=lmax)
    # Make signal map
    s_map = hp.alm2map(alm_sig, nside=nside, lmax=lmax, verbose=False)
    # Draw pixel white noise from N
    n_map = np.random.normal(loc=0.0, scale=sigma, size=s_map.size)
    # Observed map (apply mask)
    return (s_map + n_map) * mask


def N_inv_op(v, N_inv_pix, nside=NSIDE, lmax=LMAX): #operator N^-1 acts on vector v in harmonic space
    Av = hp.alm2map(v, nside=nside, lmax=lmax, pol=False, verbose=False) # A=alm2map, this is A v
    return hp.map2alm(Av* N_inv_pix, lmax=lmax, iter=3, pol=False, verbose=False) # A^T=map2alm, this is A^T N A v

def S_inv_op(v, S_inv_harm): #operator S^-1 acts on vector v in harmonic space
    return S_inv_harm * v # S^-1 is diagonal

# ----------------------------
# Load data
# ----------------------------

cmb_filename = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/data/COM_CMB_IQU-commander_2048_R3.00_full.fits")
mask_filename = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/data/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")
mask = hp.read_map(mask_filename)
cmb_map = hp.read_map(cmb_filename)
map_masked = hp.ma(cmb_map)
map_masked.mask = np.logical_not(mask)

# ----------------------------
# Theoretical Cl (replace with CAMB or Planck best-fit)
# ----------------------------

Cl_theory = cls_theory[:LMAX+1]   #Theory, not measured

# get Cl from map
cls_meas_frommap = hp.anafast(map_masked, lmax=LMAX, use_pixel_weights=True)
cls_meas_frommap[:LMIN] = 0.0


#Define S^-1 matrix: 1/Cl
a_lm = hp.map2alm(map_masked, lmax=LMAX, iter=3, use_pixel_weights=True, verbose=False)
invCl = np.zeros(LMAX+1)
invCl[2:] = 1.0 / cls_meas_frommap[LMIN:LMAX+1]   # remove l=0,1
S_inv_harm = hp.almxfl(np.ones_like(a_lm), invCl)


# define N^-1 matrix with gaussian noise in pixel space around 20-30uK
sigma = 25e-6               # K_CMB (since you used e-6)
var_pix = sigma**2          # variance, constant
N_inv_pix = mask / var_pix          # this is N^{-1}_pix diagonal weights

#act on data N^-1 a
N_inv_a = N_inv_op(a_lm, N_inv_pix, nside=NSIDE, lmax=LMAX)


# create S^-1 + N^-1 operator to use in iterative gradient inversion
def S_inv_plus_N_inv(v):
    return S_inv_op(v, S_inv_harm) + N_inv_op(v, N_inv_pix, nside=NSIDE, lmax=LMAX)
    
S_N_operator = LinearOperator((len(S_inv_harm),len(S_inv_harm)), matvec=S_inv_plus_N_inv)


# generate sky map from S and N distribution