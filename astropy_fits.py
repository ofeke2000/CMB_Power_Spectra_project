from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os

cmb_filename = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/data/COM_CMB_IQU-commander_2048_R3.00_full.fits")

f = fits.open(cmb_filename)

print(f[1].header)