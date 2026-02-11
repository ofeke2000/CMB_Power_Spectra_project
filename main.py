import healpy as hp
import numpy as np
import os
import astropy.units as u
import matplotlib.pyplot as plt
#matplotlib inline


#=====================================
# configuration
#=====================================

output_dir = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/output")
os.makedirs(output_dir, exist_ok=True)

#=====================================
# Load the CMB map
#=====================================

cmb_filename = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/data/COM_CMB_IQU-commander_2048_R3.00_full.fits")

cmb_map = hp.read_map(
    cmb_filename,
    field=0,          # temperature map (I)
    dtype=np.float64  # avoid dtype warnings
)

hp.mollview(cmb_map, min=-1e-3, max=1e-3, title="CMB only temperature map", unit="K")
plt.savefig(os.path.join(output_dir, "cmb_map.png"), dpi=300, bbox_inches='tight')
plt.clf()

#=====================================
# import mask
#=====================================

mask_filename = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/data/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")

mask = hp.read_map(mask_filename, dtype=np.float64)
cmb_map_masked = hp.ma(cmb_map)
cmb_map_masked.mask = np.logical_not(mask)

hp.mollview(cmb_map_masked, min=-1e-3, max=1e-3)
plt.savefig(os.path.join(output_dir, "cmb_map_masked.png"), dpi=300, bbox_inches='tight')
plt.clf()

#=====================================
# load published power spectrum
#=====================================

cmb_binned_spectrum_filename = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/data/COM_PowerSpect_CMB-TT-binned_R3.01.txt")  

cmb_binned_spectrum = np.loadtxt(cmb_binned_spectrum_filename)

ell_planck = cmb_binned_spectrum[:, 0]
Dl_planck  = cmb_binned_spectrum[:, 1]

#=====================================
# compute power spectrum
#=====================================

lmax = 3000
test_cls_meas_frommap = hp.anafast(cmb_map_masked, lmax=lmax, use_pixel_weights=True)
ll = np.arange(lmax+1)
sky_fraction = len(cmb_map_masked.compressed()) / len(cmb_map_masked)
print(f"The map covers {sky_fraction:.1%} of the sky")
plt.style.use("seaborn-v0_8-poster")
k2muK = 1e6

plt.plot(cmb_binned_spectrum[:,0], cmb_binned_spectrum[:,1], '--', alpha=1, label='Planck 2018 PS release')
plt.plot(ll, ll*(ll+1.)*test_cls_meas_frommap*k2muK**2/2./np.pi / sky_fraction, '--', alpha=0.6, label='Planck 2018 PS from Data Map')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell~[\mu K^2]$')
plt.grid()
plt.legend(loc='best')
plt.savefig(os.path.join(output_dir, "power_spectrum.png"), dpi=300, bbox_inches='tight')
plt.clf()

#=====================================
# correction for the beam
#=====================================

w_ell = hp.gauss_beam((5*u.arcmin).to_value(u.radian), lmax=lmax)

plt.plot(cmb_binned_spectrum[:,0], cmb_binned_spectrum[:,1], '--', alpha=1, label='Planck 2018 PS release')
plt.plot(ll, ll*(ll+1.)*test_cls_meas_frommap*k2muK**2/2./np.pi / sky_fraction / w_ell**2,
         alpha=0.6, label='Planck 2018 PS from Data Map (beam corrected)')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell~[\mu K^2]$')
plt.grid()
plt.legend(loc='best');
plt.savefig(os.path.join(output_dir, "power_spectrum_beam_corrected.png"), dpi=300, bbox_inches='tight')
plt.clf()
