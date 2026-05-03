import camb
import numpy as np

def get_theory_Cl(lmax):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    cls_theory = powers['total'][:,0]
    ells = np.arange(len(cls_theory))
    Dl = cls_theory.copy()
    Cl = np.zeros_like(Dl)
    factor = ells * (ells + 1) / (2 * np.pi)
    nonzero = factor > 0
    Cl[nonzero] = Dl[nonzero] / factor[nonzero]
    Cl_theory = Cl[:lmax+1]
    epsilon = 1e-12 * np.max(Cl_theory)
    Cl_safe = Cl_theory + epsilon
    return Cl_theory, Cl_safe
