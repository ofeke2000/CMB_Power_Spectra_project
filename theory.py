import camb
import numpy as np
from scipy.interpolate import interp1d


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
    
    # Add regularization to avoid S^{-1} -> infinity for l=0,1
    # (these modes should be marginalized out per Smith+2007 §A.3)
    Cl_safe = np.copy(Cl_theory)
    Cl_safe[0] = 1e30
    Cl_safe[1] = 1e30
    
    return Cl_theory, Cl_safe


def get_transfer_functions(lmax, N_k=200, k_min=1e-4, k_max=0.5):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTransfer = True
    pars.WantCls = True

    results_transfer = camb.get_results(pars)
    transfer_data = results_transfer.get_cmb_transfer_data()

    k_camb = None
    if hasattr(transfer_data, 'q'):
        k_camb = np.asarray(transfer_data.q)
    elif hasattr(transfer_data, 'k'):
        k_camb = np.asarray(transfer_data.k)
    else:
        raise AttributeError("CAMB transfer data does not have 'q' or 'k' attribute")

    l_vals = None
    if hasattr(transfer_data, 'l'):
        l_vals = np.asarray(transfer_data.l)
    elif hasattr(transfer_data, 'L'):
        l_vals = np.asarray(transfer_data.L)
    else:
        raise AttributeError("CAMB transfer data does not have 'l' or 'L' attribute")

    delta_raw = transfer_data.delta_p_l_k[0]

    if delta_raw.ndim == 2:
        if delta_raw.shape == (len(l_vals), len(k_camb)):
            Delta_lk = delta_raw
        elif delta_raw.shape == (len(k_camb), len(l_vals)):
            Delta_lk = delta_raw.T
        else:
            raise ValueError(f"Unexpected transfer delta shape {delta_raw.shape}")
    elif delta_raw.ndim == 3:
        delta0 = delta_raw[0]
        if delta0.shape == (len(l_vals), len(k_camb)):
            Delta_lk = delta0
        elif delta0.shape == (len(k_camb), len(l_vals)):
            Delta_lk = delta0.T
        else:
            raise ValueError(f"Unexpected transfer delta shape {delta_raw.shape}")
    else:
        raise ValueError(f"Unexpected transfer delta ndim {delta_raw.ndim}")

    k_arr = np.geomspace(k_min, k_max, N_k)
    Delta_interp = np.zeros((lmax+1, N_k))

    # First: fill sparse CAMB l-values (interpolate in k)
    l_vals_int = np.array([int(l) for l in l_vals if int(l) <= lmax])
    l_vals_set = set(l_vals_int.tolist())
    for li, l in enumerate(l_vals):
        l_int = int(l)
        if l_int <= lmax:
            f_interp = interp1d(k_camb, Delta_lk[li, :], kind='linear', bounds_error=False, fill_value=0.0)
            Delta_interp[l_int, :] = f_interp(k_arr)

    # Second: interpolate over l to fill ALL integer l values
    for l in range(2, lmax + 1):
        if l not in l_vals_set:
            # Find neighbouring CAMB l-values and interpolate
            idx = np.searchsorted(l_vals_int, l)
            idx_lo = max(idx - 1, 0)
            idx_hi = min(idx, len(l_vals_int) - 1)
            l_lo = l_vals_int[idx_lo]
            l_hi = l_vals_int[idx_hi]
            
            if l_lo == l_hi:
                # No neighbours, use the same value
                Delta_interp[l, :] = Delta_interp[l_lo, :]
            else:
                # Linear interpolation in l
                w = (l - l_lo) / (l_hi - l_lo)
                Delta_interp[l, :] = ((1.0 - w) * Delta_interp[l_lo, :] +
                                      w * Delta_interp[l_hi, :])

    A_s = 2e-9
    n_s = 0.965
    k_pivot = 0.05
    P_prim = A_s * (k_arr / k_pivot)**(n_s - 1)

    return k_arr, P_prim, Delta_interp, l_vals, k_camb, delta_raw.shape
