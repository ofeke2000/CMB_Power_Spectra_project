import numpy as np
import healpy as hp
from scipy.special import spherical_jn
from scipy.integrate import simpson

def compute_alpha_beta(LMAX, k_arr, Delta_interp, P_prim, r_arr):
    N_r = len(r_arr)
    N_k = len(k_arr)
    alpha = np.zeros((N_r, LMAX + 1))
    beta = np.zeros((N_r, LMAX + 1))
    for i_r, r in enumerate(r_arr):
        x_arr = k_arr * r
        for l in range(2, LMAX + 1):
            jl = spherical_jn(l, x_arr)
            integrand_alpha = k_arr**2 * P_prim * Delta_interp[l, :] * jl
            integrand_beta  = k_arr**2 *          Delta_interp[l, :] * jl
            alpha[i_r, l] = (2.0 / np.pi) * simpson(integrand_alpha, x=k_arr)
            beta[i_r, l]  = (2.0 / np.pi) * simpson(integrand_beta,  x=k_arr)
    return alpha, beta

def cubic_term(Cinv_a, alpha, beta, LMAX, r_arr, nside):
    N_r = len(r_arr)
    N_pix = hp.nside2npix(nside)
    d_Omega = 4.0 * np.pi / N_pix
    E_cubic_integrand = np.zeros(N_r)
    for i_r, r in enumerate(r_arr):
        al = alpha[i_r, :]
        bl = beta[i_r,  :]
        alm_A = hp.almxfl(Cinv_a, al)
        map_A = hp.alm2map(alm_A, nside=nside, lmax=LMAX)
        alm_B = hp.almxfl(Cinv_a, bl)
        map_B = hp.alm2map(alm_B, nside=nside, lmax=LMAX)
        angular_integral = np.sum(map_A**2 * map_B) * d_Omega
        E_cubic_integrand[i_r] = r**2 * angular_integral
    E_cubic = simpson(E_cubic_integrand, x=r_arr)
    return E_cubic

def linear_term_MC(Cinv_a, alpha, beta, LMAX, r_arr, nside, Cl_theory, Cl_safe, var_pix, mask, N_sim=20):
    N_r = len(r_arr)
    N_pix = hp.nside2npix(nside)
    d_Omega = 4.0 * np.pi / N_pix
    E_linear_sims = np.zeros(N_sim)
    for i_sim in range(N_sim):
        sim_alm = hp.synalm(Cl_theory, lmax=LMAX, new=True)
        Cl_noise_approx = np.full(LMAX + 1, var_pix / np.mean(mask))
        Cl_signal = Cl_theory.copy()
        wiener_l = np.zeros(LMAX + 1)
        for l in range(2, LMAX + 1):
            S = Cl_signal[l] if l < len(Cl_signal) else 0.0
            N = Cl_noise_approx[l]
            wiener_l[l] = S / (S + N) if (S + N) > 0 else 0.0
        sim_filtered = hp.almxfl(sim_alm, wiener_l / np.maximum(Cl_safe, 1e-30))
        E_cross_integrand = np.zeros(N_r)
        for i_r, r in enumerate(r_arr):
            al = alpha[i_r, :]
            bl = beta[i_r,  :]
            alm_A_sim = hp.almxfl(sim_filtered, al)
            map_A_sim  = hp.alm2map(alm_A_sim, nside=nside, lmax=LMAX)
            alm_B_data = hp.almxfl(Cinv_a, bl)
            map_B_data  = hp.alm2map(alm_B_data, nside=nside, lmax=LMAX)
            angular_cross = np.sum(map_A_sim**2 * map_B_data) * d_Omega
            E_cross_integrand[i_r] = r**2 * angular_cross
        E_linear_sims[i_sim] = simpson(E_cross_integrand, x=r_arr)
    E_linear = 3.0 * np.mean(E_linear_sims)
    return E_linear
