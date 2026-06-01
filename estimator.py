"""KSW-style local-fNL bispectrum estimator (Komatsu, Spergel & Wandelt 2005).

Symmetry-factor convention
--------------------------
The local bispectrum has three permutations of (l1, l2, l3) that collapse
to a single position-space form, contributing an overall prefactor of 3.
Both `cubic_term` and the per-sim cubic diagnostic in `linear_term_MC`
include this factor of 3.  The Fisher normalization N\mathcal{N}
N is not computed in this code; if it is added later it MUST use the same 
factor-of-3 convention so that fNL = E / N is unbiased. 
Do not change the prefactor in one function without changing it in all three.
"""
import numpy as np
import healpy as hp
from scipy.special import spherical_jn
from scipy.integrate import simpson
from filtering import build_rhs, cg_solve_Cinv_a

def compute_alpha_beta(LMAX, k_arr, Delta_interp, P_prim, r_arr):
    """Vectorized computation of alpha_l(r) and beta_l(r) kernels.

    For each r and l, integrate:
        alpha_l(r) = (2/pi) ∫ k² P_prim(k) Delta_l(k) j_l(kr) dk
        beta_l(r)  = (2/pi) ∫ k² Delta_l(k) j_l(kr) dk
    """
    N_r = len(r_arr)
    alpha = np.zeros((N_r, LMAX + 1))
    beta = np.zeros((N_r, LMAX + 1))

    # Precompute k^2 and P_prim * k^2 (used in every l, r iteration)
    k_sq = k_arr**2
    P_k_sq = P_prim * k_sq

    for i_r, r in enumerate(r_arr):
        x_arr = k_arr * r
        # Vectorize spherical Bessel evaluation over all l at once
        jl_vals = np.array([spherical_jn(l, x_arr) for l in range(LMAX + 1)])

        for l in range(2, LMAX + 1):
            jl = jl_vals[l]
            # Skip if Delta_interp[l, :] is all zero (optimization for sparse transfers)
            if np.allclose(Delta_interp[l, :], 0):
                alpha[i_r, l] = 0.0
                beta[i_r, l] = 0.0
                continue

            integrand_alpha = P_k_sq * Delta_interp[l, :] * jl
            integrand_beta = k_sq * Delta_interp[l, :] * jl
            alpha[i_r, l] = (2.0 / np.pi) * simpson(integrand_alpha, x=k_arr)
            beta[i_r, l] = (2.0 / np.pi) * simpson(integrand_beta, x=k_arr)

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
        # Factor of 3: the local bispectrum has three permutations of (l1,l2,l3)
        # that collapse to a single form A*B^2 in position space (KSW 2005 eq.17).
        # CONVENTION: this factor of 3 must be matched exactly by the Fisher
        # normalization N computed elsewhere.  fNL = E / N is unbiased only when
        # cubic_term, linear_term_MC, and N all use the same prefactor.
        angular_integral = 3.0 * np.sum(map_A * map_B**2) * d_Omega
        E_cubic_integrand[i_r] = r**2 * angular_integral
    E_cubic = simpson(E_cubic_integrand, x=r_arr)
    return E_cubic

def linear_term_MC(Cinv_a, alpha, beta, LMAX, r_arr, nside, Cl_theory, Cl_safe, var_pix, mask, N_sim=20, N_inv_pix=None, alm_size=None, alm_size_complex=None, N_inv_mean=None):
    # FILTERING CONTRACT: Cinv_a must be the result of the same CG-Wiener +
    # (1/Cl_safe) pipeline applied to the real data — i.e. the caller must have
    # already done hp.almxfl(Cinv_a_wiener, 1.0 / Cl_safe) before passing Cinv_a
    # here.  The simulation path applies those two steps explicitly below; data
    # and sims must be filtered identically or the linear-term subtraction is
    # biased.  If the caller does not guarantee this, apply the 1/Cl_safe step
    # to Cinv_a here before building alm_A_data / alm_B_data.
    import time
    N_r = len(r_arr)
    N_pix = hp.nside2npix(nside)
    d_Omega = 4.0 * np.pi / N_pix

    print(f"[LINEAR] Precomputing data alms and maps for {N_r} radial shells...")
    t_pre = time.time()
    # Precompute data pixel maps once so they are not recomputed inside the sim loop
    map_A_data = [None] * N_r
    map_B_data = [None] * N_r
    for i_r in range(N_r):
        al = alpha[i_r, :]
        bl = beta[i_r, :]
        map_A_data[i_r] = hp.alm2map(hp.almxfl(Cinv_a, al), nside=nside, lmax=LMAX)
        map_B_data[i_r] = hp.alm2map(hp.almxfl(Cinv_a, bl), nside=nside, lmax=LMAX)
    print(f"[LINEAR] Data maps precomputed in {time.time() - t_pre:.1f}s")

    E_linear_sims = np.zeros(N_sim)
    # E_cubic_sims_diag: pure-sim cubic term (factor-of-3 convention, matching
    # cubic_term).  NOTE: this is NOT the error bar on fNL.  The statistical
    # error requires the scatter of the full per-sim estimator
    # E_cubic(sim) - E_linear(sim), which needs a nested MC loop.  Use this
    # array for diagnostics only.
    E_cubic_sims_diag = np.zeros(N_sim)
    for i_sim in range(N_sim):
        print(f"[LINEAR] Starting simulation {i_sim+1}/{N_sim}...")
        t_sim = time.time()

        # Generate signal and noise in pixel space
        s_map = hp.synfast(Cl_theory, nside=nside, lmax=LMAX)
        n_map = np.random.normal(0, np.sqrt(var_pix), N_pix)

        # Create mock data and apply mask
        d_sim = (s_map + n_map) * mask

        # Build RHS and run CG solver (same as for data)
        b_real_sim = build_rhs(d_sim, N_inv_pix, LMAX)
        Cinv_a_wiener_sim, info = cg_solve_Cinv_a(
            b_real_sim, alm_size, alm_size_complex, Cl_safe,
            N_inv_pix, nside, LMAX, N_inv_mean, maxiter=200
        )

        # Apply S^{-1} to get C^{-1} filtered sim; matches the data filtering
        # contract declared above (caller passes Cinv_a = Wiener + 1/Cl_safe).
        sim_filtered = hp.almxfl(Cinv_a_wiener_sim, 1.0 / Cl_safe)

        print(f"[LINEAR] Sim {i_sim+1}: CG solved in {time.time() - t_sim:.1f}s")

        E_cross_integrand = np.zeros(N_r)
        E_cubic_sim_integrand = np.zeros(N_r)
        for i_r, r in enumerate(r_arr):
            al = alpha[i_r, :]
            bl = beta[i_r, :]
            alm_A_sim = hp.almxfl(sim_filtered, al)
            map_A_sim = hp.alm2map(alm_A_sim, nside=nside, lmax=LMAX)
            alm_B_sim = hp.almxfl(sim_filtered, bl)
            map_B_sim = hp.alm2map(alm_B_sim, nside=nside, lmax=LMAX)
            # Cross term encodes all three permutations via the 1 + 2 split;
            # no extra factor of 3 is needed here (it is already symmetrized).
            angular_cross = np.sum(
                map_A_data[i_r] * map_B_sim**2
                + 2.0 * map_A_sim * map_B_sim * map_B_data[i_r]
            ) * d_Omega
            E_cross_integrand[i_r] = r**2 * angular_cross

            # Pure-sim cubic diagnostic — factor of 3 matches cubic_term convention
            angular_cubic_sim = 3.0 * np.sum(map_A_sim * map_B_sim**2) * d_Omega
            E_cubic_sim_integrand[i_r] = r**2 * angular_cubic_sim

        E_linear_sims[i_sim] = simpson(E_cross_integrand, x=r_arr)
        E_cubic_sims_diag[i_sim] = simpson(E_cubic_sim_integrand, x=r_arr)
        print(f"[LINEAR] Sim {i_sim+1} completed in {time.time() - t_sim:.1f}s")
    E_linear = np.mean(E_linear_sims)
    print(f"[LINEAR] All {N_sim} sims done, E_linear = {E_linear:.6e}")
    return E_linear, E_cubic_sims_diag
