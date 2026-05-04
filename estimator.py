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
        # Correct: B-weighted map is squared (integral of A_alpha * A_beta^2)
        angular_integral = np.sum(map_A * map_B**2) * d_Omega
        E_cubic_integrand[i_r] = r**2 * angular_integral
    E_cubic = simpson(E_cubic_integrand, x=r_arr)
    return E_cubic

def linear_term_MC(Cinv_a, alpha, beta, LMAX, r_arr, nside, Cl_theory, Cl_safe, var_pix, mask, N_sim=20, N_inv_pix=None, alm_size=None, alm_size_complex=None, N_inv_mean=None):
    import time
    N_r = len(r_arr)
    N_pix = hp.nside2npix(nside)
    d_Omega = 4.0 * np.pi / N_pix
    
    print(f"[LINEAR] Precomputing data alms for {N_r} radial shells...")
    t_pre = time.time()
    # Precompute the data-weighted alms for each radial shell (memory efficient)
    alm_A_data = [None] * N_r
    alm_B_data = [None] * N_r
    for i_r in range(N_r):
        al = alpha[i_r, :]
        bl = beta[i_r, :]
        alm_A_data[i_r] = hp.almxfl(Cinv_a, al)
        alm_B_data[i_r] = hp.almxfl(Cinv_a, bl)
    print(f"[LINEAR] Data alms precomputed in {time.time() - t_pre:.1f}s")

    E_linear_sims = np.zeros(N_sim)
    E_cubic_sims = np.zeros(N_sim)  # Array of cubic terms for each sim (for error calculation)
    for i_sim in range(N_sim):
        print(f"[LINEAR] Starting simulation {i_sim+1}/{N_sim}...")
        t_sim = time.time()
        
        # Generate signal and noise in pixel space
        s_map = hp.synfast(Cl_theory, nside=nside, lmax=LMAX, verbose=False)
        n_map = np.random.normal(0, np.sqrt(var_pix), N_pix)
        
        # Create mock data and apply mask
        d_sim = (s_map + n_map) * mask
        
        # Build RHS and run CG solver (same as for data)
        b_real_sim = build_rhs(d_sim, N_inv_pix, LMAX)
        Cinv_a_wiener_sim, info = cg_solve_Cinv_a(
            b_real_sim, alm_size, alm_size_complex, Cl_safe, 
            N_inv_pix, nside, LMAX, N_inv_mean, maxiter=200
        )
        
        # Apply S^{-1} to get C^{-1} filtered sim
        sim_filtered = hp.almxfl(Cinv_a_wiener_sim, 1.0 / Cl_safe)
        
        print(f"[LINEAR] Sim {i_sim+1}: CG solved in {time.time() - t_sim:.1f}s")
        
        E_cross_integrand = np.zeros(N_r)
        E_cubic_sim_integrand = np.zeros(N_r)  # Pure simulation cubic term for error
        for i_r, r in enumerate(r_arr):
            al = alpha[i_r, :]
            bl = beta[i_r, :]
            alm_A_sim = hp.almxfl(sim_filtered, al)
            map_A_sim = hp.alm2map(alm_A_sim, nside=nside, lmax=LMAX)
            alm_B_sim = hp.almxfl(sim_filtered, bl)
            map_B_sim = hp.alm2map(alm_B_sim, nside=nside, lmax=LMAX)
            map_A_data_i = hp.alm2map(alm_A_data[i_r], nside=nside, lmax=LMAX)
            map_B_data_i = hp.alm2map(alm_B_data[i_r], nside=nside, lmax=LMAX)
            angular_cross = np.sum(
                map_A_data_i * map_B_sim**2
                + 2.0 * map_A_sim * map_B_sim * map_B_data_i
            ) * d_Omega
            E_cross_integrand[i_r] = r**2 * angular_cross
            
            # Pure simulation cubic term (for error): A_sim * B_sim^2
            angular_cubic_sim = np.sum(map_A_sim * map_B_sim**2) * d_Omega
            E_cubic_sim_integrand[i_r] = r**2 * angular_cubic_sim
        
        E_linear_sims[i_sim] = simpson(E_cross_integrand, x=r_arr)
        E_cubic_sims[i_sim] = simpson(E_cubic_sim_integrand, x=r_arr)
        print(f"[LINEAR] Sim {i_sim+1} completed in {time.time() - t_sim:.1f}s")
    E_linear = np.mean(E_linear_sims)
    print(f"[LINEAR] All {N_sim} sims done, E_linear = {E_linear:.6e}")
    return E_linear, E_cubic_sims
