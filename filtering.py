import healpy as hp
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import time

# ------------------------------------------------
# Restriction and Prolongation (vectorized)
# ------------------------------------------------

def restrict_alm(v_complex, lmax_fine, lmax_coarse):
    """Vectorized: extract modes l <= lmax_coarse from fine alm array."""
    # Build (l, m) index arrays for coarse modes
    ls = np.concatenate([np.full(l + 1, l, dtype=int) for l in range(lmax_coarse + 1)])
    ms = np.concatenate([np.arange(l + 1, dtype=int) for l in range(lmax_coarse + 1)])
    
    idx_fine   = hp.Alm.getidx(lmax_fine,   ls, ms)
    idx_coarse = hp.Alm.getidx(lmax_coarse, ls, ms)
    
    alm_coarse = np.zeros(hp.Alm.getsize(lmax_coarse), dtype=complex)
    alm_coarse[idx_coarse] = v_complex[idx_fine]
    return alm_coarse


def prolongate_alm(x_coarse, lmax_coarse, lmax_fine):
    """Vectorized: embed coarse alm into fine alm, zero-pad high-l."""
    # Build (l, m) index arrays for coarse modes
    ls = np.concatenate([np.full(l + 1, l, dtype=int) for l in range(lmax_coarse + 1)])
    ms = np.concatenate([np.arange(l + 1, dtype=int) for l in range(lmax_coarse + 1)])
    
    idx_coarse = hp.Alm.getidx(lmax_coarse, ls, ms)
    idx_fine   = hp.Alm.getidx(lmax_fine,   ls, ms)
    
    alm_fine = np.zeros(hp.Alm.getsize(lmax_fine), dtype=complex)
    alm_fine[idx_fine] = x_coarse[idx_coarse]
    return alm_fine


# ------------------------------------------------
# Diagonal preconditioner (base case)
# ------------------------------------------------

def diagonal_precond_complex(v_complex, Cl_safe, N_inv_mean):
    M_ell = 1.0 / (1.0 / Cl_safe + N_inv_mean)
    return hp.almxfl(v_complex, M_ell)


# ------------------------------------------------
# Multigrid preconditioner (corrected)
# ------------------------------------------------

def multigrid_precond(v_complex, lmax_fine, nside_fine,
                      Cl_safe_fine, N_inv_pix_fine, N_inv_mean_fine,
                      level=0, max_level=2,
                      n_inner_iters=(5, 3),
                      logprint=None,
                      N_inv_cache=None):
    """
    Apply one level of the multigrid preconditioner.
    
    Parameters:
    -----------
    N_inv_cache : dict, optional
        Cache of precomputed downsampled noise maps to avoid recomputation.
        Keys are (nside, lmax) tuples; values are N_inv_pix arrays.

    For l <= lmax_coarse: use coarse CG solve (prolongated back)
    For l >  lmax_coarse: use diagonal preconditioner
    """
    if N_inv_cache is None:
        N_inv_cache = {}

    if level >= max_level or nside_fine <= 32 or lmax_fine <= 50:
        if logprint:
            logprint(f"[MG level {level}] base case diagonal, "
                     f"nside={nside_fine}, lmax={lmax_fine}")
        return diagonal_precond_complex(v_complex, Cl_safe_fine, N_inv_mean_fine)

    nside_coarse = nside_fine // 2
    lmax_coarse  = lmax_fine  // 2

    Cl_safe_coarse = Cl_safe_fine[:lmax_coarse + 1]
    
    # Check cache before downsampling (Bug 7 optimization)
    cache_key = (nside_coarse, lmax_coarse)
    if cache_key in N_inv_cache:
        N_inv_pix_coarse = N_inv_cache[cache_key]
    else:
        N_inv_pix_coarse = hp.ud_grade(N_inv_pix_fine, nside_coarse)
        N_inv_cache[cache_key] = N_inv_pix_coarse
    
    N_inv_mean_coarse = np.mean(N_inv_pix_coarse[N_inv_pix_coarse > 0])

    alm_size_complex_coarse = hp.Alm.getsize(lmax_coarse)
    alm_size_coarse = 2 * alm_size_complex_coarse

    if logprint:
        logprint(f"[MG level {level}] coarse solve: "
                 f"nside={nside_coarse}, lmax={lmax_coarse}, "
                 f"n_iters={n_inner_iters[min(level, len(n_inner_iters)-1)]}")

    v_coarse = restrict_alm(v_complex, lmax_fine, lmax_coarse)
    v_coarse_real = np.concatenate([v_coarse.real, v_coarse.imag])

    def coarse_op_real(x_real):
        x_c = x_real[:alm_size_complex_coarse] + 1j * x_real[alm_size_complex_coarse:]
        s_inv = hp.almxfl(x_c, 1.0 / Cl_safe_coarse)
        map_v = hp.alm2map(x_c, nside=nside_coarse, lmax=lmax_coarse)
        map_v = np.nan_to_num(map_v * N_inv_pix_coarse)
        n_inv = hp.map2alm(map_v, lmax=lmax_coarse, iter=3)
        result = s_inv + n_inv
        return np.concatenate([result.real, result.imag])

    coarse_op = LinearOperator(
        (alm_size_coarse, alm_size_coarse),
        matvec=coarse_op_real,
        dtype=np.float64
    )

    def coarse_precond_real(x_real):
        x_c = x_real[:alm_size_complex_coarse] + 1j * x_real[alm_size_complex_coarse:]
        result_c = multigrid_precond(
            x_c,
            lmax_fine       = lmax_coarse,
            nside_fine      = nside_coarse,
            Cl_safe_fine    = Cl_safe_coarse,
            N_inv_pix_fine  = N_inv_pix_coarse,
            N_inv_mean_fine = N_inv_mean_coarse,
            level           = level + 1,
            max_level       = max_level,
            n_inner_iters   = n_inner_iters,
            logprint        = logprint,
            N_inv_cache     = N_inv_cache
        )
        return np.concatenate([result_c.real, result_c.imag])

    M_coarse = LinearOperator(
        (alm_size_coarse, alm_size_coarse),
        matvec=coarse_precond_real,
        dtype=np.float64
    )

    n_iters = n_inner_iters[min(level, len(n_inner_iters)-1)]
    solution_coarse_real, info = cg(
        coarse_op,
        v_coarse_real,
        M=M_coarse,
        maxiter=n_iters,
        rtol=0.1,
        atol=0
    )

    x_coarse = (solution_coarse_real[:alm_size_complex_coarse]
              + 1j * solution_coarse_real[alm_size_complex_coarse:])

    y_fine = prolongate_alm(x_coarse, lmax_coarse, lmax_fine)

    # Vectorized high-l diagonal correction (l > lmax_coarse)
    M_ell_fine = 1.0 / (1.0 / Cl_safe_fine + N_inv_mean_fine)
    if lmax_fine > lmax_coarse:
        ls_high = np.concatenate([np.full(l + 1, l, dtype=int) for l in range(lmax_coarse + 1, lmax_fine + 1)])
        ms_high = np.concatenate([np.arange(l + 1, dtype=int) for l in range(lmax_coarse + 1, lmax_fine + 1)])
        idx_high = hp.Alm.getidx(lmax_fine, ls_high, ms_high)
        y_fine[idx_high] = v_complex[idx_high] * M_ell_fine[ls_high]

    return y_fine


# ------------------------------------------------
# Apply the S+N operator explicitly for diagnostics
# ------------------------------------------------

def apply_S_N_operator(v_complex, Cl_safe, N_inv_pix, nside, lmax):
    s_inv = hp.almxfl(v_complex, 1.0 / Cl_safe)
    map_v = hp.alm2map(v_complex, nside=nside, lmax=lmax)
    map_v = np.nan_to_num(map_v * N_inv_pix)
    n_inv = hp.map2alm(map_v, lmax=lmax, iter=3)
    return s_inv + n_inv


# ------------------------------------------------
# Main CG solve using the corrected multigrid preconditioner
# ------------------------------------------------

def cg_solve_Cinv_a(b_real, alm_size, alm_size_complex,
                    Cl_safe, N_inv_pix, nside, lmax,
                    N_inv_mean, maxiter=200, logprint=None):

    def S_N_op_real(x_real):
        v = x_real[:alm_size_complex] + 1j * x_real[alm_size_complex:]
        s_inv = hp.almxfl(v, 1.0 / Cl_safe)
        map_v = hp.alm2map(v, nside=nside, lmax=lmax)
        map_v = np.nan_to_num(map_v * N_inv_pix)
        n_inv = hp.map2alm(map_v, lmax=lmax, iter=3)
        result = s_inv + n_inv
        return np.concatenate([result.real, result.imag])

    S_N_operator = LinearOperator(
        (alm_size, alm_size),
        matvec=S_N_op_real,
        dtype=np.float64
    )

    def precond_real(x_real):
        v_c = x_real[:alm_size_complex] + 1j * x_real[alm_size_complex:]
        result = multigrid_precond(
            v_c,
            lmax_fine       = lmax,
            nside_fine      = nside,
            Cl_safe_fine    = Cl_safe,
            N_inv_pix_fine  = N_inv_pix,
            N_inv_mean_fine = N_inv_mean,
            level           = 0,
            max_level       = 2,
            n_inner_iters   = (5, 3),
            logprint        = logprint,
            N_inv_cache     = N_inv_cache_dict
        )
        return np.concatenate([result.real, result.imag])

    # Precompute downsampled noise maps for multigrid levels (Bug 7 fix)
    N_inv_cache_dict = {}
    nside_temp = nside
    lmax_temp = lmax
    while nside_temp > 32 and lmax_temp > 50:
        nside_temp = nside_temp // 2
        lmax_temp = lmax_temp // 2
        N_inv_cache_dict[(nside_temp, lmax_temp)] = hp.ud_grade(N_inv_pix, nside_temp)
    
    M_operator = LinearOperator(
        (alm_size, alm_size),
        matvec=precond_real,
        dtype=np.float64
    )

    iteration_counter = {"count": 0}
    cg_start = time.time()

    def callback(xk):
        iteration_counter["count"] += 1
        if logprint:
            elapsed = time.time() - cg_start
            logprint(f"[CG] Iter {iteration_counter['count']} | "
                     f"time: {elapsed:.2f}s")

    if logprint:
        logprint("Starting CG with corrected multigrid preconditioner...")

    solution_real, info = cg(
        S_N_operator,
        b_real,
        M=M_operator,
        maxiter=maxiter,
        rtol=1e-6,
        atol=0,
        callback=callback
    )

    solution_complex = (solution_real[:alm_size_complex]
                      + 1j * solution_real[alm_size_complex:])

    if logprint:
        total = time.time() - cg_start
        logprint(f"CG finished: {iteration_counter['count']} iters, "
                 f"info={info}, time={total:.2f}s")

    return solution_complex, info


def build_rhs(cmb_map, N_inv_pix, lmax):
    map_rhs = cmb_map * N_inv_pix
    b = hp.map2alm(map_rhs, lmax=lmax, iter=3)
    b_real = np.concatenate([b.real, b.imag])
    return b_real
