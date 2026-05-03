import healpy as hp
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

def S_inv_op(v, Cl_safe):
    return hp.almxfl(v, 1.0/Cl_safe)

def N_inv_op(v, N_inv_pix, nside, lmax):
    map_v = hp.alm2map(v, nside=nside, lmax=lmax)
    map_v = map_v * N_inv_pix
    map_v = np.nan_to_num(map_v)
    return hp.map2alm(map_v, lmax=lmax, iter=3)

def S_N_operator_real(x_real, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax):
    v = x_real[:alm_size_complex] + 1j * x_real[alm_size_complex:]
    result_complex = S_inv_op(v, Cl_safe) + N_inv_op(v, N_inv_pix, nside, lmax)
    return np.concatenate([result_complex.real, result_complex.imag])

def build_S_N_operator(alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax):
    def matvec(x_real):
        return S_N_operator_real(x_real, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax)
    return LinearOperator((alm_size, alm_size), matvec=matvec, dtype=np.float64)

def build_rhs(cmb_map, N_inv_pix, lmax):
    map_rhs = cmb_map * N_inv_pix
    b = hp.map2alm(map_rhs, lmax=lmax, iter=3)
    b_real = np.concatenate([b.real, b.imag])
    return b_real


# --- Multigrid Preconditioner Implementation ---
def multigrid_preconditioner_real(x_real, alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax, N_inv_mean, level=0, max_level=2):
    """
    Multigrid preconditioner for (S+N)^{-1} filtering.
    Recursively solves at coarser resolution, using diagonal preconditioner at the coarsest level.
    """
    if level >= max_level or nside <= 128 or lmax <= 200:
        # Coarsest level: use diagonal preconditioner
        M_ell = 1.0 / ( (1.0/Cl_safe) + N_inv_mean )
        v_complex = x_real[:alm_size_complex] + 1j * x_real[alm_size_complex:]
        result_complex = hp.almxfl(v_complex, M_ell)
        return np.concatenate([result_complex.real, result_complex.imag])
    else:
        # Coarsify: degrade nside and lmax, and N_inv_pix
        nside_coarse = nside // 2
        lmax_coarse = lmax // 2
        # Degrade N_inv_pix to lower nside
        N_inv_pix_coarse = hp.ud_grade(N_inv_pix, nside_coarse, order_in='RING', order_out='RING', power=-2)
        # Coarsify Cl_safe
        ell = np.arange(len(Cl_safe))
        Cl_safe_coarse = np.interp(np.arange(lmax_coarse+1), ell, Cl_safe[:lmax+1])
        N_inv_mean_coarse = np.mean(N_inv_pix_coarse)
        alm_size_complex_coarse = hp.Alm.getsize(lmax_coarse)
        alm_size_coarse = 2 * alm_size_complex_coarse
        # Restrict x_real to coarse grid (simple truncation)
        v_complex = x_real[:alm_size_complex] + 1j * x_real[alm_size_complex:]
        v_alm_coarse = hp.almxfl(v_complex, np.ones_like(Cl_safe))[:alm_size_complex_coarse]
        x_real_coarse = np.concatenate([v_alm_coarse.real, v_alm_coarse.imag])
        # Recursive call
        y_real_coarse = multigrid_preconditioner_real(
            x_real_coarse, alm_size_coarse, alm_size_complex_coarse,
            Cl_safe_coarse, N_inv_pix_coarse, nside_coarse, lmax_coarse, N_inv_mean_coarse,
            level=level+1, max_level=max_level
        )
        # Prolongate back to fine grid (zero pad)
        y_complex_coarse = y_real_coarse[:alm_size_complex_coarse] + 1j * y_real_coarse[alm_size_complex_coarse:]
        y_complex_fine = np.zeros(alm_size_complex, dtype=np.complex128)
        y_complex_fine[:alm_size_complex_coarse] = y_complex_coarse
        y_real_fine = np.concatenate([y_complex_fine.real, y_complex_fine.imag])
        return y_real_fine

def cg_solve_Cinv_a(b_real, alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax, N_inv_mean, maxiter=200, logprint=None):
    S_N_operator = build_S_N_operator(alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax)
    def preconditioner_real(x_real):
        return multigrid_preconditioner_real(
            x_real, alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax, N_inv_mean, level=0, max_level=2
        )
    M_operator = LinearOperator((alm_size, alm_size), matvec=preconditioner_real, dtype=np.float64)
    iteration_counter = {"count": 0}
    import time
    cg_start_time = time.time()
    def cg_callback(xk):
        iteration_counter["count"] += 1
        if logprint:
            elapsed = time.time() - cg_start_time
            logprint(f"[CG] Iter {iteration_counter['count']} | total time: {elapsed:.2f} sec")
    if logprint:
        logprint("Starting CG solve with multigrid preconditioner...")
    solution_real, info = cg(
        S_N_operator,
        b_real,
        M=M_operator,
        maxiter=maxiter,
        rtol=1e-6,
        atol=0,
        callback=cg_callback
    )
    solution_complex = solution_real[:alm_size_complex] + 1j * solution_real[alm_size_complex:]
    if logprint:
        cg_total_time = time.time() - cg_start_time
        logprint(f"CG finished in {iteration_counter['count']} iterations, {cg_total_time:.2f} seconds.")
    return solution_complex
