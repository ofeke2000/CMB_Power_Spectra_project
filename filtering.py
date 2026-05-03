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

def cg_solve_Cinv_a(b_real, alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax, N_inv_mean, maxiter=200, logprint=None):
    S_N_operator = build_S_N_operator(alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax)
    M_ell = 1.0 / ( (1.0/Cl_safe) + N_inv_mean )
    def preconditioner_real(x_real):
        v_complex = x_real[:alm_size_complex] + 1j * x_real[alm_size_complex:]
        result_complex = hp.almxfl(v_complex, M_ell)
        return np.concatenate([result_complex.real, result_complex.imag])
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
        logprint("Starting CG solve...")
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
