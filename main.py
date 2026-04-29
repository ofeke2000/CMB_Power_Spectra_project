
# === Imports ===
import healpy as hp
import numpy as np
import os
import time
import camb
from scipy.sparse.linalg import LinearOperator, cg
import matplotlib.pyplot as plt



# === User Settings ===
NSIDE_INPUT = 2048
TARGET_NSIDE = 128
data_dir = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/data")
output_dir = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/output")
os.makedirs(output_dir, exist_ok=True)
LMAX = min(3*TARGET_NSIDE - 1, 500)
log_file_path = os.path.join(output_dir, "run_log.txt")
log_file = open(log_file_path, "w")
def logprint(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()
logprint(f"Running at NSIDE={TARGET_NSIDE}, LMAX={LMAX}")


# === Load Data ===
def load_maps():
    cmb_filename = os.path.join(data_dir, "COM_CMB_IQU-commander_2048_R3.00_full.fits")
    mask_filename = os.path.join(data_dir, "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")
    cmb_map_full = hp.read_map(cmb_filename)
    mask_full = hp.read_map(mask_filename)
    cmb_map_full = np.nan_to_num(cmb_map_full)
    cmb_map_full[~np.isfinite(cmb_map_full)] = 0.0
    return cmb_map_full, mask_full

cmb_map_full, mask_full = load_maps()


# === Downgrade Maps if Needed ===
def downgrade_maps(cmb_map_full, mask_full, nside_in, nside_out):
    if nside_out != nside_in:
        logprint("Downgrading maps...")
        cmb_map = hp.ud_grade(cmb_map_full, nside_out)
        cmb_map = np.nan_to_num(cmb_map)
        cmb_map[~np.isfinite(cmb_map)] = 0.0
        mask = hp.ud_grade(mask_full, nside_out)
        mask = (mask > 0.9).astype(float)
    else:
        cmb_map = cmb_map_full
        mask = mask_full
    return cmb_map, mask

cmb_map, mask = downgrade_maps(cmb_map_full, mask_full, NSIDE_INPUT, TARGET_NSIDE)


# === Noise Model ===
sigma = 25.0
var_pix = sigma**2
N_inv_pix = mask / var_pix
N_inv_mean = np.mean(N_inv_pix[mask > 0])

# ------------------------------------------------
# Harmonic data
# ------------------------------------------------

a_lm = hp.map2alm(cmb_map * mask, lmax=LMAX, iter=3)


# === Theoretical Cl ===
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

Cl_theory, Cl_safe = get_theory_Cl(LMAX)


# === Operators for C^{-1} Filtering ===
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

# ------------------------------------------------
# Timed operator
# ------------------------------------------------


# === CG Solve for C^{-1} a ===
alm_size_complex = hp.Alm.getsize(LMAX)
alm_size = 2 * alm_size_complex

def build_S_N_operator(alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax):
    def matvec(x_real):
        return S_N_operator_real(x_real, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax)
    return LinearOperator((alm_size, alm_size), matvec=matvec, dtype=np.float64)


# === Build RHS for CG ===
def build_rhs(cmb_map, N_inv_pix, lmax):
    map_rhs = cmb_map * N_inv_pix
    b = hp.map2alm(map_rhs, lmax=lmax, iter=3)
    b_real = np.concatenate([b.real, b.imag])
    return b_real

b_real = build_rhs(cmb_map, N_inv_pix, LMAX)
logprint("b_real magnitude: {:.3e}".format(np.linalg.norm(b_real)))

#------------------------------------------------
# last minute tests
#------------------------------------------------

def test_spd(A, size):

    x = np.random.randn(size)
    Ax = A.matvec(x)

    val = np.dot(x, Ax)

    print("x^T A x =", val)

test_spd(S_N_operator, alm_size)


# === CG Solve for C^{-1} a ===
def cg_solve_Cinv_a(b_real, alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax, maxiter=200):
    S_N_operator = build_S_N_operator(alm_size, alm_size_complex, Cl_safe, N_inv_pix, nside, lmax)
    M_ell = 1.0 / ( (1.0/Cl_safe) + N_inv_mean )
    def preconditioner_real(x_real):
        v_complex = x_real[:alm_size_complex] + 1j * x_real[alm_size_complex:]
        result_complex = hp.almxfl(v_complex, M_ell)
        return np.concatenate([result_complex.real, result_complex.imag])
    M_operator = LinearOperator((alm_size, alm_size), matvec=preconditioner_real, dtype=np.float64)
    iteration_counter = {"count": 0}
    cg_start_time = time.time()
    def cg_callback(xk):
        iteration_counter["count"] += 1
        elapsed = time.time() - cg_start_time
        logprint(f"[CG] Iter {iteration_counter['count']} | total time: {elapsed:.2f} sec")
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
    cg_total_time = time.time() - cg_start_time
    logprint(f"CG finished in {iteration_counter['count']} iterations, {cg_total_time:.2f} seconds.")
    return solution_complex

Cinv_a = cg_solve_Cinv_a(b_real, alm_size, alm_size_complex, Cl_safe, N_inv_pix, TARGET_NSIDE, LMAX)

#------------------------------------------------
# sanity check: compute residual norm ||Ax - b||
#------------------------------------------------

residual = S_N_operator_real(solution_real) - b_real
residual_norm = np.linalg.norm(residual)
logprint(f"Final residual norm ||Ax - b||: {residual_norm:.3e}")
logprint("Relative residual:", np.linalg.norm(residual) / np.linalg.norm(b_real))

#-------------------------------------------------
#finishing C
#-------------------------------------------------

final_filtered_alm = hp.almxfl(solution_complex, 1.0/Cl_safe)


# === f_NL Estimator (Eq. 2.1, unnormalized) ===
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
        map_A = hp.alm2map(alm_A, nside=nside, lmax=LMAX, verbose=False)
        alm_B = hp.almxfl(Cinv_a, bl)
        map_B = hp.alm2map(alm_B, nside=nside, lmax=LMAX, verbose=False)
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
            map_A_sim  = hp.alm2map(alm_A_sim, nside=nside, lmax=LMAX, verbose=False)
            alm_B_data = hp.almxfl(Cinv_a, bl)
            map_B_data  = hp.alm2map(alm_B_data, nside=nside, lmax=LMAX, verbose=False)
            angular_cross = np.sum(map_A_sim**2 * map_B_data) * d_Omega
            E_cross_integrand[i_r] = r**2 * angular_cross
        E_linear_sims[i_sim] = simpson(E_cross_integrand, x=r_arr)
    E_linear = 3.0 * np.mean(E_linear_sims)
    return E_linear

#=====================================
# time test
#=====================================

def benchmark_nside(nside_list):
    results = []
    
    for nside in nside_list:
        lmax = 3*nside - 1
        
        logprint(f"\nTesting NSIDE={nside}, LMAX={lmax}")
        
        test_alm = hp.synalm(np.ones(lmax+1), lmax=lmax)
        
        t0 = time.time()
        _ = hp.alm2map(test_alm, nside=nside, lmax=lmax, verbose=False)
        t1 = time.time()
        
        dt = t1 - t0
        logprint(f"Time per alm2map: {dt:.3f}s")
        
        results.append((nside, dt))
    
    return results

#results = benchmark_nside([64, 128, 256, 512, 1024, 2048])


# === Main f_NL Calculation ===
def main_fnl_pipeline():
    logprint("\n========== STARTING f_NL COMPUTATION (Eq. 2.1, unnormalized) ==========")
    # Step 1: Get transfer functions
    logprint("Computing CAMB transfer functions...")
    pars_transfer = camb.CAMBparams()
    pars_transfer.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars_transfer.InitPower.set_params(As=2e-9, ns=0.965)
    pars_transfer.set_for_lmax(LMAX, lens_potential_accuracy=0)
    pars_transfer.WantTransfer = False
    pars_transfer.WantCls = True
    data_transfer = camb.get_transfer_functions(pars_transfer)
    k_min = 1e-4
    k_max = 0.5
    N_k = 200
    k_arr = np.geomspace(k_min, k_max, N_k)
    k_pivot = 0.05
    A_s = 2e-9
    n_s = 0.965
    P_prim = A_s * (k_arr / k_pivot)**(n_s - 1)
    transfer_data = data_transfer.get_cmb_transfer_data()
    k_camb = transfer_data.q
    Delta_lk = transfer_data.delta_p_l_k[0]
    from scipy.interpolate import interp1d
    Delta_interp = np.zeros((LMAX + 1, N_k))
    for l in range(2, LMAX + 1):
        if l < Delta_lk.shape[0]:
            f_interp = interp1d(k_camb, Delta_lk[l, :], kind='linear', bounds_error=False, fill_value=0.0)
            Delta_interp[l, :] = f_interp(k_arr)
    logprint("Transfer function interpolation done.")
    # Step 2: r grid
    r_min = 100.0
    r_max = 14000.0
    N_r = 50
    r_arr = np.linspace(r_min, r_max, N_r)
    logprint(f"Radial grid: {N_r} shells from {r_min} to {r_max} Mpc")
    # Step 3: alpha, beta
    logprint("Computing alpha_l(r) and beta_l(r) ... (this takes a while)")
    alpha, beta = compute_alpha_beta(LMAX, k_arr, Delta_interp, P_prim, r_arr)
    logprint("alpha_l(r) and beta_l(r) done.")
    # Step 4: Cubic term
    logprint("Computing cubic term (Eq. 2.1 numerator) ...")
    E_cubic = cubic_term(Cinv_a, alpha, beta, LMAX, r_arr, TARGET_NSIDE)
    logprint(f"E_cubic (raw) = {E_cubic:.6e}")
    # Step 5: Linear term (mean-field, MC)
    logprint("Computing linear term (mean-field, MC approximation) ...")
    E_linear = linear_term_MC(Cinv_a, alpha, beta, LMAX, r_arr, TARGET_NSIDE, Cl_theory, Cl_safe, var_pix, mask, N_sim=20)
    logprint(f"E_linear (mean-field) = {E_linear:.6e}")
    # Step 6: Unnormalized estimator
    E_unnorm = E_cubic - E_linear
    logprint(f"\nE_cubic  = {E_cubic:.6e}")
    logprint(f"E_linear = {E_linear:.6e}")
    logprint(f"E_unnorm = E_cubic - E_linear = {E_unnorm:.6e}")
    logprint("\n========== f_NL numerator (unnormalized, Eq. 2.1) ==========")
    logprint(f"f_NL numerator = {E_unnorm:.6e}")
    log_file.close()


# === Test Function for Low Resolution ===
def test_lowres_pipeline():
    """
    Run the pipeline at low NSIDE/LMAX and check for successful completion and reasonable output.
    """
    global TARGET_NSIDE, LMAX, cmb_map, mask, N_inv_pix, N_inv_mean, alm_size_complex, alm_size, b_real, Cl_theory, Cl_safe, Cinv_a
    # Use very low resolution for fast test
    TARGET_NSIDE = 16
    LMAX = min(3*TARGET_NSIDE - 1, 40)
    logprint(f"\n[TEST] Running low-res test: NSIDE={TARGET_NSIDE}, LMAX={LMAX}")
    # Redo downgrade and noise
    cmb_map, mask = downgrade_maps(cmb_map_full, mask_full, NSIDE_INPUT, TARGET_NSIDE)
    N_inv_pix = mask / var_pix
    N_inv_mean = np.mean(N_inv_pix[mask > 0])
    alm_size_complex = hp.Alm.getsize(LMAX)
    alm_size = 2 * alm_size_complex
    b_real = build_rhs(cmb_map, N_inv_pix, LMAX)
    Cl_theory, Cl_safe = get_theory_Cl(LMAX)
    # CG solve
    Cinv_a = cg_solve_Cinv_a(b_real, alm_size, alm_size_complex, Cl_safe, N_inv_pix, TARGET_NSIDE, LMAX, maxiter=50)
    # Run main pipeline (should complete quickly)
    try:
        main_fnl_pipeline()
        logprint("[TEST] Low-res pipeline completed successfully.")
    except Exception as e:
        logprint(f"[TEST] Low-res pipeline failed: {e}")

if __name__ == "__main__":
    # Run test at low resolution first
    test_lowres_pipeline()
    # Optionally, uncomment to run full pipeline after test
    # main_fnl_pipeline()