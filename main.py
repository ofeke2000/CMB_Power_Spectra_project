

# === Imports ===
import os
import numpy as np
import healpy as hp
from io_utils import logprint, load_maps, downgrade_maps
from theory import get_theory_Cl
from filtering import build_rhs, cg_solve_Cinv_a
from estimator import compute_alpha_beta, cubic_term, linear_term_MC
from test_pipeline import test_lowres_pipeline




# === User Settings ===
NSIDE_INPUT = 2048
TARGET_NSIDE = 128
data_dir = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/data")
output_dir = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/output")
os.makedirs(output_dir, exist_ok=True)
LMAX = min(3*TARGET_NSIDE - 1, 500)
log_file_path = os.path.join(output_dir, "run_log.txt")
log_file = open(log_file_path, "w")
logprint(f"Running at NSIDE={TARGET_NSIDE}, LMAX={LMAX}", log_file)



# === Load Data ===
cmb_map_full, mask_full = load_maps(data_dir)



# === Downgrade Maps if Needed ===
cmb_map, mask = downgrade_maps(cmb_map_full, mask_full, NSIDE_INPUT, TARGET_NSIDE, log_file)



# === Noise Model ===
sigma = 25.0
var_pix = sigma**2
N_inv_pix = mask / var_pix
N_inv_mean = np.mean(N_inv_pix[mask > 0])


# (Optional) Harmonic data for diagnostics
a_lm = hp.map2alm(cmb_map * mask, lmax=LMAX, iter=3)



# === Theoretical Cl ===
Cl_theory, Cl_safe = get_theory_Cl(LMAX)



# === CG Solve for C^{-1} a ===
alm_size_complex = hp.Alm.getsize(LMAX)
alm_size = 2 * alm_size_complex
b_real = build_rhs(cmb_map, N_inv_pix, LMAX)
logprint("b_real magnitude: {:.3e}".format(np.linalg.norm(b_real)), log_file)


# (Optional) Add diagnostics or SPD test here if needed



Cinv_a = cg_solve_Cinv_a(b_real, alm_size, alm_size_complex, Cl_safe, N_inv_pix, TARGET_NSIDE, LMAX, N_inv_mean, logprint=lambda msg: logprint(msg, log_file))


# (Optional) Add residual check or diagnostics here if needed



# === Main f_NL Calculation ===
def main_fnl_pipeline():
    logprint("\n========== STARTING f_NL COMPUTATION (Eq. 2.1, unnormalized) ==========", log_file)
    # Step 1: Get transfer functions
    logprint("Computing CAMB transfer functions...", log_file)
    import camb
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
    logprint("Transfer function interpolation done.", log_file)
    # Step 2: r grid
    r_min = 100.0
    r_max = 14000.0
    N_r = 50
    r_arr = np.linspace(r_min, r_max, N_r)
    logprint(f"Radial grid: {N_r} shells from {r_min} to {r_max} Mpc", log_file)
    # Step 3: alpha, beta
    logprint("Computing alpha_l(r) and beta_l(r) ... (this takes a while)", log_file)
    alpha, beta = compute_alpha_beta(LMAX, k_arr, Delta_interp, P_prim, r_arr)
    logprint("alpha_l(r) and beta_l(r) done.", log_file)
    # Step 4: Cubic term
    logprint("Computing cubic term (Eq. 2.1 numerator) ...", log_file)
    E_cubic = cubic_term(Cinv_a, alpha, beta, LMAX, r_arr, TARGET_NSIDE)
    logprint(f"E_cubic (raw) = {E_cubic:.6e}", log_file)
    # Step 5: Linear term (mean-field, MC)
    logprint("Computing linear term (mean-field, MC approximation) ...", log_file)
    E_linear = linear_term_MC(Cinv_a, alpha, beta, LMAX, r_arr, TARGET_NSIDE, Cl_theory, Cl_safe, var_pix, mask, N_sim=20)
    logprint(f"E_linear (mean-field) = {E_linear:.6e}", log_file)
    # Step 6: Unnormalized estimator
    E_unnorm = E_cubic - E_linear
    logprint(f"\nE_cubic  = {E_cubic:.6e}", log_file)
    logprint(f"E_linear = {E_linear:.6e}", log_file)
    logprint(f"E_unnorm = E_cubic - E_linear = {E_unnorm:.6e}", log_file)
    logprint("\n========== f_NL numerator (unnormalized, Eq. 2.1) ==========", log_file)
    logprint(f"f_NL numerator = {E_unnorm:.6e}", log_file)
    log_file.close()

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

if __name__ == "__main__":
    # Run test at low resolution first
    test_lowres_pipeline(data_dir, NSIDE_INPUT, var_pix, mask_full, cmb_map_full, log_file)
    # Optionally, run full pipeline after test
    # main_fnl_pipeline()