# === Imports ===
import os
import time
import numpy as np
import healpy as hp
import camb
from io_utils import logprint, load_maps, downgrade_maps
from theory import get_theory_Cl, get_transfer_functions
from filtering import build_rhs, cg_solve_Cinv_a, apply_S_N_operator
from estimator import compute_alpha_beta, cubic_term, linear_term_MC
from test_pipeline import test_lowres_pipeline


# === User Settings ===
NSIDE_INPUT = 2048
# Option A: low-resolution testing (default, fast) 
# TARGET_NSIDE = 128 
# LMAX = min(3*TARGET_NSIDE - 1, 500)
# Option B: full-resolution production: 
TARGET_NSIDE= 2048
LMAX=min(3*TARGET_NSIDE - 1, 1000)

data_dir = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/data")
output_dir = os.path.expanduser("/home/ofeke2000/CMB_project/CMB_Power_Spectra_project/output")
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "run_log.txt")
log_file = open(log_file_path, "w")


def log(msg):
    logprint(msg, log_file)


log(f"Running at NSIDE={TARGET_NSIDE}, LMAX={LMAX}")
script_start_time = time.time()


# === Load Data ===
cmb_map_full, mask_full = load_maps(data_dir)

# --- Sanity Check 1: Input maps and mask ---
def check_map_stats(map_data, name):
    log(f"Sanity check: {name} min={np.nanmin(map_data):.3e}, max={np.nanmax(map_data):.3e}, mean={np.nanmean(map_data):.3e}, std={np.nanstd(map_data):.3e}")
    if np.any(np.isnan(map_data)):
        log(f"[ERROR] {name} contains NaNs!")
        raise ValueError(f"{name} contains NaNs")
    if np.all(map_data == 0):
        log(f"[ERROR] {name} is all zeros!")
        raise ValueError(f"{name} is all zeros")


check_map_stats(cmb_map_full, "CMB map (full)")
check_map_stats(mask_full, "Mask (full)")


# === Downgrade Maps if Needed ===
cmb_map, mask = downgrade_maps(cmb_map_full, mask_full, NSIDE_INPUT, TARGET_NSIDE, log_file)
check_map_stats(cmb_map, "CMB map (target)")
check_map_stats(mask, "Mask (target)")


# === Noise Model ===
sigma = 25.0
var_pix = sigma**2
N_inv_pix = mask / var_pix
N_inv_mean = np.mean(N_inv_pix[mask > 0])

# --- Sanity Check 2: Noise map ---
check_map_stats(N_inv_pix, "N_inv_pix")
if np.any(N_inv_pix[mask > 0] <= 0):
    log("[ERROR] N_inv_pix has non-positive values in unmasked region!")
    raise ValueError("N_inv_pix has non-positive values in unmasked region!")


# (Optional) Harmonic data for diagnostics
a_lm = hp.map2alm(cmb_map * mask, lmax=LMAX, iter=3)


# === Theoretical Cl ===
Cl_theory, Cl_safe = get_theory_Cl(LMAX)

# --- Sanity Check 3: Cl ---
def check_Cl(Cl, name):
    log(f"Sanity check: {name} min={np.min(Cl):.3e}, max={np.max(Cl):.3e}")
    if np.any(Cl < 0):
        log(f"[ERROR] {name} has negative values!")
        raise ValueError(f"{name} has negative values!")


check_Cl(Cl_theory, "Cl_theory")
check_Cl(Cl_safe, "Cl_safe")


# === CG Solve for C^{-1} a ===
alm_size_complex = hp.Alm.getsize(LMAX)
alm_size = 2 * alm_size_complex
b_real = build_rhs(cmb_map, N_inv_pix, LMAX)
log(f"b_real magnitude: {np.linalg.norm(b_real):.3e}")

# --- Sanity Check 4: b_real ---
if not np.all(np.isfinite(b_real)):
    log("[ERROR] b_real contains non-finite values!")
    raise ValueError("b_real contains non-finite values!")
if np.linalg.norm(b_real) == 0:
    log("[ERROR] b_real is all zeros!")
    raise ValueError("b_real is all zeros!")


# --- CG Solve and Sanity Check 5: CG convergence and filtered map power ---
c_inv_start_time = time.time()
Cinv_a_wiener, cg_info = cg_solve_Cinv_a(
    b_real,
    alm_size,
    alm_size_complex,
    Cl_safe,
    N_inv_pix,
    TARGET_NSIDE,
    LMAX,
    N_inv_mean,
    logprint=log
)
C_inv_time = time.time() - c_inv_start_time
log(f"C inversion time = {C_inv_time:.2f} seconds")

if cg_info != 0:
    log(f"[WARNING] CG did not converge! info={cg_info}")
    log("Proceeding anyway but results may be unreliable")

b_complex = b_real[:alm_size_complex] + 1j * b_real[alm_size_complex:]
residual = apply_S_N_operator(Cinv_a_wiener, Cl_safe, N_inv_pix, TARGET_NSIDE, LMAX) - b_complex
# For residual check, mask out marginalized l=0,1 modes since they have large residuals by design
residual_masked = residual.copy()
# Set l=0, m=0 to 0
residual_masked[0] = 0
# Set l=1, m=-1,0,1 to 0 (indices 1,2,3 in alm array)
residual_masked[1:4] = 0
rel_residual = np.linalg.norm(residual_masked) / np.linalg.norm(b_complex)
log(f"CG relative residual (l>=2): {rel_residual:.3e}")
if rel_residual > 1e-3:
    log(f"[WARNING] CG residual {rel_residual:.3e} is large!")

# CG gives Wiener filter W*a = S*(S+N)^{-1}*a
# Estimator needs C^{-1}*a = (S+N)^{-1}*a = S^{-1} * (CG result)
Cinv_a = hp.almxfl(Cinv_a_wiener, 1.0 / Cl_safe)
log("[INFO] Applied S^{-1} correction: Wiener filter → C^{-1}a")

# Check filtered map alm power
alm_power = hp.alm2cl(Cinv_a)
log(f"Sanity check: Filtered alm power min={np.min(alm_power):.3e}, max={np.max(alm_power):.3e}, mean={np.mean(alm_power):.3e}")
if np.any(~np.isfinite(alm_power)):
    log("[ERROR] Filtered alm power contains non-finite values!")
    raise ValueError("Filtered alm power contains non-finite values!")
if np.all(alm_power == 0):
    log("[ERROR] Filtered alm power is all zeros!")
    raise ValueError("Filtered alm power is all zeros!")


# === Main f_NL Calculation ===
def main_fnl_pipeline():
    log("\n========== STARTING f_NL COMPUTATION (Eq. 2.1, unnormalized) ===========")

    # ---- Transfer functions ----
    log("Computing CAMB transfer functions...")
    k_arr, P_prim, Delta_interp, l_vals, k_camb, delta_shape = get_transfer_functions(LMAX)
    log(f"Transfer function k values shape: {k_camb.shape}")
    log(f"Transfer function l values shape: {l_vals.shape}")
    log(f"Transfer function delta_p_l_k shape: {delta_shape}")
    log(f"Delta_interp shape: {Delta_interp.shape}")
    log(f"Delta_interp max: {np.max(np.abs(Delta_interp)):.3e}")
    if np.all(Delta_interp == 0):
        log("[ERROR] Delta_interp is all zeros!")
        raise ValueError("Transfer functions are all zero — check CAMB setup")

    # ---- Radial grid ----
    r_min, r_max, N_r = 100.0, 14000.0, 500
    r_arr = np.linspace(r_min, r_max, N_r)
    log(f"Radial grid: {N_r} shells from {r_min} to {r_max} Mpc")

    # ---- Alpha, beta kernels ----
    log("Computing alpha_l(r) and beta_l(r)...")
    t0 = time.time()
    alpha, beta = compute_alpha_beta(LMAX, k_arr, Delta_interp, P_prim, r_arr)
    log(f"alpha/beta done in {time.time() - t0:.1f}s")
    log(f"alpha max={np.max(np.abs(alpha)):.3e}, beta max={np.max(np.abs(beta)):.3e}")
    if np.all(alpha == 0) or np.all(beta == 0):
        log("[ERROR] alpha or beta is all zeros!")
        raise ValueError("alpha or beta is all zeros — check transfer functions")

    # ---- Cubic term ----
    log("Computing cubic term...")
    t0 = time.time()
    E_cubic = cubic_term(Cinv_a, alpha, beta, LMAX, r_arr, TARGET_NSIDE)
    log(f"E_cubic = {E_cubic:.6e}  ({time.time() - t0:.1f}s)")

    # ---- Linear term ----
    log("Computing linear term (N_sim=5)...")
    t0 = time.time()
    E_linear, E_cubic_sims = linear_term_MC(
        Cinv_a,
        alpha,
        beta,
        LMAX,
        r_arr,
        TARGET_NSIDE,
        Cl_theory,
        Cl_safe,
        var_pix,
        mask,
        N_sim=5,
        N_inv_pix=N_inv_pix,
        alm_size=alm_size,
        alm_size_complex=alm_size_complex,
        N_inv_mean=N_inv_mean
    )
    log(f"E_linear = {E_linear:.6e}  ({time.time() - t0:.1f}s)")

    # ---- Unnormalized estimator ----
    E_unnorm = E_cubic - E_linear
    
    # Calculate error: std dev of simulation cubic terms (Gaussian sims have true f_NL=0)
    unnorm_error = np.std(E_cubic_sims)
    
    log(f"\nE_cubic  = {E_cubic:.6e}")
    log(f"E_linear = {E_linear:.6e}")
    log(f"E_unnorm = E_cubic - E_linear = {E_unnorm:.6e}")
    log("\n========== f_NL numerator (unnormalized, Eq. 2.1) ===========")
    log(f"f_NL numerator = {E_unnorm:.6e} ± {unnorm_error:.6e}")

    total_run_time = time.time() - script_start_time
    log("\n========== RUN SUMMARY ===========")
    log(f"Total run time = {total_run_time:.2f} seconds")
    log(f"C inversion time = {C_inv_time:.2f} seconds")
    log(f"Unnormalized f_NL = {E_unnorm:.6e} ± {unnorm_error:.6e}")
    log_file.close()


#=====================================
# time test
#=====================================

def benchmark_nside(nside_list):
    results = []

    for nside in nside_list:
        lmax = 3*nside - 1
        log(f"\nTesting NSIDE={nside}, LMAX={lmax}")

        test_alm = hp.synalm(np.ones(lmax+1), lmax=lmax)

        t0 = time.time()
        _ = hp.alm2map(test_alm, nside=nside, lmax=lmax, verbose=False)
        t1 = time.time()

        dt = t1 - t0
        log(f"Time per alm2map: {dt:.3f}s")

        results.append((nside, dt))

    return results


#results = benchmark_nside([64, 128, 256, 512, 1024, 2048])


# === Test Function for Low Resolution ===

if __name__ == "__main__":
    # Run test at low resolution first
    #test_lowres_pipeline(data_dir, NSIDE_INPUT, var_pix, mask_full, cmb_map_full, log_file)
    # Optionally, run full pipeline after test
    main_fnl_pipeline()
