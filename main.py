import healpy as hp
import numpy as np
import os
import time
import camb
from scipy.sparse.linalg import LinearOperator, cg
import astropy.units as u
import matplotlib.pyplot as plt


# ------------------------------------------------
# USER SETTINGS
# ------------------------------------------------

NSIDE_INPUT = 2048
TARGET_NSIDE = 128        # change this to test lower resolution

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

# ------------------------------------------------
# Load data
# ------------------------------------------------

cmb_filename = os.path.join(data_dir, "COM_CMB_IQU-commander_2048_R3.00_full.fits")
mask_filename = os.path.join(data_dir, "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")
cmb_map_full = hp.read_map(cmb_filename)
mask_full    = hp.read_map(mask_filename)

cmb_map_full = np.nan_to_num(cmb_map_full)
cmb_map_full[~np.isfinite(cmb_map_full)] = 0.0

# ------------------------------------------------
# Downgrade if needed
# ------------------------------------------------

if TARGET_NSIDE != NSIDE_INPUT:
    logprint("Downgrading maps...")
    
    cmb_map = hp.ud_grade(cmb_map_full, TARGET_NSIDE)

    cmb_map = np.nan_to_num(cmb_map)
    cmb_map[~np.isfinite(cmb_map)] = 0.0
    
    mask = hp.ud_grade(mask_full, TARGET_NSIDE)
    mask = (mask > 0.9).astype(float)   # re-binarize mask
    
else:
    cmb_map = cmb_map_full
    mask    = mask_full

# ------------------------------------------------
# Noise model
# ------------------------------------------------

sigma = 25.0 
var_pix = sigma**2
N_inv_pix = mask / var_pix
N_inv_mean = np.mean(N_inv_pix[mask > 0])

# ------------------------------------------------
# Harmonic data
# ------------------------------------------------

a_lm = hp.map2alm(cmb_map * mask, lmax=LMAX, iter=3)

# ------------------------------------------------
# Theoretical Cl
# ------------------------------------------------

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(As=2e-9, ns=0.965)
pars.set_for_lmax(LMAX, lens_potential_accuracy=0)

results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

cls_theory = powers['total'][:,0]  # TT spectrum

ells = np.arange(len(cls_theory))

Dl = cls_theory.copy()

Cl = np.zeros_like(Dl)

factor = ells * (ells + 1) / (2 * np.pi)

# avoid division by zero at ℓ=0
nonzero = factor > 0
Cl[nonzero] = Dl[nonzero] / factor[nonzero]

Cl_theory = Cl[:LMAX+1]

epsilon = 1e-12 * np.max(Cl_theory)
Cl_safe = Cl_theory + epsilon

# ------------------------------------------------
# Operators
# ------------------------------------------------

def S_inv_op(v):
    return hp.almxfl(v, 1.0/Cl_safe)

def N_inv_op(v):
    map_v = hp.alm2map(v, nside=TARGET_NSIDE, lmax=LMAX)
    map_v = map_v * N_inv_pix
    
    # safety check
    map_v = np.nan_to_num(map_v)
    
    return hp.map2alm(map_v, lmax=LMAX, iter=3)

def S_N_operator_real(x_real):

    # split into real/imag
    v = x_real[:alm_size_complex] + 1j * x_real[alm_size_complex:]

    result_complex = S_inv_op(v) + N_inv_op(v)

    # return stacked real vector
    return np.concatenate([
        result_complex.real,
        result_complex.imag
    ])

# ------------------------------------------------
# Timed operator
# ------------------------------------------------

iteration_counter = {"count": 0}
start_time = time.time()

def S_inv_plus_N_inv(v):
    iteration_counter["count"] += 1
    elapsed = time.time() - start_time
    logprint(f"Iteration {iteration_counter['count']} | {elapsed:.2f} sec elapsed")
    result = S_inv_op(v) + N_inv_op(v)

    if not np.all(np.isfinite(result)):
        print("Operator produced non-finite values!")
        raise ValueError("Numerical instability detected.")

    return result

alm_size_complex = hp.Alm.getsize(LMAX)
alm_size = 2 * alm_size_complex

S_N_operator = LinearOperator(
    (alm_size, alm_size),
    matvec=S_N_operator_real,
    dtype=np.float64
)

# ------------------------------------------------
# Right-hand side b = N^{-1} d
# ------------------------------------------------

logprint("Building RHS...")

map_rhs = cmb_map * N_inv_pix
b = hp.map2alm(map_rhs, lmax=LMAX, iter=3)

b_complex = b
b_real = np.concatenate([b_complex.real, b_complex.imag])

#------------------------------------------------
# last minute tests
#------------------------------------------------

def test_spd(A, size):

    x = np.random.randn(size)
    Ax = A.matvec(x)

    val = np.dot(x, Ax)

    print("x^T A x =", val)

test_spd(S_N_operator, alm_size)

# ------------------------------------------------
# Conjugate Gradient Solve
# ------------------------------------------------


logprint("Starting CG solve...")

cg_start_time = time.time()
iteration_counter["count"] = 0  # reset counter


#Build preconditioner

M_ell = 1.0 / ( (1.0/Cl_safe) + N_inv_mean ) # approximate diagonal in harmonic space

def preconditioner_real(x_real):

    v_complex = (
        x_real[:alm_size_complex]
        + 1j * x_real[alm_size_complex:]
    )

    result_complex = hp.almxfl(v_complex, M_ell)

    return np.concatenate([
        result_complex.real,
        result_complex.imag
    ])


def cg_callback(xk):
    # This gets called once per iteration
    iter_num = iteration_counter["count"]
    elapsed = time.time() - cg_start_time
    logprint(f"[CG] Iter {iter_num} | total time: {elapsed:.2f} sec")


M_operator = LinearOperator(
    (alm_size, alm_size),
    matvec=preconditioner_real,
    dtype=np.float64
)

solution_real, info = cg(
    S_N_operator,
    b_real,
    M=M_operator,
    maxiter=200,
    rtol=1e-6,
    atol=0,
    callback=cg_callback
)

solution_complex = (
    solution_real[:alm_size_complex]
    + 1j * solution_real[alm_size_complex:]
)

cg_total_time = time.time() - cg_start_time

#-------------------------------------------------
#finishing C
#-------------------------------------------------

final_filtered_alm = hp.almxfl(solution_complex, 1.0/Cl_safe)

# ------------------------------------------------
# Results
# ------------------------------------------------

logprint("\n========== CG FINISHED ==========")

if info == 0:
    logprint("Converged successfully.")
elif info > 0:
    logprint(f"Did NOT converge. Reached {info} iterations.")
else:
    logprint("CG failed with illegal input or breakdown.")

logprint(f"Total CG iterations: {iteration_counter['count']}")
logprint(f"Total CG runtime: {cg_total_time:.2f} seconds")
logprint(f"Average time per iteration: {cg_total_time / max(iteration_counter['count'],1):.2f} sec")
logprint(f"Run completed at NSIDE={TARGET_NSIDE}, LMAX={LMAX}")

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