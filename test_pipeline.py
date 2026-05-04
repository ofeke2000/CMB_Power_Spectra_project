from io_utils import logprint, load_maps, downgrade_maps
from theory import get_theory_Cl
from filtering import build_rhs, cg_solve_Cinv_a
from estimator import compute_alpha_beta, cubic_term, linear_term_MC
import healpy as hp
import numpy as np

def test_lowres_pipeline(data_dir, nside_in, var_pix, mask_full, cmb_map_full, log_file):
    TARGET_NSIDE = 16
    LMAX = min(3*TARGET_NSIDE - 1, 40)
    logprint(f"\n[TEST] Running low-res test: NSIDE={TARGET_NSIDE}, LMAX={LMAX}", log_file)
    cmb_map, mask = downgrade_maps(cmb_map_full, mask_full, nside_in, TARGET_NSIDE, log_file)
    N_inv_pix = mask / var_pix
    N_inv_mean = np.mean(N_inv_pix[mask > 0])
    alm_size_complex = hp.Alm.getsize(LMAX)
    alm_size = 2 * alm_size_complex
    b_real = build_rhs(cmb_map, N_inv_pix, LMAX)
    Cl_theory, Cl_safe = get_theory_Cl(LMAX)
    Cinv_a, cg_info = cg_solve_Cinv_a(b_real, alm_size, alm_size_complex, Cl_safe, N_inv_pix, TARGET_NSIDE, LMAX, N_inv_mean, maxiter=50, logprint=lambda msg: logprint(msg, log_file))
    assert cg_info == 0, f"[ERROR] CG did not converge! info={cg_info}"
    # Minimal estimator test (no full pipeline for speed)
    assert np.all(np.isfinite(Cinv_a)), "Cinv_a contains non-finite values!"
    logprint("[TEST] Low-res pipeline completed successfully.", log_file)
