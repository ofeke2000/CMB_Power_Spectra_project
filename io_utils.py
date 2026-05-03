import os
import healpy as hp
import numpy as np

def logprint(msg, log_file=None):
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()

def load_maps(data_dir):
    cmb_filename = os.path.join(data_dir, "COM_CMB_IQU-commander_2048_R3.00_full.fits")
    mask_filename = os.path.join(data_dir, "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")
    cmb_map_full = hp.read_map(cmb_filename)
    mask_full = hp.read_map(mask_filename)
    cmb_map_full = np.nan_to_num(cmb_map_full)
    cmb_map_full[~np.isfinite(cmb_map_full)] = 0.0
    return cmb_map_full, mask_full

def downgrade_maps(cmb_map_full, mask_full, nside_in, nside_out, log_file=None):
    if nside_out != nside_in:
        logprint("Downgrading maps...", log_file)
        cmb_map = hp.ud_grade(cmb_map_full, nside_out)
        cmb_map = np.nan_to_num(cmb_map)
        cmb_map[~np.isfinite(cmb_map)] = 0.0
        mask = hp.ud_grade(mask_full, nside_out)
        mask = (mask > 0.9).astype(float)
    else:
        cmb_map = cmb_map_full
        mask = mask_full
    return cmb_map, mask
