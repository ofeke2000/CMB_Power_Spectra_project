# Computing the Planck CMB Temperature Power Spectrum with `healpy`

## Part I — What the Code Is Doing

This notebook reproduces the Planck 2018 CMB temperature (TT) angular power spectrum directly from the public Planck data using `healpy`.

The workflow consists of:

1. Downloading Planck data products
2. Loading the HEALPix CMB map
3. Applying a Galactic mask
4. Computing the angular power spectrum with `anafast`
5. Comparing to the published Planck spectrum

---

## Accessing the Planck Data

The notebook downloads three public Planck Release 3 files from the IRSA archive:

### 1. CMB Temperature Map

File:

```
COM_CMB_IQU-commander_2048_R3.00_full.fits
```

* Contains full-sky CMB temperature and polarization (I, Q, U).
* Produced using the Commander component-separation algorithm.
* HEALPix format.
* Resolution: NSIDE = 2048 (~50 million pixels).

Accessed via:

```python
cmb_map = hp.read_map(filename, field=0)
```

* `field=0` selects temperature (I).
* Returns a NumPy array of temperature values in Kelvin.
* Healpy automatically converts ordering (e.g., NESTED → RING if needed).

---

### 2. Planck Common Mask

File:

```
COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits
```

* Binary mask (1 = keep pixel, 0 = exclude).
* Removes regions contaminated by Galactic foreground emission.
* Covers ~77–78% of the sky.

Loaded with:

```python
mask = hp.read_map(mask_filename)
```

Applied using:

```python
map_masked = hp.ma(cmb_map)
map_masked.mask = np.logical_not(mask)
```

Here:

* `hp.ma()` creates a masked array compatible with spherical harmonic transforms.
* Masked pixels are ignored in later calculations.

---

### 3. Published Binned TT Power Spectrum

File:

```
COM_PowerSpect_CMB-TT-binned_R3.01.txt
```

Columns:

* ℓ (multipole)
* ( D_\ell )
* Lower error
* Upper error
* Best-fit model

Loaded using:

```python
cmb_binned_spectrum = np.loadtxt(filename)
```

This is used for comparison with the spectrum computed from the map.

---

## Healpy Functions Used

### `hp.read_map()`

Reads HEALPix FITS files.

Key behavior:

* Returns a 1D array of pixel values.
* Handles ordering conversion.
* Automatically detects NSIDE.

---

### `hp.mollview()`

Displays full-sky maps in Mollweide projection.

Used only for visualization.

---

### `hp.ma()`

Creates a masked HEALPix map.

Important for:

* Excluding Galactic plane contamination.
* Ensuring masked pixels do not bias harmonic transforms.

---

### `hp.anafast()`

Computes spherical harmonic power spectra.

```python
cls = hp.anafast(map_masked, lmax=lmax, use_pixel_weights=True)
```

What it does internally:

1. Computes spherical harmonic coefficients:
   [
   a_{\ell m}
   ]

2. Computes the estimator:
   [
   C_\ell = \frac{1}{2\ell+1} \sum_m |a_{\ell m}|^2
   ]

`use_pixel_weights=True` improves accuracy at high resolution.

If the map is masked, the measured spectrum is corrected approximately by dividing by:

[
f_{\text{sky}} = \text{fraction of unmasked pixels}
]

---

### Beam Correction

Planck maps are smoothed with a 5 arcminute beam.

The beam window function is computed with:

```python
w_ell = hp.gauss_beam(beam_fwhm_rad, lmax=lmax)
```

The power spectrum is corrected by dividing by ( w_\ell^2 ).

---

## Part II — The Angular Power Spectrum

The CMB temperature anisotropy field is:

[
T(\hat n)
]

defined on the sphere.

---

### Spherical Harmonic Decomposition

We expand:

[
T(\hat n) = \sum_{\ell m} a_{\ell m} Y_{\ell m}(\hat n)
]

The coefficients ( a_{\ell m} ) encode temperature fluctuations at angular scale ℓ.

---

### Definition of the Angular Power Spectrum

Assuming statistical isotropy:

[
\langle a_{\ell m} a^*_{\ell' m'} \rangle
=========================================

\delta_{\ell \ell'} \delta_{m m'} C_\ell
]

The angular power spectrum is:

[
C_\ell = \frac{1}{2\ell+1} \sum_m |a_{\ell m}|^2
]

It represents the variance of temperature fluctuations at angular scale ℓ.

---

### Physical Meaning of ℓ

Angular scale:

[
\theta \sim \frac{180^\circ}{\ell}
]

Examples:

| ℓ    | Angular scale |
| ---- | ------------- |
| 2    | ~90°          |
| 200  | ~1°           |
| 1000 | ~0.2°         |

Thus ( C_\ell ) tells us how much fluctuation power exists at each angular scale.

---

### Why Plot ( D_\ell )?

Planck plots:

[
D_\ell = \frac{\ell(\ell+1)}{2\pi} C_\ell
]

Reason:

* In the Sachs–Wolfe limit (large scales),
  [
  \ell(\ell+1)C_\ell = \text{constant}
  ]
* This produces a flat plateau at low ℓ.
* Acoustic peaks become visually clearer.

---

### Theoretical Expression

From linear cosmological perturbation theory:

[
C_\ell =
\frac{2}{\pi}
\int_0^\infty
dk ,
k^2
P_\zeta(k)
\left[T_\ell(k)\right]^2
]

Where:

* ( P_\zeta(k) ) — primordial power spectrum
* ( T_\ell(k) ) — radiation transfer function
* Spherical Bessel functions appear inside ( T_\ell(k) )

So:

* Theory integrates over Fourier modes ( k )
* Data decomposes the observed sky into ( a_{\ell m} )
* Both produce the same ( C_\ell )

---

## Summary

The notebook:

1. Loads Planck’s full-sky CMB temperature map.
2. Applies a Galactic mask.
3. Computes spherical harmonic coefficients.
4. Estimates the angular power spectrum.
5. Corrects for sky fraction and beam smoothing.
6. Reproduces the published Planck TT spectrum.

The angular power spectrum ( C_\ell ) encodes the statistical properties of primordial fluctuations and contains nearly all cosmological information accessible from the CMB temperature field.
