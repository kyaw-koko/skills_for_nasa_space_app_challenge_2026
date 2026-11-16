---
name: spherex-irsa-pipeline
description: End-to-end engineering skill based on Caltech/IPAC IRSA Tutorials (https://caltech-ipac.github.io/irsa-tutorials/) for querying SPHEREx API endpoints (TAP/SIAv2), downloading Multi-Extension FITS (MEF) cutouts in parallel, processing 6 detector arrays & 102 near-infrared spectral bands, performing image subtraction/coaddition, extracting on-the-fly spectra, and rendering time-series blink animations & Aladin/Firefly sky overlays.
---

# Caltech/IPAC IRSA Tutorials for SPHEREx — Complete Engineering Skill Set

Use this skill whenever building backend pipelines, API routes, data processors, or frontend sky visualization components for NASA's SPHEREx mission ("Planet X and SPHEREx" Web Tool).

---

## 1. Mission & Data Architecture Reference
- **Mission Scope**: SPHEREx maps the entire sky every 6 months across **102 near-infrared spectral bands** ($0.75\text{–}5.0\ \mu\text{m}$) with $6.2''/\text{pixel}$ spatial resolution.
- **Data Releases**: Always query **DR1** (`spherex_dr1`, `spherex_dr1_deep`) and **QR3** (`spherex_qr3`, `spherex_qr3_deep`). Ignore deprecated QR1/QR2 datasets unless legacy fallback is requested.
- **6 Detector Arrays (`energy_bandpassname`)**:
  - `SPHEREx-D1` (Band 1): $\lambda = 0.75\text{–}1.09\ \mu\text{m}$ ($R = 39$)
  - `SPHEREx-D2` (Band 2): $\lambda = 1.10\text{–}1.62\ \mu\text{m}$ ($R = 41$)
  - `SPHEREx-D3` (Band 3): $\lambda = 1.63\text{–}2.41\ \mu\text{m}$ ($R = 41$)
  - `SPHEREx-D4` (Band 4): $\lambda = 2.42\text{–}3.82\ \mu\text{m}$ ($R = 35$)
  - `SPHEREx-D5` (Band 5): $\lambda = 3.83\text{–}4.41\ \mu\text{m}$ ($R = 112$)
  - `SPHEREx-D6` (Band 6): $\lambda = 4.42\text{–}5.00\ \mu\text{m}$ ($R = 128$)

---

## 2. IRSA API Endpoints & Query Protocols

### 2.1 Protocol A: ADQL / TAP Endpoint (Recommended for Real-Time Cutouts & Zero Ingestion Lag)
New SPHEREx data is ingested weekly. TAP queries access the `spherex.artifact` and `spherex.plane` tables immediately upon ingestion (avoiding the ~24-hour SIAv2 indexing delay) and build direct on-the-fly **IRSA IBE Cutout URLs**:
- **Base TAP Endpoint**: `https://irsa.ipac.caltech.edu/TAP`
- **Cutout URL Pattern**: `https://irsa.ipac.caltech.edu/{a.uri}?center={ra_deg},{dec_deg}d&size={fov_deg}`

```python
from astroquery.ipac.irsa import Irsa
import astropy.units as u
from astropy.coordinates import SkyCoord

def query_spherex_cutouts_tap(ra_deg: float, dec_deg: float, size_deg: float = 0.05, detector: str = None):
    band_filter = f"AND p.energy_bandpassname = '{detector}'" if detector else ""
    query = f"""
    SELECT
        'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra_deg},{dec_deg}d&size={size_deg}' AS cutout_url,
        'https://irsa.ipac.caltech.edu/' || a.uri AS full_mef_url,
        p.time_bounds_lower AS obs_mjd_start,
        p.time_bounds_upper AS obs_mjd_end,
        p.energy_bandpassname AS detector_band,
        p.provenance_version AS version
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra_deg}, {dec_deg}), p.poly)
    {band_filter}
    ORDER BY p.time_bounds_lower ASC
    """
    results = Irsa.query_tap(query).to_table()
    return results
```

### 2.2 Protocol B: IVOA Simple Image Access V2 (SIAv2)

* **HTTP GET Endpoint**: `https://irsa.ipac.caltech.edu/SIA?COLLECTION=spherex_dr1&POS=CIRCLE+{ra}+{dec}+{radius_deg}`
* **Collections**: `spherex_dr1`, `spherex_dr1_deep`, `spherex_qr3`, `spherex_qr3_deep`
* **Python (`astroquery`)**:

```python
coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame="icrs")
results = Irsa.query_sia(pos=(coord, 50 * u.arcsec), collection="spherex_dr1")
```

---

## 3. Multi-Extension FITS (MEF) Structure & Parsing

Every SPHEREx Level-2 Spectral Image MEF (and its IBE cutout) contains **7 HDUs (0 to 6)**:

1. `hdulist[0]` (`PRIMARY`): Global metadata (`VERSION`, observation timestamps).
2. `hdulist[1]` (`IMAGE`): Surface brightness in $\text{MJy/sr}$ ($2040 \times 2040$ full frame, or cropped cutout).
   - **Primary WCS**: Astrometric sky coordinates (`RA`, `Dec`) with optical distortion polynomials.
   - **Alternate `WCSNAMEA` (`key='A'`)**: 0-based active detector pixel coordinates.
   - **Alternate `WCSNAMEW` (`key='W'`)**: Linear Variable Filter (LVF) spectral coordinate lookup (`Wavelength` and `Bandwidth` in $\mu\text{m}$), linked to HDU 6 (`WCS-WAVE`).
3. `hdulist[2]` (`FLAGS`): 32-bit integer bitmask per pixel (`NONFUNC`, `SATURATED`, `COSMIC_RAY`, `SOURCE`, contamination flags).
4. `hdulist[3]` (`VARIANCE`): Per-pixel variance estimate $(\text{MJy/sr})^2$ for SNR calculation.
5. `hdulist[4]` (`ZODI`): Zodiacal dust background model ($\text{MJy/sr}$). **CRITICAL**: `ZODI` is *not* pre-subtracted from `IMAGE`. Subtract `hdulist['IMAGE'].data - hdulist['ZODI'].data` for clean background removal.
6. `hdulist[5]` (`EPSF` / `PSF`): Effective Point Spread Function cube across 441 detector zones (Versions $\ge 7.0$ use `EPSF`; legacy $< 7.0$ use `PSF`).
7. `hdulist[6]` (`WCS-WAVE`): 1-row lookup table (`X`, `Y`, `VALUES`) driving `WCS(header, fobj=hdulist, key='W')`.

### Extracting Exact Central Wavelength at Target Coordinates

Because SPHEREx uses Linear Variable Filters (LVFs), wavelength changes continuously across pixels on the detector:

```python
from astropy.io import fits
from astropy.wcs import WCS

def process_single_cutout(cutout_url: str, target_coord: SkyCoord, remove_zodi: bool = True):
    with fits.open(cutout_url, cache=False) as hdul:
        img_hdu = hdul["IMAGE"]
        spatial_wcs = WCS(img_hdu.header)
        x_pix, y_pix = spatial_wcs.world_to_pixel(target_coord)

        # Evaluate exact wavelength (micrometers) at target pixel via WCS key="W"
        spectral_wcs = WCS(header=img_hdu.header, fobj=hdul, key="W")
        wavelength_um, bandwidth_um = spectral_wcs.pixel_to_world_values(x_pix, y_pix)

        img_data = img_hdu.data.astype(float)
        if remove_zodi and "ZODI" in hdul:
            img_data = img_data - hdul["ZODI"].data

        flags_data = hdul["FLAGS"].data if "FLAGS" in hdul else None
        return {
            "image": img_data,
            "flags": flags_data,
            "header": img_hdu.header,
            "wavelength_um": float(wavelength_um),
            "bandwidth_um": float(bandwidth_um)
        }
```

---

## 4. High-Speed Parallel Fetching & Fault Tolerance

Fetching 100–700 epochs sequentially takes 5–7 minutes. Always use `concurrent.futures.ThreadPoolExecutor` with **8 to 10 workers** (optimal for IRSA server rate limits) and retry transient HTTP/IncompleteRead errors:

```python
import concurrent.futures
import http.client
import urllib.error
from astropy.utils.data import conf

conf.remote_timeout = 120

def fetch_cutouts_parallel(rows, target_coord: SkyCoord, max_workers: int = 8):
    def worker(row):
        for attempt in range(3):
            try:
                data = process_single_cutout(row["cutout_url"], target_coord)
                data["obs_mjd"] = float(row["obs_mjd_start"])
                data["detector"] = row["detector_band"]
                return data
            except (TimeoutError, urllib.error.HTTPError, http.client.IncompleteRead):
                if attempt == 2:
                    return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker, rows))
    return [r for r in results if r is not None]
```

---

## 5. Image Co-addition, Differencing & On-The-Fly Spectra (Source Discovery Tool Patterns)

### 5.1 Common Grid Reprojection (Required Before Blinking, Adding, or Subtracting)

SPHEREx exposures have varying spacecraft roll angles. Before blinking frames as a video, stacking detector arrays (to boost SNR), or subtracting Epoch 1 from Epoch 2 (to isolate moving objects like Planet X or brown dwarfs), reproject all cutouts onto a unified North-Up tangential (`TAN`) WCS grid:

```python
from reproject import reproject_interp
import numpy as np

def align_and_diff_epochs(hdu_epoch1, hdu_epoch2, target_wcs, shape_out=(128, 128)):
    img1_aligned, _ = reproject_interp(hdu_epoch1, target_wcs, shape_out=shape_out)
    img2_aligned, _ = reproject_interp(hdu_epoch2, target_wcs, shape_out=shape_out)
    diff_image = img2_aligned - img1_aligned   # Positive/negative dipole reveals moving objects!
    coadd_image = np.nanmean([img1_aligned, img2_aligned], axis=0)
    return img1_aligned, img2_aligned, diff_image, coadd_image
```

### 5.2 Unit Conversion & On-The-Fly 102-Band Spectrum Generation

1. Convert surface brightness ($\text{MJy/sr}$) to flux density ($\mu\text{Jy}$) using pixel solid angle ($\approx (6.2'')^2$).
2. Perform aperture photometry (`photutils.aperture.CircularAperture` + `aperture_photometry`) at `(ra, dec)` across all fetched cutouts.
3. Sort extracted points by `wavelength_um` ($0.75 \to 5.0\ \mu\text{m}$) and return JSON `[{wavelength, flux_uJy, flux_err, mjd, detector, flags}]` for interactive frontend plotting.

---

## 6. Web Visualization, Multi-Mission Overlays & Benchmark Targets

### 6.1 Frontend Time-Series Blinker & Controls

* **Blink Video Player**: Render co-aligned epoch frames on an HTML5 `<canvas>` with play/pause, FPS slider ($1\text{–}15\ \text{fps}$), Contrast/Stretch controls (Asinh / ZScale), and Display Mode toggle (`Single Epoch`, `Detector Stack D1-D6`, `Epoch Difference A - B`).
* **Shareable URL State**: Synchronize view parameters in URL query strings (`?ra=133.795&dec=-7.245&fov=0.05&band=SPHEREx-D2&mode=diff`) so any sky patch can be bookmarked or embedded.
* **Telescope Overlays (Aladin Lite v3 / Firefly / Astroquery)**:
  - Overlay **Gaia DR3** catalog positions and proper motion vectors (`pmra`, `pmdec`) to distinguish known high-proper-motion stars from uncatalogued solar system objects or brown dwarfs.
  - Overlay **AllWISE / NEOWISE** infrared catalogs and **SPHEREx HiPS All-Sky Cubes**.

### 6.2 Mandatory Verification Benchmark Targets

Build quick-select UI preset buttons for the 5 official NASA Space Apps Challenge test targets:

1. **Barnard's Star** (Highest proper-motion star): `RA = 269.452075°` (`17h57m48.5s`), `Dec = +04.693391°` (`+04d41m36s`)
2. **WISE 0855** (Fast-moving coldest brown dwarf): `RA = 133.795129°` (`08h55m10.831s`), `Dec = -07.245147°` (`-07d14m42.53s`)
3. **Epsilon Indi B** (Binary brown dwarf pair): `RA = 331.068125°` (`22h04m16.35s`), `Dec = -56.782806°` (`-56d46m58.1s`)
4. **RW Aurigae** (T Tauri variable star): `RA = 76.956542°` (`05h07m49.57s`), `Dec = +30.401417°` (`+30d24m05.1s`)
5. **R Draconis** (Mira-type variable star): `RA = 248.108875°` (`16h32m26.13s`), `Dec = +66.754167°` (`+66d45m15.0s`)
