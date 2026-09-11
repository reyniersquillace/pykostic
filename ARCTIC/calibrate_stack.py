#!/usr/bin/env python3
"""
Generalized bias/dark(/flat)-calibration + stacking pipeline for APO 3.5m
ARCTIC imaging. Works for any filter/run -- point it at a data directory and
the root names of your bias, dark, (optionally) flat, and science frames.

What it does
------------
0. Every raw frame (bias, dark, flat, and science alike) is first run
   through per-quadrant overscan subtraction. ARCTIC's "Quad" readout mode
   records all 4 amplifiers' data AND overscan regions packed into one raw
   array (per-quadrant boundaries given by the DSECxx/BSECxx header
   keywords), with the overscan/gap columns and rows sitting *between* the
   real data -- e.g. a raw 2102x2050 frame is really four 1024x1024 science
   regions plus ~50 columns and ~2 rows of overscan/gap running through the
   middle. This step subtracts each quadrant's own overscan level (its
   BSECxx region) from that quadrant's data region, IN PLACE, and masks
   every pixel outside any DSECxx region (overscan strips, gaps) to NaN --
   without resizing the array or touching CRPIX/CRVAL/CD. (An earlier
   version of this trimmed the gap out and shifted CRPIX to compensate;
   that's mathematically unable to stay correct for both amplifier blocks
   simultaneously, since removing the gap requires each block to shift by a
   *different* amount, and a single CRPIX can only encode one of them.
   Masking in place sidesteps that: since nothing is renumbered, the WCS
   stays exactly as valid as the original raw header's.) If a frame's
   header has no DSECxx keywords (not quad-readout data), this step is
   skipped automatically and the frame is used as-is.
1. Builds a master bias (sigma-clipped median across all matching bias
   frames).
2. Builds a dark-current *rate* map (ADU/sec/pixel) from all matching dark
   frames. Dark current is thermal, not optical, so it doesn't depend on
   filter -- every frame matching --dark-root is used regardless of any
   filter suffix in its name. Each dark is bias-subtracted and divided by
   its own EXPTIME before averaging, so darks taken at different exposure
   times can be combined without assuming either matches the science
   EXPTIME.
3. If --flat-root is given and matching files are found (and
   --no-flat-fielding is not set), builds a master flat: each flat is
   bias- and dark-subtracted exactly like a science frame, normalized to
   its own median (so flats taken under different sky brightness combine
   cleanly), then sigma-clip median-combined. Unlike bias/dark, flats ARE
   filter-specific -- point --flat-root at flats taken in the same filter
   as --science-root. If no flat files are found, or --no-flat-fielding is
   passed, flat-fielding is skipped entirely (bias+dark calibration only).
4. Calibrates each science frame as:
       calibrated = (raw - master_bias - dark_rate*EXPTIME) / master_flat
   (the flat-division step is skipped if no flat was built) -- done as
   full-frame 2D arrays.
4.5. Detects a FIXED bad-pixel pattern across the calibrated science frames
   and masks it to NaN, in every frame, before alignment. This exists for
   defects that are baked into the detector itself (a flaky/trap pixel, a
   small hot cluster) and therefore show up identically in every science
   frame: if the frames aren't dithered (or barely are), such a defect looks
   exactly like a real, consistent source to the sigma-clipped combine in
   step 6 below, since that step can only reject a pixel that's an outlier
   RELATIVE TO THE OTHER FRAMES at that location -- it survives straight
   into the final stack otherwise. This step catches it differently: it runs
   LACosmic (via the `astroscrappy` package) independently on each raw
   calibrated frame -- before any resampling, so LACosmic's assumption of
   independent per-pixel noise actually holds -- and flags pixels by SHAPE
   (sharp/blocky, unlike a real star's smooth PSF falloff) rather than by
   amplitude or cross-frame consistency, so it catches the defect even
   though every single frame has it identically. The union of what's
   flagged across all frames (dilated by a couple of pixels to cover any
   sub-pixel drift introduced by alignment) is masked to NaN in every
   calibrated frame. Skip with --no-bad-pixel-mask; skipped automatically
   (with a warning) if `astroscrappy` isn't installed.
5. Aligns the calibrated science frames to a reference frame using
   star-pattern matching (astroalign), to correct for any drift between
   exposures.
6. Combines the aligned frames with a sigma-clipped mean (rejects cosmic
   rays / other transient outliers) into a single stacked FITS image.

Caveat: if you have only one or two dark (or flat) frames, there's very
little ability to sigma-clip out a cosmic ray or hot pixel landing in one of
them -- it can leave a fixed artifact at that pixel in every calibrated
science frame. Worth a visual sanity check of the dark-rate/master-flat
outputs if you notice odd fixed-position defects in the stack. (Step 4.5
above catches this specific case too, as long as it's consistently visible
under sky-level illumination in the science frames themselves -- it doesn't
need the defect to be visible in the bias/dark/flat.)

Usage
-----
    python calibrate_stack.py --science-root J2250_u --filter u \\
        [--data-dir /path/to/data] [--bias-root bias] [--dark-root dark] \\
        [--flat-root flat_u] [--no-flat-fielding] \\
        [--output-dir /path/to/output] \\
        [--output-bias NAME] [--output-dark NAME] [--output-flat NAME] \\
        [--output-stack NAME] [--save-calibrated-frames] \\
        [--no-bad-pixel-mask] [--gain GAIN] [--readnoise ADU] \\
        [--bpm-sigclip SIGMA] [--bpm-dilate N]

Root names are matched as "<root>*.fits" under --data-dir. Run with -h for
the full list of options and their defaults.

Requires: astropy, numpy, scipy. astroalign is optional but recommended
(pip install astroalign) for sub-pixel-accurate registration; without it,
the script falls back to integer-pixel shift alignment via cross-correlation.
astroscrappy is also optional but recommended (pip install astroscrappy) for
the fixed-bad-pixel-mask step described above; without it, that step is
skipped with a warning.
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve   # needed by the alignment fallback below,
                                        # which can trigger even when astroalign
                                        # IS installed (e.g. on a MaxIterError)

try:
    import astroalign as aa
    HAVE_ASTROALIGN = True
except ImportError:
    HAVE_ASTROALIGN = False

try:
    import astroscrappy
    HAVE_ASTROSCRAPPY = True
except ImportError:
    HAVE_ASTROSCRAPPY = False

# Below what fraction of the flat's own median a pixel is treated as
# unreliable (vignetted/dead) and masked out (set to NaN) rather than
# divided by, to avoid blowing up noise or dividing by ~zero.
FLAT_MIN_RELATIVE_VALUE = 0.1


def parse_args():
    p = argparse.ArgumentParser(
        description="Bias/dark(/flat)-calibrate and stack ARCTIC science frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default=".",
                    help="Directory containing the raw FITS files")
    p.add_argument("--bias-root", default="bias",
                    help="Root name for bias frames; matched as '<root>*.fits'")
    p.add_argument("--dark-root", default="dark",
                    help="Root name for dark frames; matched as '<root>*.fits'. "
                         "Dark current is filter-independent, so every matching "
                         "file is used regardless of any filter suffix in its name.")
    p.add_argument("--flat-root", default=None,
                    help="Root name for flat frames; matched as '<root>*.fits'. "
                         "Unlike darks, flats ARE filter-specific -- point this at "
                         "flats taken in the same filter as --science-root. If "
                         "omitted (or no files match), flat-fielding is skipped.")
    p.add_argument("--no-flat-fielding", action="store_true",
                    help="Force-skip flat-fielding even if --flat-root matches "
                         "files (e.g. you have flats sitting around from a "
                         "different run/filter and don't want them used here).")
    p.add_argument("--science-root", required=True,
                    help="Root name for science frames; matched as '<root>*.fits'")
    p.add_argument("--filter", required=True,
                    help="Filter name (e.g. 'u', 'z'). Used to build default output "
                         "filenames and recorded in the output FITS header.")
    p.add_argument("--output-dir", default=".",
                    help="Directory to write all output FITS files into")
    p.add_argument("--output-bias", default=None,
                    help="Output master bias filename (default: master_bias.fits)")
    p.add_argument("--output-dark", default=None,
                    help="Output dark-rate filename (default: dark_rate.fits)")
    p.add_argument("--output-flat", default=None,
                    help="Output master flat filename "
                         "(default: master_flat_<filter>.fits)")
    p.add_argument("--output-stack", default=None,
                    help="Output stacked science filename "
                         "(default: <science-root>_stack.fits)")
    p.add_argument("--save-calibrated-frames", action="store_true",
                    help="Also write out each individual calibrated science frame "
                         "(post bad-pixel-masking, pre-alignment, if that step ran)")
    p.add_argument("--no-bad-pixel-mask", action="store_true",
                    help="Skip automatic detection/masking of a fixed bad-pixel pattern "
                         "across the science frames (see step 4.5 in the module "
                         "docstring). On by default when astroscrappy is installed AND "
                         "flat-fielding is being used (see --force-bad-pixel-mask "
                         "otherwise).")
    p.add_argument("--force-bad-pixel-mask", action="store_true",
                    help="Run bad-pixel-mask detection even when no flat-fielding was "
                         "applied. NOT recommended: LACosmic assumes flat-fielded input, "
                         "and without a flat, ordinary pixel-to-pixel QE variation reads "
                         "as defects and can flag a large fraction of the frame. Even "
                         "with this set, the mask is only applied if the flagged fraction "
                         "comes out small (see the 1% sanity check in main()) -- "
                         "otherwise it's reported but skipped.")
    p.add_argument("--gain", type=float, default=None,
                    help="Detector gain (e-/ADU), used only for bad-pixel-mask "
                         "detection. Default: read from the GAIN header keyword if "
                         "present, else 1.0.")
    p.add_argument("--readnoise", type=float, default=None,
                    help="Detector read noise (ADU), used only for bad-pixel-mask "
                         "detection. Default: read from the RDNOISE header keyword if "
                         "present, else 15.0.")
    p.add_argument("--bpm-sigclip", type=float, default=4.5,
                    help="LACosmic detection threshold (in sigma) for the bad-pixel "
                         "mask; raise it if real stars get flagged, lower it if a "
                         "known defect isn't being caught.")
    p.add_argument("--bpm-dilate", type=int, default=2,
                    help="Number of binary-dilation iterations applied to the "
                         "bad-pixel mask, to also cover a few pixels of sub-pixel "
                         "drift introduced by the alignment step that follows.")
    return p.parse_args()


QUAD_NAMES = ["11", "12", "21", "22"]


def parse_iraf_section(value):
    """'[x1:x2,y1:y2]' (FITS/IRAF convention: 1-indexed, inclusive) ->
    ((y1,y2), (x1,x2)) as 1-indexed ints."""
    s = value.strip().lstrip("[").rstrip("]")
    xpart, ypart = s.split(",")
    x1, x2 = (int(v) for v in xpart.split(":"))
    y1, y2 = (int(v) for v in ypart.split(":"))
    return (y1, y2), (x1, x2)


def get_quadrant_layout(header):
    """Collect each quadrant's DSECxx (data) / BSECxx (overscan) sections
    from the header. Returns {} if this isn't quad-readout data (no DSECxx
    keywords present at all)."""
    layout = {}
    for q in QUAD_NAMES:
        dsec_key = f"DSEC{q}"
        if dsec_key not in header:
            continue
        dy, dx = parse_iraf_section(header[dsec_key])
        bsec_key = f"BSEC{q}"
        by, bx = parse_iraf_section(header[bsec_key]) if bsec_key in header else (None, None)
        layout[q] = dict(dsec_y=dy, dsec_x=dx, bsec_y=by, bsec_x=bx)
    return layout


def overscan_correct(data, header, overscan_stat=np.median, label=""):
    """
    Subtract each quadrant's own overscan level (its BSECxx region) from
    that quadrant's DSECxx data region, IN PLACE -- the array is not resized
    or re-indexed, and the header is not touched. Every pixel not covered by
    any quadrant's DSECxx region (the overscan strips themselves, any
    prescan columns, the narrow inter-amplifier gap) is set to NaN, since
    none of that corresponds to real sky data.

    An earlier version of this function trimmed the overscan/gap columns out
    of the array entirely and shifted CRPIX to compensate. That turned out
    to be mathematically unfixable in general: removing the gap requires a
    DIFFERENT constant pixel shift for each of the two amplifier blocks
    (they need different shifts specifically because the *fake* overscan
    columns being removed sit between them), and a single CRPIX value can
    only correctly encode one block's shift -- the other block silently
    picks up a systematic error equal to the width of the removed gap. This
    version sidesteps the whole problem: since nothing is renumbered, CRPIX/
    CRVAL/CD stay byte-for-byte what they were, and remain exactly as valid
    as the original raw header's WCS already was.

    If the header has no DSECxx keywords (not quad-readout data), returns
    data unchanged.
    """
    layout = get_quadrant_layout(header)
    if not layout:
        print(f"{label}: no DSECxx keywords found -- not quad-readout data, "
              "using frame as-is (no overscan correction applied).")
        return data

    if len(layout) != 4:
        msg = (f"{label}: found {len(layout)}/4 quadrant DSECxx keywords "
               "-- expected all 4 for quad-readout data. Skipping "
               "overscan correction for this frame (using it uncorrected, "
               "full raw overscan/gap counts included).")
        warnings.warn(msg)
        print(f"WARNING: {msg}")  # also on stdout: warnings.warn() goes to
                                   # stderr and is easy to miss/lose when
                                   # only stdout is captured or redirected
        return data

    ny, nx = data.shape
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    for q, info in layout.items():
        (y1, y2), (x1, x2) = info["dsec_y"], info["dsec_x"]
        if y2 > ny or x2 > nx:
            warnings.warn(f"{label}: DSEC{q}=[{x1}:{x2},{y1}:{y2}] falls outside "
                           f"the {nx}x{ny} array -- skipping overscan correction "
                           "for this frame.")
            return data
        if info["bsec_y"] is not None:
            (oy1, oy2), (ox1, ox2) = info["bsec_y"], info["bsec_x"]
            overscan_level = overscan_stat(data[oy1 - 1:oy2, ox1 - 1:ox2])
        else:
            overscan_level = 0.0
        out[y1 - 1:y2, x1 - 1:x2] = (
            data[y1 - 1:y2, x1 - 1:x2].astype(np.float64) - overscan_level
        ).astype(np.float32)

    n_nan = int(np.isnan(out).sum())
    print(f"{label}: quad overscan-corrected ({len(layout)} quadrant(s)) -- "
          f"{n_nan} pixel(s) outside any DSECxx region masked to NaN "
          "(overscan/gap, not real sky data). Array shape and WCS unchanged.")
    return out


def load_data(path):
    """Return (float32 data array, header) from the primary HDU of a FITS
    file, after per-quadrant overscan subtraction (see overscan_correct) --
    a no-op for non-quad-readout data. The array shape and WCS keywords are
    never modified."""
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()
    label = os.path.basename(path)
    data = overscan_correct(data, header, label=label)
    return data, header


def find_frames(data_dir, root, label):
    files = sorted(glob.glob(os.path.join(data_dir, f"{root}*.fits")))
    print(f"found {len(files)} {label} frame(s) matching '{root}*.fits' in {data_dir}")
    return files


# On-sky WCS keywords. Not meaningful for calibration products (bias/dark/
# flat frames aren't pointed at a fixed sky position in any useful sense),
# and APO's TCC has been observed to write the literal invalid string
# "+NAN" into some of these (e.g. the CD matrix) for untracked exposures
# like flats, which crashes astropy's FITS writer if left in the header.
# Stripped from calibration-product headers below.
WCS_KEYWORDS_TO_STRIP = [
    "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "CDELT1", "CDELT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
    "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    "LATPOLE", "LONPOLE",
]


def strip_wcs(header):
    """Return a copy of header with on-sky WCS keywords removed -- see
    WCS_KEYWORDS_TO_STRIP. Used for calibration-product outputs (master
    bias/dark/flat), which don't have a meaningful WCS of their own."""
    header = header.copy()
    for key in WCS_KEYWORDS_TO_STRIP:
        if key in header:
            del header[key]
    return header


def safe_writeto(hdu, path):
    """Write a FITS HDU, first silently fixing any other non-standard/
    invalid cards astropy knows how to repair (e.g. a stray malformed
    keyword from the instrument control software), rather than crashing
    outright on something that isn't actually our data."""
    hdu.verify("silentfix")
    hdu.writeto(path, overwrite=True)


def build_master_bias(bias_files):
    print(f"building master bias from {len(bias_files)} frame(s)...")
    stack = np.stack([load_data(f)[0] for f in bias_files], axis=0)
    clipped = sigma_clip(stack, sigma=3, maxiters=5, axis=0, masked=True)
    master_bias = np.ma.median(clipped, axis=0).filled(np.nanmedian(stack, axis=0))
    master_bias = master_bias.astype(np.float32)

    mean, med, std = sigma_clipped_stats(master_bias)
    print(f"  master bias stats: mean={mean:.2f}  median={med:.2f}  std={std:.2f}")
    return master_bias


def build_dark_rate(dark_files, master_bias):
    """
    Bias-subtract each dark frame, divide by its EXPTIME to get a dark-current
    RATE (ADU/sec/pixel), and average the rate maps together. Filter doesn't
    matter here since dark current is thermal, not optical -- only EXPTIME
    matters, and working in a rate sidesteps needing the darks and the
    science frames to share an exposure time.
    """
    print(f"building dark-current rate map from {len(dark_files)} frame(s): "
          f"{', '.join(os.path.basename(f) for f in dark_files)}")
    rate_maps = []
    for f in dark_files:
        data, hdr = load_data(f)
        exptime = hdr.get("EXPTIME")
        if not exptime:
            warnings.warn(f"{f}: no EXPTIME in header; skipping this dark frame.")
            continue
        if data.shape != master_bias.shape:
            sys.exit(f"{f}: shape {data.shape} does not match master bias "
                      f"shape {master_bias.shape} -- check readout modes match.")
        rate = (data - master_bias) / exptime
        print(f"  {os.path.basename(f)}: EXPTIME={exptime}s, "
              f"median rate={np.nanmedian(rate):.4f} ADU/s (NaN pixels are overscan/gap, excluded)")
        rate_maps.append(rate)

    if not rate_maps:
        sys.exit("No usable dark frames (missing EXPTIME in all of them).")

    dark_rate = np.mean(np.stack(rate_maps, axis=0), axis=0).astype(np.float32)
    print(f"  combined dark rate: median={np.nanmedian(dark_rate):.4f} ADU/s/pixel (NaN pixels are overscan/gap, excluded)")
    return dark_rate


def build_master_flat(flat_files, master_bias, dark_rate):
    """
    Bias- and dark-subtract each flat (same as a science frame), normalize
    each to its own median (so flats taken under different sky brightness
    combine cleanly), then sigma-clip median-combine. The result is
    renormalized to unit median, and near-zero/vignetted pixels are masked
    to NaN so later division doesn't blow up.
    """
    print(f"building master flat from {len(flat_files)} frame(s): "
          f"{', '.join(os.path.basename(f) for f in flat_files)}")
    normalized = []
    flat_hdr = None
    for f in flat_files:
        data, hdr = load_data(f)
        if flat_hdr is None:
            flat_hdr = hdr
        if data.shape != master_bias.shape:
            sys.exit(f"{f}: shape {data.shape} does not match master bias "
                      f"shape {master_bias.shape} -- check readout modes match.")
        exptime = hdr.get("EXPTIME")
        if not exptime:
            warnings.warn(f"{f}: no EXPTIME in header; skipping this flat frame.")
            continue
        corrected = data - master_bias - dark_rate * exptime
        # NaN-safe: `corrected` legitimately contains NaN wherever load_data's
        # overscan_correct masked out overscan/gap columns (real for any
        # quad-readout ARCTIC frame). A plain np.median poisons to NaN the
        # instant ANY pixel is NaN, which both silently defeats the `med <= 0`
        # sanity check below (NaN <= 0 is False, so it looks "fine") and
        # would make the whole normalized frame NaN -- and from there, the
        # whole master flat, and every flat-fielded science frame. Use
        # np.nanmedian so only the real (non-gap) pixels count.
        med = np.nanmedian(corrected)
        if not np.isfinite(med) or med <= 0:
            warnings.warn(f"{f}: non-finite/non-positive median ({med}) after bias/dark "
                           "subtraction; skipping this flat frame.")
            continue
        normalized.append(corrected / med)

    if not normalized:
        sys.exit("No usable flat frames (all failed EXPTIME/median checks).")

    stack = np.stack(normalized, axis=0)
    clipped = sigma_clip(stack, sigma=3, maxiters=5, axis=0, masked=True)
    master_flat = np.ma.median(clipped, axis=0).filled(np.nanmedian(stack, axis=0))
    # Same NaN-safety issue as above: overscan/gap pixels are still
    # (correctly) NaN in master_flat at this point, so this renormalization
    # must ignore them via nanmedian too, or the ENTIRE master flat -- not
    # just the gap pixels -- goes NaN.
    master_flat = (master_flat / np.nanmedian(master_flat)).astype(np.float32)

    bad = master_flat < FLAT_MIN_RELATIVE_VALUE
    if np.any(bad):
        print(f"  masking {bad.sum()} pixel(s) below {FLAT_MIN_RELATIVE_VALUE} "
              "of the flat's median (vignetted/dead) to avoid dividing by ~0")
        master_flat[bad] = np.nan

    mean, med, std = sigma_clipped_stats(master_flat, sigma=3, maxiters=5)
    print(f"  master flat stats: mean={mean:.3f}  median={med:.3f}  std={std:.3f}")
    return master_flat, flat_hdr


def get_gain_readnoise(header, gain_arg, readnoise_arg):
    """Resolve gain/readnoise for bad-pixel-mask detection: an explicit CLI
    value wins, then the header's GAIN/RDNOISE keyword, then a hardcoded
    fallback (1.0 e-/ADU, 15.0 ADU) if neither is available."""
    gain = gain_arg if gain_arg is not None else header.get("GAIN", 1.0)
    readnoise = readnoise_arg if readnoise_arg is not None else header.get("RDNOISE", 15.0)
    return float(gain), float(readnoise)


def build_bad_pixel_mask(calibrated_frames, header, gain_arg, readnoise_arg,
                          sigclip=4.5, dilate_iters=2):
    """
    Detect a FIXED bad-pixel pattern -- a hot/flaky/trap pixel (or small
    cluster) baked into the detector -- by running LACosmic (astroscrappy)
    independently on EACH raw calibrated science frame, before any
    alignment/resampling, and taking the union of what it flags across all
    of them.

    This exists because the sigma-clipped mean combine used to build the
    final stack can only reject a pixel that's an outlier RELATIVE TO THE
    OTHER FRAMES at that location. A detector defect that's present in every
    frame identically (e.g. because the science frames weren't dithered)
    looks exactly like a real, consistent source to that combine step, and
    survives into the final stack untouched. Running LACosmic per-frame
    instead catches it by SHAPE (sharp/blocky, unlike a real star's smooth
    PSF falloff), not by amplitude or cross-frame consistency, so it works
    even though every single frame has the exact same defect.

    Returns (dilated_mask, n_before) -- a single boolean mask (same shape as
    the frames) dilated by `dilate_iters` to also cover a few pixels of
    sub-pixel drift introduced by the alignment step that follows, plus the
    pixel count BEFORE dilation. Callers should sanity-check `n_before` (or
    its fraction of the frame) rather than the dilated count: dilation can
    inflate even a handful of genuinely-bad pixels by an order of magnitude
    or more, so it's a poor signal for "did detection go haywire."
    """
    gain, readnoise = get_gain_readnoise(header, gain_arg, readnoise_arg)
    print(f"bad-pixel mask: running LACosmic on {len(calibrated_frames)} raw "
          f"calibrated frame(s) (gain={gain:.2f} e-/ADU, readnoise={readnoise:.2f} ADU, "
          f"sigclip={sigclip})...")
    combined_mask = None
    for i, frame in enumerate(calibrated_frames, start=1):
        # astroscrappy expects a finite array; the overscan/gap columns are
        # already NaN at this point (real non-data, excluded elsewhere), so
        # fill them with the frame's own median for detection purposes only
        # -- this doesn't add them to the returned mask (they're not local
        # outliers relative to that fill value) and doesn't shift alignment
        # between the mask and the original array.
        safe_frame = np.where(np.isfinite(frame), frame, np.nanmedian(frame))
        mask, _ = astroscrappy.detect_cosmics(
            safe_frame, gain=gain, readnoise=readnoise,
            sigclip=sigclip, sigfrac=0.3, objlim=5.0, niter=4,
        )
        print(f"  frame {i}/{len(calibrated_frames)}: {int(mask.sum())} pixel(s) flagged")
        combined_mask = mask if combined_mask is None else (combined_mask | mask)

    n_before = int(combined_mask.sum())
    if dilate_iters > 0:
        combined_mask = binary_dilation(combined_mask, iterations=dilate_iters)
    print(f"bad-pixel mask: {n_before} pixel(s) flagged bad in at least one frame; "
          f"{int(combined_mask.sum())} pixel(s) after {dilate_iters}-iteration dilation")
    return combined_mask, n_before


def integer_shift_align(ref, img):
    """Fallback alignment: whole-pixel shift found via FFT cross-correlation."""
    f0 = ref - ref.mean()
    f1 = img - img.mean()
    corr = fftconvolve(f0, f1[::-1, ::-1], mode="same")
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    center = np.array(corr.shape) // 2
    dy, dx = np.array(peak) - center
    aligned = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
    print(f"    shift (integer-pixel fallback): dy={dy}, dx={dx}")
    return aligned


def align_frame(ref, img):
    if HAVE_ASTROALIGN:
        try:
            aligned, _ = aa.register(img, ref)
            return aligned.astype(np.float32)
        except (aa.MaxIterError, ValueError) as exc:
            # astroalign raises MaxIterError when it can't converge, and a
            # plain ValueError ("Reference stars ... less than the minimum
            # value") when a frame has too few detectable stars to match at
            # all (e.g. a short/cloudy exposure) -- both are "not enough
            # stars", just surfaced differently across astroalign versions.
            warnings.warn(f"astroalign failed to align this frame ({exc}); "
                           "falling back to integer-pixel cross-correlation.")
    return integer_shift_align(ref, img)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    out_bias = args.output_bias or "master_bias.fits"
    out_dark = args.output_dark or "dark_rate.fits"
    out_flat = args.output_flat or f"master_flat_{args.filter}.fits"
    out_stack = args.output_stack or f"{os.path.basename(args.science_root)}_stack.fits"

    bias_files = find_frames(args.data_dir, args.bias_root, "bias")
    dark_files = find_frames(args.data_dir, args.dark_root, "dark")
    flat_files = find_frames(args.data_dir, args.flat_root, "flat") if args.flat_root else []
    science_files = find_frames(args.data_dir, args.science_root, "science")

    if not bias_files:
        sys.exit(f"No bias frames found matching '{args.bias_root}*.fits' in {args.data_dir}")
    if not dark_files:
        sys.exit(f"No dark frames found matching '{args.dark_root}*.fits' in {args.data_dir}")
    if not science_files:
        sys.exit(f"No science frames found matching '{args.science_root}*.fits' "
                  f"in {args.data_dir}")
    if args.flat_root and not flat_files:
        warnings.warn(f"--flat-root '{args.flat_root}' given but no files matched "
                       "-- continuing without flat-fielding.")
    if args.no_flat_fielding and flat_files:
        print(f"--no-flat-fielding set -- ignoring {len(flat_files)} matched flat "
              "frame(s), continuing without flat-fielding.")
        flat_files = []

    # -- Master bias -------------------------------------------------- #
    master_bias = build_master_bias(bias_files)
    _, bias_hdr = load_data(bias_files[0])
    safe_writeto(
        fits.PrimaryHDU(data=master_bias, header=strip_wcs(bias_hdr)),
        os.path.join(args.output_dir, out_bias),
    )
    print(f"wrote {out_bias}")

    # -- Dark current rate (bias-subtracted, per-second) ---------------- #
    dark_rate = build_dark_rate(dark_files, master_bias)
    safe_writeto(
        fits.PrimaryHDU(data=dark_rate, header=strip_wcs(bias_hdr)),
        os.path.join(args.output_dir, out_dark),
    )
    print(f"wrote {out_dark}")

    # -- Master flat (optional) ------------------------------------------ #
    master_flat = None
    if flat_files:
        master_flat, flat_hdr = build_master_flat(flat_files, master_bias, dark_rate)
        safe_writeto(
            fits.PrimaryHDU(data=master_flat, header=strip_wcs(flat_hdr)),
            os.path.join(args.output_dir, out_flat),
        )
        print(f"wrote {out_flat}")
    else:
        print("no flat frames -- skipping flat-fielding (bias+dark calibration only)")

    # -- Calibrate science frames ---------------------------------------- #
    calibrated = []
    frame_headers = []
    ref_header = None
    exptimes = set()
    for f in science_files:
        data, hdr = load_data(f)
        if data.shape != master_bias.shape:
            sys.exit(f"{f}: shape {data.shape} does not match master bias "
                      f"shape {master_bias.shape} -- check readout modes match.")
        exptime = hdr.get("EXPTIME")
        if not exptime:
            sys.exit(f"{f}: no EXPTIME in header; can't scale dark current.")
        scaled_dark = dark_rate * exptime
        frame = data - master_bias - scaled_dark
        if master_flat is not None:
            frame = frame / master_flat
        calibrated.append(frame)
        frame_headers.append(hdr)
        exptimes.add(exptime)
        if ref_header is None:
            ref_header = hdr

    if len(exptimes) > 1:
        warnings.warn(f"Science frames have differing EXPTIME values: {exptimes}. "
                       "Dark current was scaled per-frame, but a straight combine "
                       "still assumes matched exposure times/throughput.")
    else:
        print(f"all science frames share EXPTIME = {exptimes.pop()} s")

    # -- Detect + mask a fixed bad-pixel pattern (before alignment) ------- #
    # See step 4.5 in the module docstring / build_bad_pixel_mask() for why
    # this has to happen here, pre-alignment, rather than on the final stack.
    #
    # IMPORTANT: LACosmic assumes the input is already flat-fielded -- its
    # noise model only accounts for Poisson (shot) + read noise, not ordinary
    # pixel-to-pixel QE variation, which a flat-field division normally
    # removes. Skip flat-fielding (as here whenever master_flat is None) and
    # that un-corrected few-percent QE texture reads as "sharp" structure
    # against the (now too-optimistic) noise model, causing widespread false
    # positives -- confirmed with a synthetic test: ~9% of pixels falsely
    # flagged on realistic bright-sky data without a flat, vs. 0.02% (just
    # the real defect) with the same data flat-fielded. So this step is
    # SKIPPED BY DEFAULT whenever there's no master flat, unless
    # --force-bad-pixel-mask is passed explicitly.
    bad_pixel_mask = None
    if args.no_bad_pixel_mask:
        print("--no-bad-pixel-mask set -- skipping automatic bad-pixel detection")
    elif master_flat is None and not args.force_bad_pixel_mask:
        warnings.warn(
            "No flat-fielding was applied (see the flat-field step above) -- skipping "
            "automatic bad-pixel-mask detection. LACosmic assumes flat-fielded input; "
            "without it, ordinary pixel-to-pixel QE variation gets misread as defects "
            "and can flag a large fraction of the frame (a 'grainy' look). Rerun with "
            "--flat-root once you have z'-band flats, or pass --force-bad-pixel-mask "
            "to run it anyway (not recommended without a flat -- expect false positives)."
        )
    elif not HAVE_ASTROSCRAPPY:
        warnings.warn("astroscrappy not installed -- skipping automatic bad-pixel-mask "
                       "detection (`pip install astroscrappy` to enable it). A fixed "
                       "detector defect that's identical in every science frame (e.g. "
                       "because they weren't dithered) will NOT be caught by the "
                       "sigma-clipped combine below and may survive into the stack.")
    else:
        if master_flat is None:
            warnings.warn("--force-bad-pixel-mask set with no flat-fielding -- expect "
                           "some false positives from un-corrected pixel-to-pixel QE "
                           "variation being misread as defects.")
        bad_pixel_mask, n_before_dilation = build_bad_pixel_mask(
            calibrated, ref_header, args.gain, args.readnoise,
            sigclip=args.bpm_sigclip, dilate_iters=args.bpm_dilate,
        )
        # Sanity-check on the PRE-dilation count/fraction, not the dilated
        # one -- dilation can inflate even a handful of genuinely-bad pixels
        # by an order of magnitude or more, which would otherwise trip this
        # check on perfectly legitimate detections.
        frac_before_dilation = n_before_dilation / bad_pixel_mask.size
        if frac_before_dilation > 0.01:
            n_flagged = int(bad_pixel_mask.sum())
            warnings.warn(
                f"bad-pixel mask flagged {n_before_dilation} pixel(s) "
                f"({100*frac_before_dilation:.1f}% of the frame, before dilation) -- "
                "that's far more than a real fixed-defect count and likely means the "
                "noise model doesn't match this data (check --gain/--readnoise against "
                "your real header values, or flat-fielding may be missing/insufficient). "
                "NOT applying this mask automatically -- rerun with --bpm-sigclip raised "
                "well above the default, or fix flat-fielding/gain/readnoise, and check "
                "the pre-dilation count drops to a small number of pixels before trusting it."
            )
            bad_pixel_mask = None  # don't apply it, and don't claim we did in the HISTORY card
        elif n_before_dilation:
            n_flagged = int(bad_pixel_mask.sum())
            print(f"masking {n_flagged} bad-pixel-mask pixel(s) to NaN "
                  f"in all {len(calibrated)} calibrated science frame(s)")
            for frame in calibrated:
                frame[bad_pixel_mask] = np.nan
        else:
            print("bad-pixel mask: no consistently-bad pixels found")

    if args.save_calibrated_frames:
        for f, frame, hdr in zip(science_files, calibrated, frame_headers):
            out_name = os.path.splitext(os.path.basename(f))[0] + "_calibrated.fits"
            safe_writeto(
                fits.PrimaryHDU(data=frame, header=hdr),
                os.path.join(args.output_dir, out_name),
            )

    # -- Align to the first frame --------------------------------------- #
    print("aligning frames..." + ("" if HAVE_ASTROALIGN
                                   else "  (astroalign not installed -- "
                                        "using integer-pixel fallback; "
                                        "`pip install astroalign` for better results)"))
    reference = calibrated[0]
    aligned = [reference]
    for i, img in enumerate(calibrated[1:], start=2):
        print(f"  aligning frame {i}/{len(calibrated)}: {science_files[i-1]}")
        aligned.append(align_frame(reference, img))

    # -- Combine (sigma-clipped mean) ------------------------------------ #
    print("combining aligned frames (sigma-clipped mean)...")
    cube = np.stack(aligned, axis=0)
    clipped = sigma_clip(cube, sigma=3, maxiters=5, axis=0, masked=True)
    stacked = np.ma.mean(clipped, axis=0).filled(np.nanmean(cube, axis=0))
    stacked = stacked.astype(np.float32)

    # -- Write result ------------------------------------------------------ #
    out_hdr = ref_header.copy()
    out_hdr["FILTER"] = (args.filter, "Filter used for this stack (per --filter arg)")
    out_hdr["HISTORY"] = f"Calibrated with bias root '{args.bias_root}', " \
                          f"dark root '{args.dark_root}'" + \
        (f", flat root '{args.flat_root}'" if flat_files else " (no flat-fielding)")
    out_hdr["HISTORY"] = "Master-bias subtracted (full-frame 2D subtraction)"
    out_hdr["HISTORY"] = "Dark-current subtracted (rate map from " + \
        ", ".join(os.path.basename(f) for f in dark_files) + \
        ", scaled to each frame's EXPTIME)"
    if flat_files:
        out_hdr["HISTORY"] = "Flat-fielded (master flat from " + \
            ", ".join(os.path.basename(f) for f in flat_files) + ")"
    if bad_pixel_mask is not None:
        out_hdr["HISTORY"] = (
            f"Fixed bad-pixel mask: {int(bad_pixel_mask.sum())} pixel(s) masked to NaN "
            f"in every science frame pre-alignment (LACosmic via astroscrappy, "
            f"sigclip={args.bpm_sigclip}, dilate={args.bpm_dilate} iter)"
        )
        out_hdr["BPMASKN"] = (int(bad_pixel_mask.sum()), "Bad-pixel-mask pixels masked pre-alignment")
    elif args.no_bad_pixel_mask:
        out_hdr["HISTORY"] = "Fixed bad-pixel-mask detection skipped (--no-bad-pixel-mask)"
    elif master_flat is None and not args.force_bad_pixel_mask:
        out_hdr["HISTORY"] = ("Fixed bad-pixel-mask detection skipped (no flat-fielding; "
                               "LACosmic needs flat-fielded input -- see --force-bad-pixel-mask)")
    elif not HAVE_ASTROSCRAPPY:
        out_hdr["HISTORY"] = "Fixed bad-pixel-mask detection skipped (astroscrappy not installed)"
    else:
        out_hdr["HISTORY"] = ("Fixed bad-pixel-mask detection ran but flagged too large a "
                               "fraction of the frame to trust -- NOT applied (see log)")
    out_hdr["HISTORY"] = f"Stacked from {len(science_files)} frames: " + \
        ", ".join(os.path.basename(f) for f in science_files)
    out_hdr["HISTORY"] = "Aligned via " + ("astroalign" if HAVE_ASTROALIGN
                                            else "integer-pixel cross-correlation")
    out_hdr["HISTORY"] = "Combined via 3-sigma-clipped mean"
    out_hdr["NCOMBINE"] = (len(science_files), "Number of frames combined")
    safe_writeto(
        fits.PrimaryHDU(data=stacked, header=out_hdr),
        os.path.join(args.output_dir, out_stack),
    )
    print(f"wrote {out_stack}")


if __name__ == "__main__":
    main()