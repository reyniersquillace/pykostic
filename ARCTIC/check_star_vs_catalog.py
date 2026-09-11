#!/usr/bin/env python3
"""
compare_cutouts.py

Cut the same angular region out of an ARCTIC stack and a PanSTARRS image
(default: a 30" box centered on wk_coord, the pulsar timing localization),
apply the same contrast-stretch RECIPE to each (ZScale + asinh -- applied
independently per image, since raw ARCTIC counts and PS1 flux units aren't
on the same absolute scale), and plot them side by side on a common
arcsec-offset extent so bright stars can be visually compared for alignment.

Example:
    python3 compare_cutouts.py J2250_z_stack.fits panstarrs_cutout.fits \
        --box-arcsec 30 \
        --candidate-ra "22:48:48.1926" --candidate-dec "+69:19:08.433" \
        --compare-ra "22:48:47.7159" --compare-dec "+69:19:08.270"
"""

import argparse

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
from photutils.detection import DAOStarFinder


def parse_coord(ra_str, dec_str):
    try:
        return SkyCoord(float(ra_str) * u.deg, float(dec_str) * u.deg)
    except ValueError:
        return SkyCoord(f"{ra_str} {dec_str}", unit=["hourangle", "deg"])


def clean_icrs(skycoord):
    """Strip leftover non-ICRS frame attributes (e.g. an 'equinox' carried
    over from an FK5-labeled header) so spherical_offsets_to() -- which
    requires an exact frame-attribute match, unlike separation() -- works."""
    return SkyCoord(skycoord.icrs.frame)


def detect_sources(data, nsigma=6.0, fwhm=4.0, saturation_adu=None, label=""):
    """Run DAOStarFinder and return (sky-position-ready pixel x, pixel y, flux)
    arrays, or (None, None, None) if nothing was detected. Column-name-safe
    across photutils versions (xcentroid/ycentroid -> x_centroid/y_centroid).

    If saturation_adu is given, drops any detection whose peak (background
    added back in) meets or exceeds it -- a saturated PSF's centroid is
    unreliable (flat/clipped core, possible charge-bleed smearing along the
    readout direction), and DAOStarFinder's flux-based sort otherwise puts
    exactly these unreliable detections at the top of the list. This mirrors
    refine_astrometry.py's own --saturation-adu exclusion, which is why its
    catalog check can look clean while an unfiltered comparison here doesn't:
    that check never used the saturated stars in the first place."""
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=nsigma * std)
    sources = finder(data - median)
    if sources is None or len(sources) == 0:
        return None, None, None, median, std
    if saturation_adu is not None:
        unsaturated = (sources["peak"] + median) < saturation_adu
        n_removed = int(np.sum(~unsaturated))
        if n_removed:
            print(f"{label}: excluded {n_removed} likely-saturated detection(s) "
                  f"(peak+background >= {saturation_adu} ADU) -- their centroids "
                  "aren't trustworthy for position comparison")
        sources = sources[unsaturated]
        if len(sources) == 0:
            return None, None, None, median, std
    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
    return np.asarray(sources[xcol]), np.asarray(sources[ycol]), np.asarray(sources["flux"]), median, std


def report_stars(data, wcs, center, label, n=8, nsigma=6.0, saturation_adu=None):
    px, py, flux, median, std = detect_sources(data, nsigma=nsigma,
                                                saturation_adu=saturation_adu, label=label)
    print(f"\n{label}: background median={median:.2f} std={std:.2f}")
    if px is None:
        print(f"{label}: no sources detected above {nsigma}-sigma")
        return
    order = np.argsort(flux)[::-1]
    print(f"{label}: top {min(n, len(px))} of {len(px)} detected source(s), "
          f"offset from box center:")
    for i in order[:n]:
        sky = clean_icrs(wcs.pixel_to_world(px[i], py[i]))
        dra, ddec = clean_icrs(center).spherical_offsets_to(sky)
        print(f"  flux={flux[i]:8.1f}  dRA={dra.arcsec:+7.2f}\"  dDec={ddec.arcsec:+7.2f}\"")


def cross_match_images(arctic_data, arctic_wcs, ps1_data, ps1_wcs,
                        match_radius_arcsec=20.0, nsigma=6.0,
                        arctic_saturation_adu=None, ps1_saturation_adu=None):
    """Detect field stars independently in each image and cross-match them
    directly against EACH OTHER by sky position (each using its own WCS) --
    no external catalog involved. This is the real test of whether the two
    images agree with each other, as opposed to each independently agreeing
    with a reference catalog (which doesn't guarantee they agree with each
    other if, e.g., one of them is internally fine but offset as a whole
    relative to truth in a way the catalog cross-match didn't happen to
    probe, or a naive by-eye/by-brightness star pairing was mistaken)."""
    ax, ay, aflux, _, _ = detect_sources(arctic_data, nsigma=nsigma,
                                          saturation_adu=arctic_saturation_adu, label="ARCTIC")
    px, py, pflux, _, _ = detect_sources(ps1_data, nsigma=nsigma,
                                          saturation_adu=ps1_saturation_adu, label="Pan-STARRS")
    print("\ncross-match: ARCTIC detections vs Pan-STARRS detections directly "
          "(each image's own WCS, no external catalog)")
    if ax is None or px is None:
        print("  not enough detections in one or both images to cross-match")
        return
    arctic_sky = clean_icrs(arctic_wcs.pixel_to_world(ax, ay))
    ps1_sky = clean_icrs(ps1_wcs.pixel_to_world(px, py))

    idx, sep2d, _ = arctic_sky.match_to_catalog_sky(ps1_sky)
    good = sep2d < (match_radius_arcsec * u.arcsec)
    n_good = int(np.sum(good))
    print(f"  {n_good} of {len(arctic_sky)} ARCTIC source(s) matched to a Pan-STARRS "
          f"source within {match_radius_arcsec}\"")
    if n_good == 0:
        print("  no matches at this tolerance -- try a larger --xmatch-radius-arcsec "
              "if you suspect a bigger offset, or check the two boxes actually "
              "cover overlapping sky")
        return

    dra, ddec = arctic_sky[good].spherical_offsets_to(ps1_sky[idx[good]])
    print(f"  median offset (ARCTIC -> Pan-STARRS): "
          f"dRA={np.median(dra.arcsec):+.3f}\"  dDec={np.median(ddec.arcsec):+.3f}\"  "
          f"(scatter: dRA={np.std(dra.arcsec):.3f}\" dDec={np.std(ddec.arcsec):.3f}\")")
    seps_sorted = np.sort(sep2d[good].arcsec)
    print(f"  separation distribution (sorted): "
          f"{', '.join(f'{s:.2f}' for s in seps_sorted)}\"")
    print(f"  (a real match set clusters tightly near the true offset; a spread "
          f"that fills the whole 0-{match_radius_arcsec:.0f}\" range fairly evenly "
          "means most of these are chance nearest-neighbor pairings between "
          "otherwise-unrelated detections, not real star matches -- tighten "
          "--xmatch-radius-arcsec and/or raise --detection-nsigma if so)")
    print("  per-match detail (ARCTIC flux, Pan-STARRS flux, offset, separation):")
    for k, i in enumerate(np.where(good)[0]):
        j = idx[i]
        print(f"    ARCTIC flux={aflux[i]:10.1f}  PS1 flux={pflux[j]:10.1f}  "
              f"dRA={dra.arcsec[k]:+7.2f}\"  dDec={ddec.arcsec[k]:+7.2f}\"  "
              f"sep={sep2d[i].arcsec:.2f}\"")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("arctic_file")
    p.add_argument("panstarrs_file")
    p.add_argument("--center-ra", default="342.1972329",
                    help="Box center RA, deg or sexagesimal (default: wk_coord)")
    p.add_argument("--center-dec", default="69.3196023",
                    help="Box center Dec, deg or sexagesimal (default: wk_coord)")
    p.add_argument("--box-arcsec", type=float, default=30.0,
                    help="Full width/height of the compared region, arcsec")
    p.add_argument("--candidate-ra", default=None)
    p.add_argument("--candidate-dec", default=None)
    p.add_argument("--compare-ra", default=None)
    p.add_argument("--compare-dec", default=None)
    p.add_argument("--output", default="cutout_comparison.png")
    p.add_argument("--xmatch-radius-arcsec", type=float, default=20.0,
                    help="Tolerance for directly cross-matching ARCTIC detections "
                         "against Pan-STARRS detections (own WCS each, no external "
                         "catalog) -- set generously wide on a first pass since the "
                         "point is to find out how big any real offset is, not "
                         "assume it. Once you have a sense of the true offset (or "
                         "already expect it to be small from an external catalog "
                         "check), tighten this -- a loose radius over a sparse "
                         "field lets chance nearest-neighbor pairings between "
                         "otherwise-unrelated detections dominate the statistics.")
    p.add_argument("--detection-nsigma", type=float, default=6.0,
                    help="Detection threshold (in background-sigma) for DAOStarFinder, "
                         "used by both --xmatch and the printed top-N star lists. "
                         "Raise this (e.g. 10-15) if the field is turning up spurious "
                         "low-significance 'detections' (a giveaway: DAOStarFinder's "
                         "flux column coming out negative for some of them) that "
                         "swamp a real signal in the cross-match.")
    p.add_argument("--arctic-saturation-adu", type=float, default=40000.0,
                    help="Exclude ARCTIC detections whose peak (background added "
                         "back in) meets or exceeds this many ADU -- a saturated "
                         "PSF's centroid is unreliable (flat/clipped core, possible "
                         "charge-bleed smearing), and DAOStarFinder's flux-based "
                         "sort otherwise puts exactly these at the top of the "
                         "'brightest source' list. Matches refine_astrometry.py's "
                         "own --saturation-adu default. Pass 0 or a negative number "
                         "to disable.")
    p.add_argument("--ps1-saturation-adu", type=float, default=None,
                    help="Same idea as --arctic-saturation-adu, but for the "
                         "Pan-STARRS image -- left disabled (None) by default since "
                         "its pixel values aren't in ADU and the ARCTIC saturation "
                         "level has no meaning there. Set it explicitly if you know "
                         "the right threshold for your particular Pan-STARRS product.")
    return p.parse_args()


def load_cutout(path, center, box_arcsec, label):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
    wcs = WCS(header).celestial
    scale_arcsec = np.mean(proj_plane_pixel_scales(wcs)) * 3600.0

    requested_x, requested_y = wcs.world_to_pixel(center)
    print(f"{label}: requested center pixel = ({requested_x:.1f}, {requested_y:.1f}) "
          f"in a {data.shape[1]}x{data.shape[0]} (nx,ny) array")

    size_pix = int(np.ceil(box_arcsec / scale_arcsec))
    cutout = Cutout2D(data, position=center, size=(size_pix, size_pix), wcs=wcs, mode="trim")

    achieved_center = cutout.wcs.pixel_to_world(cutout.data.shape[1] / 2, cutout.data.shape[0] / 2)
    print(f"{label}: cutout shape = {cutout.data.shape}, "
          f"actual cutout center = {achieved_center.to_string('hmsdms')} "
          f"({achieved_center.ra.deg:.6f}, {achieved_center.dec.deg:.6f}), "
          f"offset from requested center = {center.separation(achieved_center).arcsec:.2f}\"")

    return cutout.data, scale_arcsec, cutout.wcs


def plot_panel(ax, data, scale_arcsec, center, box_arcsec, title, candidate, compare):
    ny, nx = data.shape
    half_x = 0.5 * nx * scale_arcsec
    half_y = 0.5 * ny * scale_arcsec
    # Both images have RA decreasing with increasing pixel-x (confirmed via
    # their CD/PC matrices), so put +half_x on the left to keep the usual
    # East-left, North-up display consistent between the two panels.
    extent = [half_x, -half_x, -half_y, half_y]

    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(data, origin="lower", cmap="gray", norm=norm, extent=extent)
    ax.set_xlim(box_arcsec / 2, -box_arcsec / 2)
    ax.set_ylim(-box_arcsec / 2, box_arcsec / 2)
    ax.set_aspect("equal")
    ax.set_xlabel('$\\Delta$RA (")')
    ax.set_ylabel('$\\Delta$Dec (")')
    ax.set_title(title)
    ax.plot(0, 0, "+", color="cyan", ms=12, mew=1.5, label="box center")

    if candidate is not None:
        dra, ddec = center.spherical_offsets_to(candidate)
        ax.plot(dra.arcsec, ddec.arcsec, "r+", ms=12, mew=1.5, label="candidate")
    if compare is not None:
        dra, ddec = center.spherical_offsets_to(compare)
        ax.plot(dra.arcsec, ddec.arcsec, "m+", ms=12, mew=1.5, label="compare")


def main():
    args = parse_args()
    center = parse_coord(args.center_ra, args.center_dec)
    candidate = parse_coord(args.candidate_ra, args.candidate_dec) if args.candidate_ra else None
    compare = parse_coord(args.compare_ra, args.compare_dec) if args.compare_ra else None

    arctic_data, arctic_scale, arctic_wcs = load_cutout(
        args.arctic_file, center, args.box_arcsec, "ARCTIC")
    ps1_data, ps1_scale, ps1_wcs = load_cutout(
        args.panstarrs_file, center, args.box_arcsec, "Pan-STARRS")

    arctic_sat = args.arctic_saturation_adu if args.arctic_saturation_adu > 0 else None
    ps1_sat = args.ps1_saturation_adu

    report_stars(arctic_data, arctic_wcs, center, "ARCTIC", nsigma=args.detection_nsigma,
                 saturation_adu=arctic_sat)
    report_stars(ps1_data, ps1_wcs, center, "Pan-STARRS", nsigma=args.detection_nsigma,
                 saturation_adu=ps1_sat)
    cross_match_images(arctic_data, arctic_wcs, ps1_data, ps1_wcs,
                        match_radius_arcsec=args.xmatch_radius_arcsec,
                        nsigma=args.detection_nsigma,
                        arctic_saturation_adu=arctic_sat,
                        ps1_saturation_adu=ps1_sat)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_panel(axes[0], arctic_data, arctic_scale, center, args.box_arcsec,
               "ARCTIC", candidate, compare)
    plot_panel(axes[1], ps1_data, ps1_scale, center, args.box_arcsec,
               "Pan-STARRS", candidate, compare)
    axes[0].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()