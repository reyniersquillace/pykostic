#!/usr/bin/env python3
"""
map_offset_region.py

Plot the whole ARCTIC frame with the naive-WCS-vs-catalog residual for every
matched detection, color-coded by residual size, so a spatially LOCALIZED
astrometry problem (as opposed to a global/quadrant-wide one) shows up as a
visible cluster rather than something you have to infer from lists of
numbers. Also draws the amplifier quadrant boundaries (from the header's own
DSECxx layout, same convention as calibrate_stack.py) so a real-vs-defect
boundary can be checked against them directly.

This widens the net past the tight --match-radius-arcsec cross-match: for
every detected (unsaturated) star, it finds the nearest catalog star at
whatever separation, and colors by that separation. A tight cluster of
large-residual points sitting in one place (rather than scattered evenly
across the whole frame, which is what pure catalog crowding would look like)
is the signature of a real, localized problem.

Example:
    python3 map_offset_region.py J2250_z_stack.fits --catalog ps1
"""

import argparse

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
from photutils.detection import DAOStarFinder
from astroquery.sdss import SDSS
from astroquery.vizier import Vizier


def clean_icrs(skycoord):
    return SkyCoord(skycoord.icrs.frame)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stack_file")
    p.add_argument("--fwhm-pix", type=float, default=4.0)
    p.add_argument("--detection-nsigma", type=float, default=6.0)
    p.add_argument("--saturation-adu", type=float, default=40000.0)
    p.add_argument("--mask-bad-quadrant", dest="mask_bad_quadrant",
                    action="store_true", default=True,
                    help="Mask the known-bad lower-left quadrant before computing "
                         "background stats and running DAOStarFinder, same as "
                         "refine_astrometry.py -- without this, that quadrant's "
                         "contamination can skew the background statistics enough "
                         "that no detections pass DAOStarFinder's filtering at all.")
    p.add_argument("--no-mask-bad-quadrant", dest="mask_bad_quadrant", action="store_false")
    p.add_argument("--catalog", choices=["sdss", "ps1", "auto"], default="auto")
    p.add_argument("--good-thresh-arcsec", type=float, default=2.0,
                    help="Residuals at or below this are plotted as 'good' (small dots)")
    p.add_argument("--bad-thresh-arcsec", type=float, default=8.0,
                    help="Residuals at or above this are plotted as 'bad' (large red dots); "
                         "in between is an intermediate color")
    p.add_argument("--min-flux", type=float, default=None,
                    help="Only plot/analyze detections with DAOStarFinder flux >= this value. "
                         "Use this to re-run restricted to bright stars only, which should have "
                         "real catalog counterparts if the astrometry is right -- much less "
                         "affected by the chance-crowding noise from faint/below-catalog-depth "
                         "detections that dominates the unrestricted map.")
    p.add_argument("--output", default="offset_map.png")
    return p.parse_args()


def query_sdss_positions(center, radius_deg, data_release=17):
    dra = radius_deg / np.cos(np.deg2rad(center.dec.deg))
    query = f"""
    SELECT p.ra, p.dec
    FROM PhotoPrimary AS p
    WHERE p.ra BETWEEN {center.ra.deg - dra} AND {center.ra.deg + dra}
      AND p.dec BETWEEN {center.dec.deg - radius_deg} AND {center.dec.deg + radius_deg}
      AND p.type = 6 AND p.clean = 1
    """
    try:
        table = SDSS.query_sql(query, data_release=data_release)
    except Exception as exc:
        print(f"  SDSS query failed: {exc}")
        return None
    if table is None or len(table) == 0:
        return None
    return SkyCoord(table["ra"] * u.deg, table["dec"] * u.deg)


def query_ps1_positions(center, radius_deg):
    v = Vizier(columns=["RAJ2000", "DEJ2000"], row_limit=-1)
    try:
        result = v.query_region(center, radius=radius_deg * u.deg, catalog="II/349/ps1")
    except Exception as exc:
        print(f"  PS1 (VizieR) query failed: {exc}")
        return None
    if not result:
        return None
    table = result[0]
    return SkyCoord(table["RAJ2000"], table["DEJ2000"])


def parse_iraf_section(value):
    s = value.strip().lstrip("[").rstrip("]")
    xpart, ypart = s.split(",")
    x1, x2 = (int(v) for v in xpart.split(":"))
    y1, y2 = (int(v) for v in ypart.split(":"))
    return (y1, y2), (x1, x2)


def get_quadrant_boundaries(header):
    """Return the (x, y) pixel line(s) separating amplifier quadrants, from
    the header's own DSECxx layout -- so the plotted boundary is this
    dataset's ACTUAL quadrant split, not an assumed array-index midpoint."""
    xs, ys = set(), set()
    for q in ["11", "12", "21", "22"]:
        key = f"DSEC{q}"
        if key not in header:
            return None, None
        (y1, y2), (x1, x2) = parse_iraf_section(header[key])
        xs.update([x1, x2])
        ys.update([y1, y2])
    return sorted(xs), sorted(ys)


def main():
    args = parse_args()

    with fits.open(args.stack_file) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
    naive_wcs = WCS(header)
    ny, nx = data.shape

    bad_pixel_mask = np.zeros(data.shape, dtype=bool)
    if args.mask_bad_quadrant:
        bad_pixel_mask[: ny // 2, : nx // 2] = True

    mean, median, std = sigma_clipped_stats(data, mask=bad_pixel_mask, sigma=3.0)
    finder = DAOStarFinder(fwhm=args.fwhm_pix, threshold=args.detection_nsigma * std)
    sources = finder(data - median, mask=bad_pixel_mask)
    if sources is None or len(sources) == 0:
        raise SystemExit("No sources detected.")
    unsaturated = (sources["peak"] + median) < args.saturation_adu
    sources = sources[unsaturated]
    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
    det_pix_x = np.asarray(sources[xcol])
    det_pix_y = np.asarray(sources[ycol])
    det_flux = np.asarray(sources["flux"])
    print(f"{len(det_pix_x)} unsaturated detection(s)")

    if args.min_flux is not None:
        keep = det_flux >= args.min_flux
        n_dropped = int(np.sum(~keep))
        det_pix_x, det_pix_y, det_flux = det_pix_x[keep], det_pix_y[keep], det_flux[keep]
        print(f"--min-flux {args.min_flux:.0f}: dropped {n_dropped} fainter detection(s), "
              f"{len(det_pix_x)} remain")
        if len(det_pix_x) == 0:
            raise SystemExit("No detections remain after the --min-flux cut.")

    det_sky = clean_icrs(naive_wcs.pixel_to_world(det_pix_x, det_pix_y))
    center = clean_icrs(naive_wcs.pixel_to_world(nx / 2, ny / 2))
    scale_deg = proj_plane_pixel_scales(naive_wcs)[0]
    radius_deg = 0.6 * max(nx, ny) * scale_deg

    catalog_sky = None
    catalog_name = None
    if args.catalog in ("sdss", "auto"):
        catalog_sky = query_sdss_positions(center, radius_deg)
        if catalog_sky is not None:
            catalog_name = "SDSS"
    if catalog_sky is None and args.catalog in ("ps1", "auto"):
        catalog_sky = query_ps1_positions(center, radius_deg)
        if catalog_sky is not None:
            catalog_name = "Pan-STARRS1"
    if catalog_sky is None:
        raise SystemExit("No SDSS or PS1 coverage found for this field.")
    print(f"using {catalog_name}: {len(catalog_sky)} reference star(s)")

    idx, sep2d, _ = det_sky.match_to_catalog_sky(catalog_sky)
    sep_arcsec = sep2d.arcsec

    # --- Poisson chance-coincidence caveat -----------------------------------
    # Same statistical issue already identified in refine_astrometry.py's
    # wide-radius diagnostic: at this catalog's density, EVERY detection --
    # even one with perfect astrometry but no real bright-enough catalog
    # counterpart nearby (e.g. a genuine star fainter than this catalog's
    # depth, or a spurious/noise detection) -- still gets assigned "the
    # nearest catalog star" at whatever distance that happens to be. That
    # distance follows a Poisson/nearest-neighbor distribution set purely by
    # catalog density, with NOTHING to do with whether the astrometry is
    # actually right. Compare the OBSERVED good/mid/bad counts below against
    # what pure chance alone would predict before treating this map's colors
    # as evidence of a real, localized problem.
    frame_area_arcsec2 = (nx * scale_deg * 3600.0) * (ny * scale_deg * 3600.0)
    catalog_density = len(catalog_sky) / frame_area_arcsec2

    def chance_hit_prob(radius_arcsec):
        lam = catalog_density * np.pi * radius_arcsec**2
        return lam, 1.0 - np.exp(-lam)

    lam_good, p_good = chance_hit_prob(args.good_thresh_arcsec)
    lam_bad, p_not_bad = chance_hit_prob(args.bad_thresh_arcsec)
    p_bad = 1.0 - p_not_bad
    p_mid = p_not_bad - p_good
    n_det = len(det_pix_x)
    exp_good, exp_mid, exp_bad = n_det * p_good, n_det * p_mid, n_det * p_bad

    obs_good = int(np.sum(sep_arcsec <= args.good_thresh_arcsec))
    obs_mid = int(np.sum((sep_arcsec > args.good_thresh_arcsec) & (sep_arcsec < args.bad_thresh_arcsec)))
    obs_bad = int(np.sum(sep_arcsec >= args.bad_thresh_arcsec))

    print(f"\npure-chance expectation (lambda({args.good_thresh_arcsec}\")={lam_good:.2f}, "
          f"lambda({args.bad_thresh_arcsec}\")={lam_bad:.2f} at this catalog's density of "
          f"{catalog_density:.4f}/arcsec^2):")
    print(f"  if EVERY detection had no real catalog counterpart at all (pure noise/below-depth "
          f"case), you'd still expect ~{exp_good:.0f} 'good', ~{exp_mid:.0f} 'mid', ~{exp_bad:.0f} "
          "'bad' out of "
          f"{n_det} purely from nearest-neighbor crowding")
    for name, obs, exp in [("good", obs_good, exp_good), ("mid", obs_mid, exp_mid), ("bad", obs_bad, exp_bad)]:
        sigma = np.sqrt(exp) if exp > 0 else float("nan")
        z = (obs - exp) / sigma if sigma else float("nan")
        print(f"  observed {name:4s}: {obs:4d}   vs chance-only expectation {exp:6.1f}   "
              f"({z:+.2f} sigma)")
    if lam_bad > 1.0:
        print(f"  lambda({args.bad_thresh_arcsec}\") > 1: even the 'bad' (red) threshold is not "
              "selective at this catalog's density -- a large fraction of red/yellow points below "
              "are expected even with PERFECT astrometry, most likely because many detections are "
              "real stars fainter than this catalog's depth (so they have no true counterpart to "
              "match at all) rather than evidence of a spatially localized problem. Don't trust the "
              "raw color counts or their spatial spread as confirmation of anything until the "
              "brightness-stratified breakdown below is checked too.")

    # --- brightness-stratified breakdown --------------------------------------
    # If most of the red/yellow population above is just chance-crowding noise
    # from faint, below-catalog-depth detections, then BRIGHT detections
    # (which should have a real catalog counterpart if the astrometry is
    # right) should show a much lower bad/mid fraction than faint ones. If
    # bright stars show a comparably high bad fraction, that's evidence of a
    # real problem that isn't just a detection-depth artifact.
    order = np.argsort(det_flux)[::-1]
    n_split = max(1, n_det // 4)
    bright_idx = order[:n_split]
    faint_idx = order[-n_split:]
    print(f"\nbrightness-stratified check (top {n_split} vs bottom {n_split} of {n_det} by flux):")
    for label, sub in [("brightest quartile", bright_idx), ("faintest quartile", faint_idx)]:
        sub_sep = sep_arcsec[sub]
        g = int(np.sum(sub_sep <= args.good_thresh_arcsec))
        m = int(np.sum((sub_sep > args.good_thresh_arcsec) & (sub_sep < args.bad_thresh_arcsec)))
        b = int(np.sum(sub_sep >= args.bad_thresh_arcsec))
        print(f"  {label:20s}: good={g:4d} ({100*g/len(sub):4.1f}%)  mid={m:4d} "
              f"({100*m/len(sub):4.1f}%)  bad={b:4d} ({100*b/len(sub):4.1f}%)  "
              f"flux range [{det_flux[sub].min():.0f}, {det_flux[sub].max():.0f}]")
    print("  if the brightest quartile's bad% is close to the faintest quartile's, that argues "
          "against \"most bad points are just faint stars below catalog depth\" and toward a real, "
          "brightness-independent problem (e.g. the fringing artifact) affecting stars across the "
          "flux range.")

    fig, ax = plt.subplots(figsize=(9, 9))
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(data, origin="lower", cmap="gray", norm=norm)

    if args.mask_bad_quadrant:
        ax.add_patch(plt.Rectangle((0, 0), nx // 2, ny // 2, facecolor="blue",
                                    alpha=0.15, edgecolor="none", zorder=1,
                                    label="known-bad quadrant (masked out)"))

    good = sep_arcsec <= args.good_thresh_arcsec
    mid = (sep_arcsec > args.good_thresh_arcsec) & (sep_arcsec < args.bad_thresh_arcsec)
    bad = sep_arcsec >= args.bad_thresh_arcsec

    ax.scatter(det_pix_x[good], det_pix_y[good], s=15, facecolors="none",
               edgecolors="lime", linewidths=0.8,
               label=f"nearest catalog star <= {args.good_thresh_arcsec}\" ({np.sum(good)})")
    ax.scatter(det_pix_x[mid], det_pix_y[mid], s=25, facecolors="none",
               edgecolors="yellow", linewidths=1.0,
               label=f"{args.good_thresh_arcsec}\"-{args.bad_thresh_arcsec}\" ({np.sum(mid)})")
    ax.scatter(det_pix_x[bad], det_pix_y[bad], s=45, facecolors="none",
               edgecolors="red", linewidths=1.5,
               label=f">= {args.bad_thresh_arcsec}\" ({np.sum(bad)})")

    qx, qy = get_quadrant_boundaries(header)
    if qx:
        for x in qx:
            ax.axvline(x, color="cyan", lw=0.7, alpha=0.6)
        for y in qy:
            ax.axhline(y, color="cyan", lw=0.7, alpha=0.6)
        print(f"quadrant boundaries (from DSECxx): x={qx}  y={qy}")

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    ax.set_title(f"Nearest-{catalog_name}-star separation by detected position\n"
                 "(green=good, yellow=intermediate, red=large offset; cyan=quadrant edges)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")

    print("\nred (large-offset) detection pixel positions, for reference:")
    for x, y, s in sorted(zip(det_pix_x[bad], det_pix_y[bad], sep_arcsec[bad]), key=lambda t: -t[2]):
        print(f"  ({x:8.1f}, {y:8.1f})  sep={s:.2f}\"")


if __name__ == "__main__":
    main()