#!/usr/bin/env python3
"""
refine_astrometry.py

Check whether an ARCTIC stack's header WCS carries a systematic
pointing-model offset (common for a "raw", non-plate-solved WCS written
by a telescope control system), by cross-matching detected field stars
against SDSS or Pan-STARRS1 and fitting a refined WCS from those matches
(astropy.wcs.utils.fit_wcs_from_points). Then re-derives the sky position
of a candidate source (e.g. a centroided optical counterpart) under the
corrected WCS, so it can be fairly compared against an external reference
position (a PanSTARRS detection, a pulsar timing localization, etc.).

Example:
    python3 refine_astrometry.py J2250_z_stack.fits \
        --candidate-ra "22:48:48.1926" --candidate-dec "+69:19:08.433" \
        --compare-ra "22:48:47.7159" --compare-dec "+69:19:08.270"
"""

import argparse
import sys

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points, proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astroquery.sdss import SDSS
from astroquery.vizier import Vizier


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stack_file")
    p.add_argument("--fwhm-pix", type=float, default=4.0)
    p.add_argument("--detection-nsigma", type=float, default=6.0)
    p.add_argument("--saturation-adu", type=float, default=40000.0)
    p.add_argument("--match-radius-arcsec", type=float, default=3.0,
                    help="Cross-match tolerance against the catalog for the INITIAL match "
                         "(before the WCS is refined), used only to find enough stars to "
                         "fit against -- not related to the candidate's own offset.")
    p.add_argument("--mask-bad-quadrant", dest="mask_bad_quadrant",
                    action="store_true", default=True)
    p.add_argument("--no-mask-bad-quadrant", dest="mask_bad_quadrant",
                    action="store_false")
    p.add_argument("--catalog", choices=["sdss", "ps1", "auto"], default="auto")
    p.add_argument("--candidate-ra", default=None,
                    help="RA of a candidate source to re-derive under the refined WCS "
                         "(optional -- omit this and --candidate-dec to just validate/"
                         "refine the WCS itself, e.g. after a pipeline change, without "
                         "needing any specific position of interest yet)")
    p.add_argument("--candidate-dec", default=None,
                    help="Dec of a candidate source, e.g. '+69:19:08.433' or decimal degrees "
                         "(optional, see --candidate-ra)")
    p.add_argument("--compare-ra", default=None,
                    help="RA of an external reference position to compare against (optional)")
    p.add_argument("--compare-dec", default=None,
                    help="Dec of an external reference position to compare against (optional)")
    p.add_argument("--output-fits", default=None,
                    help="If given, write a copy of stack_file with the refined WCS baked "
                         "into the header (data unchanged) to this path, so DS9/other "
                         "scripts pick up the corrected astrometry automatically.")
    return p.parse_args()


# WCS-related keywords that could be left over from the ORIGINAL (naive) header
# and would otherwise conflict with / shadow the refined solution written by
# refined_wcs.to_header() (e.g. a leftover CD matrix if the new solution is
# expressed as PC+CDELT, or vice versa).
WCS_KEYWORDS_TO_STRIP = [
    "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "CDELT1", "CDELT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
    "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    "PC001001", "PC001002", "PC002001", "PC002002",
    "LATPOLE", "LONPOLE", "RADESYS", "RADECSYS", "EQUINOX",
]


def write_corrected_fits(data, original_header, refined_wcs, output_path):
    new_header = original_header.copy()
    for key in WCS_KEYWORDS_TO_STRIP:
        if key in new_header:
            del new_header[key]
    new_header.update(refined_wcs.to_header())
    new_header["HISTORY"] = "WCS refined by refine_astrometry.py via fit_wcs_from_points"
    hdu = fits.PrimaryHDU(data=data, header=new_header)
    hdu.verify("silentfix")
    hdu.writeto(output_path, overwrite=True)
    print(f"\nwrote corrected FITS file: {output_path}")


def clean_icrs(skycoord):
    """Convert to ICRS and strip leftover non-ICRS frame attributes (e.g. an
    'equinox' inherited from an FK5 origin). Without this, SkyCoord objects
    that are numerically in ICRS but still carry such leftover attributes
    fail SkyCoord.is_equivalent_frame() against a "clean" ICRS SkyCoord, which
    spherical_offsets_to() requires (unlike separation(), which is tolerant)."""
    return SkyCoord(skycoord.icrs.frame)


def parse_coord(ra_str, dec_str):
    try:
        return SkyCoord(float(ra_str) * u.deg, float(dec_str) * u.deg)
    except ValueError:
        return SkyCoord(f"{ra_str} {dec_str}", unit=["hourangle", "deg"])


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
    print(f"background: mean={mean:.2f} median={median:.2f} std={std:.2f}")

    finder = DAOStarFinder(fwhm=args.fwhm_pix, threshold=args.detection_nsigma * std)
    sources = finder(data - median, mask=bad_pixel_mask)
    if sources is None or len(sources) == 0:
        sys.exit("No sources detected -- try loosening --detection-nsigma or --fwhm-pix.")

    unsaturated = (sources["peak"] + median) < args.saturation_adu
    sources = sources[unsaturated]
    print(f"detected {len(sources)} unsaturated candidate field star(s)")
    if len(sources) < 4:
        sys.exit("Fewer than 4 usable detections -- can't reliably fit a refined WCS.")

    # photutils renamed xcentroid/ycentroid -> x_centroid/y_centroid; support both.
    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
    det_pix_x = np.asarray(sources[xcol])
    det_pix_y = np.asarray(sources[ycol])
    det_flux = np.asarray(sources["flux"])
    # Normalize to ICRS here so every downstream SkyCoord (detections, catalog
    # matches, candidate) is in the same frame class. naive_wcs.pixel_to_world()
    # returns whatever frame the header's RADECSYS/EQUINOX declare (FK5 for
    # ARCTIC's TCC-written headers), which match_to_catalog_sky()/separation()
    # transform transparently, but spherical_offsets_to() does not -- it
    # requires matching frame classes and raises ValueError otherwise.
    det_sky = clean_icrs(naive_wcs.pixel_to_world(det_pix_x, det_pix_y))

    center = clean_icrs(naive_wcs.pixel_to_world(nx / 2, ny / 2))
    scale_deg = proj_plane_pixel_scales(naive_wcs)[0]
    radius_deg = 0.6 * max(nx, ny) * scale_deg  # generous half-diagonal padding

    catalog_sky = None
    catalog_name = None
    if args.catalog in ("sdss", "auto"):
        print("querying SDSS ...")
        catalog_sky = query_sdss_positions(center, radius_deg)
        if catalog_sky is not None:
            catalog_name = "SDSS"
    if catalog_sky is None and args.catalog in ("ps1", "auto"):
        print("querying Pan-STARRS1 (VizieR) ...")
        catalog_sky = query_ps1_positions(center, radius_deg)
        if catalog_sky is not None:
            catalog_name = "Pan-STARRS1"

    if catalog_sky is None:
        sys.exit("No SDSS or PS1 coverage found for this field -- can't refine astrometry "
                  "without an external reference catalog.")
    print(f"using {catalog_name} catalog: {len(catalog_sky)} reference star(s) in field")

    idx, sep2d, _ = det_sky.match_to_catalog_sky(catalog_sky)
    good = sep2d < (args.match_radius_arcsec * u.arcsec)
    n_matched = int(np.sum(good))
    print(f"{n_matched} detection(s) matched within {args.match_radius_arcsec} arcsec of "
          f"the naive-WCS position")
    if n_matched < 4:
        sys.exit("Fewer than 4 matches -- not enough spatial coverage to constrain a refined "
                  "WCS (offset/scale/rotation). Try increasing --match-radius-arcsec if you "
                  "suspect the systematic offset itself is larger than the current tolerance.")

    # match_to_catalog_sky() already found the NEAREST catalog star for every
    # detection, not just the ones within --match-radius-arcsec -- 'good'
    # above just decides which of those get used for the fit. A detection
    # whose naive-WCS position is off by more than the match radius never
    # gets within reach of its true catalog counterpart, so it's silently
    # dropped here and invisible to every check above (the median, the
    # per-quadrant breakdown, even the per-star outlier flagging -- all of
    # them only ever look at the subset that already matched). That's a
    # structural blind spot: a star with a genuinely large individual offset
    # looks identical to a star with no catalog coverage at all. Surface the
    # excluded-but-plausibly-findable ones directly instead of discarding
    # them, using a much wider (but still bounded, to avoid crowded-field
    # false positives) search.
    wide_radius_arcsec = max(20.0, 5.0 * args.match_radius_arcsec)
    far_but_findable = (~good) & (sep2d < (wide_radius_arcsec * u.arcsec))
    n_far = int(np.sum(far_but_findable))

    # In a dense catalog, "some catalog star within wide_radius_arcsec" can
    # be nearly guaranteed by chance alone, regardless of whether it's a real
    # counterpart -- this check is only informative if it's actually
    # SELECTIVE. Model it properly as a Poisson process: lambda is the
    # expected number of catalog stars falling within wide_radius_arcsec of
    # an arbitrary point, so 1-exp(-lambda) is the chance any single
    # unmatched detection has AT LEAST ONE coincidental "match" even with no
    # real counterpart there at all. (A naive density x area x n_unmatched
    # estimate overcounts badly once lambda gets much above 1, since it
    # doesn't saturate the way "at least one hit per detection" must --
    # tested against this exact dataset's numbers before shipping this.)
    frame_area_arcsec2 = (nx * scale_deg * 3600.0) * (ny * scale_deg * 3600.0)
    catalog_density = len(catalog_sky) / frame_area_arcsec2
    n_unmatched = int(np.sum(~good))
    lam = catalog_density * np.pi * wide_radius_arcsec**2
    chance_hit_prob = 1.0 - np.exp(-lam)
    expected_chance = n_unmatched * chance_hit_prob
    print(f"\ndetections excluded from the {args.match_radius_arcsec}\" match above, but with "
          f"a nearest {catalog_name} star within {wide_radius_arcsec:.0f}\": {n_far} of "
          f"{n_unmatched} unmatched detection(s)")
    print(f"  (at this catalog's density, an arbitrary point has ~{lam:.1f} catalog star(s) "
          f"within {wide_radius_arcsec:.0f}\" on average -- a {chance_hit_prob*100:.0f}% chance "
          "any given unmatched detection finds SOME 'match' with no real counterpart there at "
          f"all, i.e. ~{expected_chance:.0f} of the {n_unmatched} unmatched detections are "
          "expected to show up below purely from crowding)")
    if lam > 1.0:
        print("  lambda > 1: this catalog is too dense for a nearest-neighbor search at this "
              "radius to be selective -- don't read individual entries below as confirmed "
              "problems. Narrow to specific bright stars you already suspect instead (e.g. via "
              "check_star_vs_catalog.py, which checks one star at a time against the catalog "
              "directly rather than nearest-neighbor matching a whole list).")
    if n_far == 0:
        print(f"  none -- every unmatched detection's nearest catalog star is beyond "
              f"{wide_radius_arcsec:.0f}\" (no coverage there, not a large-offset case)")
    else:
        # Focus on the brightest unmatched detections -- a bright star with
        # no good catalog match is far more suspicious (should easily be in
        # PS1) than a faint one, and it keeps the list from being dominated
        # by faint, below-PS1-depth detections that never had a real
        # counterpart to find in the first place.
        far_idx = np.where(far_but_findable)[0]
        far_idx = far_idx[np.argsort(det_flux[far_idx])[::-1]]
        n_show = min(20, len(far_idx))
        print(f"  brightest {n_show} of {n_far} (by ARCTIC detection flux):")
        for i in far_idx[:n_show]:
            this_dra, this_ddec = det_sky[i].spherical_offsets_to(catalog_sky[idx[i]])
            print(f"  pixel ({det_pix_x[i]:8.1f}, {det_pix_y[i]:8.1f})  flux={det_flux[i]:12.1f}  "
                  f"dRA={this_dra.arcsec:+7.2f}\"  dDec={this_ddec.arcsec:+7.2f}\"  "
                  f"sep={sep2d[i].arcsec:.2f}\"")

    matched_pix_x = det_pix_x[good]
    matched_pix_y = det_pix_y[good]
    matched_flux = det_flux[good]
    matched_sky = catalog_sky[idx[good]]

    raw_dra, raw_ddec = det_sky[good].spherical_offsets_to(matched_sky)
    print(f"naive-WCS offset vs {catalog_name} (median over {n_matched} matched stars): "
          f"dRA={np.median(raw_dra.arcsec):+.3f}\"  dDec={np.median(raw_ddec.arcsec):+.3f}\"")

    # Is the brightest end of the WELL-BEHAVED (already-matched) population
    # comparably bright to the problem stars we're chasing? If so, this isn't
    # simply "bright stars are unreliable" -- something specific to their
    # location (or some other property) is the real driver. If the matched
    # set tops out much fainter than the problem stars, brightness-linked
    # effects (saturation/nonlinearity below the hard cutoff, etc.) stay on
    # the table.
    flux_order = np.argsort(matched_flux)[::-1]
    print(f"brightest {min(10, len(matched_flux))} of the {n_matched} matched (well-behaved) "
          "stars, for comparison against known problem-star flux levels:")
    for i in flux_order[:10]:
        print(f"  pixel ({matched_pix_x[i]:8.1f}, {matched_pix_y[i]:8.1f})  "
              f"flux={matched_flux[i]:12.1f}  dRA={raw_dra.arcsec[i]:+6.2f}\"  "
              f"dDec={raw_ddec.arcsec[i]:+6.2f}\"")

    # A single global affine (TAN, no distortion) fit can look accurate in
    # the whole-frame median while still being many arcsec off in any one
    # amplifier quadrant that's under-represented among the matched stars --
    # a real risk for ARCTIC's quad-readout mosaic, where each amplifier's
    # geometric placement is not guaranteed to agree with the others at the
    # sub-arcsec level. Break the same residuals down by quadrant (split at
    # the array's half-width/half-height, same convention as
    # --mask-bad-quadrant above) so a quadrant-dependent problem shows up
    # directly instead of being averaged away.
    mid_x, mid_y = nx / 2.0, ny / 2.0
    quad_ns_ew = np.where(matched_pix_y < mid_y, "lower", "upper")
    quad_labels = np.char.add(quad_ns_ew, np.where(matched_pix_x < mid_x, "-left", "-right"))
    print(f"\nper-quadrant breakdown of naive-WCS offset vs {catalog_name} "
          "(checks whether the global fit below is hiding a quadrant-dependent "
          "systematic -- if one quadrant's offset differs sharply from the "
          "others, a single global WCS can't correct it):")
    for q in ["lower-left", "lower-right", "upper-left", "upper-right"]:
        sel = quad_labels == q
        n = int(np.sum(sel))
        if n == 0:
            print(f"  {q:12s}: 0 matched star(s) -- no data to check this quadrant")
            continue
        print(f"  {q:12s}: {n:3d} matched star(s)  "
              f"dRA={np.median(raw_dra.arcsec[sel]):+7.3f}\"  "
              f"dDec={np.median(raw_ddec.arcsec[sel]):+7.3f}\"")

    # The median (both above and per-quadrant) is robust to a handful of BAD
    # individual matches -- e.g. a detection cross-matched to the wrong
    # catalog star because a closer-but-incorrect neighbor happened to sit
    # within --match-radius-arcsec in a crowded field. That kind of outlier
    # is invisible in a median even though it's sitting right there, and
    # would silently corrupt fit_wcs_from_points below (least-squares has no
    # way to know a pair is wrong). List any individual match whose own
    # residual is well outside the pack so it can be checked directly (e.g.
    # against the actual external catalog) rather than trusted blindly.
    sep_arcsec = np.hypot(raw_dra.arcsec, raw_ddec.arcsec)
    outlier_thresh = max(2.0, 5.0 * np.median(np.abs(sep_arcsec - np.median(sep_arcsec))))
    outliers = np.where(sep_arcsec > outlier_thresh)[0]
    print(f"\nper-star residuals vs {catalog_name} (flagging any individual match "
          f">{outlier_thresh:.2f}\" from the pack -- a real outlier here means THAT "
          "specific star was probably cross-matched to the wrong catalog entry, "
          "which the median above won't reveal):")
    if len(outliers) == 0:
        print(f"  no individual match exceeds {outlier_thresh:.2f}\" -- residuals look "
              "consistent star-to-star")
    else:
        for i in outliers:
            print(f"  pixel ({matched_pix_x[i]:8.1f}, {matched_pix_y[i]:8.1f})  "
                  f"dRA={raw_dra.arcsec[i]:+7.2f}\"  dDec={raw_ddec.arcsec[i]:+7.2f}\"  "
                  f"sep={sep_arcsec[i]:.2f}\"  <-- likely mismatched to the wrong "
                  "catalog star")

    refined_wcs = fit_wcs_from_points(
        (matched_pix_x, matched_pix_y),
        matched_sky,
        proj_point="center",
        projection="TAN",
    )

    if args.output_fits:
        write_corrected_fits(data, header, refined_wcs, args.output_fits)

    if args.candidate_ra and args.candidate_dec:
        candidate = parse_coord(args.candidate_ra, args.candidate_dec)
        cand_pix_x, cand_pix_y = naive_wcs.world_to_pixel(candidate)
        corrected_candidate = refined_wcs.pixel_to_world(cand_pix_x, cand_pix_y)

        print("\n--- candidate position ---")
        print(f"naive-WCS sky position:   {candidate.to_string('hmsdms')}")
        print(f"refined-WCS sky position: {corrected_candidate.to_string('hmsdms')}")
        print(f"correction applied: {candidate.separation(corrected_candidate).arcsec:.3f} arcsec")

        if args.compare_ra and args.compare_dec:
            compare = parse_coord(args.compare_ra, args.compare_dec)
            sep_before = candidate.separation(compare)
            sep_after = corrected_candidate.separation(compare)
            print("\n--- comparison to external reference position ---")
            print(f"separation BEFORE astrometric correction: {sep_before.arcsec:.3f} arcsec")
            print(f"separation AFTER astrometric correction:  {sep_after.arcsec:.3f} arcsec")
    else:
        print("\nno --candidate-ra/--candidate-dec given -- skipping candidate "
              "position reporting. The naive-WCS-vs-catalog offset printed above "
              "(and the --output-fits file, if requested) is the actual result "
              "of this run: it tells you directly whether the current pipeline's "
              "WCS is astrometrically accurate, independent of any specific "
              "source of interest.")


if __name__ == "__main__":
    main()