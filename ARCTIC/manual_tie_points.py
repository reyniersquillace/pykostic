#!/usr/bin/env python3
"""
manual_tie_points.py

Fit a WCS solution for the ARCTIC stack from tie points YOU identify by eye:
the same star, clicked (or specified) once in the ARCTIC image and once in
the Pan-STARRS reference image. This sidesteps the crowded-field
nearest-neighbor problems that have been contaminating the automated
cross-match in refine_astrometry.py / map_offset_region.py -- a nearest-
neighbor search can't tell two nearby stars apart, but a human eye can, by
using each star's pattern of neighbors rather than just proximity.

Two ways to supply tie points:

1. INTERACTIVE (needs a real display / GUI-capable matplotlib backend):
    python3 manual_tie_points.py J2250_z_stack.fits ps1_ref.fits

   Click a star in the LEFT (ARCTIC) panel, then click the SAME star in the
   RIGHT (Pan-STARRS) panel. Each click is auto-refined to the local flux
   centroid in a small box around it, so you don't need to click pixel-
   perfectly -- just get close (within a few pixels) to the star's core.
   Repeat for as many stars as you can confidently identify (4 minimum to
   fit at all; 8-12+ spread across the WHOLE frame -- including both the
   region you already suspect is bad, and regions you think are fine --
   is much better and lets you see whether the fit residuals are uniform
   or not). Press 'u' to undo the last click, close the window when done.

2. FROM A FILE (works with no display at all -- e.g. if you're running this
   over a headless/remote terminal): prepare a plain text file, one tie
   point per line, as either
       arctic_x arctic_y  ps1_x ps1_y
   or (if you already know the sky coordinates directly, e.g. by reading
   them off a catalog for a star you can identify in the ARCTIC image)
       arctic_x arctic_y  ra_deg dec_deg
   whitespace- or comma-separated, '#' for comments/blank lines allowed.
   Then run:
    python3 manual_tie_points.py J2250_z_stack.fits ps1_ref.fits \\
        --coords-file my_tiepoints.txt

Either way, this prints each tie point's fit residual (in arcsec) so you can
spot a mis-identified star (it'll stick out as a huge outlier relative to
the others) and re-do just that one, and optionally writes a corrected FITS
file with the new WCS.

An interactive session's clicks are ALWAYS auto-saved to a text file (see
--save-coords) before fitting, even if you forget --output-fits -- so if you
like the residuals but didn't ask for a FITS file, you don't need to
re-click anything: rerun with
    python3 manual_tie_points.py J2250_z_stack.fits ps1_ref.fits \\
        --coords-file <the saved file> --output-fits corrected.fits
(add --quadrant back too, if you used it) to reproduce the exact same fit
and write the FITS this time.

Restricting to one amplifier quadrant
--------------------------------------
A single global WCS (one CRPIX/CD) is a single LINEAR map, and can't in
general be exact on both sides of the gap between amplifier blocks (the gap
is a readout-electronics artifact, not a physically uniform continuation of
the pixel grid -- see calibrate_stack.py's docstring). Pass --quadrant
{LL,UL,LR,UR} to restrict tie-point selection to one quadrant at a time (its
bounds are read from the header's own DSECxx keywords, same convention as
the rest of this pipeline) and fit each quadrant's own small WCS
independently, rather than forcing one fit to serve all of them. With
--output-fits, the written file is CROPPED to just that quadrant's pixels
(CRPIX shifted to match) rather than keeping the full frame -- since the fit
is only valid there anyway, this avoids the earlier multi-quadrant CRPIX
problem entirely by simply not asking one CRPIX to describe more than one
quadrant at a time.
"""

import argparse
import os
import sys

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points, proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("arctic_file")
    p.add_argument("ps1_file", help="A Pan-STARRS (or other trusted-WCS) reference image "
                                     "covering the same field, e.g. the file used by "
                                     "compare_cutouts.py.")
    p.add_argument("--coords-file", default=None,
                    help="Read tie points from this file instead of opening an interactive "
                         "window (see the module docstring for the format).")
    p.add_argument("--centroid-box", type=int, default=15,
                    help="Box size (pixels) used to refine each click/coordinate to the local "
                         "flux centroid before fitting.")
    p.add_argument("--no-refine", action="store_true",
                    help="Use the given/clicked coordinates as-is, skipping local centroid "
                         "refinement (useful if --coords-file already has precise centroids, "
                         "e.g. from DAOStarFinder output).")
    p.add_argument("--output-fits", default=None,
                    help="If given, write a copy of the ARCTIC file with the new fitted WCS "
                         "in its header.")
    p.add_argument("--quadrant", choices=["LL", "UL", "LR", "UR"], default=None,
                    help="Restrict tie-point selection to one ARCTIC amplifier quadrant -- "
                         "lower-left/upper-left/lower-right/upper-right, with bounds read from "
                         "the header's DSECxx keywords (falls back to a simple nx/2,ny/2 split "
                         "if those aren't present). Use this to fit an independent WCS per "
                         "quadrant instead of one global fit -- see the module docstring for "
                         "why a single WCS can't in general be exact across the amplifier gap. "
                         "In interactive mode this also crops the initial view to that quadrant "
                         "(and the corresponding sky region in the Pan-STARRS panel) and warns "
                         "if you click outside it; in --coords-file mode it warns about any "
                         "ARCTIC point falling outside the quadrant's bounds. With --output-fits, "
                         "the written file is cropped to just this quadrant's pixels (CRPIX "
                         "shifted to match) instead of keeping the full frame.")
    p.add_argument("--save-coords", default=None,
                    help="Where to save tie points collected interactively (ignored in "
                         "--coords-file mode). Defaults to an auto-generated name based on the "
                         "ARCTIC filename and quadrant. Always written before fitting, so a "
                         "good click session is never lost even if the fit or --output-fits "
                         "step fails afterward.")
    return p.parse_args()


def parse_iraf_section(value):
    """'[x1:x2,y1:y2]' (FITS/IRAF convention: 1-indexed, inclusive) ->
    ((y1,y2), (x1,x2)) as 1-indexed ints. Same convention used throughout
    this pipeline (calibrate_stack.py, map_offset_region.py)."""
    s = value.strip().lstrip("[").rstrip("]")
    xpart, ypart = s.split(",")
    x1, x2 = (int(v) for v in xpart.split(":"))
    y1, y2 = (int(v) for v in ypart.split(":"))
    return (y1, y2), (x1, x2)


def get_named_quadrant_bounds(header, shape):
    """Return {"LL": (x0, x1, y0, y1), "UL": ..., "LR": ..., "UR": ...} as
    0-indexed, half-open pixel bounds (x0 <= x < x1, y0 <= y < y1) for each
    quadrant of a quad-readout ARCTIC frame.

    This is worked out from the header's own DSECxx keywords by classifying
    each quadrant's data section as the low/high half in x and the low/high
    half in y -- it does NOT assume any particular ARCTIC qq-number-to-
    corner convention (e.g. that DSEC11 is specifically the lower-left
    quadrant), since that convention isn't documented anywhere in this
    pipeline and isn't needed: classifying by actual pixel position is both
    simpler and can't be wrong. Falls back to a plain nx/2, ny/2 split (with
    a printed warning) if DSECxx keywords aren't present at all."""
    ny, nx = shape
    sections = {}
    for q in ["11", "12", "21", "22"]:
        key = f"DSEC{q}"
        if key not in header:
            sections = None
            break
        (y1, y2), (x1, x2) = parse_iraf_section(header[key])
        sections[q] = (x1 - 1, x2, y1 - 1, y2)  # 0-indexed, half-open

    if sections is None:
        print("warning: no DSECxx keywords found in the ARCTIC header -- falling back to a "
              "plain nx/2, ny/2 split for --quadrant bounds; this may not match the real "
              "amplifier boundaries.")
        return {
            "LL": (0, nx // 2, 0, ny // 2),
            "LR": (nx // 2, nx, 0, ny // 2),
            "UL": (0, nx // 2, ny // 2, ny),
            "UR": (nx // 2, nx, ny // 2, ny),
        }

    x_mid = (min(v[0] for v in sections.values()) + max(v[1] for v in sections.values())) / 2.0
    y_mid = (min(v[2] for v in sections.values()) + max(v[3] for v in sections.values())) / 2.0
    named = {}
    for x0, x1, y0, y1 in sections.values():
        xside = "L" if (x0 + x1) / 2.0 < x_mid else "R"
        yside = "L" if (y0 + y1) / 2.0 < y_mid else "U"
        named[yside + xside] = (x0, x1, y0, y1)
    return named


def project_box(src_wcs, dst_wcs, x0, x1, y0, y1, margin_frac=0.15):
    """Project a pixel box's four corners through src_wcs to sky, then
    through dst_wcs back to pixels, returning a view (with a fractional
    margin) in dst's pixel space covering the same sky region. Direction-
    agnostic -- used both ARCTIC->PS1 and PS1->ARCTIC below."""
    corners_x = [x0, x1, x0, x1]
    corners_y = [y0, y0, y1, y1]
    sky = src_wcs.pixel_to_world(corners_x, corners_y)
    px, py = dst_wcs.world_to_pixel(sky)
    pxmin, pxmax = float(np.min(px)), float(np.max(px))
    pymin, pymax = float(np.min(py)), float(np.max(py))
    mx = max((pxmax - pxmin) * margin_frac, 5.0)
    my = max((pymax - pymin) * margin_frac, 5.0)
    return pxmin - mx, pxmax + mx, pymin - my, pymax + my


def box_area_arcsec2(wcs, x0, x1, y0, y1):
    """Angular area (arcsec^2) covered by a pixel box, from the WCS's own
    local pixel scale -- used only to compare the ARCTIC region of interest
    against the loaded Pan-STARRS file's own footprint, so whichever is
    smaller can be used as the shared view for both panels rather than
    always deferring to one side."""
    sx, sy = proj_plane_pixel_scales(wcs)[:2]  # deg/pixel
    return abs(x1 - x0) * abs(y1 - y0) * (sx * 3600.0) * (sy * 3600.0)


def refine_centroid(data, x, y, box):
    """Refine a rough (x, y) pixel position to the local flux centroid within
    a box x box window, explicitly subtracting the box's own local
    background first (a flux-weighted centroid on non-background-subtracted
    data is dominated by the flat sky level, not the star). Falls back to
    the input position if the box goes out of bounds or no positive flux
    remains after background subtraction (e.g. no clear peak nearby)."""
    ny, nx = data.shape
    half = box // 2
    x0, x1 = int(round(x)) - half, int(round(x)) + half + 1
    y0, y1 = int(round(y)) - half, int(round(y)) + half + 1
    if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
        return float(x), float(y)
    cutout = data[y0:y1, x0:x1]
    bkg = np.median(cutout)
    weights = np.clip(cutout - bkg, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = float(np.sum(xx * weights) / total)
    cy = float(np.sum(yy * weights) / total)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return float(x), float(y)
    return cx, cy


def fit_and_report(arctic_xy, sky_points, arctic_wcs_for_proj):
    """Fit a WCS from (pixel, sky) tie points and print per-point residuals.
    Returns the fitted WCS."""
    arctic_xy = np.asarray(arctic_xy, dtype=float)
    if len(arctic_xy) < 4:
        raise SystemExit(f"Need at least 4 tie points to fit a WCS (got {len(arctic_xy)}).")
    proj_point = arctic_wcs_for_proj.pixel_to_world(
        *np.mean(arctic_xy, axis=0)
    )
    new_wcs = fit_wcs_from_points(
        (arctic_xy[:, 0], arctic_xy[:, 1]), sky_points,
        proj_point=proj_point, projection="TAN",
    )
    fitted_sky = new_wcs.pixel_to_world(arctic_xy[:, 0], arctic_xy[:, 1])
    resid_arcsec = fitted_sky.separation(sky_points).arcsec
    med_resid = float(np.median(resid_arcsec))
    mad = float(np.median(np.abs(resid_arcsec - med_resid)))
    outlier_thresh = max(1.0, med_resid + 5.0 * 1.4826 * mad)
    print(f"\nfit from {len(arctic_xy)} tie point(s); per-point residual after fitting "
          "(large outliers here likely mean a mis-identified star -- redo that one):")
    for i, (xy, r) in enumerate(zip(arctic_xy, resid_arcsec)):
        flag = "  <-- outlier?" if r > outlier_thresh else ""
        print(f"  #{i+1:2d}  pixel ({xy[0]:8.1f}, {xy[1]:8.1f})  residual={r:.3f}\"{flag}")
    print(f"  median residual: {med_resid:.3f}\"   max: {np.max(resid_arcsec):.3f}\"")
    if len(arctic_xy) < 8:
        print("  NOTE: with this few points, a single mis-identified star can drag the whole "
              "fit rather than only showing up as a huge LOCAL residual (the fit is a global "
              "least-squares solution). If the median residual looks worse than you'd expect "
              "from click/centroid precision alone (which should be well under 1\"), even "
              "without an obvious outlier above, try dropping one point at a time and refitting "
              "to see which one it is.")
    return new_wcs


def load_coords_file(path):
    """Parse a tie-point text file. Returns (arctic_xy list, ps1_xy list or
    None, sky_deg list or None) -- exactly one of ps1_xy/sky_deg is set.

    Prefers an explicit '# format: pixel' or '# format: radec' directive
    comment line if present (this is what save_coords_file() below always
    writes). Falls back to a heuristic -- 4th column magnitude <= 90 and 3rd
    column in [0,360] looks like RA/Dec degrees, otherwise PS1 pixel x/y --
    ONLY for hand-written files with no directive. The heuristic is
    ambiguous for a small PS1 cutout, where pixel coordinates like (30, 45)
    would be misread as RA/Dec; the directive avoids that entirely, which is
    exactly why saved files always include it."""
    arctic_xy, other_xy = [], []
    fmt_directive = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                lower = line.lower().replace(" ", "")
                if lower.startswith("#format:pixel"):
                    fmt_directive = "pixel"
                elif lower.startswith("#format:radec"):
                    fmt_directive = "radec"
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 4:
                raise SystemExit(f"Bad line in {path} (expected 4 numbers): {line!r}")
            ax, ay, bx, by = (float(v) for v in parts)
            arctic_xy.append((ax, ay))
            other_xy.append((bx, by))
    other_xy = np.array(other_xy)

    if fmt_directive == "radec":
        return arctic_xy, None, other_xy
    if fmt_directive == "pixel":
        return arctic_xy, other_xy, None

    # No directive (a hand-written file) -- fall back to the heuristic.
    looks_like_radec = np.all(np.abs(other_xy[:, 1]) <= 90.0) and np.all(
        (other_xy[:, 0] >= 0) & (other_xy[:, 0] <= 360)
    )
    print("note: no '# format:' directive found in this coords file -- guessing "
          f"{'ra/dec' if looks_like_radec else 'pixel'} from the value ranges. Add a "
          "'# format: pixel' or '# format: radec' line to be unambiguous, especially for "
          "a small Pan-STARRS cutout where pixel coordinates can look like plausible RA/Dec.")
    if looks_like_radec:
        return arctic_xy, None, other_xy
    return arctic_xy, other_xy, None


def save_coords_file(path, arctic_xy, ps1_xy, arctic_file, ps1_file, quadrant):
    """Write collected tie points to a plain text file in the same format
    load_coords_file() reads, with an explicit format directive -- so an
    interactive session's clicks are never lost even if you forget
    --output-fits, and can be replayed later via --coords-file without
    re-clicking."""
    with open(path, "w") as f:
        f.write("# manual_tie_points.py -- saved tie points\n")
        f.write("# format: pixel\n")
        f.write(f"# arctic_file: {arctic_file}\n")
        f.write(f"# ps1_file: {ps1_file}\n")
        if quadrant:
            f.write(f"# quadrant: {quadrant}\n")
        f.write("# arctic_x arctic_y ps1_x ps1_y\n")
        for (ax, ay), (bx, by) in zip(arctic_xy, ps1_xy):
            f.write(f"{ax:.3f} {ay:.3f} {bx:.3f} {by:.3f}\n")
    print(f"\nsaved {len(arctic_xy)} tie point(s) to {path} -- rerun any time with "
          f"'--coords-file {path}' (add --output-fits to write a corrected FITS) to reuse "
          "these without re-clicking.")


class InteractivePicker:
    """Click a star in the ARCTIC panel, then the same star in the PS1
    panel, alternating; each click is refined to a local centroid. 'u' undoes
    the last click."""

    def __init__(self, arctic_data, ps1_data, centroid_box, refine,
                 quadrant=None, quad_bounds=None, arctic_view=None, ps1_view=None):
        self.arctic_data = arctic_data
        self.ps1_data = ps1_data
        self.centroid_box = centroid_box
        self.refine = refine
        self.quadrant = quadrant
        self.quad_bounds = quad_bounds  # (x0, x1, y0, y1) or None -- click validation
        self.arctic_clicks = []   # refined (x, y) in ARCTIC pixels
        self.ps1_clicks = []      # refined (x, y) in PS1 pixels
        self.pending_arctic = None
        self.markers = []

        self.fig, (self.ax_a, self.ax_p) = plt.subplots(1, 2, figsize=(14, 7))
        norm_a = ImageNormalize(arctic_data, interval=ZScaleInterval(), stretch=AsinhStretch())
        norm_p = ImageNormalize(ps1_data, interval=ZScaleInterval(), stretch=AsinhStretch())
        self.ax_a.imshow(arctic_data, origin="lower", cmap="gray", norm=norm_a)
        self.ax_p.imshow(ps1_data, origin="lower", cmap="gray", norm=norm_p)
        quad_label = f" (quadrant {quadrant})" if quadrant else ""
        self.ax_a.set_title(f"ARCTIC{quad_label} -- click a star")
        self.ax_p.set_title("Pan-STARRS -- click the SAME star")
        self.fig.suptitle("Click ARCTIC star, then its match in Pan-STARRS, repeat. "
                           "'u' = undo last pair. Close window when done.")
        # quad_bounds always draws the allowed-click boundary (when --quadrant
        # is set), independent of the actual displayed view below -- clicking
        # outside it is rejected regardless of current zoom/pan.
        if quad_bounds is not None:
            x0, x1, y0, y1 = quad_bounds
            self.ax_a.axvline(x0, color="cyan", lw=0.7, alpha=0.6)
            self.ax_a.axvline(x1, color="cyan", lw=0.7, alpha=0.6)
            self.ax_a.axhline(y0, color="cyan", lw=0.7, alpha=0.6)
            self.ax_a.axhline(y1, color="cyan", lw=0.7, alpha=0.6)
        # arctic_view/ps1_view set the initial displayed extent -- these are
        # whichever of {selected ARCTIC region, whole PS1 file} is smaller in
        # actual sky area, applied to BOTH panels, so neither panel shows a
        # region the other side can't cover.
        if arctic_view is not None:
            x0, x1, y0, y1 = arctic_view
            self.ax_a.set_xlim(x0, x1)
            self.ax_a.set_ylim(y0, y1)
        if ps1_view is not None:
            px0, px1, py0, py1 = ps1_view
            self.ax_p.set_xlim(px0, px1)
            self.ax_p.set_ylim(py0, py1)
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("key_press_event", self.onkey)

    def onclick(self, event):
        if event.inaxes is self.ax_a and self.pending_arctic is None:
            x, y = self._maybe_refine(self.arctic_data, event.xdata, event.ydata)
            if self.quad_bounds is not None:
                x0, x1, y0, y1 = self.quad_bounds
                if not (x0 <= x < x1 and y0 <= y < y1):
                    print(f"  click at ({x:.1f}, {y:.1f}) is outside quadrant "
                          f"{self.quadrant}'s bounds (x=[{x0},{x1}) y=[{y0},{y1})) -- ignored; "
                          "click inside the cyan box.")
                    return
            self.pending_arctic = (x, y)
            (m,) = self.ax_a.plot(x, y, "r+", ms=14, mew=2)
            self.markers.append(m)
            self.ax_a.set_title(f"ARCTIC -- now click the SAME star on the right "
                                 f"(pair #{len(self.arctic_clicks)+1})")
            self.fig.canvas.draw_idle()
        elif event.inaxes is self.ax_p and self.pending_arctic is not None:
            x, y = self._maybe_refine(self.ps1_data, event.xdata, event.ydata)
            self.arctic_clicks.append(self.pending_arctic)
            self.ps1_clicks.append((x, y))
            self.pending_arctic = None
            (m,) = self.ax_p.plot(x, y, "r+", ms=14, mew=2)
            self.markers.append(m)
            n = len(self.arctic_clicks)
            self.ax_a.set_title(f"ARCTIC -- click a star ({n} pair(s) so far)")
            self.fig.canvas.draw_idle()
            print(f"  pair {n}: ARCTIC ({self.arctic_clicks[-1][0]:.1f}, "
                  f"{self.arctic_clicks[-1][1]:.1f})  <->  PS1 "
                  f"({x:.1f}, {y:.1f})")

    def onkey(self, event):
        if event.key == "u" and self.arctic_clicks:
            self.arctic_clicks.pop()
            self.ps1_clicks.pop()
            for _ in range(2):
                if self.markers:
                    self.markers.pop().remove()
            self.pending_arctic = None
            print("  undid last pair.")
            self.fig.canvas.draw_idle()

    def _maybe_refine(self, data, x, y):
        if self.refine:
            return refine_centroid(data, x, y, self.centroid_box)
        return float(x), float(y)


def main():
    args = parse_args()

    with fits.open(args.arctic_file) as hdul:
        arctic_data = hdul[0].data.astype(float)
        arctic_header = hdul[0].header
    arctic_wcs = WCS(arctic_header)

    with fits.open(args.ps1_file) as hdul:
        ps1_data = hdul[0].data.astype(float)
        ps1_header = hdul[0].header
    ps1_wcs = WCS(ps1_header)

    quad_bounds = None
    if args.quadrant:
        named = get_named_quadrant_bounds(arctic_header, arctic_data.shape)
        quad_bounds = named[args.quadrant]
        x0, x1, y0, y1 = quad_bounds
        print(f"restricting to quadrant {args.quadrant}: ARCTIC pixel x=[{x0},{x1}) "
              f"y=[{y0},{y1})")

    # The ARCTIC region of interest is the selected quadrant if given,
    # otherwise the whole ARCTIC frame. Compare ITS sky area against the
    # whole loaded Pan-STARRS file's sky area, and use whichever is smaller
    # as the shared view for BOTH panels -- if the PS1 reference happens to
    # be a small cutout (smaller than the ARCTIC region), there's no point
    # trying to zoom PS1 out to match a region it doesn't cover; crop the
    # ARCTIC view down to what PS1 actually covers instead, and vice versa.
    ny_a, nx_a = arctic_data.shape
    ny_p, nx_p = ps1_data.shape
    arctic_box = quad_bounds if quad_bounds is not None else (0, nx_a, 0, ny_a)
    arctic_view = quad_bounds  # default: crop ARCTIC view to the quadrant, if any
    ps1_view = None            # default: show the whole PS1 file
    try:
        arctic_area = box_area_arcsec2(arctic_wcs, *arctic_box)
        ps1_area = box_area_arcsec2(ps1_wcs, 0, nx_p, 0, ny_p)
        print(f"sky area of interest: ARCTIC region={arctic_area:.1f} arcsec^2, "
              f"Pan-STARRS file={ps1_area:.1f} arcsec^2 -- using the smaller for "
              "the shared initial view.")
        if arctic_area <= ps1_area:
            ps1_view = project_box(arctic_wcs, ps1_wcs, *arctic_box)
            # arctic_view stays as arctic_box (quad_bounds, or None -> whole frame)
        else:
            arctic_view = project_box(ps1_wcs, arctic_wcs, 0, nx_p, 0, ny_p)
            # ps1_view stays None -> show the whole (smaller) PS1 file
    except Exception as exc:
        print(f"  (couldn't compare/compute a shared view: {exc}; falling back to each "
              "panel's own default extent)")

    if args.coords_file:
        arctic_xy, ps1_xy, sky_deg = load_coords_file(args.coords_file)
        if quad_bounds is not None:
            x0, x1, y0, y1 = quad_bounds
            for ax, ay in arctic_xy:
                if not (x0 <= ax < x1 and y0 <= ay < y1):
                    print(f"  WARNING: tie point ARCTIC ({ax:.1f}, {ay:.1f}) from "
                          f"{args.coords_file} falls outside quadrant {args.quadrant}'s bounds "
                          f"(x=[{x0},{x1}) y=[{y0},{y1})) -- including it anyway, but this "
                          "defeats the point of a per-quadrant fit; double check it.")
        if not args.no_refine:
            arctic_xy = [refine_centroid(arctic_data, x, y, args.centroid_box) for x, y in arctic_xy]
            if ps1_xy is not None:
                ps1_xy = [refine_centroid(ps1_data, x, y, args.centroid_box) for x, y in ps1_xy]
        if sky_deg is not None:
            sky_points = SkyCoord(sky_deg[:, 0] * u.deg, sky_deg[:, 1] * u.deg)
        else:
            px = np.array([p[0] for p in ps1_xy])
            py = np.array([p[1] for p in ps1_xy])
            sky_points = ps1_wcs.pixel_to_world(px, py)
            sky_points = SkyCoord(sky_points.icrs.frame)
    else:
        if not sys.stdout.isatty():
            print("warning: stdout doesn't look like a terminal -- if this hangs with no "
                  "window appearing, you likely don't have a GUI display available here; "
                  "use --coords-file instead.")
        picker = InteractivePicker(arctic_data, ps1_data, args.centroid_box, not args.no_refine,
                                    quadrant=args.quadrant, quad_bounds=quad_bounds,
                                    arctic_view=arctic_view, ps1_view=ps1_view)
        print("Click ARCTIC star, then its match in Pan-STARRS. 'u' undoes. Close window when done.")
        plt.show()
        arctic_xy = picker.arctic_clicks
        ps1_xy = picker.ps1_clicks
        if arctic_xy:
            save_path = args.save_coords or (
                f"tiepoints_{os.path.splitext(os.path.basename(args.arctic_file))[0]}"
                f"{'_' + args.quadrant if args.quadrant else ''}.txt"
            )
            save_coords_file(save_path, arctic_xy, ps1_xy, args.arctic_file, args.ps1_file,
                              args.quadrant)
        if len(arctic_xy) < 4:
            raise SystemExit(f"Only {len(arctic_xy)} tie point(s) collected -- need at least 4.")
        px = np.array([p[0] for p in ps1_xy])
        py = np.array([p[1] for p in ps1_xy])
        sky_points = ps1_wcs.pixel_to_world(px, py)
        sky_points = SkyCoord(sky_points.icrs.frame)

    new_wcs = fit_and_report(arctic_xy, sky_points, arctic_wcs)

    if args.output_fits:
        new_header = arctic_header.copy()
        wcs_header = new_wcs.to_header()

        if args.quadrant:
            # Since this WCS is only valid inside the selected quadrant anyway,
            # write out ONLY that quadrant's pixels rather than the full frame
            # -- that also sidesteps the earlier multi-quadrant CRPIX problem
            # entirely (a single CRPIX is perfectly fine for a single, actually
            # contiguous crop; the impossibility was only ever about one CRPIX
            # trying to serve multiple quadrants across the amplifier gap at
            # once). CRPIX must shift by the crop's own pixel offset so it
            # still points at the same physical reference position within the
            # smaller array.
            x0, x1, y0, y1 = quad_bounds
            data_to_write = arctic_data[y0:y1, x0:x1]
            wcs_header["CRPIX1"] = wcs_header["CRPIX1"] - x0
            wcs_header["CRPIX2"] = wcs_header["CRPIX2"] - y0
            new_header.update(wcs_header)
            # DSECxx/BSECxx describe the FULL frame's quadrant layout in the
            # ORIGINAL pixel numbering -- meaningless (and actively misleading)
            # once cropped to one quadrant, so drop them.
            for q in ["11", "12", "21", "22"]:
                for key in (f"DSEC{q}", f"BSEC{q}"):
                    if key in new_header:
                        del new_header[key]
            new_header["HISTORY"] = (
                f"Cropped to quadrant {args.quadrant} only (original-frame pixel "
                f"x=[{x0},{x1}) y=[{y0},{y1})); WCS fit from manual tie points restricted "
                "to this quadrant. DSECxx/BSECxx removed (no longer meaningful post-crop)."
            )
            print(f"\ncropped output to quadrant {args.quadrant}: {data_to_write.shape[1]}x"
                  f"{data_to_write.shape[0]} (original-frame pixel x=[{x0},{x1}) y=[{y0},{y1})); "
                  f"CRPIX shifted by (-{x0}, -{y0}) to match.")
        else:
            new_header.update(wcs_header)
            data_to_write = arctic_data

        fits.writeto(args.output_fits, data_to_write, new_header, overwrite=True)
        print(f"wrote corrected FITS file: {args.output_fits}")
        if args.quadrant:
            print(f"NOTE: this file contains ONLY quadrant {args.quadrant}'s pixels -- pixel "
                  "coordinates in it are relative to this crop, not the original full frame "
                  "(subtract the printed offset above to convert back if you need to compare "
                  "against the original file's pixel numbering).")
        print("Sanity-check this with refine_astrometry.py / check_star_vs_catalog.py before "
              "trusting it -- a WCS fit from a handful of manual points has no protection "
              "against a single mis-click other than the residual list above.")


if __name__ == "__main__":
    main()