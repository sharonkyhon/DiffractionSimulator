# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 23:37:00 2025

@author: Sharon
"""

# Diffraction simulator starter script (Spyder-friendly)
# ----------------------------------------------------
# Use Spyder's code cells ("# %%") to run sections interactively.
# This script implements Phase 1-3 basics: simple lattice, structure
# factors, and a powder-pattern generator with Gaussian peak broadening.


# %%
"""Imports and small helpers"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi, asin
from scipy.signal import fftconvolve
from scipy.special import erf

# %%
"""Physical constants & simple atomic scattering approximation
Note: here we approximate X-ray atomic form factors f_j with a simple Z-dependent scaling.
For realistic patterns, replace with lookup tables (e.g., Cromer-Mann parameters) or use pymatgen/ase.
"""
# Basic approximate X-ray form factor (very rough):
def approx_xray_form_factor(Z, q):
    """Very rough form factor: f ~ Z * exp(-alpha * q^2).
    q is scattering vector magnitude in 1/Angstrom. alpha chosen
    so f decays with q; this is only for demonstration.
    """
    alpha = 0.005 + 0.02 * (Z / 30.0)
    return Z * np.exp(-alpha * q**2)

# For neutrons, would use element-specific scattering lengths (constant w.r.t. q).

# %%
"""Lattice and reciprocal-lattice utilities"""

def generate_simple_cell(kind='fcc', a=3.566):
    """Return lattice vectors and fractional atomic positions for
    a small set of example crystals. a in Angstrom.

    Atoms may be returned as (element, frac) or (element, frac, B) where
    B is an isotropic Debye-Waller parameter (Å^2). If B is omitted, B=0.0.

    Returns: (lattice_vectors (3x3 array), atoms list)
    """
    if kind == 'sc':
        lat = np.diag([a, a, a])
        atoms = [('Cu', np.array([0.0, 0.0, 0.0]))]
    elif kind == 'fcc':
        lat = np.diag([a, a, a])
        atoms = [
            ('Si', np.array([0.0, 0.0, 0.0])),
            ('Si', np.array([0.25, 0.25, 0.25]))
        ]
    elif kind == 'rocksalt':
        lat = np.diag([a, a, a])
        # Example isotropic B factors in Å^2 (set to 0.0 if unknown)
        atoms = [
            ('Na', np.array([0.0, 0.0, 0.0]), 0.5),
            ('Cl', np.array([0.5, 0.5, 0.5]), 0.6)
        ]
    else:
        raise ValueError('Unknown cell kind')
    return np.array(lat), atoms


def reciprocal_lattice(lat):
    """Compute reciprocal lattice vectors given 3x3 lattice matrix (rows = a1,a2,a3).
    Returns 3 reciprocal vectors as rows (b1,b2,b3) in 1/Angstrom.
    """
    a1, a2, a3 = lat[0], lat[1], lat[2]
    V = np.dot(a1, np.cross(a2, a3))
    b1 = 2*pi * np.cross(a2, a3) / V
    b2 = 2*pi * np.cross(a3, a1) / V
    b3 = 2*pi * np.cross(a1, a2) / V
    return np.vstack([b1, b2, b3])

# %%
"""Generate list of (h,k,l) up to some max index or qmax.
We will generate reflections within a max |h|,|k|,|l| range and filter by
magnitude of G (scattering vector) if desired.
"""
# Note that (000) is skipped
def generate_hkl_list(hmax=4):
    hkls = []
    for h in range(-hmax, hmax+1):
        for k in range(-hmax, hmax+1):
            for l in range(-hmax, hmax+1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkls.append((h,k,l))
    return hkls

# %%
"""Structure factor and intensity calculation with DW and LP support
"""

# --- atom helpers: allow atoms to be (el, frac) or (el, frac, B) ---
def _atom_symbol(atom):
    return atom[0]

def _atom_frac(atom):
    return atom[1]

def _atom_B(atom):
    """Return isotropic B (Å^2) if provided, else 0.0."""
    if len(atom) >= 3:
        return float(atom[2])
    return 0.0


def structure_factor(h, k, l, atoms, lat, rec_lat, wavelength, particle='xray',
                     apply_DW=True, apply_LP=True):
    """Compute structure factor and intensity for h,k,l.

    New behavior:
    - atoms may be (el, frac) or (el, frac, B) where B is isotropic Debye-Waller (Å^2)
    - If apply_DW True, multiply atomic form factor by exp(-B * (sinθ / λ)^2)
    - If apply_LP True and particle=='xray', compute Lorentz-polarisation factor Lp

    Returns dict with keys: 'G','q','d','theta','F','I0','Lp','mult','I','hkl'
    Note: I0 is |F|^2 (before LP and multiplicity); I includes LP and multiplicity.
    """
    # G vector in 1/Angstrom
    G = h*rec_lat[0] + k*rec_lat[1] + l*rec_lat[2]
    q = np.linalg.norm(G)
    if q == 0:
        return None

    # d spacing: |G| = 2*pi/d -> d = 2*pi/|G|
    d = 2*pi / q

    # Bragg angle (theta) from lambda = 2 d sin theta (first order)
    arg = wavelength / (2*d)
    if abs(arg) > 1.0:
        theta = None
    else:
        theta = asin(arg)

    # compute structure factor F (complex)
    F = 0+0j
    for atom in atoms:
        el = _atom_symbol(atom)
        frac = _atom_frac(atom)
        phase = np.exp(2j * pi * (h*frac[0] + k*frac[1] + l*frac[2]))
        if particle == 'xray':
            f0 = approx_xray_form_factor(atomic_number_guess(el), q)
        else:
            f0 = neutron_scattering_length_guess(el)

        # Debye-Waller damping factor (isotropic, if theta is known)
        if apply_DW and theta is not None:
            sin_theta_over_lambda = np.sin(theta) / wavelength
            B = _atom_B(atom)
            DW = np.exp(-B * (sin_theta_over_lambda**2))
        else:
            DW = 1.0

        f = f0 * DW
        F += f * phase

    I0 = (np.abs(F))**2  # base intensity (no LP, no multiplicity)

    # Lorentz-polarisation (Bragg-Brentano formula for unpolarized X-rays)
    Lp = 1.0
    if apply_LP and particle == 'xray' and (theta is not None):
        s = np.sin(theta)
        c = np.cos(theta)
        # guard against division by zero near 0 or 90 deg
        if s <= 1e-12 or c <= 1e-12:
            Lp = 1.0
        else:
            # (1 + cos^2 2θ) / (sin^2 θ * cos θ)
            # note: this is the combined Lorentz-polarisation factor up to an overall scale.
            Lp = (1.0 + np.cos(2*theta)**2) / (s**2 * c)

    # multiplicity placeholder (will be set by simulate_powder grouping)
    mult = 1

    # final intensity placeholder (I0 * Lp * mult); simulate_powder will recompute mult
    I = I0 * Lp * mult

    return {'G':G, 'q':q, 'd':d, 'theta':theta, 'F':F, 'I0':I0, 'Lp':Lp, 'mult':mult, 'I':I, 'hkl':(h,k,l)}

# %%
"""Very small helper to guess atomic number from element symbol.
In a real project use a proper element database (pymatgen.Element).
"""

def atomic_number_guess(symbol):
    table = {
        'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
        'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
        'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30
    }
    return table.get(symbol.capitalize(), 14)  # default to Si=14 if symbol not known


def neutron_scattering_length_guess(symbol):
    # Very rough placeholder: use Z as proxy
    # Guessing neutron scattering length for that particular element
    return float(atomic_number_guess(symbol)) * 0.1

# %%
"""Powder pattern: convert list of reflections to I(2theta) and convolve with
instrumental peak shape (Gaussian)."""

def reflections_to_pattern(reflections, wavelength, two_theta_range=(0.5,90.0),
                           npoints=5000, fwhm=0.2):
    """Build continuous powder pattern I(2theta).
    reflections: list of dicts from structure_factor
    two_theta_range: tuple in degrees
    fwhm: full-width half max in degrees (instrument)
    """
    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], npoints)
    pattern = np.zeros_like(two_theta)
    for ref in reflections:
        theta = ref['theta']
        if theta is None:
            continue
        t_deg = np.degrees(2*theta)
        I = ref['I']
        # add a Gaussian centered at t_deg
        sigma = fwhm / (2*np.sqrt(2*np.log(2)))
        profile = I * np.exp(-0.5 * ((two_theta - t_deg)/sigma)**2)
        pattern += profile
    # normalize for display
    if np.max(pattern) > 0:
        pattern /= np.max(pattern)
    return two_theta, pattern

# %%
"""High-level wrapper to simulate a powder pattern from a simple cell
"""

def simulate_powder(kind='fcc', a=3.566, wavelength=1.5406, hmax=4,
                     two_theta_range=(5,90), fwhm=0.2, particle='xray',
                     apply_DW=True, apply_LP=True, d_tol=1e-5):
    """
    Simulate a powder pattern; computes multiplicity by grouping by d-spacing.

    d_tol: absolute tolerance (Å) for grouping d-spacings. Increase if want more aggressive grouping (e.g. 1e-3).
    """
    lat, atoms = generate_simple_cell(kind=kind, a=a)
    rec_lat = reciprocal_lattice(lat)
    hkls = generate_hkl_list(hmax=hmax)

    # compute raw reflections (with I0, Lp computed)
    raw_refs = []
    for (h,k,l) in hkls:
        out = structure_factor(h,k,l, atoms, lat, rec_lat, wavelength, particle,
                               apply_DW=apply_DW, apply_LP=apply_LP)
        if out is None:
            continue
        if out['theta'] is not None:
            raw_refs.append(out)

    # group reflections by d within tolerance to compute multiplicity
    raw_refs_sorted = sorted(raw_refs, key=lambda r: r['d'])
    grouped = []
    i = 0
    n = len(raw_refs_sorted)
    while i < n:
        base = raw_refs_sorted[i]
        group = [base]
        j = i + 1
        while j < n and abs(raw_refs_sorted[j]['d'] - base['d']) <= d_tol:
            group.append(raw_refs_sorted[j])
            j += 1
        # multiplicity is number of equivalent reflections in the group
        mult = len(group)
        # pick representative (first) and set multiplicity and recompute I
        rep = group[0].copy()
        rep['mult'] = mult
        # recompute I: use I0 * Lp * mult
        rep['I'] = rep['I0'] * rep['Lp'] * mult
        rep['members'] = [g['hkl'] for g in group]
        grouped.append(rep)
        i = j

    # sort by 2theta
    refs_sorted = sorted(grouped, key=lambda r: np.degrees(2*r['theta']))

    two_theta, pattern = reflections_to_pattern(refs_sorted, wavelength,
                                                 two_theta_range, npoints=3000, fwhm=fwhm)
    return {'two_theta':two_theta, 'pattern':pattern, 'refs':refs_sorted}

# %%
"""Example usage: run and plot
Run this section in Spyder to see results. Modify parameters interactively.
"""
if __name__ == '__main__':
    # Example: NaCl (rocksalt)
    res = simulate_powder(kind='rocksalt', a=5.64, wavelength=1.5406, hmax=6,
                          two_theta_range=(10,80), fwhm=0.3,
                          particle='xray', apply_DW=True, apply_LP=True, d_tol=1e-4)

    plt.figure(figsize=(8,4))
    plt.plot(res['two_theta'], res['pattern'], lw=1)
    plt.xlabel(r"$2\theta$ [$^\circ$]")
    plt.ylabel('Normalized intensity')
    plt.title('Simulated powder X-ray pattern (with L·P, DW, multiplicity)')

    # annotate a few strongest peaks
    top = sorted(res['refs'], key=lambda r: r['I'], reverse=True)[:8]
    for r in top:
        tdeg = np.degrees(2*r['theta'])
        hkl = r['hkl']
        plt.axvline(tdeg, color='gray', linestyle='--', alpha=0.4)
        plt.text(tdeg+0.15, 0.6, f"{hkl}", rotation=90, va='bottom', fontsize=8)

    plt.xlim(10,80)
    plt.tight_layout()
    plt.show()

# %%
"""Notes & next steps (in-code):
- Replace approx_xray_form_factor with Cromer-Mann or library values.
- Use spglib or CIF parsing for symmetry-aware multiplicities and systematic absences.
- Add Lorentzian/Gaussian instrument convolution (pseudo-Voigt) and background.
- Implement anisotropic Debye-Waller if needed (requires U-tensors).
"""
