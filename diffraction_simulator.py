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
from math import pi, sin, asin, sqrt
from scipy.signal import fftconvolve
from scipy.special import erf

# %%
"""Physical constants & simple atomic scattering approximation
Note: for a starter notebook we approximate X-ray atomic form factors f_j
with a simple Z-dependent scaling. For realistic patterns, replace with
lookup tables (e.g., Cromer-Mann parameters) or use pymatgen/ase.
"""
# speed of light etc. (not strictly needed here)
# Basic approximate X-ray form factor (very rough):
def approx_xray_form_factor(Z, q):
    """Very rough form factor: f ~ Z * exp(-alpha * q^2).
    q is scattering vector magnitude in 1/Angstrom. alpha chosen
    so f decays with q; this is only for demonstration.
    """
    alpha = 0.005 + 0.02 * (Z / 30.0)
    return Z * np.exp(-alpha * q**2)

# For neutrons you would use element-specific scattering lengths (constant w.r.t q).

# %%
"""Lattice and reciprocal-lattice utilities"""

def generate_simple_cell(kind='fcc', a=3.566):
    """Return lattice vectors and fractional atomic positions for
    a small set of example crystals. a in Angstrom.

    Returns: (lattice_vectors (3x3 array), atoms list of tuples (element, frac_pos))
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
        atoms = [
            ('Na', np.array([0.0, 0.0, 0.0])),
            ('Cl', np.array([0.5, 0.5, 0.5]))
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
"""Structure factor and intensity calculation
"""

def structure_factor(h,k,l, atoms, lat, rec_lat, wavelength, particle='xray'):
    """Compute structure factor F_hkl and intensity I_hkl for given h,k,l.
    atoms: list of (element, frac_pos)
    lat: real-space lattice (3x3)
    rec_lat: reciprocal lattice (3x3)
    wavelength: radiation wavelength in Angstrom
    particle: 'xray' or 'neutron' (neutron uses constant scattering lengths)

    Returns (Gvec (1/A), q=|G|, d_spacing, theta (rad), F, I)
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
        theta = None  # no reflection for this wavelength
    else:
        theta = asin(arg)

    # compute F
    F = 0+0j
    for (el, frac) in atoms:
        # convert fractional to cartesian r = frac dot lattice
        r = frac[0]*lat[0] + frac[1]*lat[1] + frac[2]*lat[2]
        phase = np.exp(2j * pi * (h*frac[0] + k*frac[1] + l*frac[2]))
        if particle == 'xray':
            f = approx_xray_form_factor(atomic_number_guess(el), q)
        else:
            f = neutron_scattering_length_guess(el)
        F += f * phase
    I = (np.abs(F))**2
    return {'G':G, 'q':q, 'd':d, 'theta':theta, 'F':F, 'I':I, 'hkl':(h,k,l)}

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
    return table.get(symbol.capitalize(), 14)  # default to Si=14


def neutron_scattering_length_guess(symbol):
    # Very rough placeholder: use Z as proxy
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
    pattern /= np.max(pattern)
    return two_theta, pattern

# %%
"""High-level wrapper to simulate a powder pattern from a simple cell
"""

def simulate_powder(kind='fcc', a=3.566, wavelength=1.5406, hmax=4,
                     two_theta_range=(5,90), fwhm=0.2, particle='xray'):
    lat, atoms = generate_simple_cell(kind=kind, a=a)
    rec_lat = reciprocal_lattice(lat)
    hkls = generate_hkl_list(hmax=hmax)
    refs = []
    for (h,k,l) in hkls:
        out = structure_factor(h,k,l, atoms, lat, rec_lat, wavelength, particle)
        if out is None:
            continue
        # Only keep reflections that satisfy Bragg (theta not None)
        if out['theta'] is not None:
            # apply simple multiplicity factor (ignore equivalent reflections for demo)
            refs.append(out)
    # sort by 2theta
    refs_sorted = sorted(refs, key=lambda r: np.degrees(2*r['theta']))
    two_theta, pattern = reflections_to_pattern(refs_sorted, wavelength,
                                                 two_theta_range, npoints=3000, fwhm=fwhm)
    return {'two_theta':two_theta, 'pattern':pattern, 'refs':refs_sorted}

# %%
"""Example usage: run and plot
Run this section in Spyder to see results. Modify parameters interactively.
"""
if __name__ == '__main__':
    res = simulate_powder(kind='rocksalt', a=5.64, wavelength=1.5406, hmax=4,
                          two_theta_range=(10,80), fwhm=0.3)
    plt.figure(figsize=(8,4))
    plt.plot(res['two_theta'], res['pattern'])
    plt.xlabel(r"$2\theta$ [$^\circ$]")
    plt.ylabel(r'Normalized intensity [$Wm^{-2}$]')
    plt.title('Simulated powder X-ray pattern')
    # annotate a few strongest peaks
    top = sorted(res['refs'], key=lambda r: r['I'], reverse=True)[:6]
    for r in top:
        tdeg = np.degrees(2*r['theta'])
        #plt.axvline(tdeg, color='gray', linestyle='--', alpha=0.5)
        hkl = r['hkl']
        #plt.text(tdeg+0.2, 0.6, f"{hkl}", rotation=90, va='bottom')
    plt.xlim(10,80)
    plt.tight_layout()
    plt.show()

# %%
"""Notes & next steps (in-code):
- Replace approx_xray_form_factor with Cromer-Mann or library values.
- Add multiplicity, Lorentz-polarization, and Debye-Waller factors.
- Read CIFs via pymatgen/ase and support arbitrary unit cells.
- Implement peak-shape models (pseudo-Voigt) and background.
"""

# %%
""" Inclusion of Lorentz-polarisation factor
"""




# %%
""" Inclusion of Debye-Waller factor 
"""
