"""
fuzzy/membership.py
Fungsi keanggotaan from scratch untuk sistem rekomendasi restoran (Zomato).
Berisi: trapezoid, triangle, interp_membership, fuzzify_all,
        mf_output, get_mf_curves, SUGENO_CONSTANTS.
Breakpoint dihitung adaptif dari distribusi data (kuartil).
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. FUNGSI KEANGGOTAAN DASAR
# ─────────────────────────────────────────────

def trapezoid(x, a, b, c, d):
    """
    Fungsi keanggotaan Trapesium.
    Naik  : a → b
    Datar : b → c  (nilai = 1)
    Turun : c → d
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    mask1 = (x >= a) & (x < b)
    if b != a:
        y[mask1] = (x[mask1] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    mask3 = (x > c) & (x <= d)
    if d != c:
        y[mask3] = (d - x[mask3]) / (d - c)
    return y


def triangle(x, a, b, c):
    """
    Fungsi keanggotaan Segitiga.
    Naik  : a → b  (puncak = 1 di b)
    Turun : b → c
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    mask1 = (x >= a) & (x <= b)
    if b != a:
        y[mask1] = (x[mask1] - a) / (b - a)
    mask2 = (x > b) & (x <= c)
    if c != b:
        y[mask2] = (c - x[mask2]) / (c - b)
    y[x == b] = 1.0
    return y


def interp_membership(universe, mf_values, x):
    """Interpolasi derajat keanggotaan untuk nilai x tunggal."""
    return float(np.interp(x, universe, mf_values))


# ─────────────────────────────────────────────
# 2. LOAD DATA & HITUNG BREAKPOINT ADAPTIF
# ─────────────────────────────────────────────

def _load_and_preprocess(csv_path='zomato.csv'):
    df = pd.read_csv(csv_path, encoding='latin-1')
    cols = ['Restaurant Name', 'Average Cost for two', 'Price range',
            'Votes', 'Has Table booking', 'Has Online delivery', 'Aggregate rating']
    data = df[cols].copy()
    data = data[data['Aggregate rating'] > 0]
    data['Has Table booking']   = data['Has Table booking'].map({'Yes': 1, 'No': 0})
    data['Has Online delivery'] = data['Has Online delivery'].map({'Yes': 1, 'No': 0})
    data = data.dropna().reset_index(drop=True)
    return data


def _get_breakpoints(series):
    return {
        'min': float(series.min()),
        'p25': float(series.quantile(0.25)),
        'p50': float(series.quantile(0.50)),
        'p75': float(series.quantile(0.75)),
        'max': float(series.max()),
    }


# Precompute breakpoint saat module diimport
_data      = _load_and_preprocess()
BP_COST    = _get_breakpoints(_data['Average Cost for two'])
BP_VOTES   = _get_breakpoints(_data['Votes'])
MAX_COST   = BP_COST['max']
MAX_VOTES  = BP_VOTES['max']


# ─────────────────────────────────────────────
# 3. UNIVERSE OF DISCOURSE
# ─────────────────────────────────────────────

U_COST   = np.arange(0, MAX_COST + 10, 10, dtype=float)
U_VOTES  = np.arange(0, MAX_VOTES + 2, 1,  dtype=float)
U_PRICE  = np.arange(1, 6, 1, dtype=float)
U_BINARY = np.array([0.0, 1.0])
U_OUTPUT = np.arange(0, 101, 1, dtype=float)


# ─────────────────────────────────────────────
# 4. MEMBERSHIP FUNCTIONS PER VARIABEL
# ─────────────────────────────────────────────

MF_HARGA = {
    'murah' : trapezoid(U_COST,  BP_COST['min'], BP_COST['min'], BP_COST['p25'], BP_COST['p50']),
    'sedang': triangle (U_COST,  BP_COST['p25'], BP_COST['p50'], BP_COST['p75']),
    'mahal' : trapezoid(U_COST,  BP_COST['p50'], BP_COST['p75'], BP_COST['max'], BP_COST['max']),
}

MF_VOTES = {
    'sedikit': trapezoid(U_VOTES, BP_VOTES['min'], BP_VOTES['min'], BP_VOTES['p25'], BP_VOTES['p50']),
    'sedang' : triangle (U_VOTES, BP_VOTES['p25'], BP_VOTES['p50'], BP_VOTES['p75']),
    'banyak' : trapezoid(U_VOTES, BP_VOTES['p50'], BP_VOTES['p75'], BP_VOTES['max'], BP_VOTES['max']),
}

MF_PRICE = {
    'rendah': trapezoid(U_PRICE, 1, 1, 1, 2),
    'sedang': triangle (U_PRICE, 1, 2, 3),
    'tinggi': trapezoid(U_PRICE, 2, 3, 4, 4),
}

MF_BINARY = {
    'tidak': triangle(U_BINARY, 0, 0, 1),
    'ya'   : triangle(U_BINARY, 0, 1, 1),
}

# Output Mamdani
mf_output = {
    'tidak_layak' : trapezoid(U_OUTPUT, 0,  0,  30, 50),
    'cukup_layak' : triangle (U_OUTPUT, 30, 50, 70),
    'sangat_layak': trapezoid(U_OUTPUT, 50, 70, 100, 100),
}

# Konstanta output Sugeno (Orde-0)
SUGENO_CONSTANTS = {
    'tidak_layak' : 20,
    'cukup_layak' : 50,
    'sangat_layak': 85,
}


# ─────────────────────────────────────────────
# 5. FUZZIFIKASI
# ─────────────────────────────────────────────

def fuzzify_all(harga, votes, price_range, booking, delivery):
    """Hitung derajat keanggotaan semua variabel input."""
    return {
        'harga'   : {k: interp_membership(U_COST,   v, harga)       for k, v in MF_HARGA.items()},
        'votes'   : {k: interp_membership(U_VOTES,  v, votes)       for k, v in MF_VOTES.items()},
        'price'   : {k: interp_membership(U_PRICE,  v, price_range) for k, v in MF_PRICE.items()},
        'booking' : {k: interp_membership(U_BINARY, v, booking)     for k, v in MF_BINARY.items()},
        'delivery': {k: interp_membership(U_BINARY, v, delivery)    for k, v in MF_BINARY.items()},
    }


# ─────────────────────────────────────────────
# 6. KURVA MF UNTUK VISUALISASI
# ─────────────────────────────────────────────

def get_mf_curves():
    return {
        'harga': {
            'x': U_COST, 'xlabel': f'Harga (Average Cost for two)',
            **MF_HARGA
        },
        'votes': {
            'x': U_VOTES, 'xlabel': 'Jumlah Votes',
            **MF_VOTES
        },
        'price_range': {
            'x': U_PRICE, 'xlabel': 'Price Range (1-4)',
            **MF_PRICE
        },
        'output': {
            'x': U_OUTPUT, 'xlabel': 'Skor Rekomendasi (0-100)',
            **mf_output
        },
    }
