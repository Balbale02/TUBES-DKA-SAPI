"""
fuzzy/mamdani.py
Inferensi Fuzzy Mamdani from scratch untuk sistem rekomendasi restoran.
Alur: Fuzzifikasi → Evaluasi Rule (AND=min, agregasi=max)
      → Defuzzifikasi (Centroid)
"""

import time
import numpy as np
from fuzzy.membership import (
    fuzzify_all, mf_output, U_OUTPUT, SUGENO_CONSTANTS
)


# ─────────────────────────────────────────────
# RULE BASE (17 rules)
# ─────────────────────────────────────────────

RULES = [
    {'conditions': [('harga','murah'),  ('votes','banyak')],                          'output': 'sangat_layak'},
    {'conditions': [('harga','murah'),  ('votes','sedang')],                          'output': 'cukup_layak'},
    {'conditions': [('harga','murah'),  ('votes','sedikit')],                         'output': 'cukup_layak'},
    {'conditions': [('harga','sedang'), ('votes','banyak')],                          'output': 'sangat_layak'},
    {'conditions': [('harga','sedang'), ('votes','sedang')],                          'output': 'cukup_layak'},
    {'conditions': [('harga','sedang'), ('votes','sedikit')],                         'output': 'tidak_layak'},
    {'conditions': [('harga','mahal'),  ('votes','banyak')],                          'output': 'cukup_layak'},
    {'conditions': [('harga','mahal'),  ('votes','sedang')],                          'output': 'tidak_layak'},
    {'conditions': [('harga','mahal'),  ('votes','sedikit')],                         'output': 'tidak_layak'},
    {'conditions': [('price','rendah'), ('votes','banyak')],                          'output': 'sangat_layak'},
    {'conditions': [('price','tinggi'), ('votes','sedikit')],                         'output': 'tidak_layak'},
    {'conditions': [('price','sedang'), ('votes','sedang')],                          'output': 'cukup_layak'},
    {'conditions': [('booking','ya'),   ('delivery','ya')],                           'output': 'sangat_layak'},
    {'conditions': [('booking','tidak'),('delivery','tidak'), ('votes','sedikit')],   'output': 'tidak_layak'},
    {'conditions': [('booking','ya'),   ('votes','sedang')],                          'output': 'cukup_layak'},
    {'conditions': [('delivery','ya'),  ('harga','murah')],                           'output': 'sangat_layak'},
    {'conditions': [('booking','tidak'),('harga','mahal')],                           'output': 'tidak_layak'},
]


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def get_rekomendasi_level(score):
    if score < 35:
        return 'TIDAK LAYAK'
    elif score < 65:
        return 'CUKUP LAYAK'
    else:
        return 'SANGAT LAYAK'


# ─────────────────────────────────────────────
# EVALUASI RULE
# ─────────────────────────────────────────────

def evaluate_rules(fz):
    """Evaluasi 17 rule. AND = min(). Agregasi MAX per label output."""
    fired = []
    alpha_per_label = {k: 0.0 for k in mf_output}

    for i, rule in enumerate(RULES):
        strengths = []
        for var, lbl in rule['conditions']:
            strengths.append(fz[var][lbl] if lbl in fz.get(var, {}) else 0.0)
        alpha = min(strengths)

        if alpha > 0:
            fired.append({
                'rule_idx'       : i + 1,
                'conditions'     : rule['conditions'],
                'output'         : rule['output'],
                'firing_strength': round(alpha, 4),
            })
            alpha_per_label[rule['output']] = max(
                alpha_per_label[rule['output']], alpha
            )

    return fired, alpha_per_label


# ─────────────────────────────────────────────
# AGREGASI & DEFUZZIFIKASI
# ─────────────────────────────────────────────

def aggregate(alpha_per_label):
    """Clipping + agregasi MAX."""
    agg = np.zeros_like(U_OUTPUT)
    for label, alpha in alpha_per_label.items():
        clipped = np.minimum(alpha, mf_output[label])
        agg = np.maximum(agg, clipped)
    return agg


def centroid(agg):
    """Defuzzifikasi Centroid (Center of Gravity)."""
    den = np.sum(agg)
    return float(np.sum(U_OUTPUT * agg) / den) if den > 0 else 50.0


# ─────────────────────────────────────────────
# PREDIKSI MAMDANI
# ─────────────────────────────────────────────

def mamdani_predict(harga, votes, price_range, booking, delivery):
    """Pipeline Mamdani lengkap untuk satu sampel."""
    t0  = time.time()
    fz  = fuzzify_all(harga, votes, price_range, booking, delivery)
    fired, alpha_per_label = evaluate_rules(fz)
    agg   = aggregate(alpha_per_label)
    score = round(centroid(agg), 4)
    return {
        'score'      : score,
        'level'      : get_rekomendasi_level(score),
        'fired_rules': fired,
        'x_values'   : U_OUTPUT,
        'aggregated' : agg,
        'runtime'    : round(time.time() - t0, 5),
    }


def mamdani_batch(df, sample_size=None):
    """Jalankan Mamdani pada seluruh/sebagian dataframe."""
    df_s = df.sample(n=sample_size, random_state=42) if sample_size else df
    actual, pred = [], []
    for _, row in df_s.iterrows():
        r = mamdani_predict(
            row['Average Cost for two'],
            row['Votes'],
            row['Price range'],
            row['Has Table booking'],
            row['Has Online delivery'],
        )
        actual.append(row['Aggregate rating'] * 20)
        pred.append(r['score'])
    return np.array(actual), np.array(pred)
