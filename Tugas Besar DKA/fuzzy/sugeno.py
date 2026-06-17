"""
fuzzy/sugeno.py
Inferensi Fuzzy Sugeno Orde-0 from scratch untuk sistem rekomendasi restoran.
Alur: Fuzzifikasi → Evaluasi Rule (AND=min)
      → Defuzzifikasi (Weighted Average dari konstanta)
"""

import time
import numpy as np
from fuzzy.membership import fuzzify_all, SUGENO_CONSTANTS
from fuzzy.mamdani    import RULES, get_rekomendasi_level


# ─────────────────────────────────────────────
# DEFUZZIFIKASI SUGENO
# ─────────────────────────────────────────────

def weighted_average(fired_rules):
    """
    z* = Σ(αᵢ × zᵢ) / Σαᵢ
    """
    num = sum(r['firing_strength'] * r['output_value'] for r in fired_rules)
    den = sum(r['firing_strength'] for r in fired_rules)
    return num / den if den > 0 else 50.0


# ─────────────────────────────────────────────
# PREDIKSI SUGENO
# ─────────────────────────────────────────────

def sugeno_predict(harga, votes, price_range, booking, delivery):
    """Pipeline Sugeno Orde-0 lengkap untuk satu sampel."""
    t0 = time.time()
    fz = fuzzify_all(harga, votes, price_range, booking, delivery)

    fired = []
    for i, rule in enumerate(RULES):
        strengths = [fz[var][lbl] for var, lbl in rule['conditions']
                     if lbl in fz.get(var, {})]
        alpha = min(strengths) if strengths else 0.0
        if alpha > 0:
            z = SUGENO_CONSTANTS[rule['output']]
            fired.append({
                'rule_idx'       : i + 1,
                'conditions'     : rule['conditions'],
                'output'         : rule['output'],
                'firing_strength': round(alpha, 4),
                'output_value'   : z,
            })

    score = round(weighted_average(fired), 4) if fired else 50.0
    return {
        'score'      : score,
        'level'      : get_rekomendasi_level(score),
        'fired_rules': fired,
        'runtime'    : round(time.time() - t0, 5),
    }


def sugeno_batch(df, sample_size=None):
    """Jalankan Sugeno pada seluruh/sebagian dataframe."""
    df_s = df.sample(n=sample_size, random_state=42) if sample_size else df
    actual, pred = [], []
    for _, row in df_s.iterrows():
        r = sugeno_predict(
            row['Average Cost for two'],
            row['Votes'],
            row['Price range'],
            row['Has Table booking'],
            row['Has Online delivery'],
        )
        actual.append(row['Aggregate rating'] * 20)
        pred.append(r['score'])
    return np.array(actual), np.array(pred)
