"""
scanner.py — BIST Fırsat Tarayıcı v2

Doğru mantık:
1. Her hissenin son 20G ve 40G hareketini şablon al
2. DİĞER hisselerin 2 yıllık geçmişinde bu şablona benzer dönemleri bul
3. O benzer dönemlerden sonra ne olmuş → konsensüs hesapla
4. Çoğunluk YÜKSELİŞ ise fırsat listesine ekle
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import streamlit as st
import time
import os
import json
from datetime import datetime

try:
    from portfolio import render_add_to_portfolio_button
except Exception:
    render_add_to_portfolio_button = None

# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def zscore(arr):
    arr = np.array(arr, dtype=float)
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-9:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def downsample_factor(window):
    """
    Uzun şablonlarda DTW maliyeti pencere uzunluğuyla KARESEL artıyor
    (band da pencereyle birlikte büyüdüğü için). Bunu sınırlamak için,
    uzun pencerelerde günlük yerine seyreltilmiş (örn. haftalık) örnekleme
    kullanıyoruz — şekil bilgisi büyük ölçüde korunuyor, etkin nokta sayısı
    ~70-90 civarında sabit kalıyor (pencere ne kadar uzarsa uzasın).
    Hedef/stop/beklenen gün gibi finansal hesaplamalar HER ZAMAN tam günlük
    veriyle yapılıyor — sadece "hangi dönem benziyor" araması seyreltiliyor.
    """
    if window <= 60:
        return 1
    return max(1, window // 75)


_WINDOW_CACHE = {}


def _get_cached_windows(ticker_key, closes, window, fut_window):
    """
    Bir hissenin tüm sliding window z-score'larını önceden hesapla ve önbelleğe al.
    Aynı tarama içinde birden fazla şablon bu hisseyi aday olarak kullanacaksa
    (20G ve 40G taramaları farklı window'lar kullansa da), tekrar hesaplamayı önler.
    """
    cache_key = (ticker_key, window, fut_window, len(closes))
    if cache_key in _WINDOW_CACHE:
        return _WINDOW_CACHE[cache_key]

    n = len(closes)
    max_start = n - window - fut_window
    if max_start < 5:
        _WINDOW_CACHE[cache_key] = None
        return None

    ds = downsample_factor(window)
    step = max(1, window // 5)
    starts = list(range(0, max_start, step))
    windows_z = np.array([zscore(closes[i:i + window][::ds]) for i in starts])

    result = {'starts': starts, 'windows_z': windows_z, 'step': step,
              'max_start': max_start, 'ds': ds}
    _WINDOW_CACHE[cache_key] = result
    return result


def clear_window_cache():
    """Yeni tarama başlarken önbelleği temizle (bellek şişmesin)."""
    _WINDOW_CACHE.clear()


# ── YENİ: Toplu vektörize ön-eleme (window bank) ───────────────────────────────

def build_window_bank(all_data, window, fut_window):
    """
    Tüm hisselerin sliding window'larını TEK bir matriste topla.
    Böylece her şablon, tüm adaylara karşı tek matris çarpımıyla karşılaştırılır
    (hisse hisse Python döngüsü yerine).
    """
    bank_meta = []   # (ticker, start_idx) listesi
    bank_rows = []   # z-score satırları

    for ticker, df in all_data.items():
        closes = df['Close'].values.astype(float)
        cache = _get_cached_windows(ticker, closes, window, fut_window)
        if cache is None:
            continue
        for start, wz in zip(cache['starts'], cache['windows_z']):
            bank_meta.append((ticker, start))
            bank_rows.append(wz)

    if not bank_rows:
        return None

    bank_matrix = np.array(bank_rows)  # (toplam_pencere, window)
    bank_centered = bank_matrix - bank_matrix.mean(axis=1, keepdims=True)
    bank_norms = np.sqrt((bank_centered ** 2).sum(axis=1)) + 1e-9

    return {
        'meta': bank_meta,
        'matrix': bank_matrix,
        'centered': bank_centered,
        'norms': bank_norms,
    }


def batch_prefilter(tpl_z, bank, exclude_ticker=None, top_k=25):
    """
    Şablonu TÜM bankaya karşı tek matris çarpımıyla karşılaştırır.
    En yüksek Pearson skorlu top_k adayı döndürür: (ticker, start_idx, pearson).
    """
    t = tpl_z - tpl_z.mean()
    t_norm = np.sqrt((t ** 2).sum()) + 1e-9

    scores = (bank['centered'] @ t) / (bank['norms'] * t_norm)  # (toplam_pencere,)

    if exclude_ticker is not None:
        mask = np.array([m[0] == exclude_ticker for m in bank['meta']])
        scores = np.where(mask, -np.inf, scores)

    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [(bank['meta'][i][0], bank['meta'][i][1], float(scores[i])) for i in top_idx]


def pearson(a, b):
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ── DTW hesaplama (banded, saf Python) ──────────────────────────────────────
def dtw_fast(s1, s2, band=None):
    n = len(s1)
    if n == 0:
        return 0.0
    band = band or max(2, n // 6)
    dtw = np.full((n + 1, n + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(n, i + band) + 1
        for j in range(j0, j1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    dist = dtw[n, n] / n
    return max(0.0, 1.0 - dist * 1.5)


def similarity_score(tpl_z, win_prices, ds=1):
    """
    Hızlı benzerlik: Pearson + DTW kombinasyonu.
    ds>1 ise win_prices, tpl_z ile aynı seyreltme oranında örneklenir —
    tpl_z zaten downsample_factor(window) ile seyreltilmiş kabul edilir.
    """
    win_z = zscore(win_prices[::ds] if ds > 1 else win_prices)
    p = (pearson(tpl_z, win_z) + 1) / 2
    if p < 0.45:
        return p * 100
    d = dtw_fast(tpl_z, win_z)
    return (0.55 * p + 0.45 * d) * 100


def daily_returns(prices):
    prices = np.array(prices, dtype=float)
    if len(prices) < 2:
        return np.zeros(1)
    return np.diff(prices) / (np.abs(prices[:-1]) + 1e-9)


def calc_rsi(prices, n=14):
    prices = np.array(prices, dtype=float)
    if len(prices) < n + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = gains[:n].mean()
    al = losses[:n].mean()
    for i in range(n, len(deltas)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
    return float(100 - 100 / (1 + ag / (al + 1e-9)))


def find_best_match(tpl_z, candidate_closes, window, fut_window, candidate_dates=None,
                     candidate_key=None, top_k=3, refine_frac=1.0):
    """
    Aday hissenin geçmişinde şablona en benzer bölgeyi bul.
    Sadece ardında yeterli gelecek verisi olan bölgeleri tara.
    Performans: candidate_key verilirse, önbellekten hazır z-score matrisini
    kullanır (vektörize Pearson ön-eleme) — DTW sadece en güçlü adaylarda çalışır.
    top_k: pearson ön-elemesinden sonra DTW ile detaylı bakılacak aday sayısı.
    refine_frac: ince ayar (refinement) penceresinin genişlik katsayısı — 1.0
    tam, 0.5 yarı yarıya daha az DTW çağrısı demek (hız/hassasiyet dengesi).
    """
    n = len(candidate_closes)
    max_start = n - window - fut_window
    if max_start < 5:
        return None

    ds = downsample_factor(window)

    cache = _get_cached_windows(candidate_key, candidate_closes, window, fut_window) \
        if candidate_key is not None else None

    if cache is not None and len(cache['starts']) > 0:
        starts = cache['starts']
        windows_z = cache['windows_z']  # (n_windows, ~window/ds) — seyreltilmiş
        step = max(1, int(cache['step'] * refine_frac))

        t = tpl_z - tpl_z.mean()
        w = windows_z - windows_z.mean(axis=1, keepdims=True)
        t_norm = np.sqrt((t ** 2).sum()) + 1e-9
        w_norms = np.sqrt((w ** 2).sum(axis=1)) + 1e-9
        pearson_scores = (w @ t) / (w_norms * t_norm)  # -1..1, shape (n_windows,)

        top_k = min(top_k, len(starts))
        top_idx = np.argpartition(-pearson_scores, top_k - 1)[:top_k]

        best_sim, best_i = -1, starts[0]
        for idx in top_idx:
            i = starts[idx]
            sim = similarity_score(tpl_z, candidate_closes[i:i + window], ds=ds)
            if sim > best_sim:
                best_sim, best_i = sim, i

        for i in range(max(0, best_i - step), min(max_start + 1, best_i + step + 1)):
            sim = similarity_score(tpl_z, candidate_closes[i:i + window], ds=ds)
            if sim > best_sim:
                best_sim, best_i = sim, i
    else:
        step = max(1, window // 5)
        best_sim, best_i = -1, 0
        for i in range(0, max_start, step):
            sim = similarity_score(tpl_z, candidate_closes[i:i + window], ds=ds)
            if sim > best_sim:
                best_sim, best_i = sim, i
        for i in range(max(0, best_i - step), min(max_start + 1, best_i + step + 1)):
            sim = similarity_score(tpl_z, candidate_closes[i:i + window], ds=ds)
            if sim > best_sim:
                best_sim, best_i = sim, i

    if best_sim < 55:
        return None

    match_closes = candidate_closes[best_i:best_i + window]
    future_closes = candidate_closes[best_i + window:best_i + window + fut_window]
    if len(future_closes) < 3:
        return None

    fut_pct = (future_closes[-1] - future_closes[0]) / (future_closes[0] + 1e-9) * 100
    fut_max = (future_closes.max() - future_closes[0]) / (future_closes[0] + 1e-9) * 100
    fut_min = (future_closes.min() - future_closes[0]) / (future_closes[0] + 1e-9) * 100

    match_date_label = None
    match_date_start = None
    match_date_end = None
    if candidate_dates is not None and best_i < len(candidate_dates):
        try:
            match_date_label = candidate_dates[best_i].strftime('%m.%Y')
            match_date_start = candidate_dates[best_i].strftime('%d.%m.%Y')
            end_idx = min(best_i + window - 1, len(candidate_dates) - 1)
            match_date_end = candidate_dates[end_idx].strftime('%d.%m.%Y')
        except Exception:
            match_date_label = None
            match_date_start = None
            match_date_end = None

    return {
        'sim': round(best_sim, 1),
        'fut_pct': round(fut_pct, 2),
        'fut_max': round(fut_max, 2),
        'fut_min': round(fut_min, 2),
        'match_closes': match_closes,
        'future_closes': future_closes,
        'match_start_idx': best_i,
        'match_date_label': match_date_label,
        'match_date_start': match_date_start,
        'match_date_end': match_date_end,
    }


def scan_single_ticker(ticker, df, all_data, window, fut_window, min_sim=60,
                        index_closes=None, bank=None, candidate_top_k=25,
                        dtw_top_k=3, refine_frac=1.0):
    """
    Tek hisse için fırsat analizi:
    - Son `window` günü şablon al
    - Diğer hisselerin geçmişinde benzer dönemleri bul (bank varsa toplu ön-eleme ile)
    - Konsensüs hesapla
    - Endeks korelasyonu kontrol et (genel piyasa hareketi mi?)
    """
    closes = df['Close'].values.astype(float)
    volumes = df['Volume'].values.astype(float)
    dates = df.index

    if len(closes) < window + 10:
        return None

    tpl_prices = closes[-window:]
    tpl_z = zscore(tpl_prices[::downsample_factor(window)])
    tpl_rets = daily_returns(tpl_prices)

    tpl_change = (tpl_prices[-1] - tpl_prices[0]) / (tpl_prices[0] + 1e-9) * 100
    tpl_rsi = calc_rsi(tpl_prices)
    current_price = float(closes[-1])

    index_corr = None
    if index_closes is not None and len(index_closes) >= window:
        idx_tpl = index_closes[-window:]
        min_len = min(len(tpl_prices), len(idx_tpl))
        stock_rets = daily_returns(tpl_prices[-min_len:])
        idx_rets = daily_returns(idx_tpl[-min_len:])
        m = min(len(stock_rets), len(idx_rets))
        if m >= 4 and np.std(stock_rets[-m:]) > 1e-9 and np.std(idx_rets[-m:]) > 1e-9:
            index_corr = float(np.corrcoef(stock_rets[-m:], idx_rets[-m:])[0, 1])

    matches = []

    if bank is not None:
        # ── Toplu vektörize ön-eleme ile aday seç, sadece top adaylarda DTW çalıştır
        candidates = batch_prefilter(tpl_z, bank, exclude_ticker=ticker, top_k=candidate_top_k)
        seen_tickers = set()
        for other_ticker, start_idx, pscore in candidates:
            if other_ticker in seen_tickers:
                continue
            seen_tickers.add(other_ticker)
            other_closes = all_data[other_ticker]['Close'].values.astype(float)
            other_dates = all_data[other_ticker].index
            result = find_best_match(tpl_z, other_closes, window, fut_window, other_dates,
                                      candidate_key=other_ticker, top_k=dtw_top_k,
                                      refine_frac=refine_frac)
            if result and result['sim'] >= min_sim:
                result['source'] = other_ticker
                matches.append(result)
    else:
        # ── Eski yol (bank verilmezse) — tüm hisseleri tek tek tara
        for other_ticker, other_df in all_data.items():
            if other_ticker == ticker:
                continue
            other_closes = other_df['Close'].values.astype(float)
            other_dates = other_df.index
            result = find_best_match(tpl_z, other_closes, window, fut_window, other_dates,
                                      candidate_key=other_ticker)
            if result and result['sim'] >= min_sim:
                result['source'] = other_ticker
                matches.append(result)

    # Bu hissenin kendi geçmişinde de ara (son window gün hariç)
    if len(closes) >= window * 3 + fut_window:
        hist_closes = closes[:-window]
        hist_dates = dates[:-window]
        result = find_best_match(tpl_z, hist_closes, window, fut_window, hist_dates,
                                  candidate_key=f"{ticker}_self", top_k=dtw_top_k,
                                  refine_frac=refine_frac)
        if result and result['sim'] >= min_sim:
            result['source'] = f"{ticker} (geçmiş)"
            matches.append(result)

    if len(matches) < 2:
        return None

    unique_periods = len(set(m.get('match_date_label') for m in matches
                              if m.get('match_date_label')))
    if len(matches) >= 4:
        from collections import defaultdict
        clusters = defaultdict(list)
        for m in matches:
            key = m.get('match_date_label') or 'unknown'
            clusters[key].append(m)
        max_per_cluster = max(1, len(matches) // 3)
        diversified = []
        taken = {k: 0 for k in clusters}
        for m in sorted(matches, key=lambda x: x['sim'], reverse=True):
            key = m.get('match_date_label') or 'unknown'
            if taken[key] < max_per_cluster:
                diversified.append(m)
                taken[key] += 1
        if len(diversified) >= 2:
            matches = diversified

    sims = np.array([m['sim'] for m in matches], dtype=float)
    weights = sims / sims.sum()
    pcts = np.array([m['fut_pct'] for m in matches])
    maxes = np.array([m['fut_max'] for m in matches])

    weighted_pct = float(np.dot(weights, pcts))
    weighted_max = float(np.dot(weights, maxes))
    up_weight = float(sum(w for w, p in zip(weights, pcts) if p > 0))
    up_count = int(sum(1 for p in pcts if p > 0))
    dispersion = float(np.std(pcts))

    # ── YENİ: Dinamik stop-loss ve beklenen süre ────────────────────────────
    # Eşleşmelerin gerçek geçmişinden türetilir — sabit/uydurma değil.
    # NOT: stop_pct POZİTİF bir yüzde (düşüş büyüklüğü) olarak tutulur —
    # daily_scan.py bunu current_price * (1 - stop_pct/100) ile kullanıyor.
    mins = np.array([m['fut_min'] for m in matches])
    weighted_min = float(np.dot(weights, mins))  # genelde negatif (düşüş)
    downside_magnitude = -weighted_min if weighted_min < 0 else 0.0
    # Güvenlik payı: en az %3, en fazla %15 (aşırı dar/geniş stop'u engelle)
    stop_pct = round(min(15.0, max(3.0, downside_magnitude)), 2)

    days_to_peak = []
    for m in matches:
        fc = m.get('future_closes')
        if fc is not None and len(fc) > 0:
            days_to_peak.append(int(np.argmax(fc)) + 1)
        else:
            days_to_peak.append(fut_window)
    expected_days = int(round(float(np.dot(weights, np.array(days_to_peak, dtype=float)))))

    if up_weight < 0.55:
        return None

    direction_conf = up_weight * 100
    disp_penalty = min(30, dispersion * 1.2)
    avg_sim = float(np.dot(weights, sims))
    sim_bonus = max(0, (avg_sim - 60) / 40 * 15)
    match_bonus = min(10, (len(matches) - 2) * 2)
    confidence = max(0, min(100,
                             direction_conf - disp_penalty + sim_bonus + match_bonus))

    index_penalty_applied = False
    if index_corr is not None and index_corr > 0.75:
        confidence = max(0, confidence - 20)
        index_penalty_applied = True

    if confidence < 45:
        return None

    target = current_price * (1 + weighted_max / 100)

    formations = []
    try:
        from formations import scan_all_formations
        fmts = scan_all_formations(tpl_prices, volumes[-window:], min_confidence=50)
        formations = [f.name for f in fmts[:2]]
    except Exception:
        pass

    regime_label = "—"
    try:
        from bist_psi import detect_regime
        reg = detect_regime(tpl_prices, volumes[-window:])
        regime_label = reg.describe()
    except Exception:
        pass

    return {
        'ticker': ticker,
        'window': window,
        'current_price': round(current_price, 2),
        'tpl_change': round(tpl_change, 2),
        'tpl_rsi': round(tpl_rsi, 1),
        'weighted_pct': round(weighted_pct, 2),
        'target': round(target, 2),
        'weighted_max': round(weighted_max, 2),
        'stop_pct': stop_pct,
        'expected_days': expected_days,
        'ml_prob': None,
        'confidence': round(confidence, 1),
        'avg_sim': round(avg_sim, 1),
        'up_count': up_count,
        'total_matches': len(matches),
        'unique_periods': unique_periods,
        'dispersion': round(dispersion, 2),
        'regime': regime_label,
        'formations': formations,
        'index_corr': round(index_corr, 2) if index_corr is not None else None,
        'index_penalty_applied': index_penalty_applied,
        'top_matches': sorted(matches, key=lambda x: x['sim'], reverse=True)[:3],
    }


def _required_fetch_period(window_options):
    """
    Seçilen en uzun şablon (window) için, hem şablonun hem de tahmin
    penceresinin (fut_window = window*1.5) sığması için gereken minimum
    veri derinliğini hesaplar ve uygun bir yfinance period string'i döner.
    Yetersiz veri çekilirse (örn. 360G şablon için sabit '2y'), o pencerede
    hiçbir hissede yeterli geçmiş kalmaz ve tarama sessizce boş sonuç verir.
    """
    if not window_options:
        return "2y"
    max_window = max(window_options)
    # Gerekli işlem günü ≈ window + window*1.5 + güvenlik payı
    required_trading_days = max_window + int(max_window * 1.5) + 40
    # ~0.69 işlem günü/takvim günü oranıyla takvim gününe çevir, payla
    required_calendar_days = required_trading_days / 0.65
    required_years = required_calendar_days / 365
    if required_years <= 1.8:
        return "2y"
    elif required_years <= 4.5:
        return "5y"
    elif required_years <= 9:
        return "10y"
    else:
        return "max"


def render_scanner(all_data_getter, bist_lists):
    st.markdown("## 🔭 BIST Fırsat Tarayıcı")
    st.caption(
        "Her hissenin **son dönem fiyat grafiğinin şeklini** (DTW + Pearson korelasyonu) "
        "diğer hisselerin geçmişiyle karşılaştırır — 'bu grafik daha önce nerede görüldü, "
        "sonrasında ne oldu' sorusuna cevap arar. Sonuçta gösterilen tarih aralığını "
        "grafikte açıp gözle doğrulayabilirsiniz."
    )
    st.caption(
        "ℹ️ Bu, sayfa menüsündeki **Pattern Matcher**'dan farklı bir araç: Pattern "
        "Matcher şekil benzerliğine ek olarak hissenin genel karakterini (volatilite, "
        "beta, trend) de karışıma katar — o yüzden iki sayfa aynı hisse için farklı "
        "sonuç verebilir, bu normaldir."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scope = st.selectbox("Kapsam", ["BIST 30", "BIST 100", "Tüm BIST"], index=1)
    with c2:
        window_options = st.multiselect(
            "Şablon Uzunlukları (gün)", [10, 20, 30, 90, 120, 180, 240, 360],
            default=[90, 120],
            help="Kısa şablonlar (10-30G) yakın vadeli/ani kırılma sinyalleri, "
                 "uzun şablonlar (90-360G) daha yavaş gelişen trendleri yakalar. "
                 "Her uzunluk ayrı taranır ve ayrı sekmede gösterilir. "
                 "fut_window otomatik olarak şablonun 1.5 katı alınır."
        )
    with c3:
        min_sim = st.slider("Min Benzerlik", 55, 85, 80, 1,
                             help="Backtesting: PSI 80+ en iyi (%%61 kazanç)")
        min_conf = st.slider("Min Güven %", 40, 80, 55, 1,
                              help="Backtesting: 55-65 bandı optimal (%66 kazanç, +5.2%)")
        max_conf = st.slider("Maks Güven %", 60, 100, 68, 1,
                              help="Anti-consensus: 65+ güven sinyalleri daha az kazanıyor")
    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        scan_btn = st.button("🔭 Tara", type="primary", use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        sort_mode = st.radio(
            "Sonuçları Sırala",
            ["🔒 Güvene göre (varsayılan)", "⚡ En Yakın Kırılmaya göre"],
            horizontal=True,
            help="'En Yakın Kırılmaya göre', geçmiş eşleşmelerin zirveye ortalama "
                 "kaç günde ulaştığına (Beklenen Gün) bakarak en çabuk hareket "
                 "etmesi beklenen hisseleri en üste getirir."
        )
        speed_mode = st.radio(
            "Hız Modu",
            ["⚖️ Dengeli", "🚀 Hızlı (daha az aday karşılaştırılır)"],
            horizontal=True,
            help="Hızlı mod, hisse başına karşılaştırılan aday sayısını ve DTW "
                 "ince-ayar taramasını azaltır — süre yaklaşık yarıya iner, "
                 "sonuçların hassasiyetinde küçük bir düşüş olabilir."
        )
    with c6:
        max_expected_days = st.slider(
            "Maks Beklenen Gün (yakın kırılma filtresi)", 3, 360, 360, 1,
            help="Sadece bu gün sayısı içinde hareket etmesi beklenen hisseleri göster. "
                 "'Hemen çıkacak' hisseler için bunu küçük tutun (örn. 10-20 gün)."
        )

    if speed_mode.startswith("🚀"):
        candidate_top_k, dtw_top_k, refine_frac = 12, 2, 0.5
    else:
        candidate_top_k, dtw_top_k, refine_frac = 25, 3, 1.0

    if not window_options:
        st.warning("En az bir şablon uzunluğu seçmelisiniz.")
        return

    st.markdown("""
    <div style='background:#FFF7ED;border:1px solid #FED7AA;border-radius:8px;
    padding:10px 14px;margin-bottom:8px;font-size:12px;color:#92400E'>
    ⚡ Önbellekli tarama: BIST 30 ~30-60sn, BIST 100 ~2-4 dk, Tüm BIST daha uzun sürebilir.
    Tarama sırasında sayfayı kapatmayın veya başka sekmeye geçmeyin —
    Streamlit bağlantısı kopabilir.<br>
    💡 Düzenli/uzun taramalar için <b>🔔 Telegram Bildirimleri</b> sayfasından günlük
    otomatik taramayı kurabilirsiniz — o zaman tarayıcıyı açık tutmanız gerekmez.<br>
    Bu araç yatırım tavsiyesi değildir.
    </div>
    """, unsafe_allow_html=True)

    if scan_btn:
        scope_map = {
            "BIST 30": bist_lists['bist30'],
            "BIST 100": bist_lists['bist100'],
            "Tüm BIST": bist_lists['all']
        }
        tickers = scope_map[scope]

        with st.spinner("Veriler yükleniyor..."):
            fetch_period = _required_fetch_period(window_options)
            all_data = all_data_getter(tickers, period=fetch_period)

            index_closes = None
            try:
                import yfinance as yf
                xu100_raw = yf.download("XU100.IS", period="2y",
                                         auto_adjust=True, progress=False, threads=False)
                if xu100_raw is not None and not xu100_raw.empty:
                    if isinstance(xu100_raw.columns, pd.MultiIndex):
                        xu100_raw.columns = xu100_raw.columns.get_level_values(0)
                    index_closes = xu100_raw['Close'].values.astype(float)
            except Exception:
                index_closes = None

        # ── Veri çekme teşhisi — sessiz başarısızlığı görünür yap ────────────
        n_requested = len(tickers)
        n_received = len(all_data)
        if n_received == 0:
            st.session_state['fetch_diagnostic'] = {
                'level': 'error',
                'msg': (
                    f"❌ {n_requested} hisse istendi ama HİÇBİRİNİN verisi alınamadı. "
                    "Bu bir 'sonuç bulunamadı' değil, veri çekme hatası — muhtemelen "
                    "yfinance/Yahoo Finance'a çok fazla hisseyi tek seferde sorduğumuz "
                    "için istek reddedildi veya zaman aşımına uğradı. Daha küçük bir "
                    "kapsamla (BIST 30/100) tekrar deneyin, veya birkaç dakika sonra "
                    "tekrar deneyin."
                )
            }
            st.error(st.session_state['fetch_diagnostic']['msg'])
            return
        elif n_received < n_requested * 0.5:
            st.session_state['fetch_diagnostic'] = {
                'level': 'warning',
                'msg': (
                    f"⚠️ {n_requested} hisse istendi, sadece {n_received} tanesinin "
                    f"verisi alınabildi (%{n_received/n_requested*100:.0f}). Sonuçlar "
                    "eksik olabilir — bazı hisseler az işlem görüyor olabilir ya da "
                    "veri sağlayıcıda geçici bir sorun var."
                )
            }
        else:
            st.session_state['fetch_diagnostic'] = {
                'level': 'ok',
                'msg': f"✅ {n_received}/{n_requested} hisse verisi alındı."
            }

        clear_window_cache()

        # ── Tarama başlamadan önce her seçili pencere için bir kez banka kur ──
        window_banks = {}
        bank_status = st.empty()
        for i, w in enumerate(window_options):
            bank_status.info(
                f"📦 Karşılaştırma bankası hazırlanıyor: {w}G şablonu "
                f"({i + 1}/{len(window_options)})..."
            )
            window_banks[w] = build_window_bank(all_data, window=w, fut_window=int(w * 1.5))
        bank_status.empty()

        # ── Chunk boyutu: pencere sayısına göre otomatik küçülür ──────────────
        chunk_size = max(3, 20 // max(1, len(window_options)))
        # Uzun pencereler (180G+) hisse başına çok daha pahalı — chunk'ı
        # küçültüp ilerleme çubuğunun/durum satırının donmuş görünmesini önle.
        max_selected_window = max(window_options)
        if max_selected_window >= 240:
            chunk_size = max(1, chunk_size // 4)
        elif max_selected_window >= 150:
            chunk_size = max(2, chunk_size // 2)

        st.session_state['scan_job'] = {
            'tickers': list(all_data.keys()),
            'all_data': all_data,
            'index_closes': index_closes,
            'windows': list(window_options),
            'window_banks': window_banks,
            'min_sim': min_sim,
            'min_conf': min_conf,
            'max_conf': max_conf,
            'scope': scope,
            'cursor': 0,
            'chunk_size': chunk_size,
            'candidate_top_k': candidate_top_k,
            'dtw_top_k': dtw_top_k,
            'refine_frac': refine_frac,
            'results': {w: [] for w in window_options},
            'start_time': time.time(),
        }
        st.session_state.pop('scan_results', None)
        st.session_state.pop('scan_windows', None)
        st.rerun()

    job = st.session_state.get('scan_job')
    if job is not None:
        total = len(job['tickers'])
        cursor = job['cursor']
        chunk_end = min(cursor + job.get('chunk_size', 20), total)
        chunk_tickers = job['tickers'][cursor:chunk_end]

        prog = st.progress(int(cursor / total * 100) if total else 0,
                            text=f"Taranıyor: {cursor}/{total} hisse")
        eta_text = st.empty()
        elapsed = time.time() - job['start_time']
        rate = cursor / elapsed if elapsed > 0 and cursor > 0 else 0
        remaining = (total - cursor) / rate if rate > 0 else 0
        found_so_far = sum(len(v) for v in job['results'].values())
        eta_text.caption(
            f"⏱️ Geçen: {elapsed:.0f}sn | Tahmini kalan: {remaining:.0f}sn | "
            f"Bulunan: {found_so_far} fırsat | "
            f"Bu sayfa otomatik ilerleyecek — kapatmayın"
        )

        timing_log = job.get('timing_log', [])
        if timing_log:
            avg_per_call = sum(t[2] for t in timing_log) / len(timing_log)
            with st.expander(f"🔍 Hız teşhisi (hisse×pencere başına ort. {avg_per_call:.2f}sn)"):
                slowest = sorted(timing_log, key=lambda t: -t[2])[:5]
                st.caption("En yavaş 5 hisse/pencere kombinasyonu:")
                for tkr, w, dur in slowest:
                    st.text(f"  {tkr} ({w}G): {dur:.2f}sn")

        if st.button("⏹️ Taramayı İptal Et", key="cancel_scan"):
            st.session_state.pop('scan_job', None)
            st.warning("Tarama iptal edildi.")
            st.rerun()

        live_status = st.empty()
        chunk_total_steps = len(chunk_tickers) * len(job['windows'])
        step_i = 0

        for ticker in chunk_tickers:
            df = job['all_data'][ticker]
            for w in job['windows']:
                step_i += 1
                live_status.caption(
                    f"🔄 İşleniyor: {ticker} ({w}G) — bu grupta {step_i}/{chunk_total_steps}"
                )
                fut_w = int(w * 1.5)
                job.setdefault('eligible_counts', {w2: [0, 0] for w2 in job['windows']})
                job['eligible_counts'].setdefault(w, [0, 0])
                job['eligible_counts'][w][1] += 1  # toplam denenen
                if len(df) >= w + fut_w + 5:
                    job['eligible_counts'][w][0] += 1  # yeterli veriye sahip
                _t0 = time.time()
                r = scan_single_ticker(ticker, df, job['all_data'],
                                        window=w, fut_window=fut_w,
                                        min_sim=job['min_sim'], index_closes=job['index_closes'],
                                        bank=job['window_banks'].get(w),
                                        candidate_top_k=job.get('candidate_top_k', 25),
                                        dtw_top_k=job.get('dtw_top_k', 3),
                                        refine_frac=job.get('refine_frac', 1.0))
                _elapsed_ticker = time.time() - _t0
                job.setdefault('timing_log', []).append((ticker, w, round(_elapsed_ticker, 3)))
                if r and job['min_conf'] <= r['confidence'] <= job['max_conf']:
                    job['results'][w].append(r)

        job['cursor'] = chunk_end
        st.session_state['scan_job'] = job

        if chunk_end < total:
            time.sleep(0.1)
            st.rerun()
        else:
            clear_window_cache()
            key_fn = lambda x: x['confidence'] * 0.5 + x['avg_sim'] * 0.3 + x['weighted_pct'] * 0.2
            sorted_results = {w: sorted(rs, key=key_fn, reverse=True)
                               for w, rs in job['results'].items()}

            total_time = time.time() - job['start_time']
            st.session_state['scan_results'] = sorted_results
            st.session_state['scan_windows'] = job['windows']
            st.session_state['scan_scope'] = job['scope']
            st.session_state['scan_duration'] = total_time
            st.session_state['eligible_counts'] = job.get('eligible_counts', {})
            st.session_state.pop('scan_job', None)

            st.success(f"✅ Tarama {total_time:.0f} saniyede tamamlandı!")
            st.rerun()

        return

    scan_duration = st.session_state.get('scan_duration')
    if scan_duration:
        st.caption(f"✅ Son tarama {scan_duration:.0f} saniyede tamamlandı.")

    fetch_diag = st.session_state.get('fetch_diagnostic')
    if fetch_diag:
        if fetch_diag['level'] == 'error':
            st.error(fetch_diag['msg'])
        elif fetch_diag['level'] == 'warning':
            st.warning(fetch_diag['msg'])
        else:
            st.caption(fetch_diag['msg'])

    eligible_counts = st.session_state.get('eligible_counts', {})
    if eligible_counts:
        eligibility_lines = []
        for w, (eligible, total) in eligible_counts.items():
            pct = (eligible / total * 100) if total else 0
            eligibility_lines.append(f"{w}G: {eligible}/{total} hissede yeterli veri (%{pct:.0f})")
        low_data_windows = [w for w, (elig, tot) in eligible_counts.items()
                             if tot and elig / tot < 0.5]
        if low_data_windows:
            st.warning(
                "⚠️ " + " | ".join(eligibility_lines) +
                f" — {', '.join(str(w) + 'G' for w in low_data_windows)} penceresinde "
                "hisselerin çoğunda bu kadar uzun bir şablon için yeterli geçmiş veri "
                "yok, bu yüzden neredeyse hiç gerçek karşılaştırma yapılamadı. Daha "
                "kısa bir şablon uzunluğu deneyin veya bu pencereyi listeden çıkarın."
            )
        else:
            st.caption("📊 " + " | ".join(eligibility_lines))

    results_by_window = st.session_state.get('scan_results', {})
    scan_windows = st.session_state.get('scan_windows', [])

    if 'scan_results' not in st.session_state:
        st.info("Ayarları yapıp 'Tara' butonuna basın.")
        return

    # ── Görüntüleme anında filtre + sıralama uygula (yeniden tarama gerekmez) ──
    filtered_results_by_window = {}
    for w in scan_windows:
        rs = [r for r in results_by_window.get(w, []) if r['expected_days'] <= max_expected_days]
        if sort_mode.startswith("⚡"):
            rs = sorted(rs, key=lambda x: x['expected_days'])
        else:
            rs = sorted(rs, key=lambda x: x['confidence'] * 0.5 + x['avg_sim'] * 0.3 + x['weighted_pct'] * 0.2,
                        reverse=True)
        filtered_results_by_window[w] = rs
    results_by_window = filtered_results_by_window

    scope_label = st.session_state.get('scan_scope', '')
    total_found = sum(len(v) for v in results_by_window.values())

    if total_found == 0:
        st.warning(
            f"**{scope_label}** taramasında kriter karşılayan hisse bulunamadı. "
            "Min Benzerlik, Min Güven veya Maks Beklenen Gün değerlerini gevşetin."
        )
        return

    found_summary = " + ".join(
        f"{len(results_by_window.get(w, []))} adet {w}G" for w in scan_windows
    )
    st.success(f"✅ **{scope_label}** — {found_summary} fırsat")

    # ── Çoklu Şablon Doğrulama ────────────────────────────────────────────────
    # Bir hisse SEÇİLİ TÜM pencerelerde aynı yönde (bullish) çıktıysa bu
    # çift/çoklu doğrulanmış güçlü bir sinyaldir — yanlış pozitif riski düşer.
    multi_confirmed = []
    if len(scan_windows) >= 2:
        by_ticker = {}
        for w in scan_windows:
            for r in results_by_window.get(w, []):
                by_ticker.setdefault(r['ticker'], {})[w] = r

        for ticker, per_window in by_ticker.items():
            if len(per_window) == len(scan_windows) and \
               all(per_window[w]['weighted_pct'] > 0 for w in scan_windows):
                combined_conf = float(np.mean([per_window[w]['confidence'] for w in scan_windows]))
                combined_sim = float(np.mean([per_window[w]['avg_sim'] for w in scan_windows]))
                avg_pct = float(np.mean([per_window[w]['weighted_pct'] for w in scan_windows]))
                multi_confirmed.append({
                    'ticker': ticker,
                    'per_window': per_window,
                    'combined_confidence': round(combined_conf, 1),
                    'combined_similarity': round(combined_sim, 1),
                    'avg_expected_pct': round(avg_pct, 2),
                })
        multi_confirmed.sort(key=lambda x: x['combined_confidence'], reverse=True)

    if multi_confirmed:
        windows_label = "+".join(f"{w}G" for w in scan_windows)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#F0FDF4,#FFFFFF);
        border:1.5px solid #0E9F6E;border-radius:10px;
        padding:14px 18px;margin:12px 0'>
        <div style='font-size:14px;font-weight:700;color:#0E9F6E'>
        ⭐ {len(multi_confirmed)} Hisse Çoklu Doğrulanmış!
        </div>
        <div style='font-size:12px;color:#555;margin-top:4px'>
        Seçili tüm pencerelerde ({windows_label}) aynı yönde sinyal verdi —
        yanlış pozitif riski normal sinyallere göre daha düşüktür.
        </div>
        </div>
        """, unsafe_allow_html=True)

    tabs_list = [f"📊 {w}G ({len(results_by_window.get(w, []))})" for w in scan_windows]
    if multi_confirmed:
        tabs_list.append(f"⭐ Çoklu Doğrulanmış ({len(multi_confirmed)})")
    all_tabs = st.tabs(tabs_list)
    window_tabs = dict(zip(scan_windows, all_tabs[:len(scan_windows)]))
    tab_multi = all_tabs[len(scan_windows)] if multi_confirmed else None

    if tab_multi is not None:
        with tab_multi:
            windows_label = "+".join(f"{w}G" for w in scan_windows)
            st.caption(
                f"Bu hisseler seçili tüm pencerelerde ({windows_label}) aynı "
                "yönde sinyal verdi. Birden fazla bağımsız zaman dilimi aynı "
                "sonuca ulaştığı için bu eşleşmeler özellikle dikkate değer."
            )
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Çoklu Doğrulanan", f"{len(multi_confirmed)} hisse")
            dc2.metric("Ort. Birleşik Güven",
                       f"%{np.mean([d['combined_confidence'] for d in multi_confirmed]):.0f}")
            dc3.metric("Ort. Beklenen Hareket",
                       f"+{np.mean([d['avg_expected_pct'] for d in multi_confirmed]):.1f}%")

            for d in multi_confirmed:
                window_blocks = ""
                for w in scan_windows:
                    rw = d['per_window'][w]
                    window_blocks += f"""
                    <div style='flex:1;background:#F9FAFB;border-radius:6px;padding:8px 10px'>
                    <div style='font-size:10px;color:#888'>{w}G ŞABLON</div>
                    <div style='font-size:13px;color:#1A1A2E'>
                    Güven: %{rw['confidence']:.0f} |
                    Beklenen: {rw['weighted_pct']:+.1f}%
                    </div>
                    </div>
                    """
                any_row = next(iter(d['per_window'].values()))
                st.markdown(f"""
                <div style='background:#FFFFFF;border:1.5px solid #0E9F6E;
                border-radius:10px;padding:14px 16px;margin:10px 0'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                <div style='font-size:20px;font-weight:800;color:#1A1A2E'>
                ⭐ {d['ticker']}
                </div>
                <div style='text-align:right'>
                <div style='font-size:10px;color:#888'>BİRLEŞİK GÜVEN</div>
                <div style='font-size:22px;font-weight:700;color:#0E9F6E'>
                %{d['combined_confidence']:.0f}
                </div>
                </div>
                </div>
                <div style='display:flex;gap:16px;margin-top:10px;flex-wrap:wrap'>
                {window_blocks}
                </div>
                <div style='margin-top:8px;font-size:13px;color:#555'>
                Güncel Fiyat: <b>{any_row['current_price']:.2f} ₺</b> &nbsp;|&nbsp;
                Ortalama Hedef Hareket: <b style='color:#0E9F6E'>
                +{d['avg_expected_pct']:.1f}%</b>
                </div>
                </div>
                """, unsafe_allow_html=True)

    for w in scan_windows:
        tab = window_tabs[w]
        results = results_by_window.get(w, [])
        wlabel = f"{w} Günlük"
        fut_label = f"~{int(w * 1.5)} gün"

        with tab:
            if not results:
                st.info("Bu vadede fırsat bulunamadı.")
                continue

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Bulunan", f"{len(results)} hisse")
            m2.metric("Ort. Güven", f"%{np.mean([r['confidence'] for r in results]):.0f}")
            m3.metric("Ort. Benzerlik", f"{np.mean([r['avg_sim'] for r in results]):.0f}")
            m4.metric("Ort. Hedef Hareket", f"+{np.mean([r['weighted_pct'] for r in results]):.1f}%")

            rows = []
            for r in results:
                fmt_str = ' / '.join(r['formations'][:2]) if r['formations'] else '—'
                top_m = r['top_matches'][0] if r.get('top_matches') else None
                if top_m:
                    _period = (f" [{top_m['match_date_start']}–{top_m['match_date_end']}]"
                               if top_m.get('match_date_start') else "")
                    top_m_str = f"{top_m['source']}{_period} (%{top_m['sim']:.0f})"
                else:
                    top_m_str = "—"
                rows.append({
                    '🏢 Hisse': r['ticker'],
                    '🔗 En Benzediği (dönem)': top_m_str,
                    '💰 Fiyat': f"{r['current_price']:.2f} ₺",
                    f'📊 Son {wlabel}': f"{r['tpl_change']:+.1f}%",
                    'RSI': f"{r['tpl_rsi']:.0f}",
                    '🎯 Hedef': f"{r['target']:.2f} ₺",
                    '⛔ Stop': f"-%{r['stop_pct']:.1f}",
                    '📈 Beklenen': f"+{r['weighted_pct']:.1f}%",
                    '⚡ Beklenen Gün': f"{r['expected_days']} gün",
                    '🔒 Güven': f"%{r['confidence']:.0f}",
                    '✅ Oy': f"{r['up_count']}/{r['total_matches']}",
                    '🔷 Formasyon': fmt_str,
                })
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown(f"#### 🃏 Detay Kartlar — {fut_label} tahmin penceresi")

            for row_i in range(0, min(len(results), 15), 3):
                row = results[row_i:row_i + 3]
                cols = st.columns(3)
                for col, r in zip(cols, row):
                    with col:
                        conf = r['confidence']
                        c_col = ('#0E9F6E' if conf >= 65 else
                                 '#E3A008' if conf >= 50 else '#E02424')
                        conf_bar = int(conf / 5)
                        fmt_html = "".join(
                            f"<span style='background:#EFF6FF;color:#1A56DB;"
                            f"font-size:9px;padding:1px 5px;border-radius:3px;margin:1px'>{f}</span>"
                            for f in r['formations'][:2]
                        ) or "<span style='font-size:10px;color:#aaa'>—</span>"

                        top_match = r['top_matches'][0] if r.get('top_matches') else None
                        other_matches = r['top_matches'][1:3] if r.get('top_matches') and len(r['top_matches']) > 1 else []
                        other_match_lines = ""
                        for m in other_matches:
                            m_color = '#0E9F6E' if m['fut_pct'] > 0 else '#E02424'
                            if m.get('match_date_start'):
                                date_label = f"{m['match_date_start']}–{m['match_date_end']}"
                            else:
                                date_label = m.get('match_date_label', '')
                            other_match_lines += (
                                f"<div style='display:flex;justify-content:space-between;"
                                f"font-size:10px;padding:2px 0'>"
                                f"<span style='color:#555'>{m['source']}"
                                f"{f' ({date_label})' if date_label else ''}</span>"
                                f"<span style='color:#888'>%{m['sim']:.0f}</span>"
                                f"<span style='color:{m_color};font-weight:600'>"
                                f"{m['fut_pct']:+.1f}%</span></div>"
                            )

                        badges_html = ""
                        if r.get('index_penalty_applied'):
                            badges_html += (
                                "<span style='background:#FEF2F2;color:#E02424;"
                                "font-size:9px;padding:1px 6px;border-radius:3px;margin-right:4px'>"
                                f"⚠️ Piyasa geneli (%{r.get('index_corr',0)*100:.0f})</span>"
                            )
                        elif r.get('index_corr') is not None and r['index_corr'] < 0.4:
                            badges_html += (
                                "<span style='background:#F0FDF4;color:#0E9F6E;"
                                "font-size:9px;padding:1px 6px;border-radius:3px;margin-right:4px'>"
                                "✅ Hisseye özgü</span>"
                            )
                        if r.get('unique_periods', 0) >= 3:
                            badges_html += (
                                "<span style='background:#EFF6FF;color:#1A56DB;"
                                "font-size:9px;padding:1px 6px;border-radius:3px'>"
                                f"📅 {r['unique_periods']} farklı dönem</span>"
                            )

                        if top_match:
                            tm_color = '#0E9F6E' if top_match['fut_pct'] > 0 else '#E02424'
                            tm_icon = '📈' if top_match['fut_pct'] > 0 else '📉'
                            if top_match.get('match_date_start'):
                                tm_date = f"{top_match['match_date_start']} – {top_match['match_date_end']}"
                            else:
                                tm_date = top_match.get('match_date_label', '')
                            top_match_html = f"""
                            <div style='background:linear-gradient(135deg,#F0F7FF,#FFFFFF);
                            border:1px solid #BFDBFE;border-radius:8px;
                            padding:8px 10px;margin:8px 0'>
                            <div style='font-size:9px;color:#1A56DB;letter-spacing:0.5px;
                            margin-bottom:3px'>🔗 EN ÇOK BENZEDİĞİ HİSSE — DÖNEM</div>
                            <div style='display:flex;justify-content:space-between;align-items:center'>
                            <div style='font-size:16px;font-weight:800;color:#1A1A2E'>
                            {top_match['source']}
                            </div>
                            <div style='font-size:14px;font-weight:700;color:#1A56DB'>
                            %{top_match['sim']:.0f}
                            </div>
                            </div>
                            {f"<div style='font-size:11px;color:#1A56DB;font-weight:600;margin-top:2px'>📅 {tm_date}</div>" if tm_date else ""}
                            <div style='font-size:11px;color:#555;margin-top:2px'>
                            O dönemden sonra: <b style='color:{tm_color}'>
                            {tm_icon} {top_match['fut_pct']:+.1f}%</b> hareket etti — bu tarihleri
                            grafikte kontrol edebilirsiniz
                            </div>
                            </div>
                            """
                        else:
                            top_match_html = ""

                        st.markdown(f"""
                        <div style='background:#FFFFFF;border:1.5px solid #E5E9F0;
                        border-radius:10px;padding:14px 12px;margin-bottom:8px'>
                        <div style='display:flex;justify-content:space-between;align-items:start'>
                        <div>
                        <div style='font-size:20px;font-weight:800;
                        color:#1A1A2E'>{r['ticker']}</div>
                        <div style='font-size:10px;color:#888'>{r['regime']}</div>
                        </div>
                        <div style='text-align:right'>
                        <div style='font-size:10px;color:#888'>GÜVEN</div>
                        <div style='font-size:20px;font-weight:700;
                        color:{c_col}'>%{conf:.0f}</div>
                        </div>
                        </div>
                        <div style='margin:6px 0'>{badges_html}</div>
                        {top_match_html}
                        <div style='display:flex;justify-content:space-between;
                        margin:10px 0;gap:4px'>
                        <div style='text-align:center'>
                        <div style='font-size:9px;color:#888'>GÜNCEL</div>
                        <div style='font-size:13px;font-weight:600'>
                        {r['current_price']:.2f} ₺</div>
                        </div>
                        <div style='text-align:center'>
                        <div style='font-size:9px;color:#888'>BEKLENEN</div>
                        <div style='font-size:13px;font-weight:700;color:#0E9F6E'>
                        +{r['weighted_pct']:.1f}%</div>
                        </div>
                        <div style='text-align:center'>
                        <div style='font-size:9px;color:#888'>HEDEF</div>
                        <div style='font-size:13px;font-weight:600;color:#0E9F6E'>
                        {r['target']:.2f} ₺</div>
                        </div>
                        <div style='text-align:center'>
                        <div style='font-size:9px;color:#888'>RSI</div>
                        <div style='font-size:13px;font-weight:600;
                        color:{"#E02424" if r["tpl_rsi"]>70 else "#0E9F6E" if r["tpl_rsi"]<30 else "#555"}'>
                        {r['tpl_rsi']:.0f}</div>
                        </div>
                        <div style='text-align:center'>
                        <div style='font-size:9px;color:#888'>⚡ BEKLENEN GÜN</div>
                        <div style='font-size:13px;font-weight:600;color:#1A56DB'>
                        {r['expected_days']} gün</div>
                        </div>
                        </div>
                        {f'''<div style='background:#F9FAFB;border-radius:6px;
                        padding:6px 8px;margin:6px 0'>
                        <div style='font-size:9px;color:#888;margin-bottom:3px'>
                        DİĞER BENZER DÖNEMLER
                        </div>
                        {other_match_lines}
                        </div>''' if other_match_lines else ""}
                        <div style='font-family:monospace;font-size:10px;color:{c_col}'>
                        {'█'*conf_bar}{'░'*(20-conf_bar)} %{conf:.0f}
                        </div>
                        <div style='margin-top:5px'>{fmt_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if render_add_to_portfolio_button is not None:
                            render_add_to_portfolio_button(
                                ticker=r['ticker'],
                                current_price=r['current_price'],
                                source=f"Fırsat Tarayıcı ({r['window']}G)",
                                signal_score=r.get('avg_sim'),
                                confidence=r.get('confidence'),
                                expected_pct=r.get('weighted_pct'),
                                key_suffix=f"scan_{r['window']}_{row_i}"
                            )
