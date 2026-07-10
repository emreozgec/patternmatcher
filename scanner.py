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
import requests

# yfinance indirmelerinin engellenmesini önlemek için User-Agent tanımlı session oluştur
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

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


_WINDOW_CACHE = {}
_RUNS_CACHE = {}

def _get_cached_top_runs(ticker_key, closes, window, fut_window, k=5):
    """
    Aday hissenin geçmişinde en çok yükseliş yaptığı top K dönemi bulup önbelleğe alır.
    Kırılım öncesi dönem eşleştirmesi (breakout_focused) için kullanılır.
    """
    cache_key = (ticker_key, window, fut_window, len(closes))
    if cache_key in _RUNS_CACHE:
        return _RUNS_CACHE[cache_key]

    n = len(closes)
    max_start = n - window - fut_window
    if max_start < 5:
        _RUNS_CACHE[cache_key] = []
        return []

    # Dinamik kırılım ve sıkışma eşikleri
    if window <= 40:
        min_peak = 15.0
        max_cons = 18.0
    elif window <= 60:
        min_peak = 20.0
        max_cons = 20.0
    else:
        min_peak = 25.0
        max_cons = 22.0

    # Rolling calculations to avoid numpy overhead inside the loop
    closes_series = pd.Series(closes)
    rolling_min = closes_series.rolling(window).min().values
    rolling_max = closes_series.rolling(window).max().values
    rolling_fut_max = closes_series.rolling(fut_window).max().values

    rises = []
    for i in range(max_start + 1):
        p_base = closes[i + window - 1]
        p_end = closes[i + window + fut_window - 1]
        
        # 1. Şablon penceresindeki sıkışma (akümülasyon) kontrolü
        tpl_min = rolling_min[i + window - 1]
        tpl_max = rolling_max[i + window - 1]
        
        # NaN kontrolü (rolling'in ilk elemanlarında NaN olabilir)
        if np.isnan(tpl_min) or np.isnan(tpl_max):
            continue
            
        tpl_range = (tpl_max - tpl_min) / (tpl_min + 1e-9) * 100
        
        if tpl_range > max_cons:
            continue  # Sıkışma yok, çok oynak veya zaten yükselmiş
            
        # 1.b. Şablon penceresinin net değişimi (çöküş/düşüş veya sert yükseliş olmamalı)
        tpl_net_change = (p_base - closes[i]) / (closes[i] + 1e-9) * 100
        if tpl_net_change < -8.0 or tpl_net_change > 12.0:
            continue  # Sıkışma döneminde sert düşüş ya da sert yükseliş yaşamış, yatay değil
            
        # 2. Gelecek penceresindeki kırılım büyüklüğü (zirve getiri) kontrolü
        peak_price = rolling_fut_max[i + window + fut_window - 1]
        if np.isnan(peak_price):
            continue
            
        peak_pct = (peak_price - p_base) / (p_base + 1e-9) * 100
        
        # Zirve yükseliş en az min_peak olmalı ve net bitiş getirisi pozitif kalmalı (>3.0%)
        pct_end = (p_end - p_base) / (p_base + 1e-9) * 100
        if peak_pct >= min_peak and pct_end > 3.0:
            rises.append((peak_pct, i))

    # Zirve yükseliş büyüklüğüne göre sırala
    rises.sort(key=lambda x: x[0], reverse=True)

    selected_indices = []
    min_distance = window
    for peak, idx in rises:
        # Çakışmayan (non-overlapping) dönemleri seç
        if all(abs(idx - sel_idx) >= min_distance for sel_idx in selected_indices):
            selected_indices.append(idx)
            if len(selected_indices) >= k:
                break

    _RUNS_CACHE[cache_key] = selected_indices
    return selected_indices

def _get_cached_windows(ticker_key, closes, window, fut_window, use_log=True):
    """
    Bir hissenin tüm sliding window z-score'larını önceden hesapla ve önbelleğe al.
    Aynı tarama içinde birden fazla şablon bu hisseyi aday olarak kullanacaksa
    tekrar hesaplamayı önler.
    """
    cache_key = (ticker_key, window, fut_window, len(closes), use_log)
    if cache_key in _WINDOW_CACHE:
        return _WINDOW_CACHE[cache_key]

    n = len(closes)
    max_start = n - window - fut_window
    if max_start < 5:
        _WINDOW_CACHE[cache_key] = None
        return None

    step = max(1, window // 5)
    starts = list(range(0, max_start, step))
    
    if use_log:
        windows_z = np.array([zscore(np.log(np.maximum(closes[i:i+window], 1e-5))) for i in starts])
    else:
        windows_z = np.array([zscore(closes[i:i+window]) for i in starts])

    result = {'starts': starts, 'windows_z': windows_z, 'step': step, 'max_start': max_start}
    _WINDOW_CACHE[cache_key] = result
    return result


def clear_window_cache():
    """Yeni tarama başlarken önbelleği temizle (bellek şişmesin)."""
    _WINDOW_CACHE.clear()
    _RUNS_CACHE.clear()

def pearson(a, b):
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# Numba JIT, Streamlit Cloud konteynerlerindeki Segmentation Fault hatalarını önlemek için devre dışı bırakıldı.
_dtw_fast_jit = None

def dtw_fast(s1, s2, band=None):
    n = len(s1)
    if n == 0:
        return 0.0
    band = band or max(2, n // 6)
    
    if _dtw_fast_jit is not None:
        try:
            dist = _dtw_fast_jit(s1, s2, band) / n
            return max(0.0, 1.0 - dist * 1.5)
        except Exception:
            pass # fallback
            
    dtw = np.full((n+1, n+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        j0 = max(1, i - band)
        j1 = min(n, i + band) + 1
        for j in range(j0, j1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    dist = dtw[n,n] / n
    return max(0.0, 1.0 - dist * 1.5)


def similarity_score(tpl_z, win_prices, use_log=True):
    """Hızlı benzerlik: Pearson + DTW kombinasyonu"""
    if use_log:
        win_z = zscore(np.log(np.maximum(win_prices, 1e-5)))
    else:
        win_z = zscore(win_prices)
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
        ag = (ag*(n-1) + gains[i]) / n
        al = (al*(n-1) + losses[i]) / n
    return float(100 - 100 / (1 + ag / (al + 1e-9)))

def calc_volatility_stop_pct(closes, n=20):
    """Kapanış fiyatlarının son N günlük standart sapmasına göre dinamik stop-loss yüzdesi hesaplar."""
    if len(closes) < n + 2:
        return 5.0 # Varsayılan %5
    rets = np.diff(closes[-n-1:]) / (closes[-n-1:-1] + 1e-9)
    std = np.std(rets)
    # 2.5 standart sapma güvenlik marjı (BIST için ideal)
    stop_pct = std * 2.5 * 100
    # Stop oranını mantıklı sınırlar içinde tutalım (%3 ile %10 arası)
    return float(np.clip(stop_pct, 3.0, 10.0))

def calc_bb_width_and_squeeze(closes, n=20, check_days=90):
    """
    20 günlük Bollinger Bant Genişliği (Band Width) hesaplar.
    Son check_days gün içindeki en dar dilime (bottom 25% percentile) bakarak
    volatilite sıkışması (squeeze) durumunu belirler.
    """
    if len(closes) < n:
        return 1.0, False
    widths = []
    start_idx = max(0, len(closes) - check_days - n)
    for i in range(start_idx, len(closes) - n + 1):
        win = closes[i : i + n]
        ma = np.mean(win)
        std = np.std(win)
        width = (4 * std) / (ma + 1e-9)
        widths.append(width)
        
    if not widths:
        return 1.0, False
    current_width = widths[-1]
    pct_25 = np.percentile(widths, 25)
    squeeze_active = current_width <= pct_25
    return current_width, squeeze_active

def find_best_match(tpl_z, candidate_closes, window, fut_window, candidate_dates=None,
                     candidate_key=None, breakout_focused=True, candidate_volumes=None,
                     use_log=True):
    """
    Aday hissenin geçmişinde şablona en benzer bölgeyi bul.
    Sadece ardında yeterli gelecek verisi olan bölgeleri tara.
    """
    n = len(candidate_closes)
    max_start = n - window - fut_window
    if max_start < 5:
        return None

    if breakout_focused:
        starts = _get_cached_top_runs(candidate_key, candidate_closes, window, fut_window, k=5)
        if not starts:
            return None
        
        best_sim, best_i = -1.0, starts[0]
        for i in starts:
            sim = similarity_score(tpl_z, candidate_closes[i:i+window], use_log=use_log)
            if sim > best_sim:
                best_sim, best_i = sim, i
    else:
        cache = _get_cached_windows(candidate_key, candidate_closes, window, fut_window, use_log=use_log) \
                if candidate_key is not None else None

        if cache is not None and len(cache['starts']) > 0:
            starts = cache['starts']
            windows_z = cache['windows_z']   # (n_windows, window)
            step = cache['step']

            # Vektörize Pearson ön-eleme — tek numpy işlemiyle tüm pencereler
            t = tpl_z - tpl_z.mean()
            w = windows_z - windows_z.mean(axis=1, keepdims=True)
            t_norm = np.sqrt((t**2).sum()) + 1e-9
            w_norms = np.sqrt((w**2).sum(axis=1)) + 1e-9
            pearson_scores = (w @ t) / (w_norms * t_norm)  # -1..1, shape (n_windows,)

            # En iyi 3 adayı DTW ile detaylı kontrol et
            top_k = min(3, len(starts))
            top_idx = np.argpartition(-pearson_scores, top_k - 1)[:top_k]

            best_sim, best_i = -1, starts[0]
            for idx in top_idx:
                i = starts[idx]
                sim = similarity_score(tpl_z, candidate_closes[i:i+window], use_log=use_log)
                if sim > best_sim:
                    best_sim, best_i = sim, i

            # İnce tarama etrafında (orijinal davranışla uyumlu hassasiyet)
            for i in range(max(0, best_i - step), min(max_start+1, best_i + step + 1)):
                sim = similarity_score(tpl_z, candidate_closes[i:i+window], use_log=use_log)
                if sim > best_sim:
                    best_sim, best_i = sim, i
        else:
            # Önbellek yoksa eski (yavaş ama güvenilir) yöntem
            step = max(1, window // 5)
            best_sim, best_i = -1, 0
            for i in range(0, max_start, step):
                sim = similarity_score(tpl_z, candidate_closes[i:i+window], use_log=use_log)
                if sim > best_sim:
                    best_sim, best_i = sim, i
            for i in range(max(0, best_i - step), min(max_start+1, best_i + step + 1)):
                sim = similarity_score(tpl_z, candidate_closes[i:i+window], use_log=use_log)
                if sim > best_sim:
                    best_sim, best_i = sim, i

    if best_sim < 55:
        return None

    match_closes = candidate_closes[best_i:best_i+window]
    future_closes = candidate_closes[best_i+window:best_i+window+fut_window]

    if len(future_closes) < 3:
        return None

    fut_pct = (future_closes[-1] - future_closes[0]) / (future_closes[0] + 1e-9) * 100
    fut_max = (future_closes.max() - future_closes[0]) / (future_closes[0] + 1e-9) * 100
    fut_min = (future_closes.min() - future_closes[0]) / (future_closes[0] + 1e-9) * 100

    match_date_label = None
    if candidate_dates is not None and best_i < len(candidate_dates):
        try:
            start_date = candidate_dates[best_i].strftime('%d.%m.%Y')
            end_idx = min(len(candidate_dates) - 1, best_i + window - 1)
            end_date = candidate_dates[end_idx].strftime('%d.%m.%Y')
            match_date_label = f"{start_date} - {end_date}"
        except Exception:
            match_date_label = None

    peak_days = int(np.argmax(future_closes))

    # Şablon sıkışma aralığı (consolidation tightness)
    tpl_min = match_closes.min()
    tpl_max = match_closes.max()
    tpl_range = (tpl_max - tpl_min) / (tpl_min + 1e-9) * 100

    # Hacim patlaması (Volume Surge) hesabı
    volume_surge_ratio = 1.0
    if candidate_volumes is not None and len(candidate_volumes) > best_i + window + 5:
        tpl_vol = candidate_volumes[best_i : best_i + window]
        breakout_vol = candidate_volumes[best_i + window : best_i + window + 5]  # Kırılımın ilk 5 günü
        avg_tpl_vol = np.mean(tpl_vol)
        avg_breakout_vol = np.mean(breakout_vol)
        if avg_tpl_vol > 1e-9:
            volume_surge_ratio = avg_breakout_vol / avg_tpl_vol

    return {
        'sim': round(best_sim, 1),
        'fut_pct': round(fut_pct, 2),
        'fut_max': round(fut_max, 2),
        'fut_min': round(fut_min, 2),
        'match_closes': match_closes,
        'future_closes': future_closes,
        'match_start_idx': best_i,
        'match_date_label': match_date_label,
        'peak_days': peak_days,
        'tpl_range': round(tpl_range, 1),
        'volume_surge_ratio': round(volume_surge_ratio, 2),
    }



def scan_single_ticker(ticker, df, all_data, window, fut_window, min_sim=60,
                       index_closes=None, bist100_set=None, breakout_focused=True,
                       max_template_change=10.0, use_log=True, start_year="2020"):
    """
    Tek hisse için fırsat analizi:
    - Son `window` günü şablon al
    - Diğer hisselerin geçmişinde benzer dönemleri bul
    - Konsensüs hesapla
    - Endeks korelasyonu kontrol et (genel piyasa hareketi mi?)
    """
    closes = df['Close'].values.astype(float)
    volumes = df['Volume'].values.astype(float)
    dates = df.index

    if len(closes) < window + 10:
        return None

    # Şablon
    tpl_prices = closes[-window:]
    if use_log:
        tpl_z = zscore(np.log(np.maximum(tpl_prices, 1e-5)))
    else:
        tpl_z = zscore(tpl_prices)
    tpl_rets = daily_returns(tpl_prices)

    # Şablon istatistikleri
    tpl_change = (tpl_prices[-1] - tpl_prices[0]) / (tpl_prices[0] + 1e-9) * 100

    # Kırılım gelmeden yakalama filtresi: Son dönemde çok yükselmiş hisseleri ele
    if max_template_change is not None and tpl_change > max_template_change:
        return None

    tpl_rsi = calc_rsi(tpl_prices)
    current_price = float(closes[-1])

    # Endeks korelasyon kontrolü — genel piyasa hareketi mi?
    index_corr = None
    if index_closes is not None and len(index_closes) >= window:
        idx_tpl = index_closes[-window:]
        min_len = min(len(tpl_prices), len(idx_tpl))
        stock_rets = daily_returns(tpl_prices[-min_len:])
        idx_rets = daily_returns(idx_tpl[-min_len:])
        m = min(len(stock_rets), len(idx_rets))
        if m >= 4 and np.std(stock_rets[-m:]) > 1e-9 and np.std(idx_rets[-m:]) > 1e-9:
            index_corr = float(np.corrcoef(stock_rets[-m:], idx_rets[-m:])[0, 1])

    # Tarih filtresi
    cutoff_date = pd.Timestamp(f"{start_year}-01-01")

    # Diğer hisselerde benzer dönem ara
    matches = []
    for other_ticker, other_df in all_data.items():
        if other_ticker == ticker:
            continue
            
        # Tüm BIST taramasında (len > 150) aday arama havuzunu sadece BIST100 ile sınırlarız.
        # Bu işlem aramayı 5 kat hızlandırır ve likit olmayan gürültülü tahtaları eler.
        if bist100_set is not None and len(all_data) > 150:
            clean_ticker = other_ticker.replace('.IS', '')
            if clean_ticker not in bist100_set and other_ticker != "XU100.IS":
                continue

        # Arama geçmişi başlangıç yılı filtresi
        candidate_df = other_df.loc[other_df.index >= cutoff_date]
        if len(candidate_df) < window + fut_window + 5:
            continue

        other_closes = candidate_df['Close'].values.astype(float)
        other_volumes = candidate_df['Volume'].values.astype(float) if 'Volume' in candidate_df.columns else None
        other_dates = candidate_df.index
        result = find_best_match(tpl_z, other_closes, window, fut_window, other_dates,
                                 candidate_key=other_ticker, breakout_focused=breakout_focused,
                                 candidate_volumes=other_volumes, use_log=use_log)
        if result and result['sim'] >= min_sim:
            result['source'] = other_ticker
            matches.append(result)


    # Bu hissenin kendi geçmişinde de ara (son window gün hariç)
    self_hist_df = df.loc[(df.index >= cutoff_date) & (df.index < dates[-window])]
    if len(self_hist_df) >= window * 2 + fut_window:
        hist_closes = self_hist_df['Close'].values.astype(float)
        hist_volumes = self_hist_df['Volume'].values.astype(float) if 'Volume' in self_hist_df.columns else None
        hist_dates = self_hist_df.index
        result = find_best_match(tpl_z, hist_closes, window, fut_window, hist_dates,
                                 candidate_key=f"{ticker}_self", breakout_focused=breakout_focused,
                                 candidate_volumes=hist_volumes, use_log=use_log)
        if result and result['sim'] >= min_sim:
            result['source'] = f"{ticker} (geçmiş)"
            matches.append(result)

    if len(matches) < 2:
        return None

    # ── Tarih Çeşitliliği Filtresi ──────────────────────────────────────────
    # Aynı döneme yığılan eşleşmeleri sınırla (genel piyasa hareketi sinyali)
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

    # Ağırlıklı konsensüs
    sims = np.array([m['sim'] for m in matches], dtype=float)
    weights = sims / sims.sum()
    pcts = np.array([m['fut_pct'] for m in matches])
    maxes = np.array([m['fut_max'] for m in matches])

    weighted_pct = float(np.dot(weights, pcts))
    weighted_max = float(np.dot(weights, maxes))
    peak_days_arr = np.array([m.get('peak_days', 0) for m in matches])
    expected_days = int(round(float(np.dot(weights, peak_days_arr))))

    up_weight = float(sum(w for w, p in zip(weights, pcts) if p > 0))
    up_count = int(sum(1 for p in pcts if p > 0))
    dispersion = float(np.std(pcts))

    # Sadece bullish konsensüs
    if up_weight < 0.55:
        return None

    # Güven skoru
    direction_conf = up_weight * 100
    disp_penalty = min(30, dispersion * 1.2)
    avg_sim = float(np.dot(weights, sims))
    sim_bonus = max(0, (avg_sim - 60) / 40 * 15)
    # Eşleşme sayısı bonusu
    match_bonus = min(10, (len(matches) - 2) * 2)
    confidence = max(0, min(100,
        direction_conf - disp_penalty + sim_bonus + match_bonus))

    # 1. Hacim Kırılımı Filtresi (Hacimli kırılımlar daha başarılıdır)
    current_vol = float(volumes[-1])
    avg_vol_20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1.0
    rel_vol = current_vol / (avg_vol_20 + 1e-9)
    vol_bonus = 0
    if rel_vol > 1.5:
        vol_bonus = 5    # Hacimli kırılım bonusu
    elif rel_vol < 0.5:
        vol_bonus = -10  # Düşük hacim cezası

    # 2. Endeks Trend Filtresi (Ayı piyasasında long sinyalleri cezalandırılır)
    index_trend_bullish = True
    index_trend_penalty = 0
    if index_closes is not None and len(index_closes) >= 20:
        index_trend_bullish = bool(index_closes[-1] >= np.mean(index_closes[-20:]))
        if not index_trend_bullish:
            index_trend_penalty = -10  # Endeks SMA20 altındaysa ceza

    confidence = max(0, min(100, confidence + vol_bonus + index_trend_penalty))

    # Endeks korelasyonu çok yüksekse güven cezalandırılır
    index_penalty_applied = False
    if index_corr is not None and index_corr > 0.75:
        confidence = max(0, confidence - 20)
        index_penalty_applied = True

    if confidence < 45:
        return None


    target = current_price * (1 + weighted_max / 100)

    # Formasyon tespiti (basit)
    formations = []
    try:
        from formations import scan_all_formations
        fmts = scan_all_formations(tpl_prices, volumes[-window:], min_confidence=50)
        formations = [f.name for f in fmts[:2]]
    except Exception:
        pass

    # Rejim
    regime_label = "—"
    try:
        from bist_psi import detect_regime
        reg = detect_regime(tpl_prices, volumes[-window:])
        regime_label = reg.describe()
    except Exception:
        pass

    reasons = []
    try:
        from rise_reason import analyze_rise_reasons
        reasons = analyze_rise_reasons(df, window=window)
    except Exception as e:
        reasons = ["📊 Teknik Düzeltme"]

    res = {
        'reasons': reasons,
        'ticker': ticker,
        'window': window,
        'current_price': round(current_price, 2),
        'tpl_change': round(tpl_change, 2),
        'tpl_rsi': round(tpl_rsi, 1),
        'weighted_pct': round(weighted_pct, 2),
        'target': round(target, 2),
        'weighted_max': round(weighted_max, 2),
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
        'expected_days': expected_days,
        'relative_volume': round(rel_vol, 2),
        'index_trend_bullish': index_trend_bullish,
        'squeeze_active': calc_bb_width_and_squeeze(closes, n=20, check_days=90)[1],
        'stop_pct': calc_volatility_stop_pct(closes, n=20),
    }

    try:
        from ml_model import get_ml_win_probability
        res['ml_prob'] = get_ml_win_probability(res)
    except Exception:
        res['ml_prob'] = None

    return res





# ── Streamlit UI ───────────────────────────────────────────────────────────────

def render_scanner(all_data_getter, bist_lists):
    st.markdown("## 🔭 BIST Fırsat Tarayıcı")
    st.caption(
        "Her hissenin son hareketini şablon alır, diğer hisselerin geçmişinde "
        "benzer dönemleri bulur ve konsensüs yükseliş olan hisseleri listeler."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scope = st.selectbox("Kapsam", ["BIST 30", "BIST 100", "Tüm BIST"], index=1)
        selected_windows = st.multiselect(
            "Şablon Vadeleri", 
            [20, 40, 60, 90, 120, 180, 250, 365], 
            default=[90, 120],
            help="Taranacak kalıp günlerini belirler. 180G ve 365G gibi uzun kalıplar için 5-10 yıllık veri indirilir."
        )
        breakout_focused = st.checkbox(
            "Sadece Kırılım Öncesi Dönemleri Eşleştir",
            value=True,
            help="Hisselerin geçmişindeki en büyük yükselişler öncesindeki kalıplarla eşleştirme yapar. Hem 100 kat hızlı çalışır hem de yükseliş potansiyeli yüksek sonuçlar üretir."
        )
        only_bullish_index = st.checkbox(
            "Sadece Endeks Pozitifken Sinyal Üret",
            value=False,
            help="BIST 100 endeksi 20 günlük hareketli ortalamasının altındayken (düşüş trendindeyken) long sinyali üretmez ve sistemi korumaya alır."
        )
    with c2:
        min_sim = st.slider("Min Benzerlik", 55, 85, 80, 1,
                     help="Backtesting: PSI 80+ en iyi (%%61 kazanç)")
        max_tpl_change = st.slider("Maks. Son Yükseliş (%)", 5, 30, 10, 1,
                     help="Hissenin son 20/40 günde halihazırda en fazla ne kadar yükselmiş olabileceğini sınırlar. Yatay/akümülasyon aşamasındaki hisseleri yakalamak için %8-%12 civarı önerilir.")
        start_year = st.selectbox("Arama Geçmişi Başlangıcı", ["2016", "2018", "2020", "2022"], index=2,
                                  help="Kalıpların aranacağı en eski tarihi sınırlar. 2020 ve sonrası yüksek enflasyonlu yeni rejimi temsil eder.")
        use_log = st.checkbox("Logaritmik (Yüzdesel) Eşleştirme", value=True,
                             help="Fiyat hareketlerini logaritmik ölçekte eşleştirerek enflasyon kaynaklı fiyat ölçeği bozunmalarını engeller.")
    with c3:
        min_conf = st.slider("Min Güven %", 40, 80, 55, 1,
                      help="Backtesting: 55-65 bandı optimal (%66 kazanç, +5.2%)")
        max_conf = st.slider("Maks Güven %", 60, 100, 68, 1,
                      help="Anti-consensus: 65+ güven sinyalleri daha az kazanıyor")
    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        scan_btn = st.button("🔭 Tara", type="primary", use_container_width=True)

    st.markdown("""
    <div style='background:#FFF7ED;border:1px solid #FED7AA;border-radius:8px;
                padding:10px 14px;margin-bottom:8px;font-size:12px;color:#92400E'>
        ⚡ Önbellekli tarama: BIST 30 ~30-60sn, BIST 100 ~2-4 dk, Tüm BIST ~6-10 dk.
        Tarama sırasında sayfayı kapatmayın veya başka sekmeye geçmeyin —
        Streamlit bağlantısı kopabilir.<br>
        💡 Düzenli tarama için <b>🔔 Telegram Bildirimleri</b> sayfasından günlük
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

        # RAM dostu parçalama (Chunking): Tüm BIST taranırken 1GB RAM'in taşmaması için
        # Hisseleri 50'şerli gruplar halinde indirip sadece Close/Volume verilerini alacağız.
        chunk_size = 50 if scope == "Tüm BIST" else len(tickers)
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        # Dinamik veri yükleme dönemi belirleme
        max_w = max(selected_windows) if selected_windows else 120
        scan_period = "10y" if max_w >= 180 else ("5y" if max_w >= 90 else "2y")

        index_closes = None
        try:
            import yfinance as yf
            xu100_raw = yf.download("XU100.IS", period=scan_period,
                                    auto_adjust=True, progress=False, threads=False, session=session)
            if xu100_raw is not None and not xu100_raw.empty:
                if isinstance(xu100_raw.columns, pd.MultiIndex):
                    xu100_raw.columns = xu100_raw.columns.get_level_values(0)
                index_closes = xu100_raw['Close'].values.astype(float)
                
                # Sadece Endeks Pozitifken Sinyal Üret kontrolü
                if only_bullish_index and len(index_closes) >= 20:
                    index_trend_bullish = bool(index_closes[-1] >= np.mean(index_closes[-20:]))
                    if not index_trend_bullish:
                        st.error("⚠️ BIST 100 endeksi düşüş trendinde (kapanış 20 günlük ortalamanın altında). 'Sadece Endeks Pozitifken Sinyal Üret' seçeneği aktif olduğu için tarama durduruldu.")
                        return
        except Exception:
            index_closes = None

        clear_window_cache()

        # Paralel Tarama hazırlığı
        start_time = time.time()
        
        # Progress bar elemanları
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        import concurrent.futures
        import gc
        
        lightweight_data = {}
        total_chunks = len(ticker_chunks)
        
        # 1. Aşama: Verilerin Parça Parça İndirilmesi ve Hafifletilmesi (0% - 50%)
        for idx, chunk in enumerate(ticker_chunks):
            progress_text.text(f"Veriler indiriliyor (Grup {idx + 1}/{total_chunks})...")
            chunk_raw = all_data_getter(chunk, period=scan_period)
            if chunk_raw:
                for t, df in chunk_raw.items():
                    if df is not None and not df.empty and 'Close' in df.columns:
                        # Sadece gerekli sütunları (Close, Volume) kopyalıyoruz (Büyük RAM tasarrufu!)
                        cols = ['Close', 'Volume'] if 'Volume' in df.columns else ['Close']
                        lightweight_df = df[cols].copy()
                        if 'Volume' not in lightweight_df.columns:
                            lightweight_df['Volume'] = 0.0
                        lightweight_data[t] = lightweight_df
            
            # Progress bar güncelle
            percent = int((idx + 1) / total_chunks * 50)
            progress_bar.progress(percent)
            
            # Gereksiz ham veriyi sil ve RAM'i boşalt
            del chunk_raw
            gc.collect()

        progress_text.text("Hisseler analiz ediliyor...")
        clear_window_cache()
        bist100_set = set(bist_lists['bist100'])


        def _process_ticker_task(ticker):
            df = lightweight_data.get(ticker)
            if df is None or len(df) < 10:
                return None
            
            ticker_results = {}
            for win in selected_windows:
                fut_win = int(win * 1.5)
                # Dinamik gevşetme için taramayı floor limit olan min_sim=55 ile yapıyoruz, filtrelemeyi sonra yapacağız
                r = scan_single_ticker(ticker, df, lightweight_data,
                                       window=win, fut_window=fut_win,
                                       min_sim=55, index_closes=index_closes,
                                       bist100_set=bist100_set, breakout_focused=breakout_focused,
                                       max_template_change=max_tpl_change,
                                       use_log=use_log, start_year=start_year)
                if r:
                    ticker_results[win] = r
            return ticker, ticker_results


        tickers_list = list(lightweight_data.keys())
        total_tickers = len(tickers_list)
        
        # Streamlit Cloud'da RAM taşmasını önlemek için Tüm BIST'te thread sayısını 4'e düşürüyoruz
        max_workers = 4 if scope == "Tüm BIST" else 8
        results_by_window = {win: [] for win in selected_windows}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_ticker_task, t): t for t in tickers_list}
            
            for idx, fut in enumerate(concurrent.futures.as_completed(futures)):
                try:
                     res = fut.result()
                     if res:
                          ticker, ticker_results = res
                          for win, r in ticker_results.items():
                               if min_conf <= r['confidence'] <= max_conf:
                                    results_by_window[win].append(r)
                except Exception as e:
                     print(f"⚠️ {futures[fut]} taranırken hata: {e}")
                
                # Progress bar güncelle
                percent = 50 + int((idx + 1) / total_tickers * 50)
                progress_bar.progress(percent)
                progress_text.text(f"Analiz ediliyor: {idx + 1}/{total_tickers} hisse")

        clear_window_cache()
        
        # Eşikleri gevşeterek en az 3 adet hisse getirme mantığı (+%5 getiri beklentisi şartıyla)
        def relax_filter(results, user_sim, user_conf):
            # Sadece getiri beklentisi en az %5.0 olanları dikkate al
            valid_results = [r for r in results if r.get('weighted_pct', 0.0) >= 5.0]
            
            # Kademeli gevşetme seviyeleri (sim, conf)
            steps = [
                (user_sim, user_conf),
                (max(55, user_sim - 5), max(45, user_conf - 5)),
                (max(55, user_sim - 10), max(45, user_conf - 10)),
                (55, 45)
            ]
            
            for sim_th, conf_th in steps:
                filtered = [r for r in valid_results if r['avg_sim'] >= sim_th and r['confidence'] >= conf_th]
                if len(filtered) >= 3:
                    return filtered
            
            # 3'e ulaşamazsa floor limitlere göre filtrelenmiş halini dön
            return [r for r in valid_results if r['avg_sim'] >= 55 and r['confidence'] >= 45]

        key_fn = lambda x: x['confidence'] * 0.5 + x['avg_sim'] * 0.3 + x['weighted_pct'] * 0.2
        results_sorted_by_window = {}
        for win in selected_windows:
            relaxed = relax_filter(results_by_window[win], min_sim, min_conf)
            sorted_res = sorted(relaxed, key=key_fn, reverse=True)
            results_sorted_by_window[win] = sorted_res
            st.session_state[f'scan_results_{win}'] = sorted_res

        total_time = time.time() - start_time
        
        st.session_state['scan_scope'] = scope
        st.session_state['scan_duration'] = total_time
        st.session_state['scan_selected_windows'] = selected_windows
        
        # Temizle
        progress_bar.empty()
        progress_text.empty()
        
        # RAM temizliği
        del lightweight_data
        gc.collect()
        
        # SQLite veritabanına otomatik kaydet
        try:
             import db_utils
             from datetime import datetime
             today_str = datetime.today().strftime('%Y-%m-%d')
             db_utils.init_db() # Veritabanının oluşturulduğundan emin ol
             for win in selected_windows:
                 for r in results_sorted_by_window[win]:
                     stop_val = round(r['current_price'] * (1 - r['stop_pct'] / 100), 2)
                     db_utils.save_signal(
                         ticker=r['ticker'],
                         window=win,
                         signal_date=today_str,
                         entry_price=r['current_price'],
                         target_price=r['target'],
                         weighted_pct=r['weighted_pct'],
                         confidence=r['confidence'],
                         avg_sim=r['avg_sim'],
                         source='manual_scan',
                         expected_days=r['expected_days'],
                         stop_price=stop_val
                     )

        except Exception as e:
             print(f"⚠️ Veritabanına kaydederken hata: {e}")
             
        st.success(f"✅ Tarama {total_time:.1f} saniyede tamamlandı!")
        st.rerun()


    # ── Devam eden iş yoksa ve daha önce bitmiş sonuç varsa devam ──────────
    scan_duration = st.session_state.get('scan_duration')
    if scan_duration:
        st.caption(f"✅ Son tarama {scan_duration:.0f} saniyede tamamlandı.")

    # ── Sonuçlar ──
    if 'scan_selected_windows' not in st.session_state:
        st.info("Ayarları yapıp 'Tara' butonuna basın.")
        return

    selected_windows = st.session_state['scan_selected_windows']
    results_by_window = {win: st.session_state.get(f'scan_results_{win}', []) for win in selected_windows}

    scope_label = st.session_state.get('scan_scope', '')
    total_found = sum(len(res) for res in results_by_window.values())

    if total_found == 0:
        st.warning(
            f"**{scope_label}** taramasında kriter karşılayan hisse bulunamadı. "
            "Min Benzerlik ve Min Güven değerlerini düşürün."
        )
        return

    col_suc, col_tg = st.columns([3, 1])
    col_suc.success(f"✅ **{scope_label}** — Toplam {total_found} adet fırsat tespiti yapıldı.")
    with col_tg:
        if st.button("📤 Telegram'a Gönder", key="manual_tg_send_btn", use_container_width=True):
            with st.spinner("Telegram'a gönderiliyor..."):
                try:
                    from telegram_utils import send_telegram_message, format_results_message
                    messages = format_results_message(results_by_window, scope_label)
                    success_count = 0
                    for m in messages:
                        if send_telegram_message(m):
                            success_count += 1
                        time.sleep(1)
                    if success_count == len(messages):
                        st.success("✅ Gönderildi!")
                    else:
                        st.warning(f"⚠️ {success_count}/{len(messages)} gönderildi.")
                except Exception as e:
                    st.error(f"❌ Hata: {e}")


    # ── Çoklu Şablon Doğrulama ──────────────────────────────────────────────
    # Taranan farklı vadelerden en az ikisinde aynı hisse, aynı yönde (bullish) çıktıysa
    # bu çift doğrulanmış güçlü bir sinyaldir — yanlış pozitif riski çok daha düşük.
    multi_confirmed = []
    if len(selected_windows) >= 2:
        all_tickers = set()
        tickers_by_win = {}
        for win in selected_windows:
            tickers_by_win[win] = {r['ticker']: r for r in results_by_window[win]}
            all_tickers.update(tickers_by_win[win].keys())
            
        for ticker in all_tickers:
            matched_windows = []
            for win in selected_windows:
                if ticker in tickers_by_win[win] and tickers_by_win[win][ticker]['weighted_pct'] > 0:
                    matched_windows.append((f"{win}G", tickers_by_win[win][ticker]))
            
        if len(matched_windows) >= 2:
            combined_conf = np.mean([w[1]['confidence'] for w in matched_windows])
            combined_sim = np.mean([w[1]['avg_sim'] for w in matched_windows])
            combined_pct = np.mean([w[1]['weighted_pct'] for w in matched_windows])
            
            multi_confirmed.append({
                'ticker': ticker,
                'windows': matched_windows,
                'combined_confidence': round(combined_conf, 1),
                'combined_similarity': round(combined_sim, 1),
                'avg_expected_pct': round(combined_pct, 2),
            })
    multi_confirmed.sort(key=lambda x: x['combined_confidence'], reverse=True)

    if multi_confirmed:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#F0FDF4,#FFFFFF);
                    border:1.5px solid #0E9F6E;border-radius:10px;
                    padding:14px 18px;margin:12px 0'>
            <div style='font-size:14px;font-weight:700;color:#0E9F6E'>
                ⭐ {len(multi_confirmed)} Hisse Çoklu Doğrulanmış!
            </div>
            <div style='font-size:12px;color:#555;margin-top:4px'>
                Hem 90 günlük hem de 120 günlük şablonlarda aynı yönde sinyal verdi —
                yanlış pozitif riski normal sinyallere göre daha düşüktür.
            </div>
        </div>
        """, unsafe_allow_html=True)

    tabs_list = [f"📊 {win} Günlük ({len(results_by_window[win])})" for win in selected_windows]
    if multi_confirmed:
        tabs_list.append(f"⭐ Çoklu Doğrulanmış ({len(multi_confirmed)})")

    all_tabs = st.tabs(tabs_list)
    tab_multi = all_tabs[-1] if multi_confirmed else None

    if tab_multi is not None:
        with tab_multi:
            st.caption(
                "Bu hisseler taranan farklı şablon vadelerinin en az ikisinde aynı "
                "yönde sinyal verdi. Farklı bağımsız zaman dilimleri aynı sonuca "
                "ulaştığı için bu eşleşmeler yüksek güvenilirlik taşır."
            )
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Çoklu Doğrulanan", f"{len(multi_confirmed)} hisse")
            dc2.metric("Ort. Birleşik Güven",
                      f"%{np.mean([d['combined_confidence'] for d in multi_confirmed]):.0f}")
            dc3.metric("Ort. Beklenen Hareket",
                      f"+{np.mean([d['avg_expected_pct'] for d in multi_confirmed]):.1f}%")

            for d in multi_confirmed:
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
                    <div style='display:flex;gap:16px;margin-top:10px'>
                """ + "".join(f"""
                        <div style='flex:1;background:#F9FAFB;border-radius:6px;padding:8px 10px'>
                            <div style='font-size:10px;color:#888'>{w[0]} ŞABLON</div>
                            <div style='font-size:13px;color:#1A1A2E'>
                                Güven: %{w[1]['confidence']:.0f} |
                                Beklenen: {w[1]['weighted_pct']:+.1f}%
                            </div>
                        </div>
                """ for w in d['windows']) + f"""
                    </div>
                    <div style='margin-top:8px;font-size:13px;color:#555'>
                        Ortalama Hedef: <b style='color:#0E9F6E'>+{d['avg_expected_pct']:.1f}%</b> &nbsp;|&nbsp;
                        Zaman Dilimleri: <b>{" &middot; ".join(w[0] for w in d['windows'])}</b>
                    </div>

                </div>
                """, unsafe_allow_html=True)

    for idx, win in enumerate(selected_windows):
        tab = all_tabs[idx]
        results = results_by_window[win]
        wlabel = f"{win} Günlük"
        fut_label = f"~{int(win * 1.5)} gün"
        with tab:
            if not results:
                st.info("Bu vadede fırsat bulunamadı.")
                continue

            # Özet metrikler
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Bulunan", f"{len(results)} hisse")
            m2.metric("Ort. Güven", f"%{np.mean([r['confidence'] for r in results]):.0f}")
            m3.metric("Ort. Benzerlik", f"{np.mean([r['avg_sim'] for r in results]):.0f}")
            m4.metric("Ort. Hedef Hareket", f"+{np.mean([r['weighted_pct'] for r in results]):.1f}%")

            # Tablo
            rows = []
            for r in results:
                fmt_str = ' / '.join(r['formations'][:2]) if r['formations'] else '—'
                top_m = r['top_matches'][0] if r.get('top_matches') else None
                top_m_str = f"{top_m['source']} (%{top_m['sim']:.0f})" if top_m else "—"
                ml_prob_val = f"%{r['ml_prob']:.0f}" if r.get('ml_prob') is not None else "—"
                reasons_str = ", ".join(r.get('reasons', [])) if r.get('reasons') else "—"
                rows.append({
                    '🏢 Hisse':      r['ticker'],
                    '🔍 Yükseliş Nedeni': reasons_str,
                    '🔗 En Benzediği': top_m_str,
                    '💰 Fiyat':      f"{r['current_price']:.2f} ₺",
                    '🛑 Stop-Loss':  f"{r['current_price'] * (1 - r['stop_pct'] / 100):.2f} ₺ (-{r['stop_pct']:.1f}%)",
                    '📊 Son {wlabel}': f"{r['tpl_change']:+.1f}%",
                    'RSI':           f"{r['tpl_rsi']:.0f}",
                    '🎯 Hedef':      f"{r['target']:.2f} ₺",
                    '📈 Beklenen':   f"+{r['weighted_pct']:.1f}%",
                    '⏳ Tahmini Vade': f"~{r.get('expected_days', 0)} gün",
                    '🔒 Güven':      f"%{r['confidence']:.0f}",
                    '🤖 ML Olasılık': ml_prob_val,
                    '✅ Oy':         f"{r['up_count']}/{r['total_matches']}",
                    '🔷 Formasyon':  fmt_str,
                })



            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown(f"#### 🃏 Detay Kartlar — {fut_label} tahmin penceresi")

            # 3 sütun kart
            for row_i in range(0, min(len(results), 15), 3):
                row = results[row_i:row_i+3]
                cols = st.columns(3)
                for col, r in zip(cols, row):
                    with col:
                        conf = r['confidence']
                        conf_emoji = "🟢" if conf >= 65 else "🟡" if conf >= 50 else "🔴"

                        with st.container(border=True):
                            # Başlık satırı: hisse adı + güven
                            h1, h2 = st.columns([2, 1])
                            with h1:
                                st.markdown(f"### {r['ticker']}")
                                st.caption(r['regime'])
                            with h2:
                                st.metric("Güven", f"%{conf:.0f}")

                            # Sebepler (Capsule rozetleri)
                            if r.get('reasons'):
                                badges_html = " ".join(f"<span style='background-color:#EBF5FF; color:#1E40AF; font-size:11px; font-weight:600; padding:3px 8px; border-radius:12px; margin-right:4px; display:inline-block'>{reason}</span>" for reason in r['reasons'])
                                st.markdown(f"<div style='margin-bottom:10px'>{badges_html}</div>", unsafe_allow_html=True)

                            # Rozetler (piyasa geneli mi / hisseye özgü mü, dönem çeşitliliği, hacim kırılımı, endeks trend)
                            badge_bits = []
                            if r.get('squeeze_active'):
                                badge_bits.append("⚡ Sıkışma (Squeeze)")
                            if r.get('index_penalty_applied'):
                                badge_bits.append(f"⚠️ Piyasa geneli (%{r.get('index_corr', 0)*100:.0f})")
                            elif r.get('index_corr') is not None and r['index_corr'] < 0.4:
                                badge_bits.append("✅ Hisseye özgü")
                            if r.get('unique_periods', 0) >= 3:
                                badge_bits.append(f"📅 {r['unique_periods']} farklı dönem")
                            if r.get('relative_volume', 1.0) > 1.5:
                                badge_bits.append("🔥 Hacimli Kırılım")
                            elif r.get('relative_volume', 1.0) < 0.5:
                                badge_bits.append("💤 Düşük Hacim")
                            if not r.get('index_trend_bullish', True):
                                badge_bits.append("⚠️ Endeks Negatif")

                            if badge_bits:
                                st.caption(" · ".join(badge_bits))

                            # ML Başarı Olasılığı
                            if r.get('ml_prob') is not None:
                                prob_val = r['ml_prob']
                                prob_color = "#0E9F6E" if prob_val >= 60 else ("#D97706" if prob_val >= 50 else "#EF5350")
                                st.markdown(f"""
                                <div style='background:#F9FAFB; border:1px solid #E5E7EB; border-radius:6px; padding:6px 12px; display:flex; justify-content:space-between; align-items:center; margin-bottom:10px'>
                                    <span style='font-size:13px; font-weight:600; color:#374151'>🤖 ML Başarı Olasılığı</span>
                                    <span style='font-size:15px; font-weight:700; color:{prob_color}'>%{prob_val:.0f}</span>
                                </div>
                                """, unsafe_allow_html=True)

                            # En çok benzediği hisse
                            top_match = r['top_matches'][0] if r.get('top_matches') else None
                            if top_match:
                                tm_date = top_match.get('match_date_label', '')
                                tm_sign = "📈" if top_match['fut_pct'] > 0 else "📉"
                                surge_label = " | 🔥 Hacim Onaylı" if top_match.get('volume_surge_ratio', 1.0) > 1.5 else ""
                                st.info(
                                    f"**🔗 En çok benzediği:** {top_match['source']}"
                                    f"{f' ({tm_date})' if tm_date else ''} — %{top_match['sim']:.0f} benzerlik\n\n"
                                    f"O dönemden sonra: {tm_sign} **{top_match['fut_pct']:+.1f}%** hareket etti\n\n"
                                    f"📊 Sıkışma: %{top_match.get('tpl_range', 0.0):.1f} | Zirve: +%{top_match.get('fut_max', 0.0):.1f}{surge_label}"
                                )

                            # Ana metrikler
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Güncel", f"{r['current_price']:.2f} ₺")
                            m2.metric("Beklenen", f"+{r['weighted_pct']:.1f}%")
                            m3.metric("Hedef", f"{r['target']:.2f} ₺",
                                      delta=f"~{r.get('expected_days', 0)} günde" if r.get('expected_days') else None,
                                      delta_color="off")

                            rsi_label = "Aşırı satım" if r['tpl_rsi'] < 30 else "Aşırı alım" if r['tpl_rsi'] > 70 else None
                            m4.metric("RSI", f"{r['tpl_rsi']:.0f}", delta=rsi_label, delta_color="off")

                            # Dinamik Stop-Loss ve Hedef Gösterimi
                            stop_price = r['current_price'] * (1 - r['stop_pct'] / 100)
                            st.markdown(
                                f"<div style='font-size:12px;color:#374151;margin:6px 0'>"
                                f"🛑 <b>Dinamik Stop-Loss:</b> {stop_price:.2f} ₺ (-{r['stop_pct']:.1f}%)<br>"
                                f"🎯 <b>Hedef Fiyat:</b> {r['target']:.2f} ₺ (+{r['weighted_pct']:.1f}%)"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                            # Diğer benzer dönemler
                            other_matches = (r['top_matches'][1:3]
                                              if r.get('top_matches') and len(r['top_matches']) > 1 else [])
                            if other_matches:
                                with st.expander("Diğer benzer dönemler"):
                                    for m in other_matches:
                                        date_label = m.get('match_date_label', '')
                                        sign = "🟢" if m['fut_pct'] > 0 else "🔴"
                                        surge_label = " (🔥 Hacim Onaylı)" if m.get('volume_surge_ratio', 1.0) > 1.5 else ""
                                        st.write(
                                            f"{sign} **{m['source']}**"
                                            f"{f' ({date_label})' if date_label else ''} "
                                            f"— %{m['sim']:.0f} benzerlik → {m['fut_pct']:+.1f}% "
                                            f"(Sıkışma: %{m.get('tpl_range', 0.0):.1f} | Zirve: +%{m.get('fut_max', 0.0):.1f}){surge_label}"
                                        )

                            # Güven barı
                            st.progress(int(conf) / 100, text=f"{conf_emoji} Güven: %{conf:.0f}")

                            # Formasyonlar
                            if r['formations']:
                                st.caption("🔷 " + " · ".join(r['formations'][:2]))

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
