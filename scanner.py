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

    step = max(1, window // 5)
    starts = list(range(0, max_start, step))
    windows_z = np.array([zscore(closes[i:i+window]) for i in starts])

    result = {'starts': starts, 'windows_z': windows_z, 'step': step, 'max_start': max_start}
    _WINDOW_CACHE[cache_key] = result
    return result


def clear_window_cache():
    """Yeni tarama başlarken önbelleği temizle (bellek şişmesin)."""
    _WINDOW_CACHE.clear()

def pearson(a, b):
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def dtw_fast(s1, s2, band=None):
    n = len(s1)
    if n == 0:
        return 0.0
    band = band or max(2, n // 6)
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

def similarity_score(tpl_z, win_prices):
    """Hızlı benzerlik: Pearson + DTW kombinasyonu"""
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

def find_best_match(tpl_z, candidate_closes, window, fut_window, candidate_dates=None,
                     candidate_key=None):
    """
    Aday hissenin geçmişinde şablona en benzer bölgeyi bul.
    Sadece ardında yeterli gelecek verisi olan bölgeleri tara.

    Performans: candidate_key verilirse, önbellekten hazır z-score matrisini
    kullanır (vektörize Pearson ön-eleme) — DTW sadece en güçlü adaylarda çalışır.
    """
    n = len(candidate_closes)
    max_start = n - window - fut_window
    if max_start < 5:
        return None

    cache = _get_cached_windows(candidate_key, candidate_closes, window, fut_window) \
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
            sim = similarity_score(tpl_z, candidate_closes[i:i+window])
            if sim > best_sim:
                best_sim, best_i = sim, i

        # İnce tarama etrafında (orijinal davranışla uyumlu hassasiyet)
        for i in range(max(0, best_i - step), min(max_start+1, best_i + step + 1)):
            sim = similarity_score(tpl_z, candidate_closes[i:i+window])
            if sim > best_sim:
                best_sim, best_i = sim, i
    else:
        # Önbellek yoksa eski (yavaş ama güvenilir) yöntem
        step = max(1, window // 5)
        best_sim, best_i = -1, 0
        for i in range(0, max_start, step):
            sim = similarity_score(tpl_z, candidate_closes[i:i+window])
            if sim > best_sim:
                best_sim, best_i = sim, i
        for i in range(max(0, best_i - step), min(max_start+1, best_i + step + 1)):
            sim = similarity_score(tpl_z, candidate_closes[i:i+window])
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
            match_date_label = candidate_dates[best_i].strftime('%m.%Y')
        except Exception:
            match_date_label = None

    return {
        'sim': round(best_sim, 1),
        'fut_pct': round(fut_pct, 2),
        'fut_max': round(fut_max, 2),
        'fut_min': round(fut_min, 2),
        'match_closes': match_closes,
        'future_closes': future_closes,
        'match_start_idx': best_i,
        'match_date_label': match_date_label,
    }


def scan_single_ticker(ticker, df, all_data, window, fut_window, min_sim=60,
                       index_closes=None):
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
    tpl_z = zscore(tpl_prices)
    tpl_rets = daily_returns(tpl_prices)

    # Şablon istatistikleri
    tpl_change = (tpl_prices[-1] - tpl_prices[0]) / (tpl_prices[0] + 1e-9) * 100
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

    # Diğer hisselerde benzer dönem ara
    matches = []
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
        hist_closes = closes[:-window]  # Son window günü hariç tut
        hist_dates = dates[:-window]
        result = find_best_match(tpl_z, hist_closes, window, fut_window, hist_dates,
                                 candidate_key=f"{ticker}_self")
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

    return {
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
    }


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
    with c2:
        min_sim = st.slider("Min Benzerlik", 55, 85, 80, 1,
                     help="Backtesting: PSI 80+ en iyi (%%61 kazanç)")
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

    CHUNK_SIZE = 20  # Her rerun'da bu kadar hisse işlenir — bağlantı canlı kalır

    if scan_btn:
        scope_map = {
            "BIST 30": bist_lists['bist30'],
            "BIST 100": bist_lists['bist100'],
            "Tüm BIST": bist_lists['all']
        }
        tickers = scope_map[scope]

        with st.spinner("Veriler yükleniyor..."):
            all_data = all_data_getter(tickers, period="2y")

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

        clear_window_cache()

        # Yeni tarama oturumu başlat — chunk işleme için state hazırla
        st.session_state['scan_job'] = {
            'tickers': list(all_data.keys()),
            'all_data': all_data,
            'index_closes': index_closes,
            'min_sim': min_sim,
            'min_conf': min_conf,
            'max_conf': max_conf,
            'scope': scope,
            'cursor': 0,
            'results_20': [],
            'results_40': [],
            'start_time': time.time(),
        }
        st.session_state.pop('scan_results_20', None)
        st.session_state.pop('scan_results_40', None)
        st.rerun()

    # ── Devam eden tarama işi varsa chunk'lar halinde işle ──────────────────
    job = st.session_state.get('scan_job')
    if job is not None:
        total = len(job['tickers'])
        cursor = job['cursor']
        chunk_end = min(cursor + CHUNK_SIZE, total)
        chunk_tickers = job['tickers'][cursor:chunk_end]

        prog = st.progress(int(cursor / total * 100) if total else 0,
                           text=f"Taranıyor: {cursor}/{total} hisse")
        eta_text = st.empty()
        elapsed = time.time() - job['start_time']
        rate = cursor / elapsed if elapsed > 0 and cursor > 0 else 0
        remaining = (total - cursor) / rate if rate > 0 else 0
        eta_text.caption(
            f"⏱️ Geçen: {elapsed:.0f}sn | Tahmini kalan: {remaining:.0f}sn | "
            f"Bulunan: {len(job['results_20']) + len(job['results_40'])} fırsat | "
            f"Bu sayfa otomatik ilerleyecek — kapatmayın"
        )

        if st.button("⏹️ Taramayı İptal Et", key="cancel_scan"):
            st.session_state.pop('scan_job', None)
            st.warning("Tarama iptal edildi.")
            st.rerun()

        for ticker in chunk_tickers:
            df = job['all_data'][ticker]
            r20 = scan_single_ticker(ticker, df, job['all_data'],
                                     window=20, fut_window=30,
                                     min_sim=job['min_sim'], index_closes=job['index_closes'])
            if r20 and job['min_conf'] <= r20['confidence'] <= job['max_conf']:
                job['results_20'].append(r20)

            r40 = scan_single_ticker(ticker, df, job['all_data'],
                                     window=40, fut_window=60,
                                     min_sim=job['min_sim'], index_closes=job['index_closes'])
            if r40 and job['min_conf'] <= r40['confidence'] <= job['max_conf']:
                job['results_40'].append(r40)

        job['cursor'] = chunk_end
        st.session_state['scan_job'] = job

        if chunk_end < total:
            # Daha hisse var — kısa bekleme sonrası otomatik devam et
            time.sleep(0.1)
            st.rerun()
        else:
            # Tarama tamamlandı
            clear_window_cache()
            key_fn = lambda x: x['confidence'] * 0.5 + x['avg_sim'] * 0.3 + x['weighted_pct'] * 0.2
            results_20 = sorted(job['results_20'], key=key_fn, reverse=True)
            results_40 = sorted(job['results_40'], key=key_fn, reverse=True)

            total_time = time.time() - job['start_time']
            st.session_state['scan_results_20'] = results_20
            st.session_state['scan_results_40'] = results_40
            st.session_state['scan_scope'] = job['scope']
            st.session_state['scan_duration'] = total_time
            st.session_state.pop('scan_job', None)  # all_data dahil ağır veriyi serbest bırak
            st.success(f"✅ Tarama {total_time:.0f} saniyede tamamlandı!")
            st.rerun()
        return  # Chunk işlenirken aşağıdaki sonuç bölümünü gösterme

    # ── Devam eden iş yoksa ve daha önce bitmiş sonuç varsa devam ──────────
    scan_duration = st.session_state.get('scan_duration')
    if scan_duration:
        st.caption(f"✅ Son tarama {scan_duration:.0f} saniyede tamamlandı.")

    # ── Sonuçlar ──
    r20 = st.session_state.get('scan_results_20', [])
    r40 = st.session_state.get('scan_results_40', [])

    if 'scan_results_20' not in st.session_state:
        st.info("Ayarları yapıp 'Tara' butonuna basın.")
        return

    scope_label = st.session_state.get('scan_scope', '')
    total_found = len(r20) + len(r40)

    if total_found == 0:
        st.warning(
            f"**{scope_label}** taramasında kriter karşılayan hisse bulunamadı. "
            "Min Benzerlik ve Min Güven değerlerini düşürün."
        )
        return

    st.success(f"✅ **{scope_label}** — {len(r20)} kısa vadeli + {len(r40)} orta vadeli fırsat")

    # ── Çoklu Şablon Doğrulama ──────────────────────────────────────────────
    # Hem 20G hem 40G taramasında aynı hisse, aynı yönde (bullish) çıktıysa
    # bu çift doğrulanmış güçlü bir sinyaldir — yanlış pozitif riski çok daha düşük.
    tickers_20 = {r['ticker']: r for r in r20}
    tickers_40 = {r['ticker']: r for r in r40}
    dual_confirmed = []
    for ticker in set(tickers_20.keys()) & set(tickers_40.keys()):
        r_20 = tickers_20[ticker]
        r_40 = tickers_40[ticker]
        # İkisi de bullish konsensüs (weighted_pct > 0) olmalı
        if r_20['weighted_pct'] > 0 and r_40['weighted_pct'] > 0:
            combined_conf = (r_20['confidence'] + r_40['confidence']) / 2
            combined_sim = (r_20['avg_sim'] + r_40['avg_sim']) / 2
            dual_confirmed.append({
                'ticker': ticker,
                'r20': r_20,
                'r40': r_40,
                'combined_confidence': round(combined_conf, 1),
                'combined_similarity': round(combined_sim, 1),
                'avg_expected_pct': round((r_20['weighted_pct'] + r_40['weighted_pct']) / 2, 2),
            })
    dual_confirmed.sort(key=lambda x: x['combined_confidence'], reverse=True)

    if dual_confirmed:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#F0FDF4,#FFFFFF);
                    border:1.5px solid #0E9F6E;border-radius:10px;
                    padding:14px 18px;margin:12px 0'>
            <div style='font-size:14px;font-weight:700;color:#0E9F6E'>
                ⭐ {len(dual_confirmed)} Hisse Çift Doğrulanmış!
            </div>
            <div style='font-size:12px;color:#555;margin-top:4px'>
                Hem 20 günlük hem 40 günlük şablonda aynı yönde sinyal verdi —
                yanlış pozitif riski normal sinyallere göre daha düşüktür.
            </div>
        </div>
        """, unsafe_allow_html=True)

    tabs_list = ["📊 Kısa Vadeli — 20G ({})".format(len(r20)),
                 "📈 Orta Vadeli — 40G ({})".format(len(r40))]
    if dual_confirmed:
        tabs_list.append(f"⭐ Çift Doğrulanmış ({len(dual_confirmed)})")

    all_tabs = st.tabs(tabs_list)
    tab20, tab40 = all_tabs[0], all_tabs[1]
    tab_dual = all_tabs[2] if dual_confirmed else None

    if tab_dual is not None:
        with tab_dual:
            st.caption(
                "Bu hisseler hem kısa (20G) hem orta (40G) vadeli taramada aynı "
                "yönde sinyal verdi. İki bağımsız zaman dilimi aynı sonuca "
                "ulaştığı için bu eşleşmeler özellikle dikkate değer."
            )
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Çift Doğrulanan", f"{len(dual_confirmed)} hisse")
            dc2.metric("Ort. Birleşik Güven",
                      f"%{np.mean([d['combined_confidence'] for d in dual_confirmed]):.0f}")
            dc3.metric("Ort. Beklenen Hareket",
                      f"+{np.mean([d['avg_expected_pct'] for d in dual_confirmed]):.1f}%")

            for d in dual_confirmed:
                r20_data = d['r20']
                r40_data = d['r40']
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
                        <div style='flex:1;background:#F9FAFB;border-radius:6px;padding:8px 10px'>
                            <div style='font-size:10px;color:#888'>20G ŞABLON</div>
                            <div style='font-size:13px;color:#1A1A2E'>
                                Güven: %{r20_data['confidence']:.0f} |
                                Beklenen: {r20_data['weighted_pct']:+.1f}%
                            </div>
                        </div>
                        <div style='flex:1;background:#F9FAFB;border-radius:6px;padding:8px 10px'>
                            <div style='font-size:10px;color:#888'>40G ŞABLON</div>
                            <div style='font-size:13px;color:#1A1A2E'>
                                Güven: %{r40_data['confidence']:.0f} |
                                Beklenen: {r40_data['weighted_pct']:+.1f}%
                            </div>
                        </div>
                    </div>
                    <div style='margin-top:8px;font-size:13px;color:#555'>
                        Güncel Fiyat: <b>{r20_data['current_price']:.2f} ₺</b> &nbsp;|&nbsp;
                        Ortalama Hedef Hareket: <b style='color:#0E9F6E'>
                        +{d['avg_expected_pct']:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    for tab, results, wlabel, fut_label in [
        (tab20, r20, "20 Günlük", "~30 gün"),
        (tab40, r40, "40 Günlük", "~60 gün"),
    ]:
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
                rows.append({
                    '🏢 Hisse':      r['ticker'],
                    '🔗 En Benzediği': top_m_str,
                    '💰 Fiyat':      f"{r['current_price']:.2f} ₺",
                    '📊 Son {wlabel}': f"{r['tpl_change']:+.1f}%",
                    'RSI':           f"{r['tpl_rsi']:.0f}",
                    '🎯 Hedef':      f"{r['target']:.2f} ₺",
                    '📈 Beklenen':   f"+{r['weighted_pct']:.1f}%",
                    '🔒 Güven':      f"%{r['confidence']:.0f}",
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

                            # Rozetler (piyasa geneli mi / hisseye özgü mü, dönem çeşitliliği)
                            badge_bits = []
                            if r.get('index_penalty_applied'):
                                badge_bits.append(f"⚠️ Piyasa geneli (%{r.get('index_corr', 0)*100:.0f})")
                            elif r.get('index_corr') is not None and r['index_corr'] < 0.4:
                                badge_bits.append("✅ Hisseye özgü")
                            if r.get('unique_periods', 0) >= 3:
                                badge_bits.append(f"📅 {r['unique_periods']} farklı dönem")
                            if badge_bits:
                                st.caption(" · ".join(badge_bits))

                            # En çok benzediği hisse
                            top_match = r['top_matches'][0] if r.get('top_matches') else None
                            if top_match:
                                tm_date = top_match.get('match_date_label', '')
                                tm_sign = "📈" if top_match['fut_pct'] > 0 else "📉"
                                st.info(
                                    f"**🔗 En çok benzediği:** {top_match['source']}"
                                    f"{f' ({tm_date})' if tm_date else ''} — %{top_match['sim']:.0f} benzerlik\n\n"
                                    f"O dönemden sonra: {tm_sign} **{top_match['fut_pct']:+.1f}%** hareket etti"
                                )

                            # Ana metrikler
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Güncel", f"{r['current_price']:.2f} ₺")
                            m2.metric("Beklenen", f"+{r['weighted_pct']:.1f}%")
                            m3.metric("Hedef", f"{r['target']:.2f} ₺")
                            rsi_label = "Aşırı satım" if r['tpl_rsi'] < 30 else "Aşırı alım" if r['tpl_rsi'] > 70 else None
                            m4.metric("RSI", f"{r['tpl_rsi']:.0f}", delta=rsi_label, delta_color="off")

                            # Diğer benzer dönemler
                            other_matches = (r['top_matches'][1:3]
                                              if r.get('top_matches') and len(r['top_matches']) > 1 else [])
                            if other_matches:
                                with st.expander("Diğer benzer dönemler"):
                                    for m in other_matches:
                                        date_label = m.get('match_date_label', '')
                                        sign = "🟢" if m['fut_pct'] > 0 else "🔴"
                                        st.write(
                                            f"{sign} **{m['source']}**"
                                            f"{f' ({date_label})' if date_label else ''} "
                                            f"— %{m['sim']:.0f} benzerlik → {m['fut_pct']:+.1f}%"
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
