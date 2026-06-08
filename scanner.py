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

# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def zscore(arr):
    arr = np.array(arr, dtype=float)
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-9:
        return np.zeros_like(arr)
    return (arr - mu) / sigma

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

def find_best_match(tpl_z, candidate_closes, window, fut_window):
    """
    Aday hissenin geçmişinde şablona en benzer bölgeyi bul.
    Sadece ardında yeterli gelecek verisi olan bölgeleri tara.
    """
    n = len(candidate_closes)
    max_start = n - window - fut_window
    if max_start < 5:
        return None

    step = max(1, window // 5)
    best_sim, best_i = -1, 0

    # Kaba tarama
    for i in range(0, max_start, step):
        sim = similarity_score(tpl_z, candidate_closes[i:i+window])
        if sim > best_sim:
            best_sim, best_i = sim, i

    # İnce tarama etrafında
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

    return {
        'sim': round(best_sim, 1),
        'fut_pct': round(fut_pct, 2),
        'fut_max': round(fut_max, 2),
        'fut_min': round(fut_min, 2),
        'match_closes': match_closes,
        'future_closes': future_closes,
    }


def scan_single_ticker(ticker, df, all_data, window, fut_window, min_sim=60):
    """
    Tek hisse için fırsat analizi:
    - Son `window` günü şablon al
    - Diğer hisselerin geçmişinde benzer dönemleri bul
    - Konsensüs hesapla
    """
    closes = df['Close'].values.astype(float)
    volumes = df['Volume'].values.astype(float)

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

    # Diğer hisselerde benzer dönem ara
    matches = []
    for other_ticker, other_df in all_data.items():
        if other_ticker == ticker:
            continue
        other_closes = other_df['Close'].values.astype(float)
        result = find_best_match(tpl_z, other_closes, window, fut_window)
        if result and result['sim'] >= min_sim:
            result['source'] = other_ticker
            matches.append(result)

    # Bu hissenin kendi geçmişinde de ara (son window gün hariç)
    if len(closes) >= window * 3 + fut_window:
        hist_closes = closes[:-window]  # Son window günü hariç tut
        result = find_best_match(tpl_z, hist_closes, window, fut_window)
        if result and result['sim'] >= min_sim:
            result['source'] = f"{ticker} (geçmiş)"
            matches.append(result)

    if len(matches) < 2:
        return None

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
        'dispersion': round(dispersion, 2),
        'regime': regime_label,
        'formations': formations,
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
        min_sim = st.slider("Min Benzerlik", 55, 80, 60, 1)
    with c3:
        min_conf = st.slider("Min Güven %", 40, 75, 48, 1)
    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        scan_btn = st.button("🔭 Tara", type="primary", use_container_width=True)

    st.markdown("""
    <div style='background:#FFF7ED;border:1px solid #FED7AA;border-radius:8px;
                padding:10px 14px;margin-bottom:8px;font-size:12px;color:#92400E'>
        ⚠️ BIST 100 ~3-5 dk, Tüm BIST ~8-15 dk sürebilir.
        İlk denemede BIST 30 veya BIST 100 önerilir.
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

        prog = st.progress(0, text="Veriler yükleniyor...")
        with st.spinner(""):
            all_data = all_data_getter(tickers, period="2y")
        prog.progress(15, text=f"{len(all_data)} hisse yüklendi. Tarama başlıyor...")

        results_20, results_40 = [], []
        total = len(all_data)

        for idx, (ticker, df) in enumerate(all_data.items()):
            pct = 15 + int((idx / total) * 80)
            prog.progress(pct, text=f"Taranan: {ticker} ({idx+1}/{total})")

            # 20 günlük şablon
            r20 = scan_single_ticker(ticker, df, all_data,
                                     window=20, fut_window=30,
                                     min_sim=min_sim)
            if r20 and r20['confidence'] >= min_conf:
                results_20.append(r20)

            # 40 günlük şablon
            r40 = scan_single_ticker(ticker, df, all_data,
                                     window=40, fut_window=60,
                                     min_sim=min_sim)
            if r40 and r40['confidence'] >= min_conf:
                results_40.append(r40)

        # Sırala
        key_fn = lambda x: x['confidence'] * 0.5 + x['avg_sim'] * 0.3 + x['weighted_pct'] * 0.2
        results_20.sort(key=key_fn, reverse=True)
        results_40.sort(key=key_fn, reverse=True)

        prog.progress(100, text="✅ Tamamlandı!")
        import time; time.sleep(0.4); prog.empty()

        st.session_state['scan_results_20'] = results_20
        st.session_state['scan_results_40'] = results_40
        st.session_state['scan_scope'] = scope
        st.rerun()

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

    tab20, tab40 = st.tabs([
        f"📊 Kısa Vadeli — 20G ({len(r20)} hisse)",
        f"📈 Orta Vadeli — 40G ({len(r40)} hisse)",
    ])

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
                rows.append({
                    '🏢 Hisse':      r['ticker'],
                    '💰 Fiyat':      f"{r['current_price']:.2f} ₺",
                    '📊 Son {wlabel}': f"{r['tpl_change']:+.1f}%",
                    'RSI':           f"{r['tpl_rsi']:.0f}",
                    '🎯 Hedef':      f"{r['target']:.2f} ₺",
                    '📈 Beklenen':   f"+{r['weighted_pct']:.1f}%",
                    '🔒 Güven':      f"%{r['confidence']:.0f}",
                    '🔗 Benzerlik':  f"{r['avg_sim']:.0f}",
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
                        c_col = ('#0E9F6E' if conf >= 65 else
                                 '#E3A008' if conf >= 50 else '#E02424')
                        conf_bar = int(conf / 5)
                        fmt_html = "".join(
                            f"<span style='background:#EFF6FF;color:#1A56DB;"
                            f"font-size:9px;padding:1px 5px;border-radius:3px;margin:1px'>{f}</span>"
                            for f in r['formations'][:2]
                        ) or "<span style='font-size:10px;color:#aaa'>—</span>"

                        # Top 3 eşleşme
                        match_lines = ""
                        for m in r['top_matches'][:3]:
                            m_color = '#0E9F6E' if m['fut_pct'] > 0 else '#E02424'
                            match_lines += (
                                f"<div style='display:flex;justify-content:space-between;"
                                f"font-size:10px;padding:1px 0'>"
                                f"<span style='color:#555'>{m['source']}</span>"
                                f"<span style='color:#888'>%{m['sim']:.0f}</span>"
                                f"<span style='color:{m_color};font-weight:600'>"
                                f"{m['fut_pct']:+.1f}%</span></div>"
                            )

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
                            </div>

                            <div style='background:#F9FAFB;border-radius:6px;
                                        padding:6px 8px;margin:6px 0'>
                                <div style='font-size:9px;color:#888;margin-bottom:3px'>
                                    BENZER DÖNEMLER VE SONRASI
                                </div>
                                {match_lines}
                            </div>

                            <div style='font-family:monospace;font-size:10px;color:{c_col}'>
                                {'█'*conf_bar}{'░'*(20-conf_bar)} %{conf:.0f}
                            </div>
                            <div style='margin-top:5px'>{fmt_html}</div>
                        </div>
                        """, unsafe_allow_html=True)
