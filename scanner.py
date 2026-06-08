"""
scanner.py — BIST Fırsat Tarayıcı

Her hissenin son 20 ve 40 günlük hareketini şablon olarak alır.
Geçmişteki benzer hareketlerde konsensüs YÜKSELİŞ olan hisseleri listeler.
BIST-PSI v2 kullanır.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from bist_psi import BISTPSI, detect_regime
from formations import scan_all_formations, formation_summary_score
import streamlit as st

# ── Tarama parametreleri ───────────────────────────────────────────────────────

SCAN_WINDOWS = {
    'kisa': 20,   # Kısa vadeli şablon
    'orta': 40,   # Orta vadeli şablon
}

MIN_PSI_SCORE   = 65.0   # Minimum BIST-PSI benzerlik skoru
MIN_CONFIDENCE  = 55.0   # Minimum konsensüs güven skoru
FUTURE_MULT     = 1.5    # Gelecek penceresi = şablon * 1.5
MIN_HIST_DATA   = 120    # Tarihsel karşılaştırma için minimum gün


# ── Yardımcı: tek hisse için fırsat analizi ───────────────────────────────────

def analyze_opportunity(ticker: str,
                        df: pd.DataFrame,
                        window: int,
                        psi_engine: BISTPSI,
                        all_data: Dict[str, pd.DataFrame],
                        min_psi: float = MIN_PSI_SCORE) -> Optional[Dict]:
    """
    Tek bir hisse için fırsat analizi:
    1. Son `window` günü şablon al
    2. Tüm hisselerin geçmişinde benzer hareketleri bul
    3. Konsensüs YÜKSELİŞ ise fırsat olarak kaydet
    """
    closes = df['Close'].values.astype(float)
    volumes = df['Volume'].values.astype(float)
    n = len(closes)

    if n < window + 10:
        return None

    # Şablon: son `window` gün
    tpl_prices = closes[-window:]
    tpl_volumes = volumes[-window:]

    # Formasyon analizi
    formations = scan_all_formations(tpl_prices, tpl_volumes, min_confidence=45)
    fmt_summary = formation_summary_score(formations)

    # Piyasa rejimi
    regime = detect_regime(tpl_prices, tpl_volumes)

    # Referans veri: diğer hisseler + bu hissenin geçmişi
    candidates = {}
    fut_window = min(int(window * FUTURE_MULT), 60)

    # Bu hissenin kendi geçmişi
    if n >= window + fut_window + 10:
        max_start = n - window - fut_window
        step = max(1, window // 5)
        best_sim, best_start = -1, 0

        for i in range(0, max_start, step):
            w_prices = closes[i:i+window]
            from bist_psi import zscore, dtw_sim
            tpl_z = zscore(tpl_prices)
            win_z = zscore(w_prices)
            sim = dtw_sim(tpl_z, win_z) * 100
            if sim > best_sim:
                best_sim, best_start = sim, i

        if best_sim >= min_psi:
            candidates[f"{ticker}_self"] = (
                closes[best_start:best_start+window],
                volumes[best_start:best_start+window],
                closes[best_start+window:best_start+window+fut_window],
            )

    # Diğer hisseler
    for other_ticker, other_df in all_data.items():
        if other_ticker == ticker:
            continue
        other_closes = other_df['Close'].values.astype(float)
        other_vols = other_df['Volume'].values.astype(float)
        if len(other_closes) < window + fut_window + 10:
            continue

        # Hızlı DTW tarama
        from bist_psi import zscore, dtw_sim
        tpl_z = zscore(tpl_prices)
        max_start = len(other_closes) - window - fut_window
        step = max(1, window // 5)
        best_sim, best_start = -1, 0

        for i in range(0, max_start, step):
            w_z = zscore(other_closes[i:i+window])
            sim = dtw_sim(tpl_z, w_z) * 100
            if sim > best_sim:
                best_sim, best_start = sim, i

        if best_sim >= min_psi * 0.85:  # Kaba eşik
            candidates[other_ticker] = (
                other_closes[best_start:best_start+window],
                other_vols[best_start:best_start+window],
                other_closes[best_start+window:best_start+window+fut_window],
            )

    if not candidates:
        return None

    # BIST-PSI ile tam skor hesapla ve konsensüs al
    match_results = []
    for cand_key, (cand_prices, cand_vols, future) in candidates.items():
        try:
            score, psi_result = psi_engine.compute(
                tpl_prices, tpl_volumes, cand_prices, cand_vols
            )
            if score >= min_psi and len(future) > 1:
                fut_pct = (future[-1] - future[0]) / (future[0] + 1e-9) * 100
                fut_max = (future.max() - future[0]) / (future[0] + 1e-9) * 100
                match_results.append({
                    'ticker': cand_key,
                    'psi_score': score,
                    'fut_pct': fut_pct,
                    'fut_max': fut_max,
                    'regime': psi_result.regime.name,
                    'mahalanobis': psi_result.mahalanobis_sim,
                    'ensemble_ratio': psi_result.ensemble_detail['vote_ratio'],
                })
        except Exception:
            pass

    if len(match_results) < 2:
        return None

    # Konsensüs hesapla
    weights = np.array([r['psi_score'] for r in match_results], dtype=float)
    weights = weights / weights.sum()
    pcts = np.array([r['fut_pct'] for r in match_results])

    weighted_pct = float(np.dot(weights, pcts))
    up_weight = sum(w for w, p in zip(weights, pcts) if p > 0)
    up_count = sum(1 for p in pcts if p > 0)
    total = len(pcts)

    # Sadece bullish konsensüs
    if up_weight < 0.55:
        return None

    # Güven skoru
    direction_conf = up_weight * 100
    dispersion = float(np.std(pcts))
    dispersion_penalty = min(35, dispersion * 1.5)
    avg_psi = float(np.dot(weights, [r['psi_score'] for r in match_results]))
    sim_bonus = max(0, (avg_psi - 65) / 35 * 15)
    confidence = max(0, min(100, direction_conf - dispersion_penalty + sim_bonus))

    if confidence < MIN_CONFIDENCE:
        return None

    # Hedef fiyat
    current_price = float(closes[-1])
    maxs = np.array([r['fut_max'] for r in match_results])
    weighted_max = float(np.dot(weights, maxs))
    target = current_price * (1 + weighted_max / 100)

    # Top formasyon
    top_formations = [f.name for f in formations[:2]] if formations else []

    return {
        'ticker': ticker,
        'window': window,
        'window_label': f"{window}G",
        'current_price': round(current_price, 2),
        'weighted_pct': round(weighted_pct, 2),
        'target': round(target, 2),
        'confidence': round(confidence, 1),
        'avg_psi': round(avg_psi, 1),
        'up_count': up_count,
        'total_matches': total,
        'dispersion': round(dispersion, 2),
        'regime': regime.name,
        'regime_label': regime.describe(),
        'formations': top_formations,
        'fmt_signal': fmt_summary['dominant_signal'],
        'match_results': match_results[:3],
    }


# ── Ana tarama fonksiyonu ─────────────────────────────────────────────────────

def run_opportunity_scan(
    all_data: Dict[str, pd.DataFrame],
    windows: List[int] = None,
    min_psi: float = MIN_PSI_SCORE,
    min_confidence: float = MIN_CONFIDENCE,
    progress_callback=None,
) -> Dict[str, List[Dict]]:
    """
    Tüm hisseleri 20 ve 40 günlük şablon ile tara.
    Bullish fırsatları döndür.

    Returns:
        {'kisa': [...], 'orta': [...]}
    """
    if windows is None:
        windows = [20, 40]

    psi_engine = BISTPSI()

    # Kovaryans matrisini tüm hisselerden güncelle
    feat_vectors = []
    from bist_psi import extract_features, build_feature_vector
    for ticker, df in list(all_data.items())[:100]:
        try:
            closes = df['Close'].values.astype(float)[-40:]
            vols = df['Volume'].values.astype(float)[-40:]
            feat = extract_features(closes, vols)
            feat_vectors.append(build_feature_vector(feat))
        except Exception:
            pass
    if len(feat_vectors) >= 5:
        psi_engine.update_covariance(feat_vectors)

    results = {w: [] for w in windows}
    total = len(all_data)

    for idx, (ticker, df) in enumerate(all_data.items()):
        if progress_callback:
            progress_callback(idx, total, ticker)

        for window in windows:
            try:
                opp = analyze_opportunity(
                    ticker, df, window, psi_engine,
                    all_data, min_psi
                )
                if opp and opp['confidence'] >= min_confidence:
                    results[window].append(opp)
            except Exception:
                pass

    # Güven skoruna göre sırala
    for window in windows:
        results[window].sort(
            key=lambda x: (x['confidence'] * 0.5 + x['avg_psi'] * 0.5),
            reverse=True
        )

    return results


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def render_scanner(all_data_getter, bist_list):
    """
    Fırsat Tarayıcı sayfası.
    all_data_getter: fn(tickers) -> {ticker: df}
    """
    st.markdown("## 🔭 BIST Fırsat Tarayıcı")
    st.caption(
        "Tüm BIST hisselerini BIST-PSI v2 ile tarar. "
        "Geçmişteki benzer hareketlerde konsensüs yükseliş olan hisseleri listeler."
    )
    st.divider()

    # Ayarlar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scope = st.selectbox("Kapsam", ["BIST 30", "BIST 100", "Tüm BIST"],
                             index=2)
    with c2:
        min_psi = st.slider("Min BIST-PSI", 55, 85, 65, 1)
    with c3:
        min_conf = st.slider("Min Güven %", 45, 80, 55, 1)
    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        scan_btn = st.button("🔭 Tara", type="primary", use_container_width=True)

    st.markdown("""
    <div style='background:#FFF7ED;border:1px solid #FED7AA;border-radius:8px;
                padding:10px 14px;margin-bottom:12px;font-size:12px;color:#92400E'>
        ⚠️ <b>Tüm BIST taraması</b> ~500 hisse için 3-8 dakika sürebilir.
        BIST 100 ile başlamanızı öneririz.
        Bu araç yatırım tavsiyesi değildir — kendi analizinizle destekleyin.
    </div>
    """, unsafe_allow_html=True)

    if scan_btn:
        scope_map = {"BIST 30": bist_list['bist30'],
                     "BIST 100": bist_list['bist100'],
                     "Tüm BIST": bist_list['all']}
        tickers = scope_map[scope]

        prog_bar = st.progress(0, text="Veriler yükleniyor...")
        status_ph = st.empty()

        with st.spinner(""):
            all_data = all_data_getter(tickers, period="2y")

        prog_bar.progress(20, text=f"{len(all_data)} hisse yüklendi. Tarama başlıyor...")

        scan_results = {'kisa': [], 'orta': []}
        total = len(all_data)

        # Manuel iterasyon — progress göster
        psi_engine = BISTPSI()
        from bist_psi import extract_features, build_feature_vector
        feat_vecs = []
        for ticker, df in list(all_data.items())[:80]:
            try:
                c = df['Close'].values.astype(float)[-40:]
                v = df['Volume'].values.astype(float)[-40:]
                feat_vecs.append(build_feature_vector(extract_features(c, v)))
            except Exception:
                pass
        if len(feat_vecs) >= 5:
            psi_engine.update_covariance(feat_vecs)

        for idx, (ticker, df) in enumerate(all_data.items()):
            pct = 20 + int((idx / total) * 75)
            prog_bar.progress(pct, text=f"Taranan: {ticker} ({idx+1}/{total})")

            for window, wlabel in [(20, 'kisa'), (40, 'orta')]:
                try:
                    opp = analyze_opportunity(
                        ticker, df, window, psi_engine, all_data, min_psi
                    )
                    if opp and opp['confidence'] >= min_conf:
                        scan_results[wlabel].append(opp)
                except Exception:
                    pass

        for wlabel in ['kisa', 'orta']:
            scan_results[wlabel].sort(
                key=lambda x: x['confidence'] * 0.5 + x['avg_psi'] * 0.5,
                reverse=True
            )

        prog_bar.progress(100, text="✅ Tarama tamamlandı!")
        import time; time.sleep(0.5); prog_bar.empty(); status_ph.empty()

        st.session_state['scan_results'] = scan_results
        st.session_state['scan_scope'] = scope
        st.rerun()

    # Sonuçlar
    scan_results = st.session_state.get('scan_results')
    if not scan_results:
        st.info("Taramayı başlatmak için 'Tara' butonuna basın.")
        return

    total_found = len(scan_results.get(20, [])) + len(scan_results.get(40, []))
    if total_found == 0:
        st.warning("Belirlenen kriterlerde fırsat bulunamadı. Eşikleri düşürmeyi deneyin.")
        return

    scope_label = st.session_state.get('scan_scope', '')
    st.success(f"✅ **{scope_label}** taraması tamamlandı — "
               f"{len(scan_results.get(20,[]))} kısa vadeli + "
               f"{len(scan_results.get(40,[]))} orta vadeli fırsat bulundu.")

    tab_kisa, tab_orta = st.tabs([
        f"📊 Kısa Vadeli — 20G ({len(scan_results.get(20,[]))} hisse)",
        f"📈 Orta Vadeli — 40G ({len(scan_results.get(40,[]))} hisse)",
    ])

    for tab, wkey, wlabel in [
        (tab_kisa, 20, "20 Günlük"),
        (tab_orta, 40, "40 Günlük"),
    ]:
        with tab:
            opps = scan_results.get(wkey, [])
            if not opps:
                st.info("Bu vadede kriterleri karşılayan hisse bulunamadı.")
                continue

            render_opportunity_table(opps, wlabel)


def render_opportunity_table(opps: List[Dict], label: str):
    """Fırsat listesini tablo ve kartlar halinde göster."""

    # Özet istatistikler
    avg_conf = np.mean([o['confidence'] for o in opps])
    avg_psi = np.mean([o['avg_psi'] for o in opps])
    avg_target = np.mean([o['weighted_pct'] for o in opps])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bulunan Fırsat", f"{len(opps)} hisse")
    m2.metric("Ort. Güven", f"%{avg_conf:.0f}")
    m3.metric("Ort. BIST-PSI", f"{avg_psi:.0f}")
    m4.metric("Ort. Hedef Hareket", f"+{avg_target:.1f}%")

    st.markdown("---")

    # Tablo
    rows = []
    for o in opps:
        reg_icons = {'trend_bull':'📈','trend_bear':'📉','sideways':'↔️',
                     'high_vol':'⚡','low_vol':'😴'}
        fmt_str = ' / '.join(o['formations'][:2]) if o['formations'] else '—'
        rows.append({
            '🏢 Hisse':          o['ticker'],
            '💰 Fiyat':          f"{o['current_price']:.2f} ₺",
            '🎯 Hedef':          f"{o['target']:.2f} ₺",
            '📈 Beklenen':       f"+{o['weighted_pct']:.1f}%",
            '🔒 Güven':          f"%{o['confidence']:.0f}",
            '🧮 BIST-PSI':       f"{o['avg_psi']:.0f}",
            '✅ Eşleşme':        f"{o['up_count']}/{o['total_matches']}",
            '📊 Rejim':          f"{reg_icons.get(o['regime'],'')} {o['regime_label'].split()[0]}",
            '🔷 Formasyon':      fmt_str,
        })

    df_table = pd.DataFrame(rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🃏 Detay Kartlar")

    # Kartlar — 3 sütun
    cols_per_row = 3
    for row_start in range(0, len(opps), cols_per_row):
        row_opps = opps[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, o in zip(cols, row_opps):
            with col:
                conf = o['confidence']
                psi = o['avg_psi']
                conf_color = ('#0E9F6E' if conf >= 70 else
                              '#E3A008' if conf >= 55 else '#E02424')
                reg_icons = {'trend_bull':'📈','trend_bear':'📉','sideways':'↔️',
                             'high_vol':'⚡','low_vol':'😴'}
                reg_icon = reg_icons.get(o['regime'], '📊')
                fmt_html = ""
                for f in o['formations'][:2]:
                    fmt_html += f"<span style='background:#EFF6FF;color:#1A56DB;font-size:9px;padding:1px 5px;border-radius:3px;margin:1px'>{f}</span>"

                st.markdown(f"""
                <div style='background:#FFFFFF;border:1.5px solid #E5E9F0;border-radius:10px;
                            padding:14px 12px;margin-bottom:8px'>
                    <div style='display:flex;justify-content:space-between;align-items:start'>
                        <div>
                            <div style='font-size:18px;font-weight:800;color:#1A1A2E'>{o['ticker']}</div>
                            <div style='font-size:11px;color:#888'>{reg_icon} {o['regime_label']}</div>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-size:10px;color:#888'>GÜVEN</div>
                            <div style='font-size:20px;font-weight:700;color:{conf_color}'>%{conf:.0f}</div>
                        </div>
                    </div>
                    <div style='margin:10px 0;display:flex;justify-content:space-between'>
                        <div>
                            <div style='font-size:10px;color:#888'>GÜNCEL</div>
                            <div style='font-size:14px;font-weight:600'>{o['current_price']:.2f} ₺</div>
                        </div>
                        <div style='text-align:center'>
                            <div style='font-size:10px;color:#888'>BEKLENEN</div>
                            <div style='font-size:14px;font-weight:700;color:#0E9F6E'>+{o['weighted_pct']:.1f}%</div>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-size:10px;color:#888'>HEDEF</div>
                            <div style='font-size:14px;font-weight:600;color:#0E9F6E'>{o['target']:.2f} ₺</div>
                        </div>
                    </div>
                    <div style='display:flex;justify-content:space-between;font-size:11px;color:#555;
                                background:#F9FAFB;border-radius:6px;padding:6px 8px;margin:6px 0'>
                        <span>BIST-PSI: <b>{psi:.0f}</b></span>
                        <span>Eşleşme: <b>{o['up_count']}/{o['total_matches']}</b></span>
                        <span>Dağılım: ±{o['dispersion']:.1f}%</span>
                    </div>
                    <div style='margin-top:6px'>{fmt_html if fmt_html else "<span style='font-size:10px;color:#aaa'>Formasyon yok</span>"}</div>
                </div>
                """, unsafe_allow_html=True)
