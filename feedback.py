"""
feedback.py — Eşleşme Kalite Geri Bildirimi

Kullanıcı, Pattern Matcher'da bulunan eşleşmelere 👍/👎 verebilir.
Bu geri bildirimler session_state'te biriktirilir ve zamanla:
  - Hangi boyutların (DTW, getiri, hacim, momentum, karakter, vb.) gerçekten
    "iyi" eşleşmelerle korele olduğunu gösterir
  - Basit bir ağırlık ayarlama önerisi üretir (tam ML değil, açıklanabilir
    bir istatistiksel geri besleme katmanı)

Not: Veriler sadece session_state'te tutulur (oturum bazlı). Kalıcı
saklama isteniyorsa ileride bir dosya/veritabanına genişletilebilir.
"""

import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional


# ── Veri yönetimi ──────────────────────────────────────────────────────────────

def init_feedback():
    if 'match_feedback' not in st.session_state:
        st.session_state.match_feedback = []   # list of feedback dicts


def record_feedback(ticker: str,
                    symbol: str,
                    rating: str,             # 'good' | 'bad'
                    similarity: float,
                    breakdown: Dict,
                    fut_pct: float,
                    char_score: Optional[float] = None,
                    fut_compat: Optional[float] = None,
                    corr_score: Optional[float] = None,
                    regime_match: Optional[bool] = None,
                    source_page: str = "Pattern Matcher"):
    """Bir eşleşme için kullanıcı oyu kaydet."""
    init_feedback()

    entry = {
        'ticker': ticker,
        'symbol': symbol,
        'rating': rating,
        'similarity': similarity,
        'breakdown': dict(breakdown) if breakdown else {},
        'fut_pct': fut_pct,
        'char_score': char_score,
        'fut_compat': fut_compat,
        'corr_score': corr_score,
        'regime_match': regime_match,
        'source_page': source_page,
        'timestamp': datetime.now().strftime('%d.%m.%Y %H:%M'),
    }
    st.session_state.match_feedback.append(entry)


def get_feedback_for(symbol: str, ticker: str) -> Optional[str]:
    """Bu şablon+eşleşme kombinasyonu için daha önce oy verilmiş mi?"""
    init_feedback()
    for f in reversed(st.session_state.match_feedback):
        if f['symbol'] == symbol and f['ticker'] == ticker:
            return f['rating']
    return None


def get_all_feedback() -> List[Dict]:
    init_feedback()
    return st.session_state.match_feedback


def clear_feedback():
    st.session_state.match_feedback = []


# ── Analiz ──────────────────────────────────────────────────────────────────────

DIMENSION_LABELS = {
    'fiyat_dtw': 'Fiyat Şekli (DTW)',
    'fiyat_pearson': 'Fiyat Şekli (Pearson)',
    'getiri': 'Günlük Getiri Dağılımı',
    'hacim': 'Hacim Profili',
    'momentum': 'Momentum (RSI/MACD)',
    'formasyon': 'Formasyon Uyumu',
    'mtf_score': 'Çoklu Zaman Dilimi',
    'karakter': 'Hisse Karakteri',
    'gelecek_uyum': 'Gelecek Hareket Uyumu',
    'korelasyon': 'Tarihsel Korelasyon',
}


def analyze_feedback() -> Dict:
    """
    Toplanan geri bildirimleri analiz et:
    - Her boyutun "iyi" oylarda ortalaması vs "kötü" oylarda ortalaması
    - Fark büyükse o boyut gerçekten ayırt edici demektir
    - Önerilen ağırlık ayarlaması (mevcut ağırlığa göre nispi artır/azalt)
    """
    feedback = get_all_feedback()
    if len(feedback) < 5:
        return {
            'sufficient_data': False,
            'n_good': sum(1 for f in feedback if f['rating'] == 'good'),
            'n_bad': sum(1 for f in feedback if f['rating'] == 'bad'),
            'dimension_analysis': [],
        }

    good = [f for f in feedback if f['rating'] == 'good']
    bad = [f for f in feedback if f['rating'] == 'bad']

    dimension_keys = set()
    for f in feedback:
        dimension_keys.update(f['breakdown'].keys())

    dim_analysis = []
    for dim in sorted(dimension_keys):
        good_vals = [f['breakdown'].get(dim) for f in good if f['breakdown'].get(dim) is not None]
        bad_vals = [f['breakdown'].get(dim) for f in bad if f['breakdown'].get(dim) is not None]

        if len(good_vals) < 2 or len(bad_vals) < 2:
            continue

        avg_good = float(np.mean(good_vals))
        avg_bad = float(np.mean(bad_vals))
        diff = avg_good - avg_bad
        discriminative_power = abs(diff)

        if diff > 5:
            suggestion = "Agirligi artirilabilir - iyi eslesmelerde belirgin yuksek"
        elif diff < -5:
            suggestion = "Agirligi azaltilabilir - kotu eslesmelerde de yuksek cikiyor"
        else:
            suggestion = "Ayirt edici degil, mevcut agirlik korunabilir"

        dim_analysis.append({
            'dimension': dim,
            'label': DIMENSION_LABELS.get(dim, dim),
            'avg_good': round(avg_good, 1),
            'avg_bad': round(avg_bad, 1),
            'diff': round(diff, 1),
            'discriminative_power': round(discriminative_power, 1),
            'suggestion': suggestion,
        })

    dim_analysis.sort(key=lambda x: x['discriminative_power'], reverse=True)

    good_fut = [f['fut_pct'] for f in good if f.get('fut_pct') is not None]
    bad_fut = [f['fut_pct'] for f in bad if f.get('fut_pct') is not None]
    avg_fut_good = round(float(np.mean(good_fut)), 1) if good_fut else None
    avg_fut_bad = round(float(np.mean(bad_fut)), 1) if bad_fut else None

    return {
        'sufficient_data': True,
        'n_good': len(good),
        'n_bad': len(bad),
        'dimension_analysis': dim_analysis,
        'avg_fut_pct_good': avg_fut_good,
        'avg_fut_pct_bad': avg_fut_bad,
    }


# ── UI bileşenleri ──────────────────────────────────────────────────────────────

def render_feedback_buttons(symbol: str, ticker: str, similarity: float,
                            breakdown: Dict, fut_pct: float,
                            char_score: float = None, fut_compat: float = None,
                            corr_score: float = None, regime_match: bool = None,
                            source_page: str = "Pattern Matcher",
                            key_suffix: str = ""):
    """Bir eşleşme kartının altına 👍/👎 butonları koy."""
    init_feedback()
    existing = get_feedback_for(symbol, ticker)

    btn_key_good = f"fb_good_{symbol}_{ticker}_{key_suffix}"
    btn_key_bad = f"fb_bad_{symbol}_{ticker}_{key_suffix}"

    c1, c2 = st.columns(2)
    with c1:
        good_label = "İyi Eşleşme" if existing != 'good' else "İşaretlendi (İyi)"
        if st.button(good_label, key=btn_key_good, use_container_width=True,
                     type="primary" if existing == 'good' else "secondary"):
            record_feedback(ticker, symbol, 'good', similarity, breakdown, fut_pct,
                           char_score, fut_compat, corr_score, regime_match, source_page)
            st.rerun()
    with c2:
        bad_label = "Kötü Eşleşme" if existing != 'bad' else "İşaretlendi (Kötü)"
        if st.button(bad_label, key=btn_key_bad, use_container_width=True,
                     type="primary" if existing == 'bad' else "secondary"):
            record_feedback(ticker, symbol, 'bad', similarity, breakdown, fut_pct,
                           char_score, fut_compat, corr_score, regime_match, source_page)
            st.rerun()


def render_feedback_summary_page():
    """Tüm geri bildirimlerin toplandığı, analiz edildiği sayfa."""
    init_feedback()
    feedback = get_all_feedback()

    st.markdown("## Eşleşme Kalite Geri Bildirimi")
    st.caption(
        "Pattern Matcher'da bulduğunuz eşleşmelere verdiğiniz İyi/Kötü oylar "
        "burada toplanır. Yeterli veri birikince hangi analiz boyutlarının "
        "gerçekten değerli olduğunu görebilirsiniz."
    )
    st.divider()

    if not feedback:
        st.info(
            "Henüz geri bildirim yok. Pattern Matcher sayfasında bir eşleşmenin "
            "detayına girip altındaki butonlarla oy verin."
        )
        return

    m1, m2, m3 = st.columns(3)
    n_good = sum(1 for f in feedback if f['rating'] == 'good')
    n_bad = sum(1 for f in feedback if f['rating'] == 'bad')
    m1.metric("Toplam Oy", len(feedback))
    m2.metric("İyi", n_good)
    m3.metric("Kötü", n_bad)

    analysis = analyze_feedback()

    if not analysis['sufficient_data']:
        st.warning(
            f"Anlamlı analiz için en az 5 oy gerekli (şu an {len(feedback)}). "
            "Daha fazla eşleşmeyi değerlendirin."
        )
    else:
        st.divider()
        st.markdown("### Boyut Analizi")
        st.caption(
            "Her boyutun 'iyi' olarak işaretlenen eşleşmelerdeki ortalama skoru "
            "ile 'kötü' işaretlenenlerdeki ortalaması karşılaştırılıyor."
        )

        if analysis.get('avg_fut_pct_good') is not None and analysis.get('avg_fut_pct_bad') is not None:
            vc1, vc2 = st.columns(2)
            vc1.metric("İyi İşaretlenenlerin Ort. Sonraki Hareketi",
                      f"{analysis['avg_fut_pct_good']:+.1f}%")
            vc2.metric("Kötü İşaretlenenlerin Ort. Sonraki Hareketi",
                      f"{analysis['avg_fut_pct_bad']:+.1f}%")

        for dim in analysis['dimension_analysis']:
            diff = dim['diff']
            color = '#0E9F6E' if diff > 5 else ('#E02424' if diff < -5 else '#888')
            bar_good = int(dim['avg_good'] / 5)
            bar_bad = int(dim['avg_bad'] / 5)

            st.markdown(
                "<div style='background:#FAFAFA;border:1px solid #E5E9F0;border-radius:8px;"
                "padding:10px 14px;margin:6px 0'>"
                "<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-weight:600;font-size:13px'>{dim['label']}</span>"
                f"<span style='color:{color};font-weight:700;font-size:13px'>{diff:+.1f} fark</span>"
                "</div>"
                f"<div style='font-size:11px;color:#555;margin-top:4px'>"
                f"İyi ort: %{dim['avg_good']:.0f} {'█'*bar_good}{'░'*(20-bar_good)}</div>"
                f"<div style='font-size:11px;color:#888;margin-top:2px'>"
                f"Kötü ort: %{dim['avg_bad']:.0f} {'█'*bar_bad}{'░'*(20-bar_bad)}</div>"
                f"<div style='font-size:11px;color:{color};margin-top:4px;font-style:italic'>"
                f"{dim['suggestion']}</div>"
                "</div>",
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown("### Tüm Oylar")
    import pandas as pd
    rows = []
    for f in reversed(feedback):
        icon = "İyi" if f['rating'] == 'good' else "Kötü"
        rows.append({
            'Oy': icon,
            'Şablon': f['symbol'],
            'Eşleşen': f['ticker'],
            'Benzerlik': f"%{f['similarity']:.0f}",
            'Sonraki Hareket': f"{f['fut_pct']:+.1f}%",
            'Kaynak': f['source_page'],
            'Tarih': f['timestamp'],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.button("Tüm Geri Bildirimleri Temizle"):
        clear_feedback()
        st.rerun()
