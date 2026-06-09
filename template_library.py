"""
template_library.py — Şablon Kütüphanesi

Şablonları session state'te saklar.
Her şablona: hisse, tarih, notlar, etiket, skor geçmişi kaydedilir.
Kaydedilen şablon ile hemen tarama veya karşılaştırma yapılabilir.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Optional


# ── Veri yapısı ────────────────────────────────────────────────────────────────

def init_library():
    """Session state'te kütüphaneyi başlat."""
    if 'template_library' not in st.session_state:
        st.session_state.template_library = []


def save_template(symbol: str,
                  start_date: str,
                  end_date: str,
                  prices: np.ndarray,
                  volumes: np.ndarray,
                  name: str = "",
                  notes: str = "",
                  tags: List[str] = None,
                  scan_results: List[Dict] = None,
                  consensus: Dict = None,
                  regime: str = "",
                  formations: List[str] = None) -> str:
    """Şablonu kütüphaneye kaydet. Template ID döndürür."""
    init_library()

    template_id = f"{symbol}_{start_date}_{datetime.now().strftime('%H%M%S')}"
    pct_change = float((prices[-1] - prices[0]) / (prices[0] + 1e-9) * 100)

    entry = {
        'id':           template_id,
        'symbol':       symbol,
        'start_date':   start_date,
        'end_date':     end_date,
        'name':         name or f"{symbol} — {start_date} / {end_date}",
        'notes':        notes,
        'tags':         tags or [],
        'prices':       prices.tolist(),
        'volumes':      volumes.tolist(),
        'n_days':       len(prices),
        'pct_change':   round(pct_change, 2),
        'created_at':   datetime.now().strftime('%d.%m.%Y %H:%M'),
        'regime':       regime,
        'formations':   formations or [],
        'scan_history': [],   # Geçmiş tarama sonuçları
    }

    # İlk tarama sonucunu da kaydet
    if scan_results or consensus:
        entry['scan_history'].append({
            'date':     datetime.now().strftime('%d.%m.%Y %H:%M'),
            'results':  scan_results or [],
            'consensus': consensus or {},
        })

    st.session_state.template_library.append(entry)
    return template_id


def update_scan_history(template_id: str,
                        scan_results: List[Dict],
                        consensus: Dict):
    """Mevcut şablona yeni tarama sonucu ekle."""
    init_library()
    for t in st.session_state.template_library:
        if t['id'] == template_id:
            t['scan_history'].append({
                'date':      datetime.now().strftime('%d.%m.%Y %H:%M'),
                'results':   scan_results,
                'consensus': consensus,
            })
            break


def delete_template(template_id: str):
    init_library()
    st.session_state.template_library = [
        t for t in st.session_state.template_library
        if t['id'] != template_id
    ]


def get_library() -> List[Dict]:
    init_library()
    return st.session_state.template_library


# ── Görselleştirme ─────────────────────────────────────────────────────────────

def fig_template_preview(prices: np.ndarray, symbol: str,
                          start_date: str, end_date: str,
                          height: int = 160) -> go.Figure:
    """Küçük şablon önizleme grafiği."""
    prices = np.array(prices)
    pct = (prices[-1] - prices[0]) / (prices[0] + 1e-9) * 100
    color = '#0E9F6E' if pct >= 0 else '#E02424'
    x = list(range(len(prices)))

    fig = go.Figure(go.Scatter(
        x=x, y=prices, mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'{"rgba(14,159,110,0.08)" if pct>=0 else "rgba(224,36,36,0.08)"}',
        hovertemplate='Gün %{x}: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=5, r=5, t=25, b=5),
        height=height,
        title=dict(
            text=f'{symbol} | {start_date}→{end_date} | {pct:+.1f}%',
            font=dict(size=10, color='#555')
        ),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def fig_comparison(templates: List[Dict], selected_ids: List[str]) -> go.Figure:
    """Seçili şablonları normalize edip üst üste karşılaştır."""
    selected = [t for t in templates if t['id'] in selected_ids]
    if not selected:
        return go.Figure()

    colors = ['#1A56DB','#E3A008','#0E9F6E','#9061F9','#E02424']
    fig = go.Figure()

    for i, t in enumerate(selected):
        prices = np.array(t['prices'])
        mu, sigma = prices.mean(), prices.std()
        z = (prices - mu) / (sigma + 1e-9)
        pct = t['pct_change']
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=list(range(len(z))),
            y=z,
            name=f"{t['symbol']} {t['start_date']} ({pct:+.1f}%)",
            line=dict(color=color, width=2),
            hovertemplate=f"{t['symbol']}: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
        title=dict(
            text='Şablon Karşılaştırması — Normalize',
            font=dict(size=13, color='#1A1A2E')
        ),
        legend=dict(orientation='h', y=1.12, font=dict(size=10)),
        hovermode='x unified',
        xaxis=dict(title='Gün', gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(title='Z-Score', gridcolor='rgba(0,0,0,0.05)'),
    )
    return fig


def fig_score_history(template: Dict) -> Optional[go.Figure]:
    """Şablonun tarama geçmişindeki konsensüs skorlarını göster."""
    history = template.get('scan_history', [])
    if len(history) < 2:
        return None

    dates = [h['date'] for h in history]
    scores = []
    directions = []
    for h in history:
        c = h.get('consensus', {})
        scores.append(c.get('confidence', 0))
        directions.append(c.get('direction', '—'))

    colors = ['#0E9F6E' if d == 'YÜKSELİŞ' else
              '#E02424' if d == 'DÜŞÜŞ' else '#888'
              for d in directions]

    fig = go.Figure(go.Bar(
        x=dates, y=scores,
        marker_color=colors,
        text=[f"{s:.0f}%\n{d}" for s, d in zip(scores, directions)],
        textposition='outside',
        hovertemplate='%{x}<br>Güven: %{y:.0f}%<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=10, r=10, t=40, b=30),
        height=220,
        title=dict(text='Tarama Geçmişi — Güven Skoru', font=dict(size=12)),
        yaxis=dict(range=[0, 115], title='Güven %'),
        showlegend=False,
    )
    return fig


# ── Ana sayfa render ───────────────────────────────────────────────────────────

def render_library(fetch_ticker_fn, find_patterns_fn,
                   calc_consensus_fn, fetch_batch_fn,
                   all_bist_lists: Dict):
    """Şablon Kütüphanesi sayfası."""
    init_library()
    library = get_library()

    st.markdown("## 📚 Şablon Kütüphanesi")
    st.caption("Kaydettiğiniz şablonları yönetin, karşılaştırın ve yeniden tarayın.")
    st.divider()

    if not library:
        st.info(
            "Henüz kaydedilmiş şablon yok. "
            "**Pattern Matcher** sayfasında şablon seçtikten sonra "
            "'💾 Şablonu Kaydet' butonunu kullanın."
        )
        return

    # ── Özet metrikler ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Şablon", len(library))
    m2.metric("Farklı Hisse", len(set(t['symbol'] for t in library)))
    all_tags = [tag for t in library for tag in t['tags']]
    m3.metric("Etiket", len(set(all_tags)))
    total_scans = sum(len(t['scan_history']) for t in library)
    m4.metric("Toplam Tarama", total_scans)

    st.divider()

    # ── Karşılaştırma modu ────────────────────────────────────────────────────
    st.markdown("### 🔍 Şablon Karşılaştırması")
    st.caption("Birden fazla şablon seçerek yapısal benzerliklerini karşılaştırın.")

    compare_ids = st.multiselect(
        "Karşılaştırılacak şablonlar:",
        options=[t['id'] for t in library],
        format_func=lambda tid: next(
            (t['name'] for t in library if t['id'] == tid), tid),
        max_selections=5,
        key="compare_select"
    )

    if len(compare_ids) >= 2:
        st.plotly_chart(fig_comparison(library, compare_ids),
                        use_container_width=True)

        # Benzerlik matrisi
        selected = [t for t in library if t['id'] in compare_ids]
        if len(selected) >= 2:
            st.markdown("##### Benzerlik Matrisi")
            n_sel = len(selected)
            matrix = np.zeros((n_sel, n_sel))
            for i, t1 in enumerate(selected):
                for j, t2 in enumerate(selected):
                    if i == j:
                        matrix[i][j] = 100.0
                    else:
                        p1 = np.array(t1['prices'])
                        p2 = np.array(t2['prices'])
                        min_len = min(len(p1), len(p2))
                        z1 = (p1[-min_len:] - p1[-min_len:].mean()) / (p1[-min_len:].std() + 1e-9)
                        z2 = (p2[-min_len:] - p2[-min_len:].mean()) / (p2[-min_len:].std() + 1e-9)
                        if np.std(z1) > 1e-9 and np.std(z2) > 1e-9:
                            corr = float(np.corrcoef(z1, z2)[0,1])
                        else:
                            corr = 0.0
                        matrix[i][j] = round((corr + 1) / 2 * 100, 1)

            labels = [t['symbol'] + '\n' + t['start_date'] for t in selected]
            df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
            st.dataframe(df_matrix.style.background_gradient(
                cmap='RdYlGn', vmin=0, vmax=100),
                use_container_width=True)

    st.divider()

    # ── Şablon listesi ────────────────────────────────────────────────────────
    st.markdown("### 📋 Kayıtlı Şablonlar")

    # Filtre
    fc1, fc2 = st.columns(2)
    with fc1:
        filter_symbol = st.text_input("Hisse filtrele", placeholder="THYAO...").upper()
    with fc2:
        all_tags_list = list(set(all_tags))
        filter_tag = st.selectbox("Etiket filtrele",
                                   ["Tümü"] + all_tags_list) if all_tags_list else "Tümü"

    filtered = library
    if filter_symbol:
        filtered = [t for t in filtered if filter_symbol in t['symbol']]
    if filter_tag and filter_tag != "Tümü":
        filtered = [t for t in filtered if filter_tag in t['tags']]

    if not filtered:
        st.warning("Filtreyle eşleşen şablon bulunamadı.")
        return

    for t in filtered:
        with st.expander(
            f"**{t['symbol']}** — {t['name']} | "
            f"{t['n_days']} gün | {t['pct_change']:+.1f}% | "
            f"{t['created_at']}",
            expanded=False
        ):
            col_prev, col_info = st.columns([2, 3])

            with col_prev:
                prices_arr = np.array(t['prices'])
                st.plotly_chart(
                    fig_template_preview(prices_arr, t['symbol'],
                                         t['start_date'], t['end_date']),
                    use_container_width=True
                )

            with col_info:
                st.markdown(f"**📅 Tarih:** {t['start_date']} → {t['end_date']}")
                st.markdown(f"**📊 Uzunluk:** {t['n_days']} gün | "
                            f"**Değişim:** {t['pct_change']:+.1f}%")
                if t['regime']:
                    st.markdown(f"**📈 Rejim:** {t['regime']}")
                if t['formations']:
                    st.markdown(f"**🔷 Formasyonlar:** {', '.join(t['formations'])}")
                if t['tags']:
                    tag_html = " ".join(
                        f"<span style='background:#EFF6FF;color:#1A56DB;"
                        f"padding:2px 8px;border-radius:4px;font-size:11px'>{tag}</span>"
                        for tag in t['tags']
                    )
                    st.markdown(f"**🏷️ Etiketler:** {tag_html}",
                                unsafe_allow_html=True)
                if t['notes']:
                    st.markdown(f"**📝 Notlar:** {t['notes']}")

            # Tarama geçmişi
            if t['scan_history']:
                st.markdown(f"**📊 Tarama Geçmişi:** {len(t['scan_history'])} tarama")
                hist_fig = fig_score_history(t)
                if hist_fig:
                    st.plotly_chart(hist_fig, use_container_width=True)

                # Son tarama özeti
                last = t['scan_history'][-1]
                c = last.get('consensus', {})
                if c:
                    direction = c.get('direction', '—')
                    d_color = ('#0E9F6E' if direction == 'YÜKSELİŞ' else
                               '#E02424' if direction == 'DÜŞÜŞ' else '#888')
                    st.markdown(
                        f"Son tarama ({last['date']}): "
                        f"<span style='color:{d_color};font-weight:600'>{direction}</span> | "
                        f"Güven: **%{c.get('confidence', 0):.0f}** | "
                        f"Beklenen: **{c.get('weighted_pct', 0):+.1f}%**",
                        unsafe_allow_html=True
                    )

            # Aksiyonlar
            st.markdown("---")
            ac1, ac2, ac3, ac4 = st.columns(4)

            # Yeniden tara
            with ac1:
                if st.button("🔍 Yeniden Tara", key=f"rescan_{t['id']}",
                             use_container_width=True, type="primary"):
                    st.session_state['library_action'] = {
                        'action': 'rescan',
                        'template': t
                    }
                    st.rerun()

            # Pattern Matcher'a yükle
            with ac2:
                if st.button("📊 Matcher'a Yükle", key=f"load_{t['id']}",
                             use_container_width=True):
                    st.session_state['library_action'] = {
                        'action': 'load',
                        'template': t
                    }
                    st.session_state['_goto_page'] = "🔍 Pattern Matcher"
                    st.rerun()

            # Notu güncelle
            with ac3:
                if st.button("✏️ Notu Düzenle", key=f"edit_{t['id']}",
                             use_container_width=True):
                    st.session_state[f'editing_{t["id"]}'] = True

            # Sil
            with ac4:
                if st.button("🗑️ Sil", key=f"del_{t['id']}",
                             use_container_width=True):
                    delete_template(t['id'])
                    st.rerun()

            # Not düzenleme formu
            if st.session_state.get(f'editing_{t["id"]}'):
                new_notes = st.text_area("Notlar", value=t['notes'],
                                         key=f"notes_{t['id']}")
                new_tags_str = st.text_input(
                    "Etiketler (virgülle ayırın)",
                    value=", ".join(t['tags']),
                    key=f"tags_{t['id']}"
                )
                if st.button("💾 Kaydet", key=f"save_edit_{t['id']}"):
                    for lib_t in st.session_state.template_library:
                        if lib_t['id'] == t['id']:
                            lib_t['notes'] = new_notes
                            lib_t['tags'] = [
                                tag.strip()
                                for tag in new_tags_str.split(',')
                                if tag.strip()
                            ]
                            break
                    st.session_state[f'editing_{t["id"]}'] = False
                    st.rerun()

    # ── Yeniden tarama işlemi ─────────────────────────────────────────────────
    action = st.session_state.get('library_action')
    if action and action['action'] == 'rescan':
        t = action['template']
        st.divider()
        st.markdown(f"### 🔄 Yeniden Tarama — {t['name']}")

        scope = st.selectbox("Kapsam", ["BIST 30", "BIST 100", "Tüm BIST"],
                             key="rescan_scope")
        min_sim = st.slider("Min Benzerlik", 55, 85, 65, key="rescan_sim")

        if st.button("▶️ Taramayı Başlat", type="primary"):
            scope_map = {
                "BIST 30":   all_bist_lists['bist30'],
                "BIST 100":  all_bist_lists['bist100'],
                "Tüm BIST":  all_bist_lists['all']
            }
            tickers = scope_map[scope]

            with st.spinner("Veri yükleniyor..."):
                all_data = fetch_batch_fn(tickers, period="2y")

            with st.spinner(f"{len(all_data)} hisse taranıyor..."):
                prices_arr  = np.array(t['prices'])
                volumes_arr = np.array(t['volumes'])
                matches = find_patterns_fn(
                    prices_arr, volumes_arr, all_data,
                    top_n=5, min_sim=min_sim
                )

            if matches:
                current_price = float(prices_arr[-1])
                consensus = calc_consensus_fn(matches, current_price)
                update_scan_history(t['id'], matches, consensus or {})
                st.session_state['library_action'] = None
                st.success(f"Tarama tamamlandı — {len(matches)} eşleşme bulundu.")
                st.rerun()
            else:
                st.warning("Eşleşme bulunamadı. Eşiği düşürün.")

        if st.button("❌ İptal", key="cancel_rescan"):
            st.session_state['library_action'] = None
            st.rerun()
