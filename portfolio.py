"""
portfolio.py — Portföy Simülasyonu (Aşama 3)

"Bu sinyale göre gerçekten girseydim, bugün param ne olurdu?"

Backtesting'in tekil/canlı versiyonu:
- Kullanıcı bir tarama sonucunu (Pattern Matcher / Fırsat Tarayıcı / Kütüphane)
  "Portföye Ekle" diyerek sanal pozisyona dönüştürür
- Giriş fiyatı, tarih, miktar (TL bazlı) kaydedilir
- Güncel fiyat her açılışta çekilir, kâr/zarar canlı hesaplanır
- Stop-loss / hedef / süre dolumu kuralları opsiyonel olarak izlenir
- Kapanan pozisyonlar geçmişe taşınır, performans özeti çıkar
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
# VERİ YÖNETİMİ — Session state tabanlı sanal portföy
# ══════════════════════════════════════════════════════════════════════════════

def init_portfolio():
    if 'portfolio_positions' not in st.session_state:
        st.session_state.portfolio_positions = []   # açık + kapalı pozisyonlar
    if 'portfolio_cash_log' not in st.session_state:
        st.session_state.portfolio_cash_log = []     # sermaye geçmişi


def add_position(ticker: str,
                 entry_price: float,
                 entry_date: str,
                 amount_try: float,
                 source: str = "Manuel",
                 signal_score: float = None,
                 confidence: float = None,
                 expected_pct: float = None,
                 stop_pct: float = None,
                 target_pct: float = None,
                 notes: str = "") -> str:
    """Portföye yeni sanal pozisyon ekle."""
    init_portfolio()

    pos_id = f"{ticker}_{entry_date}_{datetime.now().strftime('%H%M%S')}"
    shares = amount_try / entry_price if entry_price > 0 else 0

    position = {
        'id':            pos_id,
        'ticker':        ticker,
        'entry_price':   round(entry_price, 2),
        'entry_date':    entry_date,
        'amount_try':    round(amount_try, 2),
        'shares':        round(shares, 4),
        'source':        source,
        'signal_score':  signal_score,
        'confidence':    confidence,
        'expected_pct':  expected_pct,
        'stop_pct':      stop_pct,
        'target_pct':    target_pct,
        'stop_price':    round(entry_price * (1 - stop_pct/100), 2) if stop_pct else None,
        'target_price':  round(entry_price * (1 + target_pct/100), 2) if target_pct else None,
        'notes':         notes,
        'status':        'open',     # open / closed
        'exit_price':    None,
        'exit_date':     None,
        'exit_reason':   None,
        'created_at':    datetime.now().strftime('%d.%m.%Y %H:%M'),
    }
    st.session_state.portfolio_positions.append(position)
    return pos_id


def close_position(pos_id: str, exit_price: float, exit_reason: str = "Manuel Kapatma"):
    """Bir pozisyonu kapat."""
    init_portfolio()
    for p in st.session_state.portfolio_positions:
        if p['id'] == pos_id:
            p['status']      = 'closed'
            p['exit_price']  = round(exit_price, 2)
            p['exit_date']   = date.today().strftime('%Y-%m-%d')
            p['exit_reason'] = exit_reason
            break


def delete_position(pos_id: str):
    init_portfolio()
    st.session_state.portfolio_positions = [
        p for p in st.session_state.portfolio_positions if p['id'] != pos_id
    ]


def get_positions(status: Optional[str] = None) -> List[Dict]:
    init_portfolio()
    if status is None:
        return st.session_state.portfolio_positions
    return [p for p in st.session_state.portfolio_positions if p['status'] == status]


# ══════════════════════════════════════════════════════════════════════════════
# CANLI FİYAT GÜNCELLEME
# ══════════════════════════════════════════════════════════════════════════════

def update_live_prices(positions: List[Dict], fetch_ticker_fn) -> List[Dict]:
    """
    Açık pozisyonların güncel fiyatlarını çek ve kâr/zarar hesapla.
    Stop-loss / hedef tetiklenmişse otomatik kapatma önerisi işaretle.
    """
    enriched = []
    price_cache = {}

    for p in positions:
        p = dict(p)  # kopya
        if p['status'] == 'open':
            ticker = p['ticker']
            if ticker not in price_cache:
                try:
                    df = fetch_ticker_fn(ticker, period="5d")
                    price_cache[ticker] = float(df['Close'].iloc[-1]) if df is not None and len(df) > 0 else None
                except Exception:
                    price_cache[ticker] = None

            current_price = price_cache[ticker]
            p['current_price'] = current_price

            if current_price is not None:
                pct_change = (current_price - p['entry_price']) / (p['entry_price'] + 1e-9) * 100
                pnl_try    = p['amount_try'] * pct_change / 100
                p['pct_change'] = round(pct_change, 2)
                p['pnl_try']    = round(pnl_try, 2)
                p['current_value'] = round(p['amount_try'] + pnl_try, 2)

                # Stop/Hedef tetiklendi mi?
                trigger = None
                if p.get('stop_price') and current_price <= p['stop_price']:
                    trigger = 'stop'
                elif p.get('target_price') and current_price >= p['target_price']:
                    trigger = 'target'
                p['trigger'] = trigger
            else:
                p['pct_change'] = None
                p['pnl_try'] = None
                p['current_value'] = None
                p['trigger'] = None
        else:
            # Kapalı pozisyon — sabit sonuç
            if p.get('exit_price') and p.get('entry_price'):
                pct_change = (p['exit_price'] - p['entry_price']) / (p['entry_price'] + 1e-9) * 100
                p['pct_change'] = round(pct_change, 2)
                p['pnl_try'] = round(p['amount_try'] * pct_change / 100, 2)
                p['current_value'] = round(p['amount_try'] + p['pnl_try'], 2)
            p['current_price'] = p.get('exit_price')
            p['trigger'] = None

        enriched.append(p)

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANS METRİKLERİ
# ══════════════════════════════════════════════════════════════════════════════

def calc_portfolio_metrics(positions: List[Dict]) -> Dict:
    """Tüm portföy için özet metrikler."""
    if not positions:
        return {
            'total_invested': 0, 'total_current_value': 0, 'total_pnl': 0,
            'total_pnl_pct': 0, 'open_count': 0, 'closed_count': 0,
            'win_count': 0, 'loss_count': 0, 'win_rate': 0,
            'best_position': None, 'worst_position': None,
        }

    total_invested = sum(p['amount_try'] for p in positions)
    total_current  = sum(p.get('current_value', p['amount_try']) or p['amount_try']
                         for p in positions)
    total_pnl      = total_current - total_invested
    total_pnl_pct  = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    open_positions   = [p for p in positions if p['status'] == 'open']
    closed_positions = [p for p in positions if p['status'] == 'closed']

    closed_with_pnl = [p for p in closed_positions if p.get('pct_change') is not None]
    win_count  = sum(1 for p in closed_with_pnl if p['pct_change'] > 0)
    loss_count = sum(1 for p in closed_with_pnl if p['pct_change'] <= 0)
    win_rate   = (win_count / len(closed_with_pnl) * 100) if closed_with_pnl else 0

    all_with_pnl = [p for p in positions if p.get('pct_change') is not None]
    best  = max(all_with_pnl, key=lambda p: p['pct_change']) if all_with_pnl else None
    worst = min(all_with_pnl, key=lambda p: p['pct_change']) if all_with_pnl else None

    return {
        'total_invested':      round(total_invested, 2),
        'total_current_value': round(total_current, 2),
        'total_pnl':           round(total_pnl, 2),
        'total_pnl_pct':       round(total_pnl_pct, 2),
        'open_count':          len(open_positions),
        'closed_count':        len(closed_positions),
        'win_count':           win_count,
        'loss_count':          loss_count,
        'win_rate':            round(win_rate, 1),
        'best_position':       best,
        'worst_position':      worst,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİKLER
# ══════════════════════════════════════════════════════════════════════════════

def fig_portfolio_pie(positions: List[Dict]) -> go.Figure:
    """Portföy dağılım pasta grafiği — hisse bazlı yatırım tutarı."""
    open_pos = [p for p in positions if p['status'] == 'open']
    if not open_pos:
        return go.Figure()

    tickers = [p['ticker'] for p in open_pos]
    amounts = [p['amount_try'] for p in open_pos]

    fig = go.Figure(go.Pie(
        labels=tickers, values=amounts,
        textinfo='label+percent',
        textfont_size=10,
        marker=dict(colors=['#1A56DB','#E3A008','#0E9F6E','#9061F9','#E02424',
                            '#E8734A','#4ECDC4','#FF6B6B','#A8E6CF','#74B9FF']),
    ))
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        title=dict(text='Portföy Dağılımı (Açık Pozisyonlar)', font=dict(size=12, color='#1A1A2E')),
    )
    return fig


def fig_pnl_bar(positions: List[Dict]) -> go.Figure:
    """Pozisyon bazlı kâr/zarar bar grafiği."""
    pos_with_pnl = [p for p in positions if p.get('pct_change') is not None]
    if not pos_with_pnl:
        return go.Figure()

    pos_with_pnl.sort(key=lambda p: p['pct_change'])
    labels = [f"{p['ticker']} ({p['status'][:1].upper()})" for p in pos_with_pnl]
    values = [p['pct_change'] for p in pos_with_pnl]
    colors = ['#0E9F6E' if v > 0 else '#E02424' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in values],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))
    fig.add_vline(x=0, line_dash='dash', line_color='rgba(0,0,0,0.3)')
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(220, len(pos_with_pnl) * 32 + 80),
        title=dict(text='Pozisyon Bazlı Kâr/Zarar %', font=dict(size=12, color='#1A1A2E')),
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)', ticksuffix='%'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
        showlegend=False,
    )
    return fig


def fig_equity_timeline(positions: List[Dict], initial_value: float = 0) -> go.Figure:
    """Zaman içinde toplam portföy değerinin gelişimi (basitleştirilmiş)."""
    events = []
    for p in positions:
        events.append((p['entry_date'], p['amount_try'], 'in'))
        if p['status'] == 'closed' and p.get('exit_date'):
            pnl = p.get('pnl_try', 0) or 0
            events.append((p['exit_date'], p['amount_try'] + pnl, 'out'))

    if not events:
        return go.Figure()

    events.sort(key=lambda e: e[0])
    dates = []
    cum_invested = initial_value
    cum_value = initial_value
    timeline = []

    for edate, amount, etype in events:
        if etype == 'in':
            cum_invested += amount
            cum_value += amount
        timeline.append((edate, cum_invested))

    if not timeline:
        return go.Figure()

    x = [t[0] for t in timeline]
    y = [t[1] for t in timeline]

    fig = go.Figure(go.Scatter(
        x=x, y=y, mode='lines+markers',
        line=dict(color='#1A56DB', width=2),
        marker=dict(size=6),
        hovertemplate='%{x}: %{y:,.0f} ₺<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        title=dict(text='Yatırım Zaman Çizelgesi', font=dict(size=12, color='#1A1A2E')),
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)', type='date'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.05)', ticksuffix=' ₺'),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DIŞARIDAN POZİSYON EKLEME ARAYÜZÜ (diğer sayfalardan çağrılır)
# ══════════════════════════════════════════════════════════════════════════════

def render_add_to_portfolio_button(ticker: str,
                                   current_price: float,
                                   source: str,
                                   signal_score: float = None,
                                   confidence: float = None,
                                   expected_pct: float = None,
                                   key_suffix: str = ""):
    """
    Diğer sayfalarda (Pattern Matcher, Fırsat Tarayıcı, Kütüphane) kullanılacak
    küçük 'Portföye Ekle' butonu + miktar/stop/hedef formu.
    """
    init_portfolio()
    form_key = f"add_portfolio_{ticker}_{key_suffix}"

    if st.button(f"💼 Portföye Ekle", key=f"btn_{form_key}", use_container_width=True):
        st.session_state[f'show_{form_key}'] = True

    if st.session_state.get(f'show_{form_key}'):
        with st.form(key=f"form_{form_key}"):
            st.markdown(f"**{ticker}** — Sanal pozisyon ekle")
            c1, c2 = st.columns(2)
            amount = c1.number_input("Yatırım Tutarı (₺)", min_value=100.0,
                                     value=10000.0, step=500.0)
            entry_price_input = c2.number_input("Giriş Fiyatı (₺)",
                                                min_value=0.01,
                                                value=float(current_price),
                                                step=0.01)
            c3, c4 = st.columns(2)
            use_stop = c3.checkbox("Stop-Loss kullan", value=True)
            stop_pct = c3.number_input("Stop %", min_value=1.0, max_value=30.0,
                                       value=5.0, step=0.5) if use_stop else None
            use_target = c4.checkbox("Hedef kullan", value=True)
            target_pct = c4.number_input("Hedef %", min_value=1.0, max_value=50.0,
                                         value=10.0, step=0.5) if use_target else None
            notes = st.text_input("Not (isteğe bağlı)", value=f"{source} sinyali")

            submitted = st.form_submit_button("✅ Ekle", type="primary")
            if submitted:
                add_position(
                    ticker=ticker,
                    entry_price=entry_price_input,
                    entry_date=date.today().strftime('%Y-%m-%d'),
                    amount_try=amount,
                    source=source,
                    signal_score=signal_score,
                    confidence=confidence,
                    expected_pct=expected_pct,
                    stop_pct=stop_pct,
                    target_pct=target_pct,
                    notes=notes,
                )
                st.session_state[f'show_{form_key}'] = False
                st.success(f"✅ {ticker} portföye eklendi!")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ANA SAYFA
# ══════════════════════════════════════════════════════════════════════════════

def render_portfolio(fetch_ticker_fn):
    init_portfolio()

    st.markdown("## 💼 Portföy Simülasyonu")
    st.caption(
        "\"Bu sinyale göre gerçekten girseydim, bugün param ne olurdu?\" "
        "Sanal pozisyonlarınızı takip edin, canlı kâr/zarar görün."
    )
    st.divider()

    all_positions = get_positions()

    if not all_positions:
        st.info(
            "Henüz pozisyon eklenmedi. **Pattern Matcher**, **Fırsat Tarayıcı** veya "
            "**Şablon Kütüphanesi** sayfalarında bulduğunuz sinyalleri "
            "'💼 Portföye Ekle' butonuyla buraya taşıyabilirsiniz. "
            "Aşağıdan manuel pozisyon da ekleyebilirsiniz."
        )

    # ── Manuel pozisyon ekleme ──
    with st.expander("➕ Manuel Pozisyon Ekle"):
        with st.form("manual_add_position"):
            c1, c2, c3 = st.columns(3)
            m_ticker = c1.text_input("Hisse Kodu", placeholder="THYAO").upper()
            m_amount = c2.number_input("Tutar (₺)", min_value=100.0, value=10000.0, step=500.0)
            m_entry  = c3.number_input("Giriş Fiyatı (₺)", min_value=0.01, value=100.0, step=0.01)
            c4, c5, c6 = st.columns(3)
            m_date   = c4.date_input("Giriş Tarihi", value=date.today())
            m_stop   = c5.number_input("Stop %", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
            m_target = c6.number_input("Hedef %", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
            m_notes  = st.text_input("Not", value="Manuel giriş")

            if st.form_submit_button("✅ Ekle", type="primary"):
                if m_ticker:
                    add_position(
                        ticker=m_ticker, entry_price=m_entry,
                        entry_date=m_date.strftime('%Y-%m-%d'),
                        amount_try=m_amount, source="Manuel",
                        stop_pct=m_stop if m_stop > 0 else None,
                        target_pct=m_target if m_target > 0 else None,
                        notes=m_notes,
                    )
                    st.success(f"✅ {m_ticker} eklendi!")
                    st.rerun()
                else:
                    st.warning("Hisse kodu girin.")

    if not all_positions:
        return

    # ── Canlı fiyat güncelle ──
    refresh = st.button("🔄 Fiyatları Güncelle", type="primary")
    if refresh or 'portfolio_enriched' not in st.session_state:
        with st.spinner("Güncel fiyatlar çekiliyor..."):
            enriched = update_live_prices(all_positions, fetch_ticker_fn)
        st.session_state['portfolio_enriched'] = enriched
    else:
        enriched = st.session_state['portfolio_enriched']

    st.divider()

    # ── Özet metrikler ──
    metrics = calc_portfolio_metrics(enriched)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Toplam Yatırım", f"{metrics['total_invested']:,.0f} ₺")
    m2.metric("Güncel Değer", f"{metrics['total_current_value']:,.0f} ₺",
              delta=f"{metrics['total_pnl']:+,.0f} ₺")
    m3.metric("Toplam Getiri", f"{metrics['total_pnl_pct']:+.1f}%")
    m4.metric("Açık / Kapalı", f"{metrics['open_count']} / {metrics['closed_count']}")
    m5.metric("Kazanç Oranı", f"%{metrics['win_rate']:.0f}" if metrics['closed_count'] > 0 else "—")

    # Tetiklenen stop/hedef uyarıları
    triggered = [p for p in enriched if p.get('trigger')]
    if triggered:
        for p in triggered:
            icon = "🎯" if p['trigger'] == 'target' else "🛑"
            label = "Hedefe ulaştı" if p['trigger'] == 'target' else "Stop-Loss tetiklendi"
            st.warning(
                f"{icon} **{p['ticker']}** — {label}! "
                f"Güncel: {p['current_price']:.2f} ₺ "
                f"({p['pct_change']:+.1f}%). Pozisyonu kapatmayı düşünün."
            )

    st.divider()

    # ── Grafikler ──
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.plotly_chart(fig_portfolio_pie(enriched), use_container_width=True)
    with gcol2:
        st.plotly_chart(fig_pnl_bar(enriched), use_container_width=True)

    if metrics['best_position'] or metrics['worst_position']:
        bc1, bc2 = st.columns(2)
        if metrics['best_position']:
            b = metrics['best_position']
            bc1.success(f"🏆 En İyi: **{b['ticker']}** ({b['pct_change']:+.1f}%)")
        if metrics['worst_position']:
            w = metrics['worst_position']
            bc2.error(f"📉 En Kötü: **{w['ticker']}** ({w['pct_change']:+.1f}%)")

    st.divider()

    # ── Pozisyon listesi ──
    tab_open, tab_closed = st.tabs([
        f"📂 Açık Pozisyonlar ({metrics['open_count']})",
        f"📁 Kapalı Pozisyonlar ({metrics['closed_count']})"
    ])

    with tab_open:
        open_pos = [p for p in enriched if p['status'] == 'open']
        if not open_pos:
            st.info("Açık pozisyon yok.")
        for p in open_pos:
            _render_position_card(p, fetch_ticker_fn)

    with tab_closed:
        closed_pos = [p for p in enriched if p['status'] == 'closed']
        if not closed_pos:
            st.info("Kapalı pozisyon yok.")
        for p in closed_pos:
            _render_position_card(p, fetch_ticker_fn)


def _render_position_card(p: Dict, fetch_ticker_fn):
    """Tek bir pozisyon için detay kart."""
    pct = p.get('pct_change')
    color = '#0E9F6E' if (pct or 0) >= 0 else '#E02424'
    icon = '📈' if (pct or 0) >= 0 else '📉'

    with st.container():
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            st.markdown(f"**{p['ticker']}** — {p['source']}")
            st.caption(f"Giriş: {p['entry_date']} @ {p['entry_price']:.2f} ₺")
            if p.get('notes'):
                st.caption(f"📝 {p['notes']}")
        with c2:
            if p['status'] == 'open':
                cp = p.get('current_price')
                st.markdown(f"Güncel: **{cp:.2f} ₺**" if cp else "Güncel: —")
            else:
                st.markdown(f"Çıkış: **{p.get('exit_price', 0):.2f} ₺** ({p.get('exit_reason','')})")
            st.markdown(f"Tutar: {p['amount_try']:,.0f} ₺")
        with c3:
            if pct is not None:
                st.markdown(f"<span style='color:{color};font-size:18px;font-weight:700'>"
                           f"{icon} {pct:+.1f}%</span>", unsafe_allow_html=True)
                st.caption(f"{p.get('pnl_try', 0):+,.0f} ₺")
            else:
                st.markdown("—")
            if p.get('stop_price') or p.get('target_price'):
                st.caption(
                    f"🛑{p['stop_price']:.2f} / 🎯{p['target_price']:.2f}"
                    if p.get('stop_price') and p.get('target_price') else ""
                )
        with c4:
            if p['status'] == 'open':
                if st.button("🔒 Kapat", key=f"close_{p['id']}", use_container_width=True):
                    st.session_state[f'closing_{p["id"]}'] = True
            if st.button("🗑️", key=f"del_{p['id']}", use_container_width=True):
                delete_position(p['id'])
                st.session_state.pop('portfolio_enriched', None)
                st.rerun()

        if st.session_state.get(f'closing_{p["id"]}'):
            exit_p = st.number_input(
                "Çıkış Fiyatı", min_value=0.01,
                value=float(p.get('current_price') or p['entry_price']),
                key=f"exitprice_{p['id']}"
            )
            cc1, cc2 = st.columns(2)
            if cc1.button("✅ Onayla", key=f"confirm_close_{p['id']}", type="primary"):
                close_position(p['id'], exit_p, "Manuel Kapatma")
                st.session_state[f'closing_{p["id"]}'] = False
                st.session_state.pop('portfolio_enriched', None)
                st.rerun()
            if cc2.button("❌ Vazgeç", key=f"cancel_close_{p['id']}"):
                st.session_state[f'closing_{p["id"]}'] = False
                st.rerun()

        st.divider()
