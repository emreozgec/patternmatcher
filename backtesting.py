"""
backtesting.py — BIST-PSI Backtesting Motoru

Algoritmanın geçmişteki isabetini ölçer.
İki çıkış stratejisini karşılaştırır:
  A) Sabit süre: sinyal günü gir, N gün sonra çık
  B) Stop-loss / hedef: önce hangisine ulaşırsa çık

Metrikler:
  - Toplam getiri, yıllık getiri
  - Sharpe oranı, Sortino oranı
  - Maksimum drawdown
  - Kazanç oranı (win rate)
  - Ortalama kazanç / kayıp
  - Profit factor
  - Her işlem detayı
  - Equity curve grafiği
  - Sinyal dağılımı grafiği
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# VERİ YAPILARI
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """Tek bir alım sinyali."""
    ticker:       str
    entry_date:   str
    entry_price:  float
    signal_score: float    # BIST-PSI skoru
    confidence:   float    # Konsensüs güven skoru
    expected_pct: float    # Beklenen hareket %
    window:       int      # Şablon uzunluğu (gün)


@dataclass
class Trade:
    """Gerçekleşmiş bir işlem."""
    ticker:        str
    entry_date:    str
    exit_date:     str
    entry_price:   float
    exit_price:    float
    exit_reason:   str     # 'target' / 'stop' / 'time' / 'eod'
    hold_days:     int
    pct_return:    float
    signal_score:  float
    confidence:    float
    expected_pct:  float


@dataclass
class BacktestResult:
    """Backtesting sonucu."""
    strategy_name:   str
    trades:          List[Trade]
    equity_curve:    List[float]
    equity_dates:    List[str]
    total_return:    float
    annual_return:   float
    sharpe:          float
    sortino:         float
    max_drawdown:    float
    win_rate:        float
    avg_win:         float
    avg_loss:        float
    profit_factor:   float
    total_trades:    int
    winning_trades:  int
    losing_trades:   int
    avg_hold_days:   float
    best_trade:      float
    worst_trade:     float
    params:          Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# SİNYAL ÜRETİCİ — Geçmişteki tüm pattern eşleşmelerini bul
# ══════════════════════════════════════════════════════════════════════════════

def generate_historical_signals(
    all_data:       Dict[str, pd.DataFrame],
    find_patterns_fn,
    window:         int   = 20,
    min_psi:        float = 65.0,
    min_confidence: float = 50.0,
    step_days:      int   = 5,
    max_signals:    int   = 200,
) -> List[Signal]:
    """
    Tüm hisseler için geçmişteki pattern sinyallerini üret.

    Her hisse için 2 yıllık veriyi kaydırarak tarar:
    - Her `step_days` günde bir o güne kadarki veriyi şablon olarak al
    - Pattern eşleşmesi varsa sinyal kaydet
    - Gerçekte o günden sonra ne olduğunu bilmiyormuş gibi davran (look-ahead yok)
    """
    signals = []

    for ticker, df in all_data.items():
        closes  = df['Close'].values.astype(float)
        volumes = df['Volume'].values.astype(float)
        dates   = [d.strftime('%Y-%m-%d') for d in df.index]
        n_total = len(closes)

        if n_total < window * 3:
            continue

        # Geçmişte her step_days'te bir sinyal üret
        # Minimum: window + 30 gün gelecek verisi gerekli
        start_idx = window + 10
        end_idx   = n_total - 30

        for i in range(start_idx, end_idx, step_days):
            tpl_prices  = closes[i-window:i]
            tpl_volumes = volumes[i-window:i]

            # Pattern eşleşmesi — sadece i'den önceki veriyle (gerçekçi)
            hist_data = {}
            for other_ticker, other_df in all_data.items():
                if other_ticker == ticker:
                    continue
                oc = other_df['Close'].values.astype(float)
                ov = other_df['Volume'].values.astype(float)
                # Sadece o tarihe kadar olan veriyi kullan (look-ahead yok)
                other_end = min(i, len(oc))
                if other_end >= window * 2 + 20:
                    hist_data[other_ticker] = pd.DataFrame({
                        'Close':  oc[:other_end],
                        'Volume': ov[:other_end],
                    }, index=other_df.index[:other_end])

            try:
                matches = find_patterns_fn(
                    tpl_prices, tpl_volumes, hist_data,
                    top_n=5, min_sim=min_psi, future_mult=1.5
                )
            except Exception:
                continue

            if not matches:
                continue

            # Konsensüs hesapla
            weights = np.array([m['similarity'] for m in matches], dtype=float)
            if weights.sum() < 1e-9:
                continue
            weights /= weights.sum()
            pcts = np.array([m['fut_pct'] for m in matches])

            weighted_pct  = float(np.dot(weights, pcts))
            up_weight     = float(sum(w for w, p in zip(weights, pcts) if p > 0))
            direction_conf = up_weight * 100
            dispersion     = float(np.std(pcts))
            disp_penalty   = min(30, dispersion * 1.2)
            confidence     = max(0, direction_conf - disp_penalty)

            # Sadece bullish sinyaller
            if confidence < min_confidence or weighted_pct <= 0:
                continue

            avg_psi = float(np.dot(weights, [m['similarity'] for m in matches]))

            signals.append(Signal(
                ticker       = ticker,
                entry_date   = dates[i],
                entry_price  = float(closes[i]),
                signal_score = round(avg_psi, 1),
                confidence   = round(confidence, 1),
                expected_pct = round(weighted_pct, 2),
                window       = window,
            ))

            if len(signals) >= max_signals:
                return signals

    # Tarihe göre sırala
    signals.sort(key=lambda s: s.entry_date)
    return signals


# ══════════════════════════════════════════════════════════════════════════════
# STRATEJİ A: SABİT SÜRE
# ══════════════════════════════════════════════════════════════════════════════

def backtest_fixed_hold(
    signals:   List[Signal],
    all_data:  Dict[str, pd.DataFrame],
    hold_days: int   = 20,
    init_cap:  float = 100_000.0,
    pos_size:  float = 0.10,   # Portföyün %10'u her işlemde
) -> BacktestResult:
    """
    Strateji A: Sinyal günü gir, `hold_days` gün sonra çık.
    """
    trades = []
    capital = init_cap
    equity  = [capital]
    dates   = [signals[0].entry_date if signals else '2024-01-01']

    for sig in signals:
        df = all_data.get(sig.ticker)
        if df is None:
            continue

        closes = df['Close'].values.astype(float)
        idx_list = [i for i, d in enumerate(df.index)
                    if d.strftime('%Y-%m-%d') == sig.entry_date]
        if not idx_list:
            continue
        entry_idx = idx_list[0]
        exit_idx  = min(entry_idx + hold_days, len(closes) - 1)

        if exit_idx <= entry_idx:
            continue

        entry_price = float(closes[entry_idx])
        exit_price  = float(closes[exit_idx])
        exit_date   = df.index[exit_idx].strftime('%Y-%m-%d')
        pct_ret     = (exit_price - entry_price) / (entry_price + 1e-9) * 100
        hold        = exit_idx - entry_idx

        trade_cap   = capital * pos_size
        profit      = trade_cap * pct_ret / 100
        capital    += profit

        trades.append(Trade(
            ticker       = sig.ticker,
            entry_date   = sig.entry_date,
            exit_date    = exit_date,
            entry_price  = round(entry_price, 2),
            exit_price   = round(exit_price, 2),
            exit_reason  = 'time',
            hold_days    = hold,
            pct_return   = round(pct_ret, 2),
            signal_score = sig.signal_score,
            confidence   = sig.confidence,
            expected_pct = sig.expected_pct,
        ))
        equity.append(capital)
        dates.append(exit_date)

    return _calc_metrics("Sabit Süre", trades, equity, dates, init_cap,
                         {'hold_days': hold_days, 'pos_size': pos_size})


# ══════════════════════════════════════════════════════════════════════════════
# STRATEJİ B: STOP-LOSS / HEDEF
# ══════════════════════════════════════════════════════════════════════════════

def backtest_sl_tp(
    signals:    List[Signal],
    all_data:   Dict[str, pd.DataFrame],
    stop_pct:   float = 5.0,    # Stop-loss %
    target_pct: float = 10.0,   # Hedef %
    max_days:   int   = 30,     # Maksimum tutma süresi
    init_cap:   float = 100_000.0,
    pos_size:   float = 0.10,
) -> BacktestResult:
    """
    Strateji B: Stop-loss veya hedefe ulaşınca çık, yoksa max_days sonra çık.
    """
    trades = []
    capital = init_cap
    equity  = [capital]
    dates   = [signals[0].entry_date if signals else '2024-01-01']

    for sig in signals:
        df = all_data.get(sig.ticker)
        if df is None:
            continue

        closes = df['Close'].values.astype(float)
        idx_list = [i for i, d in enumerate(df.index)
                    if d.strftime('%Y-%m-%d') == sig.entry_date]
        if not idx_list:
            continue
        entry_idx = idx_list[0]

        entry_price  = float(closes[entry_idx])
        stop_price   = entry_price * (1 - stop_pct / 100)
        target_price = entry_price * (1 + target_pct / 100)

        exit_price  = entry_price
        exit_reason = 'time'
        exit_idx    = min(entry_idx + max_days, len(closes) - 1)

        for j in range(entry_idx + 1, exit_idx + 1):
            price = float(closes[j])
            if price <= stop_price:
                exit_price  = price
                exit_reason = 'stop'
                exit_idx    = j
                break
            if price >= target_price:
                exit_price  = price
                exit_reason = 'target'
                exit_idx    = j
                break
        else:
            exit_price = float(closes[exit_idx])

        exit_date = df.index[exit_idx].strftime('%Y-%m-%d')
        pct_ret   = (exit_price - entry_price) / (entry_price + 1e-9) * 100
        hold      = exit_idx - entry_idx

        trade_cap = capital * pos_size
        profit    = trade_cap * pct_ret / 100
        capital  += profit

        trades.append(Trade(
            ticker       = sig.ticker,
            entry_date   = sig.entry_date,
            exit_date    = exit_date,
            entry_price  = round(entry_price, 2),
            exit_price   = round(exit_price, 2),
            exit_reason  = exit_reason,
            hold_days    = hold,
            pct_return   = round(pct_ret, 2),
            signal_score = sig.signal_score,
            confidence   = sig.confidence,
            expected_pct = sig.expected_pct,
        ))
        equity.append(capital)
        dates.append(exit_date)

    return _calc_metrics("Stop/Hedef", trades, equity, dates, init_cap,
                         {'stop_pct': stop_pct, 'target_pct': target_pct,
                          'max_days': max_days, 'pos_size': pos_size})


# ══════════════════════════════════════════════════════════════════════════════
# METRİK HESAPLAMA
# ══════════════════════════════════════════════════════════════════════════════

def _calc_metrics(name: str, trades: List[Trade], equity: List[float],
                  dates: List[str], init_cap: float,
                  params: Dict) -> BacktestResult:
    if not trades:
        return BacktestResult(
            strategy_name=name, trades=[], equity_curve=[init_cap],
            equity_dates=dates[:1], total_return=0, annual_return=0,
            sharpe=0, sortino=0, max_drawdown=0, win_rate=0,
            avg_win=0, avg_loss=0, profit_factor=0, total_trades=0,
            winning_trades=0, losing_trades=0, avg_hold_days=0,
            best_trade=0, worst_trade=0, params=params
        )

    rets = [t.pct_return for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    total_return = (equity[-1] - init_cap) / init_cap * 100

    # Yıllık getiri (CAGR)
    if len(dates) >= 2:
        try:
            d1 = pd.to_datetime(dates[0])
            d2 = pd.to_datetime(dates[-1])
            years = max((d2 - d1).days / 365.25, 0.1)
            annual_return = ((equity[-1] / init_cap) ** (1 / years) - 1) * 100
        except Exception:
            annual_return = total_return
    else:
        annual_return = total_return

    # Sharpe (günlük getirilerden)
    ret_arr = np.array(rets)
    if ret_arr.std() > 1e-9:
        sharpe = float(ret_arr.mean() / ret_arr.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Sortino (sadece negatif sapma)
    neg_rets = ret_arr[ret_arr < 0]
    if len(neg_rets) > 0 and neg_rets.std() > 1e-9:
        sortino = float(ret_arr.mean() / neg_rets.std() * np.sqrt(252))
    else:
        sortino = sharpe

    # Max drawdown
    eq_arr = np.array(equity)
    peak   = eq_arr[0]
    max_dd = 0.0
    for e in eq_arr:
        if e > peak:
            peak = e
        dd = (peak - e) / (peak + 1e-9) * 100
        if dd > max_dd:
            max_dd = dd

    # Win rate
    win_rate = len(wins) / len(rets) * 100 if rets else 0

    # Avg win/loss
    avg_win  = float(np.mean(wins))   if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    # Profit factor
    gross_profit = sum(w for w in wins)
    gross_loss   = abs(sum(l for l in losses))
    profit_factor = gross_profit / (gross_loss + 1e-9)

    avg_hold = float(np.mean([t.hold_days for t in trades]))

    return BacktestResult(
        strategy_name  = name,
        trades         = trades,
        equity_curve   = equity,
        equity_dates   = dates,
        total_return   = round(total_return, 2),
        annual_return  = round(annual_return, 2),
        sharpe         = round(sharpe, 2),
        sortino        = round(sortino, 2),
        max_drawdown   = round(max_dd, 2),
        win_rate       = round(win_rate, 1),
        avg_win        = round(avg_win, 2),
        avg_loss       = round(avg_loss, 2),
        profit_factor  = round(profit_factor, 2),
        total_trades   = len(trades),
        winning_trades = len(wins),
        losing_trades  = len(losses),
        avg_hold_days  = round(avg_hold, 1),
        best_trade     = round(max(rets), 2) if rets else 0,
        worst_trade    = round(min(rets), 2) if rets else 0,
        params         = params,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİKLER
# ══════════════════════════════════════════════════════════════════════════════

def fig_equity_curve(results: List[BacktestResult], init_cap: float) -> go.Figure:
    """İki stratejinin equity eğrisini karşılaştır."""
    colors = ['#1A56DB', '#0E9F6E', '#E3A008']
    fig = go.Figure()

    for i, r in enumerate(results):
        if not r.equity_curve:
            continue
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x    = r.equity_dates,
            y    = r.equity_curve,
            name = f"{r.strategy_name} ({r.total_return:+.1f}%)",
            line = dict(color=c, width=2),
            hovertemplate = '%{x}: %{y:,.0f} ₺<extra>' + r.strategy_name + '</extra>'
        ))

    # Buy & Hold referans çizgisi (düz)
    if results and results[0].equity_dates:
        fig.add_hline(y=init_cap, line_dash='dot',
                      line_color='rgba(0,0,0,0.2)',
                      annotation_text='Başlangıç Sermayesi',
                      annotation_font_size=10)

    fig.update_layout(
        template     = 'plotly_white',
        paper_bgcolor= 'rgba(0,0,0,0)',
        plot_bgcolor = '#FFFFFF',
        margin       = dict(l=10, r=10, t=40, b=10),
        height       = 360,
        title        = dict(text='Equity Eğrisi Karşılaştırması',
                            font=dict(size=13, color='#1A1A2E')),
        legend       = dict(orientation='h', y=1.12, font=dict(size=11)),
        hovermode    = 'x unified',
        xaxis        = dict(gridcolor='rgba(0,0,0,0.05)', type='date',
                            tickformat='%b %Y'),
        yaxis        = dict(gridcolor='rgba(0,0,0,0.05)', ticksuffix=' ₺'),
    )
    return fig


def fig_trade_distribution(result: BacktestResult) -> go.Figure:
    """İşlem getiri dağılımı histogram."""
    rets = [t.pct_return for t in result.trades]
    if not rets:
        return go.Figure()

    reasons = [t.exit_reason for t in result.trades]
    reason_counts = {
        'target': reasons.count('target'),
        'stop':   reasons.count('stop'),
        'time':   reasons.count('time'),
    }
    reason_labels = {'target': '🎯 Hedef', 'stop': '🛑 Stop', 'time': '⏰ Süre'}
    reason_colors = ['#0E9F6E', '#E02424', '#E3A008']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Getiri Dağılımı', 'Çıkış Nedenleri'],
        specs=[[{"type": "histogram"}, {"type": "pie"}]]
    )

    # Histogram
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=20,
        marker_color='#1A56DB', opacity=0.7,
        name='Getiri %',
        hovertemplate='%{x:.1f}%: %{y} işlem<extra></extra>'
    ), row=1, col=1)
    fig.add_vline(x=0, line_dash='dash', line_color='rgba(0,0,0,0.3)',
                  row=1, col=1)

    # Pie
    fig.add_trace(go.Pie(
        labels=[reason_labels[k] for k in reason_counts if reason_counts[k] > 0],
        values=[v for v in reason_counts.values() if v > 0],
        marker_colors=reason_colors,
        textinfo='percent+label',
        textfont_size=10,
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        template     = 'plotly_white',
        paper_bgcolor= 'rgba(0,0,0,0)',
        plot_bgcolor = '#FFFFFF',
        margin       = dict(l=10, r=10, t=40, b=10),
        height       = 300,
        title        = dict(text=f'{result.strategy_name} — İşlem Analizi',
                            font=dict(size=12, color='#1A1A2E')),
        showlegend   = False,
    )
    return fig


def fig_signal_score_vs_return(result: BacktestResult) -> go.Figure:
    """BIST-PSI skoru ile gerçekleşen getiri arasındaki ilişki."""
    if not result.trades:
        return go.Figure()

    x = [t.signal_score for t in result.trades]
    y = [t.pct_return for t in result.trades]
    colors = ['#0E9F6E' if r > 0 else '#E02424' for r in y]
    labels = [t.ticker for t in result.trades]

    fig = go.Figure(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(color=colors, size=8, opacity=0.7,
                    line=dict(color='white', width=1)),
        text=labels,
        hovertemplate='%{text}<br>PSI: %{x:.0f}<br>Getiri: %{y:.1f}%<extra></extra>'
    ))

    # Trend çizgisi
    if len(x) >= 3:
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(min(x), max(x), 50)
        y_line = np.polyval(z, x_line)
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode='lines',
            line=dict(color='#1A56DB', width=1.5, dash='dash'),
            name='Trend', showlegend=False,
            hovertemplate='Trend: %{y:.1f}%<extra></extra>'
        ))

    fig.add_hline(y=0, line_dash='dot', line_color='rgba(0,0,0,0.2)')

    fig.update_layout(
        template     = 'plotly_white',
        paper_bgcolor= 'rgba(0,0,0,0)',
        plot_bgcolor = '#FFFFFF',
        margin       = dict(l=10, r=10, t=40, b=10),
        height       = 300,
        title        = dict(text='BIST-PSI Skoru vs Gerçekleşen Getiri',
                            font=dict(size=12, color='#1A1A2E')),
        xaxis        = dict(title='BIST-PSI Skoru', gridcolor='rgba(0,0,0,0.05)'),
        yaxis        = dict(title='Getiri %', gridcolor='rgba(0,0,0,0.05)',
                            ticksuffix='%'),
    )
    return fig


def fig_monthly_returns(result: BacktestResult) -> go.Figure:
    """Aylık getiri ısı haritası."""
    if not result.trades:
        return go.Figure()

    monthly = {}
    for t in result.trades:
        try:
            d = pd.to_datetime(t.exit_date)
            key = (d.year, d.month)
            if key not in monthly:
                monthly[key] = []
            monthly[key].append(t.pct_return)
        except Exception:
            pass

    if not monthly:
        return go.Figure()

    years  = sorted(set(k[0] for k in monthly))
    months = list(range(1, 13))
    month_names = ['Oca','Şub','Mar','Nis','May','Haz',
                   'Tem','Ağu','Eyl','Eki','Kas','Ara']

    z = []
    for yr in years:
        row = []
        for mo in months:
            rets = monthly.get((yr, mo), [])
            row.append(round(np.mean(rets), 1) if rets else None)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=month_names, y=[str(y) for y in years],
        colorscale=[[0,'#E02424'],[0.5,'#FFFFFF'],[1,'#0E9F6E']],
        zmid=0,
        text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z],
        texttemplate='%{text}',
        textfont_size=10,
        hovertemplate='%{y} %{x}: %{z:.1f}%<extra></extra>',
        showscale=True,
    ))

    fig.update_layout(
        template     = 'plotly_white',
        paper_bgcolor= 'rgba(0,0,0,0)',
        margin       = dict(l=10, r=10, t=40, b=10),
        height       = max(150, len(years) * 40 + 80),
        title        = dict(text='Aylık Ortalama Getiri Isı Haritası',
                            font=dict(size=12, color='#1A1A2E')),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT SAYFASI
# ══════════════════════════════════════════════════════════════════════════════

def render_backtest(fetch_batch_fn, find_patterns_fn, all_bist_lists: Dict):
    st.markdown("## 📊 Backtesting Motoru")
    st.caption(
        "BIST-PSI algoritmasının geçmişteki isabetini ölç. "
        "İki çıkış stratejisini karşılaştır."
    )
    st.divider()

    # Parametreler
    st.markdown("### ⚙️ Parametreler")
    col1, col2, col3 = st.columns(3)
    with col1:
        scope      = st.selectbox("Hisse Evreni", ["BIST 30","BIST 100","Tüm BIST"], index=0)
        window     = st.selectbox("Şablon Uzunluğu", [10, 20, 30, 40], index=1)
        min_psi    = st.slider("Min BIST-PSI", 55, 80, 62, 1)
    with col2:
        min_conf   = st.slider("Min Güven %", 40, 75, 50, 1)
        hold_days  = st.slider("Sabit Tutma (gün)", 5, 60, 20, 5)
        pos_size   = st.slider("Pozisyon Büyüklüğü %", 5, 25, 10, 5) / 100
    with col3:
        stop_pct   = st.slider("Stop-Loss %", 2, 15, 5, 1)
        target_pct = st.slider("Hedef %", 5, 30, 10, 1)
        max_days   = st.slider("Maks. Tutma (gün)", 10, 60, 30, 5)
        init_cap   = st.number_input("Başlangıç Sermayesi ₺",
                                      min_value=10_000, max_value=10_000_000,
                                      value=100_000, step=10_000)

    st.markdown("""
    <div style='background:#FFF7ED;border:1px solid #FED7AA;border-radius:8px;
                padding:10px 14px;font-size:12px;color:#92400E;margin-bottom:8px'>
        ⚠️ BIST 30 ~5-10 dk, BIST 100 ~20-30 dk sürebilir. İlk denemede BIST 30 öneririz.<br>
        Bu analiz geçmiş performansa dayanır — gelecek getiri garantisi değildir.
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("▶️ Backtesti Başlat", type="primary",
                        use_container_width=False)

    if run_btn:
        scope_map = {"BIST 30": all_bist_lists['bist30'],
                     "BIST 100": all_bist_lists['bist100'],
                     "Tüm BIST": all_bist_lists['all']}
        tickers = scope_map[scope]

        prog = st.progress(0, text="Veriler yükleniyor...")
        with st.spinner(""):
            all_data = fetch_batch_fn(tickers, period="2y")
        prog.progress(20, text=f"{len(all_data)} hisse yüklendi. Sinyaller üretiliyor...")

        signals = generate_historical_signals(
            all_data        = all_data,
            find_patterns_fn= find_patterns_fn,
            window          = window,
            min_psi         = min_psi,
            min_confidence  = min_conf,
            step_days       = 5,
            max_signals     = 150,
        )
        prog.progress(70, text=f"{len(signals)} sinyal üretildi. Backtest hesaplanıyor...")

        if not signals:
            prog.empty()
            st.warning("Sinyal üretilemedi. Parametreleri gevşetin.")
            return

        res_fixed = backtest_fixed_hold(
            signals, all_data, hold_days, init_cap, pos_size
        )
        res_sl_tp = backtest_sl_tp(
            signals, all_data, stop_pct, target_pct, max_days, init_cap, pos_size
        )

        prog.progress(100, text="✅ Tamamlandı!")
        import time; time.sleep(0.3); prog.empty()

        st.session_state['bt_results']  = [res_fixed, res_sl_tp]
        st.session_state['bt_signals']  = signals
        st.session_state['bt_all_data'] = all_data
        st.session_state['bt_init_cap'] = init_cap
        st.rerun()

    # ── Sonuçlar ──────────────────────────────────────────────────────────────
    results  = st.session_state.get('bt_results')
    signals  = st.session_state.get('bt_signals', [])
    init_cap_saved = st.session_state.get('bt_init_cap', 100_000)

    if not results:
        st.info("Parametreleri ayarlayıp 'Backtesti Başlat' butonuna basın.")
        return

    st.divider()
    st.markdown(f"### 📈 Sonuçlar — {len(signals)} sinyal, 2 strateji")

    # Karşılaştırma tablosu
    st.markdown("#### 🔢 Strateji Karşılaştırması")
    comp_data = []
    for r in results:
        comp_data.append({
            'Strateji':          r.strategy_name,
            'Toplam Getiri':     f"{r.total_return:+.1f}%",
            'Yıllık Getiri':     f"{r.annual_return:+.1f}%",
            'Sharpe':            f"{r.sharpe:.2f}",
            'Sortino':           f"{r.sortino:.2f}",
            'Max Drawdown':      f"-{r.max_drawdown:.1f}%",
            'Kazanç Oranı':      f"{r.win_rate:.1f}%",
            'Ort. Kazanç':       f"+{r.avg_win:.1f}%",
            'Ort. Kayıp':        f"{r.avg_loss:.1f}%",
            'Profit Factor':     f"{r.profit_factor:.2f}",
            'İşlem Sayısı':      r.total_trades,
            'Ort. Tutma (gün)':  f"{r.avg_hold_days:.0f}",
            'En İyi İşlem':      f"+{r.best_trade:.1f}%",
            'En Kötü İşlem':     f"{r.worst_trade:.1f}%",
        })

    st.dataframe(pd.DataFrame(comp_data).set_index('Strateji'),
                 use_container_width=True)

    # Equity eğrisi
    st.markdown("#### 📉 Equity Eğrisi")
    st.plotly_chart(fig_equity_curve(results, init_cap_saved),
                    use_container_width=True)

    # Sekme detayları
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 İşlem Dağılımı",
        "🎯 PSI vs Getiri",
        "📅 Aylık Performans",
        "📋 İşlem Listesi"
    ])

    with tab1:
        for r in results:
            st.plotly_chart(fig_trade_distribution(r), use_container_width=True)

    with tab2:
        for r in results:
            if r.trades:
                st.plotly_chart(fig_signal_score_vs_return(r),
                                use_container_width=True)

    with tab3:
        for r in results:
            if r.trades:
                st.plotly_chart(fig_monthly_returns(r), use_container_width=True)

    with tab4:
        for r in results:
            if not r.trades:
                continue
            st.markdown(f"##### {r.strategy_name}")
            trade_rows = []
            for t in r.trades:
                color = '🟢' if t.pct_return > 0 else '🔴'
                trade_rows.append({
                    '': color,
                    'Hisse':     t.ticker,
                    'Giriş':     t.entry_date,
                    'Çıkış':     t.exit_date,
                    'Giriş ₺':   f"{t.entry_price:.2f}",
                    'Çıkış ₺':   f"{t.exit_price:.2f}",
                    'Getiri':    f"{t.pct_return:+.1f}%",
                    'Neden':     t.exit_reason,
                    'Gün':       t.hold_days,
                    'PSI':       f"{t.signal_score:.0f}",
                    'Güven':     f"{t.confidence:.0f}%",
                    'Beklenen':  f"+{t.expected_pct:.1f}%",
                })
            st.dataframe(pd.DataFrame(trade_rows),
                         use_container_width=True, hide_index=True)

    # Algoritma kalibrasyon özeti
    st.divider()
    st.markdown("#### 🔬 Algoritma Kalibrasyon Özeti")
    best = max(results, key=lambda r: r.sharpe)

    all_trades = results[0].trades + results[1].trades
    if not all_trades:
        return

    psi_scores  = np.array([t.signal_score for t in all_trades])
    conf_scores = np.array([t.confidence   for t in all_trades])
    returns     = np.array([t.pct_return   for t in all_trades])

    corr      = float(np.corrcoef(psi_scores,  returns)[0,1]) if np.std(psi_scores)  > 0 else 0.0
    conf_corr = float(np.corrcoef(conf_scores, returns)[0,1]) if np.std(conf_scores) > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("PSI → Getiri Korelasyonu",  f"{corr:.3f}",
              help=">0.15 anlamlı, <0 ters etki")
    k2.metric("Güven → Getiri Korelasyonu", f"{conf_corr:.3f}",
              help=">0.15 anlamlı, <0 anti-consensus etkisi")
    k3.metric("En İyi Strateji", best.strategy_name,
              delta=f"Sharpe: {best.sharpe:.2f}")
    k4.metric("Toplam Sinyal", len(signals))

    # Akıllı yorum
    if len(signals) < 50:
        st.warning(
            f"⚠️ **İstatistiksel güç yetersiz** ({len(signals)} sinyal). "
            "Korelasyon sonuçları güvenilir değil. "
            "BIST 100 ile min PSI 55 ayarıyla tekrar deneyin."
        )
    elif corr > 0.15 and conf_corr > 0.15:
        st.success(
            f"✅ **Algoritma iyi kalibre**: PSI ({corr:.2f}) ve güven ({conf_corr:.2f}) "
            "skoru yüksek olanlar daha iyi getiri sağlıyor."
        )
    elif corr < -0.10 or conf_corr < -0.10:
        st.info(
            f"💡 **Anti-consensus etkisi tespit edildi** "
            f"(PSI: {corr:.2f}, Güven: {conf_corr:.2f}). "
            "Bu bilinen bir piyasa davranışı: çok belirgin patternler fiyata zaten yansımış olabilir. "
            "Sharpe oranı yüksekse sistem çalışıyor demektir — "
            "sadece güven skoru filtresini **tersine** kullanmayı deneyin."
        )
    else:
        st.warning(
            f"⚠️ **Zayıf korelasyon** (PSI: {corr:.2f}, Güven: {conf_corr:.2f}). "
            "PSI eşiğini değiştirerek tekrar test edin."
        )

    st.divider()
    st.markdown("#### 🎯 Optimal Parametre Analizi")

    # PSI bandına göre ortalama getiri
    st.markdown("**PSI Skoruna Göre Ortalama Getiri**")
    bins_psi = [(55,65),(65,72),(72,80),(80,100)]
    psi_band_data = []
    for lo, hi in bins_psi:
        mask = (psi_scores >= lo) & (psi_scores < hi)
        if mask.sum() >= 3:
            avg_ret  = float(returns[mask].mean())
            win_rate = float((returns[mask] > 0).mean() * 100)
            psi_band_data.append({
                'PSI Bandı':   f"{lo}-{hi}",
                'Sinyal Sayısı': int(mask.sum()),
                'Ort. Getiri': f"{avg_ret:+.1f}%",
                'Kazanç Oranı': f"{win_rate:.0f}%",
                'Öneri': '✅ Kullan' if avg_ret > 0 and win_rate > 50 else '⚠️ Dikkat'
            })
    if psi_band_data:
        st.dataframe(pd.DataFrame(psi_band_data),
                     use_container_width=True, hide_index=True)

    # Güven bandına göre ortalama getiri
    st.markdown("**Güven Skoruna Göre Ortalama Getiri**")
    bins_conf = [(40,55),(55,65),(65,75),(75,100)]
    conf_band_data = []
    for lo, hi in bins_conf:
        mask = (conf_scores >= lo) & (conf_scores < hi)
        if mask.sum() >= 3:
            avg_ret  = float(returns[mask].mean())
            win_rate = float((returns[mask] > 0).mean() * 100)
            conf_band_data.append({
                'Güven Bandı':   f"{lo}-{hi}",
                'Sinyal Sayısı': int(mask.sum()),
                'Ort. Getiri': f"{avg_ret:+.1f}%",
                'Kazanç Oranı': f"{win_rate:.0f}%",
                'Öneri': '✅ Kullan' if avg_ret > 0 and win_rate > 50 else '⚠️ Dikkat'
            })
    if conf_band_data:
        st.dataframe(pd.DataFrame(conf_band_data),
                     use_container_width=True, hide_index=True)

    # Optimal PSI ve güven önerisi
    best_psi_band  = None
    best_psi_ret   = -999
    best_conf_band = None
    best_conf_ret  = -999

    for lo, hi in bins_psi:
        mask = (psi_scores >= lo) & (psi_scores < hi)
        if mask.sum() >= 5:
            avg = float(returns[mask].mean())
            if avg > best_psi_ret:
                best_psi_ret, best_psi_band = avg, f"{lo}-{hi}"

    for lo, hi in bins_conf:
        mask = (conf_scores >= lo) & (conf_scores < hi)
        if mask.sum() >= 5:
            avg = float(returns[mask].mean())
            if avg > best_conf_ret:
                best_conf_ret, best_conf_band = avg, f"{lo}-{hi}"

    if best_psi_band and best_conf_band:
        st.success(
            f"🎯 **Optimal Parametre Önerisi:** "
            f"PSI bandı **{best_psi_band}** (ort. getiri {best_psi_ret:+.1f}%) — "
            f"Güven bandı **{best_conf_band}** (ort. getiri {best_conf_ret:+.1f}%). "
            f"Bu bandları Pattern Matcher ve Fırsat Tarayıcı'da kullanın."
        )
