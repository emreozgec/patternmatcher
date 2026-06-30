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

def _zscore_fast(arr: np.ndarray) -> np.ndarray:
    mu, sigma = arr.mean(), arr.std()
    return (arr - mu) / (sigma + 1e-9)


def _build_windows_matrix(closes: np.ndarray, window: int,
                           step: int, max_end: int) -> np.ndarray:
    """
    Bir hissenin tüm sliding window'larını matris olarak döndür.
    Her satır = bir window (z-score normalize edilmiş).
    Sonuç shape: (n_windows, window)
    """
    indices = range(0, max_end - window, step)
    if not indices:
        return np.empty((0, window))
    rows = []
    for i in indices:
        seg = closes[i:i+window]
        mu, sigma = seg.mean(), seg.std()
        rows.append((seg - mu) / (sigma + 1e-9))
    return np.array(rows, dtype=float)


def _batch_pearson(tpl_z: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Şablon ile matristeki tüm window'ların Pearson korelasyonunu
    tek numpy işlemiyle hesapla — döngü yok.
    tpl_z:  (window,)
    matrix: (n_windows, window)
    Döndürür: (n_windows,) → 0-100 skor
    """
    if matrix.shape[0] == 0:
        return np.array([])
    # Her satırı normalize et
    m = matrix - matrix.mean(axis=1, keepdims=True)
    t = tpl_z - tpl_z.mean()
    t_norm  = np.sqrt((t**2).sum()) + 1e-9
    m_norms = np.sqrt((m**2).sum(axis=1)) + 1e-9
    corrs = (m @ t) / (m_norms * t_norm)
    return (corrs + 1) / 2 * 100  # 0-100


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
    Vektörize sinyal üretimi — numpy matris işlemleriyle ~10x hızlı.
    BIST 100 için ~1 dakika.
    """
    signals = []
    fut_win = min(int(window * 1.5), 45)

    # Tüm hisseleri numpy'a çevir
    ticker_data = {}
    for ticker, df in all_data.items():
        closes = df['Close'].values.astype(float)
        dates  = [d.strftime('%Y-%m-%d') for d in df.index]
        if len(closes) >= window * 3 + fut_win:
            ticker_data[ticker] = (closes, dates)

    tickers = list(ticker_data.keys())
    n_tickers = len(tickers)

    # Her tarih noktasında sinyal ara
    # Referans hisseyi döngüde, diğerleri için matris işlemi
    for t_idx, (ticker, (closes, dates)) in enumerate(ticker_data.items()):
        n_total   = len(closes)
        start_idx = window + 10
        end_idx   = n_total - fut_win - 5

        # Bu hisse için sinyal noktaları
        for i in range(start_idx, end_idx, step_days):
            tpl_z = _zscore_fast(closes[i-window:i])

            matches = []

            # Diğer hisselerde matris ile toplu ara
            for other_ticker, (oc, _) in ticker_data.items():
                if other_ticker == ticker:
                    continue
                # Bu tarihe kadar olan veri (look-ahead yok)
                avail = min(i, len(oc) - fut_win - 1)
                if avail < window * 2:
                    continue

                step_w = max(1, window // 3)
                matrix = _build_windows_matrix(oc, window, step_w, avail)
                if matrix.shape[0] == 0:
                    continue

                sims = _batch_pearson(tpl_z, matrix)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])

                if best_sim < min_psi * 0.82:
                    continue

                # Gelecek hareketi
                win_start  = best_idx * step_w
                fut_start  = win_start + window
                fut_end    = min(fut_start + fut_win, len(oc))
                if fut_end - fut_start < 3:
                    continue

                future  = oc[fut_start:fut_end]
                fut_pct = (future[-1] - future[0]) / (future[0] + 1e-9) * 100

                matches.append({
                    'similarity': round(best_sim, 1),
                    'fut_pct':    round(fut_pct, 2),
                })

            if len(matches) < 2:
                continue

            # Konsensüs
            sims_arr = np.array([m['similarity'] for m in matches])
            weights  = sims_arr / (sims_arr.sum() + 1e-9)
            pcts     = np.array([m['fut_pct'] for m in matches])

            weighted_pct = float(np.dot(weights, pcts))
            up_weight    = float(sum(w for w,p in zip(weights,pcts) if p > 0))
            confidence   = max(0.0, up_weight*100 - min(30, float(np.std(pcts))*1.2))

            if confidence < min_confidence or weighted_pct <= 0:
                continue

            # Sinyalin kendi PSI skoru da eşiği karşılamalı (önceki sürümde
            # sadece bireysel aday eşleşmeler gevşek filtreleniyordu, asıl
            # sinyal skoru hiç kontrol edilmiyordu — bu yüzden PSI eşiği
            # sonuçları etkilemiyordu)
            final_signal_score = float(np.dot(weights, sims_arr))
            if final_signal_score < min_psi:
                continue

            signals.append(Signal(
                ticker       = ticker,
                entry_date   = dates[i],
                entry_price  = float(closes[i]),
                signal_score = round(final_signal_score, 1),
                confidence   = round(confidence, 1),
                expected_pct = round(weighted_pct, 2),
                window       = window,
            ))

            if len(signals) >= max_signals:
                signals.sort(key=lambda s: s.entry_date)
                return signals

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

# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH — Otomatik Parametre Optimizasyonu
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GridSearchResult:
    """Tek bir parametre kombinasyonunun sonucu."""
    params: Dict
    n_signals: int
    sharpe: float
    sortino: float
    win_rate: float
    total_return: float
    max_drawdown: float
    profit_factor: float
    psi_correlation: float    # PSI skoru ile getiri korelasyonu


def run_grid_search(
    all_data: Dict[str, pd.DataFrame],
    find_patterns_fn,
    psi_values: List[float],
    conf_bands: List[Tuple[float, float]],
    window_values: List[int],
    hold_days: int = 20,
    init_cap: float = 100_000.0,
    pos_size: float = 0.10,
    max_signals_per_combo: int = 100,
    step_days: int = 5,
    progress_callback=None,
) -> List[GridSearchResult]:
    """
    Birden fazla parametre kombinasyonunu otomatik dener.
    Her kombinasyon için: sinyal üret → backtest (sabit süre) → metrik hesapla.

    Toplam kombinasyon = len(psi_values) × len(conf_bands) × len(window_values)
    Çok büyük gridler süre alır — UI tarafında bu sınırlanır.
    """
    results = []
    combos = [
        (psi, conf_band, window)
        for psi in psi_values
        for conf_band in conf_bands
        for window in window_values
    ]
    total_combos = len(combos)

    for idx, (psi, (conf_lo, conf_hi), window) in enumerate(combos):
        if progress_callback:
            progress_callback(idx, total_combos, psi, conf_lo, conf_hi, window)

        try:
            signals = generate_historical_signals(
                all_data=all_data,
                find_patterns_fn=find_patterns_fn,
                window=window,
                min_psi=psi,
                min_confidence=conf_lo,
                step_days=step_days,
                max_signals=max_signals_per_combo,
            )
            # Üst güven sınırını da uygula (anti-consensus filtresi)
            signals = [s for s in signals if s.confidence <= conf_hi]

            if len(signals) < 5:
                results.append(GridSearchResult(
                    params={'psi': psi, 'conf_band': f"{conf_lo}-{conf_hi}", 'window': window},
                    n_signals=len(signals), sharpe=0, sortino=0, win_rate=0,
                    total_return=0, max_drawdown=0, profit_factor=0,
                    psi_correlation=0,
                ))
                continue

            bt = backtest_fixed_hold(signals, all_data, hold_days, init_cap, pos_size)

            # PSI-getiri korelasyonu
            if bt.trades:
                psi_scores = np.array([t.signal_score for t in bt.trades])
                rets = np.array([t.pct_return for t in bt.trades])
                if np.std(psi_scores) > 0 and np.std(rets) > 0:
                    corr = float(np.corrcoef(psi_scores, rets)[0, 1])
                else:
                    corr = 0.0
            else:
                corr = 0.0

            results.append(GridSearchResult(
                params={'psi': psi, 'conf_band': f"{conf_lo}-{conf_hi}", 'window': window},
                n_signals=len(signals),
                sharpe=bt.sharpe,
                sortino=bt.sortino,
                win_rate=bt.win_rate,
                total_return=bt.total_return,
                max_drawdown=bt.max_drawdown,
                profit_factor=bt.profit_factor,
                psi_correlation=round(corr, 3),
            ))
        except Exception:
            results.append(GridSearchResult(
                params={'psi': psi, 'conf_band': f"{conf_lo}-{conf_hi}", 'window': window},
                n_signals=0, sharpe=0, sortino=0, win_rate=0,
                total_return=0, max_drawdown=0, profit_factor=0,
                psi_correlation=0,
            ))

    return results


def fig_grid_heatmap(grid_results: List[GridSearchResult],
                     metric: str = 'sharpe') -> go.Figure:
    """
    Grid search sonuçlarını ısı haritası olarak göster.
    X ekseni: güven bandı, Y ekseni: PSI değeri, renk: seçilen metrik.
    Window değerleri ayrı subplot'larda gösterilir.
    """
    windows = sorted(set(r.params['window'] for r in grid_results))
    n_windows = len(windows)

    fig = make_subplots(
        rows=1, cols=n_windows,
        subplot_titles=[f"Window={w}" for w in windows],
        horizontal_spacing=0.08
    )

    metric_label = {
        'sharpe': 'Sharpe Oranı', 'win_rate': 'Kazanç Oranı %',
        'total_return': 'Toplam Getiri %', 'psi_correlation': 'PSI Korelasyonu'
    }.get(metric, metric)

    for col_idx, window in enumerate(windows, start=1):
        subset = [r for r in grid_results if r.params['window'] == window]
        psi_vals = sorted(set(r.params['psi'] for r in subset))
        conf_bands = sorted(set(r.params['conf_band'] for r in subset))

        z = []
        for psi in psi_vals:
            row = []
            for cb in conf_bands:
                match = next((r for r in subset
                             if r.params['psi'] == psi and r.params['conf_band'] == cb), None)
                row.append(getattr(match, metric) if match else None)
            z.append(row)

        fig.add_trace(go.Heatmap(
            z=z, x=conf_bands, y=[str(p) for p in psi_vals],
            colorscale='RdYlGn', zmid=0 if metric in ('sharpe','total_return','psi_correlation') else None,
            text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in z],
            texttemplate='%{text}', textfont_size=9,
            showscale=(col_idx == n_windows),
            hovertemplate='PSI: %{y}<br>Güven: %{x}<br>' + metric_label + ': %{z:.2f}<extra></extra>'
        ), row=1, col=col_idx)

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=50, b=10),
        height=320,
        title=dict(text=f'Grid Search Isı Haritası — {metric_label}',
                  font=dict(size=13, color='#1A1A2E')),
    )
    for i in range(1, n_windows + 1):
        fig.update_xaxes(title='Güven Bandı', row=1, col=i, tickfont=dict(size=9))
        if i == 1:
            fig.update_yaxes(title='Min PSI', row=1, col=i)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTESTING — Overfitting Kontrolü
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardPeriod:
    """Tek bir zaman dilimi sonucu."""
    period_label: str
    start_date: str
    end_date: str
    n_signals: int
    sharpe: float
    win_rate: float
    total_return: float
    max_drawdown: float


def run_walk_forward(
    all_data: Dict[str, pd.DataFrame],
    find_patterns_fn,
    window: int,
    min_psi: float,
    min_confidence: float,
    max_confidence: float,
    n_periods: int = 3,
    hold_days: int = 20,
    init_cap: float = 100_000.0,
    pos_size: float = 0.10,
    progress_callback=None,
) -> List[WalkForwardPeriod]:
    """
    Veriyi `n_periods` eşit zaman dilimine böler, her dilimde AYNI parametrelerle
    ayrı backtest çalıştırır. Sonuçların tutarlılığı, parametrelerin belirli bir
    döneme aşırı uyum (overfitting) göstermediğinin kanıtıdır.

    Mantık: Her hissenin veri serisini n_periods parçaya böl, her parçayı
    bağımsız bir "mini evren" gibi ele alıp sinyal üret + backtest yap.
    """
    # Tüm hisselerin ortak tarih aralığını bul
    all_dates = []
    for df in all_data.values():
        if len(df) > 0:
            all_dates.append((df.index.min(), df.index.max()))

    if not all_dates:
        return []

    global_start = max(d[0] for d in all_dates)
    global_end = min(d[1] for d in all_dates)
    total_days = (global_end - global_start).days

    if total_days < n_periods * 60:
        n_periods = max(1, total_days // 60)

    period_length = total_days // n_periods
    results = []

    for p_idx in range(n_periods):
        if progress_callback:
            progress_callback(p_idx, n_periods)

        period_start = global_start + pd.Timedelta(days=p_idx * period_length)
        period_end = global_start + pd.Timedelta(days=(p_idx + 1) * period_length)
        if p_idx == n_periods - 1:
            period_end = global_end

        # Bu döneme ait alt-veri setini oluştur
        period_data = {}
        for ticker, df in all_data.items():
            sub = df.loc[(df.index >= period_start) & (df.index <= period_end)]
            if len(sub) >= window * 3:
                period_data[ticker] = sub

        if len(period_data) < 5:
            results.append(WalkForwardPeriod(
                period_label=f"Dönem {p_idx+1}",
                start_date=period_start.strftime('%d.%m.%Y'),
                end_date=period_end.strftime('%d.%m.%Y'),
                n_signals=0, sharpe=0, win_rate=0, total_return=0, max_drawdown=0,
            ))
            continue

        try:
            signals = generate_historical_signals(
                all_data=period_data,
                find_patterns_fn=find_patterns_fn,
                window=window,
                min_psi=min_psi,
                min_confidence=min_confidence,
                step_days=5,
                max_signals=80,
            )
            signals = [s for s in signals if s.confidence <= max_confidence]

            if len(signals) < 3:
                results.append(WalkForwardPeriod(
                    period_label=f"Dönem {p_idx+1}",
                    start_date=period_start.strftime('%d.%m.%Y'),
                    end_date=period_end.strftime('%d.%m.%Y'),
                    n_signals=len(signals), sharpe=0, win_rate=0,
                    total_return=0, max_drawdown=0,
                ))
                continue

            bt = backtest_fixed_hold(signals, period_data, hold_days, init_cap, pos_size)

            results.append(WalkForwardPeriod(
                period_label=f"Dönem {p_idx+1}",
                start_date=period_start.strftime('%d.%m.%Y'),
                end_date=period_end.strftime('%d.%m.%Y'),
                n_signals=len(signals),
                sharpe=bt.sharpe,
                win_rate=bt.win_rate,
                total_return=bt.total_return,
                max_drawdown=bt.max_drawdown,
            ))
        except Exception:
            results.append(WalkForwardPeriod(
                period_label=f"Dönem {p_idx+1}",
                start_date=period_start.strftime('%d.%m.%Y'),
                end_date=period_end.strftime('%d.%m.%Y'),
                n_signals=0, sharpe=0, win_rate=0, total_return=0, max_drawdown=0,
            ))

    return results


def fig_walk_forward_consistency(periods: List[WalkForwardPeriod]) -> go.Figure:
    """Dönemler arası Sharpe ve Win Rate tutarlılığını göster."""
    labels = [f"{p.period_label}\n{p.start_date[3:]}" for p in periods]
    sharpes = [p.sharpe for p in periods]
    win_rates = [p.win_rate for p in periods]

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Sharpe Oranı', 'Kazanç Oranı %'])

    colors_sharpe = ['#0E9F6E' if s > 0 else '#E02424' for s in sharpes]
    fig.add_trace(go.Bar(x=labels, y=sharpes, marker_color=colors_sharpe,
                         text=[f"{s:.2f}" for s in sharpes], textposition='outside',
                         showlegend=False), row=1, col=1)

    colors_wr = ['#0E9F6E' if w >= 50 else '#E3A008' for w in win_rates]
    fig.add_trace(go.Bar(x=labels, y=win_rates, marker_color=colors_wr,
                         text=[f"{w:.0f}%" for w in win_rates], textposition='outside',
                         showlegend=False), row=1, col=2)

    fig.add_hline(y=0, line_dash='dash', line_color='rgba(0,0,0,0.2)', row=1, col=1)
    fig.add_hline(y=50, line_dash='dash', line_color='rgba(0,0,0,0.2)', row=1, col=2)

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        title=dict(text='Walk-Forward Tutarlılık Analizi', font=dict(size=13, color='#1A1A2E')),
    )
    return fig


def render_backtest(fetch_batch_fn, find_patterns_fn, all_bist_lists: Dict):
    st.markdown("## 📊 Backtesting Motoru")
    st.caption(
        "BIST-PSI algoritmasının geçmişteki isabetini ölç, parametreleri "
        "otomatik optimize et, tutarlılığı doğrula."
    )
    st.divider()

    tab_manual, tab_grid, tab_walk = st.tabs([
        "🎯 Manuel Backtest",
        "🔬 Grid Search (Otomatik Optimizasyon)",
        "🚶 Walk-Forward (Tutarlılık Testi)"
    ])

    with tab_manual:
        _render_manual_backtest(fetch_batch_fn, find_patterns_fn, all_bist_lists)

    with tab_grid:
        _render_grid_search(fetch_batch_fn, find_patterns_fn, all_bist_lists)

    with tab_walk:
        _render_walk_forward(fetch_batch_fn, find_patterns_fn, all_bist_lists)


def _render_manual_backtest(fetch_batch_fn, find_patterns_fn, all_bist_lists: Dict):
    # Parametreler
    st.markdown("### ⚙️ Parametreler")
    col1, col2, col3 = st.columns(3)
    with col1:
        scope      = st.selectbox("Hisse Evreni", ["BIST 30","BIST 100","Tüm BIST"], index=0)
        window     = st.selectbox("Şablon Uzunluğu", [10, 20, 30, 40], index=2,
                       help="Grid Search: 30 gün kategorik olarak en iyi (Sharpe 6.92 vs 3.5)")
        min_psi    = st.slider("Min BIST-PSI", 55, 85, 70, 1,
                        help="Grid Search doğrulaması: PSI ~70 optimal (Sharpe 6.92, %%69 kazanç)")
    with col2:
        min_conf   = st.slider("Min Güven %", 40, 80, 50, 1,
                        help="Grid Search: 50-65 bandı optimal (%%69 kazanç, +31.2%%)")
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
        ⚡ <b>Hız ipucu:</b> BIST 30 ~1-2 dk, BIST 100 ~3-5 dk.
        Backtesting hızlandırılmış Pearson algoritması kullanır.
        Tüm BIST için max_signals=150 otomatik uygulanır.<br>
        ⚠️ Bu analiz geçmiş performansa dayanır — gelecek getiri garantisi değildir.
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


def _render_grid_search(fetch_batch_fn, find_patterns_fn, all_bist_lists: Dict):
    st.markdown("### 🔬 Otomatik Parametre Optimizasyonu")
    st.caption(
        "Birden fazla PSI eşiği, güven bandı ve şablon uzunluğu kombinasyonunu "
        "otomatik dener. Hangi kombinasyonun en yüksek Sharpe/kazanç oranını "
        "verdiğini ısı haritası ile gösterir."
    )

    gc1, gc2 = st.columns(2)
    with gc1:
        g_scope = st.selectbox("Hisse Evreni", ["BIST 30", "BIST 100"], index=0,
                               key="grid_scope",
                               help="Grid search çok sayıda kombinasyon test ettiği için BIST 30 önerilir")
        g_hold_days = st.slider("Sabit Tutma Süresi (gün)", 5, 40, 20, 5, key="grid_hold")
    with gc2:
        g_init_cap = st.number_input("Başlangıç Sermayesi ₺", min_value=10_000,
                                     max_value=10_000_000, value=100_000,
                                     step=10_000, key="grid_cap")
        g_max_signals = st.slider("Kombinasyon Başına Maks Sinyal", 30, 150, 60, 10,
                                  key="grid_maxsig",
                                  help="Düşük tutmak grid search'ü hızlandırır")

    st.markdown("##### 📅 Veri Dönemi")
    st.caption(
        "Walk-Forward testinde performans son dönemde düştüyse, sadece o "
        "dönemi seçip parametreleri güncel piyasaya göre yeniden optimize edebilirsiniz."
    )
    g_period_mode = st.radio(
        "Test edilecek dönem:",
        ["Tüm Veri (2 yıl)", "Sadece Son Dönem"],
        horizontal=True, key="grid_period_mode"
    )
    g_recent_days = None
    if g_period_mode == "Sadece Son Dönem":
        g_recent_days = st.slider(
            "Son kaç gün?", 60, 365, 240, 30, key="grid_recent_days",
            help="Örn. Walk-Forward'daki son dönemle kıyaslamak için ~240 gün (8 ay) seçin"
        )

    st.markdown("##### Test Edilecek Değer Aralıkları")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        psi_options = st.multiselect(
            "PSI Eşikleri", [65, 70, 75, 80, 85],
            default=[70, 75, 80], key="grid_psi_vals"
        )
    with pc2:
        conf_band_options = st.multiselect(
            "Güven Bandı Seçenekleri",
            ["45-60", "50-65", "55-68", "60-75"],
            default=["50-65", "55-68"], key="grid_conf_vals"
        )
    with pc3:
        window_options = st.multiselect(
            "Şablon Uzunlukları", [10, 15, 20, 30, 40],
            default=[20, 30], key="grid_window_vals"
        )

    n_combos = len(psi_options) * len(conf_band_options) * len(window_options)
    if n_combos > 0:
        est_minutes = n_combos * 0.7  # kabaca tahmin
        st.info(f"📊 **{n_combos}** kombinasyon test edilecek. Tahmini süre: ~{est_minutes:.0f} dakika.")

    if n_combos > 24:
        st.warning("⚠️ 24'ten fazla kombinasyon önerilmez — süre çok uzayabilir. Seçimleri azaltın.")

    grid_btn = st.button("🔬 Grid Search Başlat", type="primary",
                         disabled=(n_combos == 0 or n_combos > 24))

    if grid_btn:
        conf_bands_parsed = []
        for cb in conf_band_options:
            lo, hi = cb.split("-")
            conf_bands_parsed.append((float(lo), float(hi)))

        scope_map = {"BIST 30": all_bist_lists['bist30'], "BIST 100": all_bist_lists['bist100']}
        tickers = scope_map[g_scope]

        prog = st.progress(0, text="Veriler yükleniyor...")
        with st.spinner(""):
            all_data = fetch_batch_fn(tickers, period="2y")

        # Sadece son dönem seçildiyse veriyi kırp
        if g_recent_days is not None:
            trimmed = {}
            for ticker, df in all_data.items():
                if len(df) == 0:
                    continue
                cutoff = df.index.max() - pd.Timedelta(days=g_recent_days)
                sub = df.loc[df.index >= cutoff]
                if len(sub) >= 40:  # Anlamlı analiz için minimum veri
                    trimmed[ticker] = sub
            all_data = trimmed
            prog.progress(10, text=f"{len(all_data)} hisse — son {g_recent_days} gün filtrelendi. "
                                   f"Grid search başlıyor...")
        else:
            prog.progress(10, text=f"{len(all_data)} hisse yüklendi. Grid search başlıyor...")

        def _progress_cb(idx, total, psi, conf_lo, conf_hi, window):
            pct = 10 + int((idx / total) * 85)
            prog.progress(pct, text=f"Test: PSI={psi}, Güven={conf_lo}-{conf_hi}, "
                                    f"Window={window} ({idx+1}/{total})")

        grid_results = run_grid_search(
            all_data=all_data,
            find_patterns_fn=find_patterns_fn,
            psi_values=psi_options,
            conf_bands=conf_bands_parsed,
            window_values=window_options,
            hold_days=g_hold_days,
            init_cap=g_init_cap,
            max_signals_per_combo=g_max_signals,
            progress_callback=_progress_cb,
        )

        prog.progress(100, text="✅ Grid search tamamlandı!")
        import time; time.sleep(0.3); prog.empty()

        st.session_state['grid_results'] = grid_results
        st.session_state['grid_period_label'] = (
            f"Son {g_recent_days} gün" if g_recent_days is not None else "Tüm Veri (2 yıl)"
        )
        st.rerun()

    grid_results = st.session_state.get('grid_results')
    if not grid_results:
        st.info("Değer aralıklarını seçip 'Grid Search Başlat' butonuna basın.")
        return

    st.divider()
    period_label = st.session_state.get('grid_period_label', 'Tüm Veri')
    st.markdown(f"#### 📋 Sonuçlar — {len(grid_results)} kombinasyon test edildi")
    st.caption(f"📅 Test edilen dönem: **{period_label}**")

    # En iyi kombinasyonlar
    valid_results = [r for r in grid_results if r.n_signals >= 5]
    if not valid_results:
        st.warning("Hiçbir kombinasyonda yeterli sinyal üretilemedi (min 5 sinyal gerekli).")
        return

    sorted_by_sharpe = sorted(valid_results, key=lambda r: r.sharpe, reverse=True)
    best = sorted_by_sharpe[0]

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("🏆 En İyi PSI", f"{best.params['psi']}")
    bc2.metric("🏆 En İyi Güven Bandı", best.params['conf_band'])
    bc3.metric("🏆 En İyi Window", f"{best.params['window']} gün")
    bc4.metric("🏆 Sharpe", f"{best.sharpe:.2f}")

    st.success(
        f"✅ **Önerilen parametre kombinasyonu:** PSI={best.params['psi']}, "
        f"Güven={best.params['conf_band']}, Window={best.params['window']} gün — "
        f"Sharpe: {best.sharpe:.2f}, Kazanç Oranı: %{best.win_rate:.0f}, "
        f"Toplam Getiri: {best.total_return:+.1f}%, Sinyal Sayısı: {best.n_signals}"
    )

    # Isı haritaları
    st.markdown("##### 🗺️ Isı Haritaları")
    heatmap_metric = st.radio(
        "Metrik", ["sharpe", "win_rate", "total_return", "psi_correlation"],
        format_func=lambda x: {
            'sharpe': 'Sharpe Oranı', 'win_rate': 'Kazanç Oranı',
            'total_return': 'Toplam Getiri', 'psi_correlation': 'PSI Korelasyonu'
        }[x],
        horizontal=True, key="grid_heatmap_metric"
    )
    st.plotly_chart(fig_grid_heatmap(grid_results, heatmap_metric),
                    use_container_width=True)

    # Detaylı tablo
    st.markdown("##### 📊 Tüm Sonuçlar")
    table_rows = []
    for r in sorted_by_sharpe:
        table_rows.append({
            'PSI': r.params['psi'],
            'Güven Bandı': r.params['conf_band'],
            'Window': r.params['window'],
            'Sinyal #': r.n_signals,
            'Sharpe': f"{r.sharpe:.2f}",
            'Sortino': f"{r.sortino:.2f}",
            'Kazanç %': f"{r.win_rate:.0f}",
            'Toplam Getiri': f"{r.total_return:+.1f}%",
            'Max DD': f"-{r.max_drawdown:.1f}%",
            'Profit Factor': f"{r.profit_factor:.2f}",
            'PSI Korelasyon': f"{r.psi_correlation:+.3f}",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def _render_walk_forward(fetch_batch_fn, find_patterns_fn, all_bist_lists: Dict):
    st.markdown("### 🚶 Walk-Forward Tutarlılık Testi")
    st.caption(
        "Aynı parametrelerle veriyi birden fazla zaman dilimine bölüp ayrı ayrı "
        "test eder. Sonuçlar dönemler arasında tutarlıysa (hepsi pozitif Sharpe, "
        "benzer kazanç oranı), parametreleriniz overfitting değil — gerçekten "
        "genelleşebilir bir kalıp yakalamış demektir."
    )

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        w_scope = st.selectbox("Hisse Evreni", ["BIST 30", "BIST 100"], index=0, key="wf_scope")
        w_window = st.selectbox("Şablon Uzunluğu", [10, 20, 30, 40], index=2, key="wf_window",
                       help="Grid Search: 30 gün en iyi sonucu verdi")
    with wc2:
        w_psi = st.slider("Min PSI", 55, 85, 70, 1, key="wf_psi")
        w_conf_lo = st.slider("Min Güven %", 40, 70, 50, 1, key="wf_conf_lo")
    with wc3:
        w_conf_hi = st.slider("Maks Güven %", 60, 90, 65, 1, key="wf_conf_hi")
        w_n_periods = st.slider("Dönem Sayısı", 2, 6, 3, 1, key="wf_periods",
                                help="Veri bu kadar eşit parçaya bölünür")

    w_hold_days = st.slider("Sabit Tutma Süresi (gün)", 5, 40, 20, 5, key="wf_hold")
    w_init_cap = st.number_input("Başlangıç Sermayesi ₺ (dönem başına)",
                                 min_value=10_000, max_value=10_000_000,
                                 value=100_000, step=10_000, key="wf_cap")

    st.markdown("""
    <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
                padding:10px 14px;font-size:12px;color:#1E40AF;margin:8px 0'>
        💡 Önce <b>Grid Search</b> sekmesinde en iyi parametreleri bulun,
        sonra burada o parametrelerle tutarlılığı doğrulayın.
    </div>
    """, unsafe_allow_html=True)

    wf_btn = st.button("🚶 Walk-Forward Testi Başlat", type="primary")

    if wf_btn:
        scope_map = {"BIST 30": all_bist_lists['bist30'], "BIST 100": all_bist_lists['bist100']}
        tickers = scope_map[w_scope]

        prog = st.progress(0, text="Veriler yükleniyor...")
        with st.spinner(""):
            all_data = fetch_batch_fn(tickers, period="2y")
        prog.progress(15, text=f"{len(all_data)} hisse yüklendi. Dönemlere bölünüyor...")

        def _wf_progress(p_idx, n_periods):
            pct = 15 + int((p_idx / n_periods) * 80)
            prog.progress(pct, text=f"Dönem {p_idx+1}/{n_periods} test ediliyor...")

        wf_results = run_walk_forward(
            all_data=all_data,
            find_patterns_fn=find_patterns_fn,
            window=w_window,
            min_psi=w_psi,
            min_confidence=w_conf_lo,
            max_confidence=w_conf_hi,
            n_periods=w_n_periods,
            hold_days=w_hold_days,
            init_cap=w_init_cap,
            progress_callback=_wf_progress,
        )

        prog.progress(100, text="✅ Walk-forward testi tamamlandı!")
        import time; time.sleep(0.3); prog.empty()

        st.session_state['wf_results'] = wf_results
        st.rerun()

    wf_results = st.session_state.get('wf_results')
    if not wf_results:
        st.info("Parametreleri ayarlayıp 'Walk-Forward Testi Başlat' butonuna basın.")
        return

    st.divider()
    st.markdown(f"#### 📋 Sonuçlar — {len(wf_results)} dönem")

    # Tutarlılık değerlendirmesi
    valid_periods = [p for p in wf_results if p.n_signals >= 3]
    if len(valid_periods) >= 2:
        sharpes = [p.sharpe for p in valid_periods]
        win_rates = [p.win_rate for p in valid_periods]
        positive_sharpe_count = sum(1 for s in sharpes if s > 0)
        sharpe_std = float(np.std(sharpes))
        avg_win_rate = float(np.mean(win_rates))

        consistency_pct = positive_sharpe_count / len(valid_periods) * 100

        if consistency_pct >= 75 and sharpe_std < 1.5:
            st.success(
                f"✅ **Yüksek Tutarlılık**: {len(valid_periods)} dönemin "
                f"{positive_sharpe_count}'ünde pozitif Sharpe (%{consistency_pct:.0f}). "
                f"Ortalama kazanç oranı: %{avg_win_rate:.0f}. "
                f"Bu parametreler overfitting değil, genelleşebilir görünüyor."
            )
        elif consistency_pct >= 50:
            st.warning(
                f"⚠️ **Orta Tutarlılık**: {len(valid_periods)} dönemin "
                f"{positive_sharpe_count}'ünde pozitif Sharpe (%{consistency_pct:.0f}). "
                f"Bazı dönemlerde zayıf performans var — dikkatli kullanın."
            )
        else:
            st.error(
                f"❌ **Düşük Tutarlılık**: Sadece {positive_sharpe_count}/{len(valid_periods)} "
                f"dönemde pozitif Sharpe (%{consistency_pct:.0f}). "
                f"Bu parametreler muhtemelen belirli bir döneme özel optimize "
                f"edilmiş (overfitting riski yüksek). Farklı parametreler deneyin."
            )

    st.plotly_chart(fig_walk_forward_consistency(wf_results), use_container_width=True)

    st.markdown("##### 📊 Dönem Detayları")
    period_rows = []
    for p in wf_results:
        period_rows.append({
            'Dönem': p.period_label,
            'Başlangıç': p.start_date,
            'Bitiş': p.end_date,
            'Sinyal #': p.n_signals,
            'Sharpe': f"{p.sharpe:.2f}",
            'Kazanç %': f"{p.win_rate:.0f}",
            'Toplam Getiri': f"{p.total_return:+.1f}%",
            'Max DD': f"-{p.max_drawdown:.1f}%",
        })
    st.dataframe(pd.DataFrame(period_rows), use_container_width=True, hide_index=True)
