import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ── Sayfa ayarı ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIST Pattern Matcher",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; }
.stButton > button { border-radius: 6px; font-weight: 500; }
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E9F0;
    border-radius: 8px;
    padding: 12px !important;
}
.match-card {
    background: #FFFFFF;
    border: 1.5px solid #E5E9F0;
    border-radius: 10px;
    padding: 16px 12px;
    text-align: center;
    transition: all 0.2s;
    cursor: pointer;
}
.match-card:hover { border-color: #1A56DB; box-shadow: 0 2px 8px rgba(26,86,219,0.1); }
.match-card.selected { border-color: #1A56DB; background: #EFF6FF; }
</style>
""", unsafe_allow_html=True)

# ── BIST hisse listeleri ───────────────────────────────────────────────────────
BIST30 = [
    "AKBNK","ARCLK","ASELS","BIMAS","EKGYO","EREGL","FROTO","GARAN","HALKB",
    "ISCTR","KCHOL","KOZAA","KOZAL","KRDMD","MGROS","ODAS","PETKM","PGSUS",
    "SAHOL","SISE","SOKM","TAVHL","TCELL","THYAO","TKFEN","TOASO","TTKOM",
    "TUPRS","VAKBN","YKBNK"
]

BIST100 = list(set(BIST30 + [
    "AEFES","AGESA","AKGRT","AKSA","AKSEN","ALARK","ALBRK","ALFAS","ANELE",
    "ANHYT","ANSGR","ASTOR","AYGAZ","BAGFS","BERA","BFREN","BRISA","BTCIM",
    "BUCIM","CIMSA","CLEBI","CMBTN","DEVA","DOHOL","ECILC","ECZYT","EDIP",
    "EGEEN","EGGUB","EKIZ","ENKAI","ERBOS","ERSU","ESCAR","EUPWR","FENER",
    "GENIL","GENTS","GEREL","GESAN","GLYHO","GMTAS","GOODY","GOZDE","GRSEL",
    "GUBRF","GWIND","HATEK","HEKTS","HLGYO","HOROZ","HUBVC","HURGZ","IEYHO",
    "IHEVA","IHLAS","INDES","INFO","INVEO","IPEKE","ISBIR","ISYAT","IZENR",
    "JANTS","KAREL","KARSN","KARTN","KCHOL","KERVT","KLKIM","KLMSN","KMPUR",
    "KONTR","KOPOL","KORDS","KOZAA","KOZAL","KRDMD","LOGO","MAVI","MEDTR",
    "MGROS","MPARK","NATEN","NETAS","NUHCM","ODAS","OTKAR","OYAKC","PETKM",
    "PETUN","PGSUS","PKART","POLHO","PRKAB","PRKME","RUBNS","SAHOL","SANEL",
    "SANFM","SARKY","SASA","SELEC","SELGD","SISE","SKBNK","SKTAS","SOKM",
    "TATGD","TAVHL","TCELL","THYAO","TKFEN","TOASO","TTKOM","TTRAK","TUPRS",
    "ULUSE","VAKBN","VAKKO","VESBE","YKBNK","YUNSA","ZEDUR"
]))

ALL_BIST = list(set(BIST100 + [
    "ACSEL","ADEL","AGYO","AKFEN","AKMGY","ALKIM","ALKLC","ARDYZ","ARSAN",
    "ATAGY","ATEKS","ATLAS","AVGYO","AYCES","AYEN","AZTEK","BASGZ","BAYRK",
    "BIENY","BJKAS","BNTAS","BOBET","BORLS","BRKSN","BRKVY","BRMEN","BSOKE",
    "BURCE","BURVA","BVSAN","CANTE","CARFA","CEMAS","CEMTS","CEOEM","COSMO",
    "CRDFA","CRFSA","CUSAN","CVKMD","DAGHL","DARDL","DENGE","DGGYO","DITAS",
    "DMSAS","DNISI","DOBUR","DOCO","DOGUB","DOKTA","DURDO","DYOBY","DZGYO",
    "EDATA","EDIP","EGEEN","EGPRO","EGSER","ELITE","EMKEL","EMNIS","EPLAS",
    "ESCOM","ESEN","ETILR","ETYAT","EUHOL","EUREN","EUYO","FENER","FLAP",
    "FONET","FORMT","FORTE","FZLGY","GARFA","GEDIK","GEDZA","GENIL","GLBMD",
    "GLRYH","GOKNR","GOLTS","GRTRK","GSDDE","GSDHO","GSRAY","HDFGS","HEDEF",
    "HKTM","HRKET","HTTBT","HUNER","ICBCT","IDEAS","IDGYO","IHAAS","IHGZT",
    "IHLGM","IHYAY","IMASM","INTEM","ISATR","ITTFK","IZFAS","IZINV","IZMDC",
    "KARSN","KATMR","KAYSE","KCAER","KFEIN","KGYO","KIMMR","KLGYO","KLRHO",
    "KLSER","KNFRT","KONAK","KOPOL","KOTON","KRONT","KRPLS","KRSTL","KRTEK",
    "KRVGD","KSTUR","KTLEV","KTSK","KUTPO","KUYAS","LIDER","LIDFA","LINK",
    "LKMNH","LMKDC","LRSHO","LUKSK","MAALT","MAGEN","MAKIM","MAKTK","MANAS",
    "MARBL","MARKA","MARTI","MEGMT","MEPET","MERCN","MERIT","MERKO","METRO",
    "METUR","MIATK","MIPAZ","MMCAS","MNDRS","MNDTR","MOBTL","MOGAN","MRDIN",
    "MRGYO","MRSHL","MSGYO","MTRKS","MZHLD","NIBAS","NILYT","NTHOL","NTTUR",
    "NUGYO","OBAMS","OBASE","ODINE","OFSYM","ONCSM","ORCAY","ORGE","ORMA",
    "OSMEN","OSTIM","OYYAT","OZGYO","OZKGY","OZRDN","OZSUB","PAGYO","PAMEL",
    "PAPIL","PARSN","PASEU","PCKMT","PCYOT","PEGYO","PENGD","PENTA","PINSU",
    "PKENT","PLTUR","PNLSN","POLTK","PRDGS","PRZMA","PSDTC","PTOFS","RTALB",
    "RYGYO","SAMAT","SANKO","SAYAS","SDTTR","SEGYO","SEKFK","SEKUR","SELVA",
    "SEYKM","SILVR","SNKRN","SODSN","SONME","SRVGY","SUWEN","TBORG","TDGYO",
    "TEKTU","TGSAS","TLMAN","TMSN","TOASO","TRCAS","TRGYO","TRILC","TSGYO",
    "TUCLK","TUKAS","TUREX","TURGG","TURSG","TZNGY","ULUFA","ULUUN","UNLU",
    "USAK","UTPYA","UZERB","VAKFN","VANGD","VBTYZ","VERUS","VKFYO","VKGYO",
    "VRGYO","WNDMR","YATAS","YAYLA","YGYO","YIGIT","YKSLN","ZRGYO"
]))

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker(symbol, period="1y"):
    try:
        ticker = symbol if symbol.endswith(".IS") else symbol + ".IS"
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty or len(df) < 20:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_batch(tickers, period="2y"):
    results = {}
    symbols = [t + ".IS" for t in tickers]
    try:
        raw = yf.download(symbols, period=period, auto_adjust=True,
                          group_by='ticker', progress=False)
        for t in tickers:
            try:
                sym = t + ".IS"
                if sym in raw.columns.get_level_values(0):
                    df = raw[sym][['Open','High','Low','Close','Volume']].dropna()
                    df.index = pd.to_datetime(df.index)
                    if len(df) >= 40:
                        results[t] = df
            except:
                pass
    except:
        pass
    return results

def zscore(arr):
    arr = np.array(arr, dtype=float)
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-9:
        return np.zeros_like(arr)
    return (arr - mu) / sigma

def daily_returns(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) < 2:
        return np.zeros(len(arr))
    rets = np.diff(arr) / (np.abs(arr[:-1]) + 1e-9)
    return rets

def pearson(a, b):
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def dtw_score(s1, s2, band=None):
    n = len(s1)
    band = band or max(2, n // 8)
    dtw = np.full((n+1, n+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        j0 = max(1, i - band)
        j1 = min(n, i + band) + 1
        for j in range(j0, j1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    dist = dtw[n, n] / n
    return max(0.0, 1.0 - dist)

def similarity(tpl_z, tpl_ret, win_z, win_ret):
    p_z = (pearson(tpl_z, win_z) + 1) / 2       # 0-1
    p_ret = (pearson(tpl_ret, win_ret) + 1) / 2  # 0-1
    d = dtw_score(tpl_z, win_z)                   # 0-1
    # Ağırlık: %35 pearson-z + %40 DTW + %25 getiri korelasyonu
    score = 0.35 * p_z + 0.40 * d + 0.25 * p_ret
    return round(score * 100, 1)

def find_patterns(template, all_data, top_n=5, min_sim=70, future_mult=1.5):
    tpl = np.array(template, dtype=float)
    tpl_z = zscore(tpl)
    tpl_ret = daily_returns(tpl)
    n = len(tpl)
    fut_win = min(int(n * future_mult), 90)
    results = []

    for ticker, df in all_data.items():
        closes = df['Close'].values.astype(float)
        dates = df.index
        if len(closes) < n + fut_win + 10:
            continue

        max_i = len(closes) - n - fut_win
        step = max(1, n // 6)

        # Kaba tarama
        best_sim, best_i = -1, 0
        for i in range(0, max_i, step):
            w = closes[i:i+n]
            wz = zscore(w)
            wr = daily_returns(w)
            s = similarity(tpl_z, tpl_ret, wz, wr)
            if s > best_sim:
                best_sim, best_i = s, i

        # İnce tarama ±step
        for i in range(max(0, best_i - step), min(max_i+1, best_i + step + 1)):
            w = closes[i:i+n]
            wz = zscore(w)
            wr = daily_returns(w)
            s = similarity(tpl_z, tpl_ret, wz, wr)
            if s > best_sim:
                best_sim, best_i = s, i

        if best_sim < min_sim:
            continue

        ms, me = best_i, best_i + n
        match_closes = closes[ms:me]
        match_dates = dates[ms:me]
        future_closes = closes[me:me+fut_win]
        future_dates = dates[me:me+fut_win]

        fut_pct = 0.0
        fut_max = 0.0
        fut_min = 0.0
        if len(future_closes) > 1:
            fut_pct = (future_closes[-1] - future_closes[0]) / future_closes[0] * 100
            fut_max = (future_closes.max() - future_closes[0]) / future_closes[0] * 100
            fut_min = (future_closes.min() - future_closes[0]) / future_closes[0] * 100

        results.append({
            'ticker': ticker,
            'similarity': best_sim,
            'ms': ms, 'me': me,
            'match_closes': match_closes,
            'match_dates': match_dates,
            'future_closes': future_closes,
            'future_dates': future_dates,
            'fut_pct': round(fut_pct, 2),
            'fut_max': round(fut_max, 2),
            'fut_min': round(fut_min, 2),
            'fut_win': fut_win,
            'all_closes': closes,
            'all_dates': dates,
            'start_date': pd.Timestamp(match_dates[0]).strftime('%d.%m.%Y'),
            'end_date': pd.Timestamp(match_dates[-1]).strftime('%d.%m.%Y'),
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

# ── Grafikler ─────────────────────────────────────────────────────────────────

COLORS = ['#1A56DB','#E3A008','#0E9F6E','#9061F9','#E02424']

def chart_opts():
    return dict(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
        legend=dict(orientation='h', y=1.12, font=dict(size=11)),
        xaxis=dict(gridcolor='rgba(0,0,0,0.06)', showgrid=True),
        yaxis=dict(gridcolor='rgba(0,0,0,0.06)', showgrid=True),
    )

def fig_main(df, symbol, sel_start=None, sel_end=None):
    dates = [d.strftime('%Y-%m-%d') for d in df.index]
    closes = df['Close'].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=closes, name=symbol,
        line=dict(color='#1A56DB', width=2),
        hovertemplate='%{x}: %{y:.2f} ₺<extra></extra>'
    ))

    if sel_start and sel_end:
        mask = (df.index >= pd.Timestamp(sel_start)) & (df.index <= pd.Timestamp(sel_end))
        seg_df = df[mask]
        if len(seg_df) > 0:
            seg_dates = [d.strftime('%Y-%m-%d') for d in seg_df.index]
            fig.add_trace(go.Scatter(
                x=seg_dates, y=seg_df['Close'].values,
                name='Seçili Şablon',
                line=dict(color='#E3A008', width=3),
                hovertemplate='%{x}: %{y:.2f} ₺<extra>Şablon</extra>'
            ))
            fig.add_vrect(
                x0=seg_dates[0], x1=seg_dates[-1],
                fillcolor='rgba(227,160,8,0.08)', line_width=0,
                annotation_text='Şablon', annotation_position='top left',
                annotation_font_color='#E3A008', annotation_font_size=10
            )

    opts = chart_opts()
    opts.update(dict(height=360, title=dict(
        text=f'<b>{symbol}</b> — Şablon Seçimi',
        font=dict(size=14, color='#1A1A2E')
    )))
    fig.update_layout(**opts)
    fig.update_xaxes(type='date', tickformat='%b %Y', tickangle=-30,
                     tickfont=dict(size=10))
    fig.update_yaxes(ticksuffix=' ₺', tickfont=dict(size=10))
    return fig

def fig_history(result, symbol):
    """Hissenin TÜM geçmişi — eşleşen bölge + sonrası işaretli"""
    closes = result['all_closes']
    dates_raw = result['all_dates']
    dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates_raw]
    ms, me = result['ms'], result['me']
    fut = result['future_closes']
    fut_pct = result['fut_pct']
    c_fut = '#0E9F6E' if fut_pct >= 0 else '#E02424'
    icon = '▲' if fut_pct >= 0 else '▼'

    fig = go.Figure()

    # Tüm geçmiş
    fig.add_trace(go.Scatter(
        x=dates, y=closes, name=result['ticker'],
        line=dict(color='rgba(100,130,180,0.45)', width=1.5),
        hovertemplate='%{x}: %{y:.2f} ₺<extra></extra>'
    ))

    # Eşleşen bölge
    fig.add_trace(go.Scatter(
        x=dates[ms:me], y=closes[ms:me],
        name=f'Eşleşen Bölge (%{result["similarity"]})',
        line=dict(color='#E3A008', width=3.5),
        hovertemplate='%{x}: %{y:.2f} ₺<extra>Eşleşen</extra>'
    ))

    # Sonraki hareket
    if len(fut) > 1:
        fut_dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in result['future_dates']]
        fig.add_trace(go.Scatter(
            x=fut_dates, y=fut,
            name=f'Sonraki Hareket ({fut_pct:+.1f}%)',
            line=dict(color=c_fut, width=2.5, dash='dot'),
            hovertemplate='%{x}: %{y:.2f} ₺<extra>Sonrası</extra>'
        ))

    # Vurgu bölgeleri
    if ms < me and len(dates) > me:
        fig.add_vrect(x0=dates[ms], x1=dates[me-1],
                      fillcolor='rgba(227,160,8,0.08)', line_width=0,
                      annotation_text=f'Eşleşme\n{result["start_date"]}',
                      annotation_position='top left',
                      annotation_font_color='#E3A008', annotation_font_size=9)
        if len(fut) > 1:
            fig.add_vrect(
                x0=dates[me], x1=dates[min(me+len(fut)-1, len(dates)-1)],
                fillcolor=f'{"rgba(14,159,110,0.06)" if fut_pct>=0 else "rgba(224,36,36,0.06)"}',
                line_width=0,
                annotation_text=f'{icon} {fut_pct:+.1f}%',
                annotation_position='top right',
                annotation_font_color=c_fut, annotation_font_size=9
            )

    opts = chart_opts()
    opts.update(dict(height=340, title=dict(
        text=f'<b>{result["ticker"]}</b> — Tüm Geçmiş | '
             f'Eşleşme: <b>{result["start_date"]} → {result["end_date"]}</b>',
        font=dict(size=13, color='#1A1A2E')
    )))
    fig.update_layout(**opts)
    fig.update_xaxes(type='date', tickformat='%b %Y', tickangle=-30,
                     tickfont=dict(size=10))
    fig.update_yaxes(ticksuffix=' ₺', tickfont=dict(size=10))
    return fig

def fig_normalize(template, results, symbol):
    """Normalize overlay — şablon + tüm eşleşmeler + sonraki hareketler"""
    tpl_z = zscore(np.array(template, dtype=float))
    n = len(tpl_z)
    x_tpl = list(range(n))

    fig = go.Figure()

    # Şablon
    fig.add_trace(go.Scatter(
        x=x_tpl, y=tpl_z, name=f'{symbol} (Şablon)',
        line=dict(color='#1A1A2E', width=3),
        hovertemplate='Şablon: %{y:.2f}<extra></extra>'
    ))

    for i, r in enumerate(results):
        c = COLORS[i % len(COLORS)]
        seg_z = zscore(r['match_closes'])
        fig.add_trace(go.Scatter(
            x=x_tpl, y=seg_z,
            name=f"{r['ticker']} (%{r['similarity']})",
            line=dict(color=c, width=1.8, dash='dot'),
            opacity=0.8,
            hovertemplate=f"{r['ticker']}: %{{y:.2f}}<extra></extra>"
        ))

        if len(r['future_closes']) > 2:
            fut = r['future_closes']
            last = float(seg_z[-1])
            fut_z = zscore(fut)
            fut_scaled = [last + v * 0.35 for v in fut_z]
            x_fut = list(range(n, n + len(fut_scaled)))
            fig.add_trace(go.Scatter(
                x=x_fut, y=fut_scaled,
                name=f"{r['ticker']} sonrası ({r['fut_pct']:+.1f}%)",
                line=dict(color=c, width=1.5, dash='longdash'),
                opacity=0.55,
                hovertemplate=f"{r['ticker']} sonrası: %{{y:.2f}}<extra></extra>"
            ))

    fig.add_vline(x=n - 0.5, line_dash='dash',
                  line_color='rgba(0,0,0,0.25)', line_width=2,
                  annotation_text='← Geçmiş | Tahmin →',
                  annotation_font_color='#555', annotation_font_size=10)
    fig.add_vrect(x0=0, x1=n-1, fillcolor='rgba(227,160,8,0.04)', line_width=0)

    opts = chart_opts()
    opts.update(dict(height=380, title=dict(
        text='Normalize Karşılaştırma — Eşleşen Bölgeler + Sonraki Hareketler',
        font=dict(size=13, color='#1A1A2E')
    )))
    fig.update_layout(**opts)
    fig.update_xaxes(title='Gün', tickfont=dict(size=10))
    fig.update_yaxes(title='Z-Score', tickfont=dict(size=10))
    return fig

def fig_compare(result, template, symbol):
    """Şablon vs eşleşen bölge normalize karşılaştırması"""
    tpl_z = zscore(np.array(template, dtype=float))
    seg_z = zscore(result['match_closes'])
    n = len(tpl_z)
    x = list(range(n))
    fut_pct = result['fut_pct']
    c_fut = '#0E9F6E' if fut_pct >= 0 else '#E02424'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=tpl_z, name=f'{symbol} (Şablon)',
        line=dict(color='#1A56DB', width=2.5),
        hovertemplate='Şablon: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=seg_z,
        name=f"{result['ticker']} — Eşleşen (%{result['similarity']})",
        line=dict(color='#E3A008', width=2, dash='dot'),
        hovertemplate=f"{result['ticker']}: %{{y:.2f}}<extra></extra>"
    ))

    if len(result['future_closes']) > 2:
        fut = result['future_closes']
        last = float(seg_z[-1])
        fut_z = zscore(fut)
        fut_scaled = [last + v * 0.35 for v in fut_z]
        x_fut = list(range(n, n + len(fut_scaled)))
        fig.add_trace(go.Scatter(
            x=x_fut, y=fut_scaled,
            name=f"Sonraki Hareket ({fut_pct:+.1f}%)",
            line=dict(color=c_fut, width=2, dash='longdash'),
            hovertemplate=f'Sonrası: %{{y:.2f}}<extra></extra>'
        ))
        fig.add_vline(x=n - 0.5, line_dash='dash',
                      line_color='rgba(0,0,0,0.2)', line_width=1.5,
                      annotation_text=f'{"▲" if fut_pct>=0 else "▼"} {fut_pct:+.1f}%',
                      annotation_font_color=c_fut, annotation_font_size=11)

    opts = chart_opts()
    opts.update(dict(height=280, title=dict(
        text=f'Şablon Uyumu — {symbol} vs {result["ticker"]}',
        font=dict(size=12, color='#1A1A2E')
    )))
    fig.update_layout(**opts)
    fig.update_xaxes(title='Gün', tickfont=dict(size=10))
    fig.update_yaxes(title='Z-Score', tickfont=dict(size=10))
    return fig

# ── Ana uygulama ──────────────────────────────────────────────────────────────

def main():
    # Başlık
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.markdown("## 📊 BIST Pattern Matcher")
        st.caption("Hisse senedi şablon eşleştirme — geçmişteki benzer hareketleri bul, sonrasını gör")
    with col_s:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.divider()

    # ── ADIM 1: Hisse seç ──
    st.markdown("### 1️⃣ Hisse ve Dönem")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbol = st.text_input("Hisse kodu", placeholder="THYAO, EREGL, ASELS...",
                                key="symbol_input").strip().upper()
    with c2:
        period = st.selectbox("Dönem", ["6mo","1y","2y"],
                               format_func=lambda x: {"6mo":"6 Ay","1y":"1 Yıl","2y":"2 Yıl"}[x],
                               index=1)
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        load = st.button("📥 Yükle", type="primary", use_container_width=True)

    # Hızlı seçim
    samples = ["THYAO","EREGL","ASELS","GARAN","BIMAS","KCHOL","SASA","TOASO","TUPRS","AKBNK"]
    cols = st.columns(len(samples))
    for i, s in enumerate(samples):
        if cols[i].button(s, key=f"q_{s}", use_container_width=True):
            st.session_state["symbol_input"] = s
            st.rerun()

    # Veri yükle
    if load and symbol:
        with st.spinner(f"{symbol} yükleniyor..."):
            df = fetch_ticker(symbol, period)
        if df is None:
            st.error(f"'{symbol}' bulunamadı. Hisse kodu doğru mu?")
            return
        st.session_state["df"] = df
        st.session_state["symbol"] = symbol
        st.session_state["matches"] = None
        st.session_state["selected"] = None

    df = st.session_state.get("df")
    sym = st.session_state.get("symbol", "")
    if df is None:
        st.info("Bir hisse seçin ve 'Yükle' butonuna basın.")
        return

    st.divider()

    # ── ADIM 2: Tarih seçimi ──
    st.markdown("### 2️⃣ Şablon Aralığı")
    st.caption("Grafikte incelemek istediğiniz fiyat hareketini tarih seçicilerle belirleyin.")

    date_list = [d.date() for d in df.index]
    mid = len(date_list) // 2

    c_s, c_e = st.columns(2)
    with c_s:
        sel_start = st.date_input("📍 Başlangıç", value=date_list[max(0, mid-15)],
                                   min_value=date_list[0], max_value=date_list[-2])
    with c_e:
        sel_end = st.date_input("🏁 Bitiş", value=date_list[min(len(date_list)-1, mid+15)],
                                 min_value=date_list[1], max_value=date_list[-1])

    if sel_start >= sel_end:
        st.warning("Başlangıç tarihi bitiş tarihinden önce olmalı.")
        return

    sel_start_ts = pd.Timestamp(sel_start)
    sel_end_ts = pd.Timestamp(sel_end)
    segment = df.loc[sel_start_ts:sel_end_ts]['Close']

    if len(segment) < 5:
        st.warning("En az 5 günlük aralık seçin.")
        return

    # Grafik
    fig = fig_main(df, sym, sel_start_ts, sel_end_ts)
    st.plotly_chart(fig, use_container_width=True)

    # Şablon istatistikleri
    pct = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0] * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Şablon Uzunluğu", f"{len(segment)} gün")
    m2.metric("Başlangıç", f"{segment.iloc[0]:.2f} ₺")
    m3.metric("Bitiş", f"{segment.iloc[-1]:.2f} ₺")
    m4.metric("Değişim", f"{pct:+.1f}%")

    st.divider()

    # ── ADIM 3: Tarama ──
    st.markdown("### 3️⃣ BIST Tarama")

    c_scope, c_sim, c_btn = st.columns([2, 1, 1])
    with c_scope:
        scope = st.radio("Kapsam", ["BIST 30", "BIST 100", "Tüm BIST"],
                          horizontal=True)
    with c_sim:
        min_sim = st.slider("Min. Benzerlik %", 60, 90, 72, 2)
    with c_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        scan = st.button("🔍 Tara", type="primary", use_container_width=True)

    if scan:
        if scope == "BIST 30":
            scan_list = [t for t in BIST30 if t != sym]
        elif scope == "BIST 100":
            scan_list = [t for t in BIST100 if t != sym]
        else:
            scan_list = [t for t in ALL_BIST if t != sym]

        prog = st.progress(0, text=f"📥 {len(scan_list)} hisse indiriliyor...")
        with st.spinner(""):
            all_data = fetch_batch(scan_list, period="2y")
        prog.progress(50, text=f"🔍 {len(all_data)} hisse taranıyor...")
        template = segment.values.astype(float)
        matches = find_patterns(template, all_data, top_n=5, min_sim=min_sim)
        prog.progress(100, text="✅ Tamamlandı!")
        import time; time.sleep(0.4); prog.empty()

        st.session_state["matches"] = matches
        st.session_state["template"] = template
        st.session_state["selected"] = None
        st.rerun()

    # ── SONUÇLAR ──
    matches = st.session_state.get("matches")
    template = st.session_state.get("template")
    if matches is None:
        return

    st.divider()

    if len(matches) == 0:
        st.warning(f"**%{min_sim}** üzeri benzerlik bulunamadı. Eşiği düşürün veya farklı aralık deneyin.")
        return

    st.markdown(f"### 📊 En Benzer {len(matches)} Hisse")
    st.caption("Bir hisseye tıklayarak detaylı analiz görün.")

    # Kartlar
    card_cols = st.columns(len(matches))
    for i, r in enumerate(matches):
        c = COLORS[i % len(COLORS)]
        is_sel = st.session_state.get("selected") == r['ticker']
        fut_pct = r['fut_pct']
        c_fut = '#0E9F6E' if fut_pct >= 0 else '#E02424'
        icon = '▲' if fut_pct >= 0 else '▼'

        with card_cols[i]:
            border = f'2px solid {c}' if is_sel else '1.5px solid #E5E9F0'
            bg = '#F0F7FF' if is_sel else '#FFFFFF'
            st.markdown(f"""
            <div style='background:{bg};border:{border};border-radius:10px;
                        padding:14px 10px;text-align:center'>
                <div style='font-size:16px;font-weight:700;color:#1A1A2E'>{r['ticker']}</div>
                <div style='font-size:10px;color:#888;margin:3px 0'>
                    {r['start_date']}<br>{r['end_date']}
                </div>
                <div style='margin:8px 0'>
                    <div style='font-size:10px;color:#888;letter-spacing:1px'>BENZERLİK</div>
                    <div style='font-size:26px;font-weight:700;color:{c}'>%{r['similarity']}</div>
                </div>
                <div style='margin:6px 0'>
                    <div style='font-size:10px;color:#888'>SONRASI</div>
                    <div style='font-size:20px;font-weight:700;color:{c_fut}'>{icon} {fut_pct:+.1f}%</div>
                </div>
                <div style='font-size:11px;color:#888'>
                    ↑{r['fut_max']:+.1f}% / ↓{r['fut_min']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            lbl = "✓ Seçili" if is_sel else "Detay →"
            if st.button(lbl, key=f"sel_{r['ticker']}_{i}", use_container_width=True,
                         type="primary" if is_sel else "secondary"):
                st.session_state["selected"] = None if is_sel else r['ticker']
                st.rerun()

    # ── Detay ──
    selected = st.session_state.get("selected")
    if selected:
        sel = next((r for r in matches if r['ticker'] == selected), None)
        if sel:
            st.divider()
            st.markdown(f"### 🔎 {selected} — Detay Analiz")

            tab1, tab2, tab3 = st.tabs([
                "📅 Tarihsel Konum",
                "🔍 Şablon Uyumu",
                "📈 Tüm Eşleşmeler"
            ])

            with tab1:
                st.plotly_chart(fig_history(sel, sym), use_container_width=True)
                st.caption(f"**{selected}** hissesinin tüm geçmişi. Sarı bölge eşleşen dönem, noktalı çizgi sonraki hareketi gösteriyor.")

            with tab2:
                st.plotly_chart(fig_compare(sel, template, sym), use_container_width=True)
                st.caption("Şablon (mavi) ve eşleşen bölge (sarı) normalize edilmiş halde. Noktalı çizgi o dönemden sonra ne olduğunu gösteriyor.")

            with tab3:
                st.plotly_chart(fig_normalize(template, matches, sym), use_container_width=True)
                st.caption("Tüm eşleşmeler üst üste. Dikey çizginin sağı geçmişteki 'sonraki hareket' — tahmin için referans.")

            # İstatistik kutusu
            st.markdown("#### 📋 Özet İstatistikler")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Benzerlik", f"%{sel['similarity']}")
            s2.metric("Sonraki Hareket", f"{sel['fut_pct']:+.1f}%")
            s3.metric("Maks. Kazanç", f"+{sel['fut_max']:.1f}%")
            s4.metric("Maks. Kayıp", f"{sel['fut_min']:.1f}%")
            s5.metric("Süre", f"{sel['fut_win']} gün")

    else:
        # Kimse seçilmediyse overlay göster
        st.divider()
        st.markdown("#### 📈 Normalize Karşılaştırma")
        st.plotly_chart(fig_normalize(template, matches, sym), use_container_width=True)

if __name__ == "__main__":
    main()
