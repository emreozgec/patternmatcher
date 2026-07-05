import streamlit as st
from formations import scan_all_formations, formation_summary_score
from bist_psi import BISTPSI, detect_regime
from scanner import render_scanner
from template_library import (
    render_library, save_template, init_library, get_library
)
from backtesting import render_backtest
from portfolio import render_portfolio, render_add_to_portfolio_button
from feedback import render_feedback_buttons, render_feedback_summary_page
from character_similarity import (
    build_character_profile, character_similarity,
    future_behavior_compatibility, historical_correlation, correlation_score
)
from performance import render_performance_dashboard
import db_utils

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests

# yfinance indirmelerinin engellenmesini önlemek için User-Agent tanımlı session oluştur
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})


st.set_page_config(
    page_title="BIST Pattern Matcher",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; background: transparent !important; }
header [data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1.5rem; }
.stButton > button { border-radius: 6px; font-weight: 500; }
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E9F0;
    border-radius: 8px;
    padding: 12px !important;
}
/* Mobilde sidebar açma okunu belirgin yap */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: block !important;
    background: #1A56DB !important;
    border-radius: 0 8px 8px 0 !important;
    opacity: 1 !important;
}
[data-testid="collapsedControl"] svg {
    fill: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# ── BIST listeleri ─────────────────────────────────────────────────────────────
BIST30 = ["AKBNK","ARCLK","ASELS","BIMAS","EKGYO","EREGL","FROTO","GARAN","HALKB",
          "ISCTR","KCHOL","KOZAA","KOZAL","KRDMD","MGROS","ODAS","PETKM","PGSUS",
          "SAHOL","SISE","SOKM","TAVHL","TCELL","THYAO","TKFEN","TOASO","TTKOM",
          "TUPRS","VAKBN","YKBNK"]

BIST100 = list(set(BIST30 + [
    "AEFES","AGESA","AKGRT","AKSA","AKSEN","ALARK","ALBRK","ALFAS","ANELE",
    "ANHYT","ANSGR","ASTOR","AYGAZ","BAGFS","BERA","BFREN","BRISA","BTCIM",
    "BUCIM","CIMSA","CLEBI","DEVA","DOHOL","ECILC","ECZYT","EDIP","EGEEN",
    "EGGUB","ENKAI","ERBOS","EUPWR","FENER","GENIL","GENTS","GEREL","GESAN",
    "GLYHO","GMTAS","GOODY","GOZDE","GRSEL","GUBRF","GWIND","HATEK","HEKTS",
    "HLGYO","HOROZ","HUBVC","HURGZ","INDES","INFO","INVEO","ISBIR","ISYAT",
    "IZENR","KAREL","KARSN","KARTN","KERVT","KLKIM","KLMSN","KONTR","KOPOL",
    "KORDS","LOGO","MAVI","MEDTR","NATEN","NETAS","NUHCM","OTKAR","OYAKC",
    "PETUN","PKART","POLHO","PRKAB","SARKY","SASA","SELEC","SELGD","SKBNK",
    "SKTAS","SOKM","TATGD","TCELL","TKFEN","TTRAK","TUPRS","ULUSE","VESBE","YUNSA"]))

ALL_BIST = list(set(BIST100 + [
    "ACSEL","ADEL","AGYO","AKFEN","AKMGY","ALKIM","ARDYZ","ARSAN","ATAGY",
    "ATEKS","ATLAS","AVGYO","AYCES","AYEN","BAGFS","BASGZ","BAYRK","BIENY",
    "BJKAS","BNTAS","BOBET","BORLS","BRKSN","BRKVY","BRMEN","BSOKE","BURCE",
    "BURVA","BVSAN","CANTE","CARFA","CEMAS","CEMTS","CEOEM","COSMO","CRDFA",
    "CRFSA","CUSAN","CVKMD","DAGHL","DARDL","DENGE","DGGYO","DITAS","DMSAS",
    "DNISI","DOBUR","DOCO","DOGUB","DOKTA","DURDO","DYOBY","DZGYO","EDATA",
    "EGPRO","EGSER","ELITE","EMKEL","EMNIS","EPLAS","ESCOM","ESEN","ETILR",
    "ETYAT","EUHOL","EUREN","EUYO","FLAP","FONET","FORMT","FORTE","FZLGY",
    "GARFA","GEDIK","GEDZA","GLBMD","GLRYH","GOKNR","GOLTS","GRTRK","GSDDE",
    "GSDHO","GSRAY","HDFGS","HEDEF","HKTM","HRKET","HTTBT","HUNER","ICBCT",
    "IDEAS","IDGYO","IHAAS","IHEVA","IHLAS","IHLGM","IHYAY","IMASM","INTEM",
    "ISATR","ITTFK","IZFAS","IZINV","IZMDC","KATMR","KAYSE","KCAER","KFEIN",
    "KGYO","KIMMR","KLGYO","KLRHO","KLSER","KMPUR","KNFRT","KONAK","KOTON",
    "KRONT","KRPLS","KRSTL","KRTEK","KRVGD","KSTUR","KTLEV","KTSK","KUTPO",
    "KUYAS","LIDER","LIDFA","LINK","LKMNH","LMKDC","LRSHO","LUKSK","MAALT",
    "MAGEN","MAKIM","MAKTK","MANAS","MARBL","MARKA","MARTI","MEGMT","MEPET",
    "MERCN","MERIT","MERKO","METRO","METUR","MIATK","MIPAZ","MMCAS","MNDRS",
    "MNDTR","MOBTL","MOGAN","MRDIN","MRGYO","MRSHL","MSGYO","MTRKS","MZHLD",
    "NIBAS","NILYT","NTHOL","NTTUR","NUGYO","OBAMS","OBASE","ODINE","OFSYM",
    "ONCSM","ORCAY","ORGE","ORMA","OSMEN","OSTIM","OYYAT","OZGYO","OZKGY",
    "OZRDN","OZSUB","PAGYO","PAMEL","PAPIL","PARSN","PASEU","PCKMT","PCYOT",
    "PEGYO","PENGD","PENTA","PINSU","PKENT","PLTUR","PNLSN","POLTK","PRDGS",
    "PRZMA","PSDTC","PTOFS","RTALB","RUBNS","RYGYO","SAMAT","SANFM","SANKO",
    "SAYAS","SDTTR","SEGYO","SEKFK","SEKUR","SELVA","SEYKM","SILVR","SNKRN",
    "SODSN","SONME","SRVGY","SUWEN","TBORG","TDGYO","TEKTU","TGSAS","TLMAN",
    "TMSN","TRCAS","TRGYO","TRILC","TSGYO","TUCLK","TUKAS","TUREX","TURGG",
    "TURSG","TZNGY","ULUFA","ULUUN","UNLU","USAK","UTPYA","UZERB","VAKFN",
    "VANGD","VBTYZ","VERUS","VKFYO","VKGYO","VRGYO","WNDMR","YATAS","YAYLA",
    "YGYO","YIGIT","YKSLN","ZEDUR","ZRGYO"]))

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: VERİ ÇEKME
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker(symbol, period="1y"):
    try:
        ticker = symbol if symbol.endswith(".IS") else symbol + ".IS"
        today = datetime.today().strftime('%Y-%m-%d')
        df = yf.download(ticker, period=period, end=today,
                         auto_adjust=True, progress=False, threads=False, session=session)
        if df.empty or len(df) < 10:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except:
        return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_batch(tickers, period="2y"):
    results = {}
    symbols = [t + ".IS" for t in tickers]
    try:
        today = datetime.today().strftime('%Y-%m-%d')
        raw = yf.download(symbols, period=period, end=today,
                          auto_adjust=True, group_by='ticker', progress=False, session=session)
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

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: TEKNİK İNDİKATÖR HESAPLAMA
# ══════════════════════════════════════════════════════════════════════════════

def zscore(arr):
    arr = np.array(arr, dtype=float)
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-9:
        return np.zeros_like(arr)
    return (arr - mu) / sigma

def daily_returns(prices):
    prices = np.array(prices, dtype=float)
    if len(prices) < 2:
        return np.zeros(1)
    return np.diff(prices) / (np.abs(prices[:-1]) + 1e-9)

def calc_rsi(prices, n=14):
    prices = np.array(prices, dtype=float)
    if len(prices) < n + 1:
        return np.full(len(prices), 50.0)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.full(len(prices), 50.0)
    ag = gains[:n].mean()
    al = losses[:n].mean()
    for i in range(n, len(deltas)):
        ag = (ag * (n-1) + gains[i]) / n
        al = (al * (n-1) + losses[i]) / n
        rs = ag / (al + 1e-9)
        rsi[i+1] = 100 - 100 / (1 + rs)
    return rsi

def calc_macd(prices, fast=12, slow=26, sig=9):
    prices = np.array(prices, dtype=float)
    def ema(x, n):
        k = 2/(n+1)
        e = [x[0]]
        for v in x[1:]:
            e.append(v*k + e[-1]*(1-k))
        return np.array(e)
    if len(prices) < slow:
        return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
    e12 = ema(prices, fast)
    e26 = ema(prices, slow)
    macd = e12 - e26
    signal = ema(macd, sig)
    hist = macd - signal
    return macd, signal, hist

def calc_volume_profile(volumes, prices, n_bins=5):
    """Hacim profilini normalize et — hangi fiyat seviyelerinde hacim yoğun"""
    volumes = np.array(volumes, dtype=float)
    prices = np.array(prices, dtype=float)
    if volumes.sum() < 1e-9:
        return np.zeros(n_bins)
    p_min, p_max = prices.min(), prices.max()
    if p_max == p_min:
        return np.zeros(n_bins)
    bins = np.linspace(p_min, p_max, n_bins + 1)
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (prices >= bins[i]) & (prices < bins[i+1])
        profile[i] = volumes[mask].sum()
    total = profile.sum()
    return profile / total if total > 0 else profile

def detect_formations(prices, volumes):
    """Geriye dönük uyumluluk için basit skor dict döndür."""
    prices = np.array(prices, dtype=float)
    volumes = np.array(volumes, dtype=float)
    formations = scan_all_formations(prices, volumes, min_confidence=40)
    summary = formation_summary_score(formations)
    return {
        'double_top': max((f.confidence/100 for f in formations if 'Double Top' in f.name), default=0.0),
        'double_bottom': max((f.confidence/100 for f in formations if 'Double Bottom' in f.name), default=0.0),
        'head_shoulders': max((f.confidence/100 for f in formations if 'Head' in f.name), default=0.0),
        'ascending_channel': max((f.confidence/100 for f in formations if 'Ascending' in f.name), default=0.0),
        'descending_channel': max((f.confidence/100 for f in formations if 'Descending' in f.name), default=0.0),
        'breakout_up': max((f.confidence/100 for f in formations if 'bullish' == f.direction and f.confidence > 60), default=0.0),
        'breakout_down': max((f.confidence/100 for f in formations if 'bearish' == f.direction and f.confidence > 60), default=0.0),
    }

def extract_features(prices, volumes):
    """
    Tüm özellikleri tek bir fonksiyonda hesapla.
    Returns: dict of feature arrays/scalars
    """
    prices = np.array(prices, dtype=float)
    volumes = np.array(volumes, dtype=float)
    n = len(prices)

    rets = daily_returns(prices)
    rsi = calc_rsi(prices)
    macd, signal, hist = calc_macd(prices)
    vol_norm = volumes / (volumes.mean() + 1e-9)
    formations = detect_formations(prices, volumes)

    return {
        'price_z': zscore(prices),
        'returns': rets,
        'returns_mean': rets.mean(),
        'returns_std': rets.std(),
        'returns_skew': float(pd.Series(rets).skew()),
        'rsi': rsi,
        'rsi_mean': rsi.mean(),
        'rsi_end': rsi[-1],
        'macd_hist': hist,
        'macd_trend': 1.0 if hist[-1] > 0 else -1.0,
        'volume_z': zscore(vol_norm),
        'volume_trend': np.polyfit(np.arange(n), vol_norm, 1)[0],
        'vol_profile': calc_volume_profile(volumes, prices),
        'formations': formations,
        'price_range': (prices.max() - prices.min()) / (prices.mean() + 1e-9),
    }

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: ÇOK BOYUTLU BENZERLİK MOTORU
# ══════════════════════════════════════════════════════════════════════════════

def dtw_similarity(s1, s2, band=None):
    """DTW mesafesi → 0-1 benzerlik skoru"""
    n = len(s1)
    if n == 0:
        return 0.0
    band = band or max(2, n // 8)
    dtw = np.full((n+1, n+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        j0 = max(1, i-band)
        j1 = min(n, i+band) + 1
        for j in range(j0, j1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    dist = dtw[n,n] / n
    return max(0.0, 1.0 - dist)

def pearson_sim(a, b):
    """Pearson korelasyonu → 0-1"""
    if len(a) != len(b) or len(a) < 3:
        return 0.5
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.5
    r = float(np.corrcoef(a, b)[0,1])
    return (r + 1) / 2

def formation_similarity(f1, f2):
    """İki formasyon seti arasındaki benzerlik"""
    keys = list(f1.keys())
    if not keys:
        return 0.5
    diffs = [abs(f1[k] - f2[k]) for k in keys]
    return 1.0 - min(1.0, np.mean(diffs))

def composite_score(feat_tpl, feat_win):
    """
    6 boyutlu kompozit benzerlik skoru.
    Her boyut 0-1 arası → ağırlıklı ortalama.

    Ağırlıklar (toplam=1.0):
    - Fiyat şekli (DTW):        0.28
    - Fiyat şekli (Pearson):    0.12
    - Günlük getiri dağılımı:   0.20
    - Hacim profili:            0.15
    - Momentum (RSI+MACD):      0.15
    - Formasyonlar:             0.10
    """
    # 1. Fiyat şekli — DTW
    s_dtw = dtw_similarity(feat_tpl['price_z'], feat_win['price_z'])

    # 2. Fiyat şekli — Pearson
    s_pearson = pearson_sim(feat_tpl['price_z'], feat_win['price_z'])

    # 3. Günlük getiri dağılımı
    # a) Getiri serisinin korelasyonu
    min_len = min(len(feat_tpl['returns']), len(feat_win['returns']))
    if min_len >= 3:
        s_ret_corr = pearson_sim(feat_tpl['returns'][:min_len], feat_win['returns'][:min_len])
    else:
        s_ret_corr = 0.5
    # b) İstatistik benzerliği (ortalama, std, skewness)
    s_ret_mean = 1 - min(1.0, abs(feat_tpl['returns_mean'] - feat_win['returns_mean']) / (abs(feat_tpl['returns_mean']) + 0.001))
    s_ret_std  = 1 - min(1.0, abs(feat_tpl['returns_std'] - feat_win['returns_std']) / (feat_tpl['returns_std'] + 0.001))
    s_ret_skew = 1 - min(1.0, abs(feat_tpl['returns_skew'] - feat_win['returns_skew']) / (abs(feat_tpl['returns_skew']) + 0.5))
    s_returns = 0.5 * s_ret_corr + 0.2 * s_ret_mean + 0.2 * s_ret_std + 0.1 * s_ret_skew

    # 4. Hacim profili
    s_vol_profile = pearson_sim(feat_tpl['vol_profile'], feat_win['vol_profile'])
    s_vol_trend = 1 - min(1.0, abs(feat_tpl['volume_trend'] - feat_win['volume_trend']))
    s_vol_shape = pearson_sim(feat_tpl['volume_z'], feat_win['volume_z'])
    s_volume = 0.4 * s_vol_profile + 0.3 * s_vol_trend + 0.3 * s_vol_shape

    # 5. Momentum (RSI + MACD)
    min_len2 = min(len(feat_tpl['rsi']), len(feat_win['rsi']))
    s_rsi_shape = pearson_sim(feat_tpl['rsi'][:min_len2], feat_win['rsi'][:min_len2]) if min_len2 >= 3 else 0.5
    s_rsi_level = 1 - min(1.0, abs(feat_tpl['rsi_end'] - feat_win['rsi_end']) / 50.0)
    min_len3 = min(len(feat_tpl['macd_hist']), len(feat_win['macd_hist']))
    s_macd = pearson_sim(feat_tpl['macd_hist'][:min_len3], feat_win['macd_hist'][:min_len3]) if min_len3 >= 3 else 0.5
    s_macd_dir = 1.0 if feat_tpl['macd_trend'] == feat_win['macd_trend'] else 0.0
    s_momentum = 0.3 * s_rsi_shape + 0.2 * s_rsi_level + 0.3 * s_macd + 0.2 * s_macd_dir

    # 6. Formasyonlar
    s_formation = formation_similarity(feat_tpl['formations'], feat_win['formations'])

    # Ağırlıklı kompozit
    score = (
        0.28 * s_dtw +
        0.12 * s_pearson +
        0.20 * s_returns +
        0.15 * s_volume +
        0.15 * s_momentum +
        0.10 * s_formation
    )
    return round(score * 100, 1), {
        'fiyat_dtw': round(s_dtw * 100, 1),
        'fiyat_pearson': round(s_pearson * 100, 1),
        'getiri': round(s_returns * 100, 1),
        'hacim': round(s_volume * 100, 1),
        'momentum': round(s_momentum * 100, 1),
        'formasyon': round(s_formation * 100, 1),
    }

def piecewise_zscore(arr: np.ndarray, segments: int = 4) -> np.ndarray:
    """
    OPT-2: Piecewise normalizasyon.
    Seriyi `segments` parçaya böler, her parçayı ayrı z-score yapar.
    Trend içindeki lokal benzerlikleri standart z-score'dan çok daha iyi yakalar.
    """
    arr = np.array(arr, dtype=float)
    n = len(arr)
    if n < segments * 2:
        return zscore(arr)
    result = np.zeros(n)
    seg_size = n // segments
    for i in range(segments):
        start = i * seg_size
        end = n if i == segments - 1 else (i + 1) * seg_size
        segment = arr[start:end]
        mu, sigma = segment.mean(), segment.std()
        result[start:end] = (segment - mu) / (sigma + 1e-9)
    return result


def hierarchical_search(tpl_z: np.ndarray, tpl_pz: np.ndarray,
                         closes: np.ndarray, n: int, max_i: int) -> tuple:
    """
    OPT-1: 3 geçişli hiyerarşik tarama.

    Geçiş 1 (çok hızlı): step=n//3 → top 5 bölge seç
    Geçiş 2 (orta):      step=n//8 → top 5 bölge etrafında ±2*step
    Geçiş 3 (hassas):    step=1    → en iyi bölge etrafında ±n//8

    Standart tek geçişe kıyasla ~3x daha az hesaplama, çok daha az atlama.
    """
    if max_i <= 0:
        return 0, -1.0

    # ── Geçiş 1: Kaba tarama ──────────────────────────────────────────────────
    step1 = max(1, n // 3)
    candidates_1 = []
    for i in range(0, max_i, step1):
        w_z = zscore(closes[i:i+n])
        # Sadece Pearson — çok hızlı
        p = pearson_sim(tpl_z, w_z) * 100
        candidates_1.append((i, p))
    candidates_1.sort(key=lambda x: x[1], reverse=True)
    top5 = [c[0] for c in candidates_1[:5]]

    # ── Geçiş 2: Orta tarama ─────────────────────────────────────────────────
    step2 = max(1, n // 8)
    candidates_2 = []
    seen = set()
    for center in top5:
        for i in range(max(0, center - step2*2), min(max_i+1, center + step2*2 + 1), step2):
            if i in seen:
                continue
            seen.add(i)
            w_z = zscore(closes[i:i+n])
            w_pz = piecewise_zscore(closes[i:i+n])
            # DTW + Pearson kombinasyonu
            d = dtw_similarity(tpl_z, w_z) * 100
            p = pearson_sim(tpl_pz, w_pz) * 100
            score = 0.6 * d + 0.4 * p
            candidates_2.append((i, score))
    if not candidates_2:
        return 0, -1.0
    candidates_2.sort(key=lambda x: x[1], reverse=True)
    best_mid_i = candidates_2[0][0]

    # ── Geçiş 3: Hassas tarama ────────────────────────────────────────────────
    step3 = max(1, n // 8)
    best_i, best_score = best_mid_i, -1.0
    for i in range(max(0, best_mid_i - step3), min(max_i+1, best_mid_i + step3 + 1)):
        w_z = zscore(closes[i:i+n])
        w_pz = piecewise_zscore(closes[i:i+n])
        d = dtw_similarity(tpl_z, w_z) * 100
        p = pearson_sim(tpl_pz, w_pz) * 100
        score = 0.6 * d + 0.4 * p
        if score > best_score:
            best_score, best_i = score, i

    return best_i, best_score


def regime_weight(tpl_regime: str, match_regime: str) -> float:
    """
    OPT-3: Koşullu rejim ağırlıklandırma.
    Şablon ile eşleşen dönemin rejimi aynıysa bonus ağırlık.
    Farklı rejimse ceza uygula.
    """
    if tpl_regime == match_regime:
        return 1.25   # %25 bonus
    # Yakın rejimler
    compatible = {
        'trend_bull': {'trend_bull', 'high_vol'},
        'trend_bear': {'trend_bear', 'high_vol'},
        'sideways':   {'sideways', 'low_vol'},
        'high_vol':   {'high_vol', 'trend_bull', 'trend_bear'},
        'low_vol':    {'low_vol', 'sideways'},
    }
    if match_regime in compatible.get(tpl_regime, set()):
        return 1.05   # %5 bonus
    return 0.85       # %15 ceza


def segment_similarity_map(tpl_z: np.ndarray, win_z: np.ndarray,
                            segments: int = 8) -> np.ndarray:
    """
    OPT-4: Segment bazlı benzerlik haritası.
    Şablon ve pencereyi `segments` parçaya böler.
    Her segment için Pearson benzerliği hesaplar.
    Grafik görselleştirmesi için 0-100 arası segment skorları döndürür.
    """
    n = len(tpl_z)
    seg_size = max(1, n // segments)
    scores = np.zeros(segments)
    for i in range(segments):
        s = i * seg_size
        e = n if i == segments - 1 else (i + 1) * seg_size
        t_seg = tpl_z[s:e]
        w_seg = win_z[s:e]
        if len(t_seg) >= 2 and len(w_seg) >= 2:
            scores[i] = (pearson_sim(t_seg, w_seg)) * 100
    return scores


def multi_timeframe_check(tpl_prices: np.ndarray, win_prices: np.ndarray) -> float:
    """
    OPT-5: Çoklu zaman dilimi kontrolü.
    Haftalık (5 gün ortalaması) ve aylık (20 gün ortalaması) agregasyonlar üzerinde
    ek benzerlik skoru hesaplar.
    Günlük gürültüden arındırılmış yapısal benzerliği ölçer.
    """
    def aggregate(arr, window):
        arr = np.array(arr, dtype=float)
        if len(arr) < window:
            return arr
        return np.array([arr[i:i+window].mean()
                         for i in range(0, len(arr)-window+1, window//2)])

    scores = []
    for w in [5, 10]:   # Haftalık ve iki haftalık
        t_agg = aggregate(tpl_prices, w)
        win_agg = aggregate(win_prices, w)
        if len(t_agg) >= 3 and len(win_agg) >= 3:
            min_len = min(len(t_agg), len(win_agg))
            t_z = zscore(t_agg[-min_len:])
            w_z = zscore(win_agg[-min_len:])
            scores.append(pearson_sim(t_z, w_z) * 100)

    return float(np.mean(scores)) if scores else 50.0


def find_patterns(template_prices, template_volumes, all_data,
                  top_n=5, min_sim=65, future_mult=1.5):
    """
    BIST-PSI Pattern Matching — 5 Optimizasyon ile:

    OPT-1: Hiyerarşik 3-geçişli tarama (daha az atlama, daha hızlı)
    OPT-2: Piecewise z-score normalizasyon (lokal trend benzerliği)
    OPT-3: Rejim bazlı koşullu ağırlıklandırma (aynı rejim → bonus)
    OPT-4: Segment benzerlik haritası (grafik için 8 segmentli analiz)
    OPT-5: Çoklu zaman dilimi kontrolü (günlük + haftalık + iki haftalık)
    """
    tpl_prices = np.array(template_prices, dtype=float)
    tpl_volumes = np.array(template_volumes, dtype=float)
    n = len(tpl_prices)
    fut_win = min(int(n * future_mult), 90)

    # Template özellikleri
    feat_tpl = extract_features(tpl_prices, tpl_volumes)
    tpl_z   = feat_tpl['price_z']
    tpl_pz  = piecewise_zscore(tpl_prices)          # OPT-2

    # Şablon rejimi (OPT-3 için)
    try:
        from bist_psi import detect_regime as _dr
        tpl_regime = _dr(tpl_prices, tpl_volumes).name
    except Exception:
        tpl_regime = 'sideways'

    results = []

    for ticker, df in all_data.items():
        closes  = df['Close'].values.astype(float)
        volumes = df['Volume'].values.astype(float)
        dates   = df.index

        if len(closes) < n + fut_win + 10:
            continue

        max_i = len(closes) - n - fut_win

        # OPT-1: Hiyerarşik tarama
        best_i, quick_score = hierarchical_search(tpl_z, tpl_pz, closes, n, max_i)
        if quick_score < min_sim * 0.75:
            continue

        # Tam kompozit skor
        w_prices  = closes[best_i:best_i+n]
        w_volumes = volumes[best_i:best_i+n]
        feat_win  = extract_features(w_prices, w_volumes)
        full_s, breakdown = composite_score(feat_tpl, feat_win)

        # OPT-5: Çoklu zaman dilimi bonusu
        mtf_score = multi_timeframe_check(tpl_prices, w_prices)
        full_s = full_s * 0.88 + mtf_score * 0.12

        # OPT-3: Rejim ağırlığı
        try:
            from bist_psi import detect_regime as _dr
            win_regime = _dr(w_prices, w_volumes).name
        except Exception:
            win_regime = 'sideways'
        reg_w = regime_weight(tpl_regime, win_regime)
        full_s = min(100.0, full_s * reg_w)

        breakdown['mtf_score'] = round(mtf_score, 1)
        breakdown['regime_match'] = win_regime == tpl_regime

        ms, me = best_i, best_i + n

        if full_s < min_sim * 0.80:   # Gevşek ön filtre — final skor sonra hesaplanacak
            continue
        match_closes  = closes[ms:me]
        match_volumes = volumes[ms:me]
        match_dates   = dates[ms:me]
        future_closes = closes[me:me+fut_win]
        future_dates  = dates[me:me+fut_win]

        fut_pct = fut_max = fut_min = 0.0
        if len(future_closes) > 1:
            fut_pct = (future_closes[-1] - future_closes[0]) / future_closes[0] * 100
            fut_max = (future_closes.max() - future_closes[0]) / future_closes[0] * 100
            fut_min = (future_closes.min() - future_closes[0]) / future_closes[0] * 100

        # OPT-4: Segment benzerlik haritası
        win_z = zscore(match_closes)
        seg_map = segment_similarity_map(tpl_z, win_z, segments=8)

        # Karakter benzerliği ve gelecek uyumluluğu
        char_score = 50.0
        fut_compat = 50.0
        corr_score = 50.0
        try:
            tpl_profile = build_character_profile(tpl_prices, tpl_volumes)
            win_profile = build_character_profile(w_prices, w_volumes)
            char_s, char_bd = character_similarity(tpl_profile, win_profile)
            char_score = char_s

            if len(future_closes) > 3:
                fut_compat = future_behavior_compatibility(
                    tpl_prices, future_closes, tpl_profile, win_profile
                )

            corr = historical_correlation(tpl_prices, w_prices)
            corr_score = correlation_score(corr)
        except Exception:
            char_bd = {}

        # Final skora karakter, gelecek uyumu ve korelasyonu dahil et
        # %75 mevcut skor + %10 karakter + %10 gelecek uyumu + %5 korelasyon
        full_s = (0.75 * full_s +
                  0.10 * char_score +
                  0.10 * fut_compat +
                  0.05 * corr_score)
        full_s = round(min(100.0, full_s), 1)

        if full_s < min_sim:
            continue

        breakdown['karakter']    = round(char_score, 1)
        breakdown['gelecek_uyum'] = round(fut_compat, 1)
        breakdown['korelasyon']  = round(corr_score, 1)

        results.append({
            'ticker':       ticker,
            'similarity':   full_s,
            'breakdown':    breakdown,
            'seg_map':      seg_map,
            'regime_match': win_regime == tpl_regime,
            'mtf_score':    round(mtf_score, 1),
            'char_score':   round(char_score, 1),
            'fut_compat':   round(fut_compat, 1),
            'corr_score':   round(corr_score, 1),
            'ms': ms, 'me': me,
            'match_closes':  match_closes,
            'match_volumes': match_volumes,
            'match_dates':   match_dates,
            'future_closes': future_closes,
            'future_dates':  future_dates,
            'fut_pct':  round(fut_pct, 2),
            'fut_max':  round(fut_max, 2),
            'fut_min':  round(fut_min, 2),
            'fut_win':  fut_win,
            'all_closes': closes,
            'all_dates':  dates,
            'start_date': pd.Timestamp(match_dates[0]).strftime('%d.%m.%Y'),
            'end_date':   pd.Timestamp(match_dates[-1]).strftime('%d.%m.%Y'),
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)

    # ── Tarih Çeşitliliği Filtresi ──────────────────────────────────────────
    # Eğer çok sayıda sonuç aynı dar tarih aralığına yığılmışsa (genel piyasa
    # hareketi sinyali), bu kümeden sadece en iyi 1-2 sonucu tut ve diğer
    # tarihlerdeki eşleşmelere de yer aç — sonuç çeşitliliğini artır.
    if len(results) > top_n:
        from collections import defaultdict
        date_clusters = defaultdict(list)
        for r in results:
            # Ayı bazlı kümeleme (aynı ay-yıl aynı küme sayılır)
            cluster_key = r['start_date'][3:]  # "GG.AA.YYYY" -> "AA.YYYY"
            date_clusters[cluster_key].append(r)

        diversified = []
        cluster_take = {k: 0 for k in date_clusters}
        max_per_cluster = max(1, top_n // 3)  # Aynı dönemden en fazla bu kadar al

        for r in results:
            cluster_key = r['start_date'][3:]
            if cluster_take[cluster_key] < max_per_cluster:
                diversified.append(r)
                cluster_take[cluster_key] += 1
            if len(diversified) >= top_n:
                break

        # Yeterli çeşitlilik bulunamadıysa orijinal sıralamadan tamamla
        if len(diversified) < top_n:
            remaining = [r for r in results if r not in diversified]
            diversified.extend(remaining[:top_n - len(diversified)])

        results = diversified
        results.sort(key=lambda x: x['similarity'], reverse=True)

    return results[:top_n]

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3B: KONSENSÜS ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════

def calc_consensus(matches, current_price):
    """
    Ağırlıklı konsensüs analizi.
    Grid Search ile doğrulanmış sonuçlar uygulanmış (2 bağımsız test, tutarlı):
    - Window=30 gün şablonlar en iyi performansı veriyor
    - PSI ~65 civarı optimal (son dönem testinde doğrulandı — Sharpe 6.73).
      Yüksek PSI (75+) son dönemde performansı sert düşürüyor (anti-consensus
      etkisi güçlenmiş), bu yüzden aşırı yüksek PSI'a ekstra bonus verilmiyor
    - Güven bandı 50-65 optimal (eskiden sanılan 55-68'den daha geniş ve iyi)
    """
    if not matches:
        return None

    # Temel ağırlıklar: PSI skoru
    raw_weights = np.array([r['similarity'] for r in matches], dtype=float)

    # Anti-consensus düzeltmesi — Güncel Grid Search doğrulaması (son dönem):
    # PSI 60-70 bandı en iyi performansı veriyor (Sharpe 6.73 @ PSI=65).
    # PSI arttıkça performans SERT düşüyor (70→Sharpe 3.1, 75→Sharpe 0.8).
    # Bu yüzden artık düşük-orta PSI'a en yüksek bonus, yüksek PSI'a ceza var.
    psi_bonus = np.array([
        1.20 if 60 <= r['similarity'] < 72 else   # Doğrulanmış optimal bant
        0.90 if r['similarity'] >= 72 else          # Yüksek PSI — performans düşüyor
        0.85                                         # Çok düşük PSI — ceza
        for r in matches
    ], dtype=float)

    weights = raw_weights * psi_bonus
    weights = weights / (weights.sum() + 1e-9)

    pcts = np.array([r['fut_pct'] for r in matches], dtype=float)
    maxs = np.array([r['fut_max'] for r in matches], dtype=float)
    mins = np.array([r['fut_min'] for r in matches], dtype=float)

    weighted_pct = float(np.dot(weights, pcts))
    weighted_max = float(np.dot(weights, maxs))
    weighted_min = float(np.dot(weights, mins))
    target_high  = current_price * (1 + weighted_max / 100)
    target_low   = current_price * (1 + weighted_min / 100)

    up_weight   = float(sum(w for w, p in zip(weights, pcts) if p > 0))
    down_weight = float(sum(w for w, p in zip(weights, pcts) if p <= 0))
    up_count    = int(sum(1 for p in pcts if p > 0))
    down_count  = int(sum(1 for p in pcts if p <= 0))

    if up_weight > 0.6:
        direction = "YÜKSELİŞ"
    elif down_weight > 0.6:
        direction = "DÜŞÜŞ"
    else:
        direction = "KARARSIZ"

    # Güven skoru — Grid Search doğrulaması: 50-65 bandı optimal
    direction_conf    = max(up_weight, down_weight) * 100
    dispersion        = float(np.std(pcts))
    dispersion_penalty = min(40, dispersion * 2)
    avg_sim           = float(np.dot(weights, [r['similarity'] for r in matches]))
    sim_bonus         = max(0, (avg_sim - 65) / 35 * 20)

    confidence = max(0, min(100,
        direction_conf - dispersion_penalty + sim_bonus))

    # Optimal bant göstergesi — doğrulanmış: 50-65
    in_optimal_band = 50 <= confidence <= 65
    band_label = "✅ Optimal Bant" if in_optimal_band else (
        "⚠️ Yüksek Güven (anti-consensus)" if confidence > 65 else
        "⚠️ Düşük Güven"
    )

    return {
        'direction':       direction,
        'confidence':      round(confidence, 1),
        'band_label':      band_label,
        'in_optimal_band': in_optimal_band,
        'weighted_pct':    round(weighted_pct, 2),
        'target_high':     round(target_high, 2),
        'target_low':      round(target_low, 2),
        'target_pct_high': round(weighted_max, 2),
        'target_pct_low':  round(weighted_min, 2),
        'agreement_ratio': f"{max(up_count,down_count)}/{len(matches)}",
        'up_count':        up_count,
        'down_count':      down_count,
        'dispersion':      round(dispersion, 2),
        'avg_similarity':  round(avg_sim, 1),
        'individual': [
            {
                'ticker': r['ticker'],
                'pct':    r['fut_pct'],
                'weight': round(float(w)*100, 1),
                'sim':    r['similarity'],
                'psi_bonus': '⭐' if r['similarity'] >= 80 else ''
            }
            for r, w in zip(matches, weights)
        ]
    }


def fig_consensus_chart(consensus, matches, template_closes, symbol):
    """Fan grafiği — ağırlıklı ortalama + bireysel senaryolar"""
    n = len(template_closes)
    tpl_z = zscore(np.array(template_closes, dtype=float))

    fig = go.Figure()

    # Gri arka plan senaryolar
    for i, r in enumerate(matches):
        if len(r['future_closes']) > 1:
            seg_z = zscore(r['match_closes'])
            last = float(seg_z[-1])
            fut_z = zscore(r['future_closes'])
            fut_s = [last + v * 0.35 for v in fut_z]
            x_fut = list(range(n, n + len(fut_s)))
            c = '#0E9F6E' if r['fut_pct'] >= 0 else '#E02424'
            fig.add_trace(go.Scatter(
                x=x_fut, y=fut_s,
                name=f"{r['ticker']} ({r['fut_pct']:+.1f}%)",
                line=dict(color=c, width=1.2, dash='dot'),
                opacity=0.4,
                hovertemplate=f"{r['ticker']}: %{{y:.2f}}<extra></extra>"
            ))

    # Şablon
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=tpl_z,
        name=f'{symbol} (Şablon)',
        line=dict(color='#1A1A2E', width=3),
        hovertemplate='Şablon: %{y:.2f}<extra></extra>'
    ))

    # Ağırlıklı konsensüs çizgisi
    # En uzun future uzunluğunu bul
    max_fut = max((len(r['future_closes']) for r in matches if len(r['future_closes']) > 1), default=0)
    if max_fut > 1:
        weights = np.array([r['similarity'] for r in matches], dtype=float)
        weights = weights / weights.sum()

        # Her t anı için ağırlıklı ortalama normalize fiyat
        consensus_line = []
        seg_z_last = float(zscore(matches[0]['match_closes'])[-1])

        for t in range(max_fut):
            vals = []
            ws = []
            for r, w in zip(matches, weights):
                if len(r['future_closes']) > t + 1:
                    seg_z = zscore(r['match_closes'])
                    last = float(seg_z[-1])
                    fut_z = zscore(r['future_closes'])
                    if t < len(fut_z):
                        vals.append(last + fut_z[t] * 0.35)
                        ws.append(w)
            if vals:
                ws_arr = np.array(ws)
                ws_arr = ws_arr / ws_arr.sum()
                consensus_line.append(float(np.dot(ws_arr, vals)))

        if consensus_line:
            x_cons = list(range(n, n + len(consensus_line)))
            c_main = '#0E9F6E' if consensus['weighted_pct'] >= 0 else '#E02424'
            fig.add_trace(go.Scatter(
                x=x_cons, y=consensus_line,
                name=f'Konsensüs ({consensus["weighted_pct"]:+.1f}%)',
                line=dict(color=c_main, width=3, dash='solid'),
                hovertemplate=f'Konsensüs: %{{y:.2f}}<extra></extra>'
            ))

    fig.add_vline(x=n - 0.5, line_dash='dash',
                  line_color='rgba(0,0,0,0.2)', line_width=2,
                  annotation_text='← Geçmiş | Konsensüs →',
                  annotation_font_color='#555', annotation_font_size=10)
    fig.add_vrect(x0=0, x1=n-1, fillcolor='rgba(227,160,8,0.04)', line_width=0)

    layout = base_layout(360, 'Konsensüs Fan Grafiği — Ağırlıklı Ortalama + Bireysel Senaryolar')
    layout['xaxis']['title'] = 'Gün'
    layout['yaxis']['title'] = 'Z-Score'
    fig.update_layout(**layout)
    return fig



# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: GRAFİKLER
# ══════════════════════════════════════════════════════════════════════════════

COLORS = ['#1A56DB','#E3A008','#0E9F6E','#9061F9','#E02424']

def base_layout(height=360, title=""):
    return dict(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode='x unified',
        height=height,
        legend=dict(orientation='h', y=1.12, font=dict(size=11)),
        title=dict(text=title, font=dict(size=13, color='#1A1A2E')),
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)', showgrid=True,
                   tickfont=dict(size=10)),
        yaxis=dict(gridcolor='rgba(0,0,0,0.05)', showgrid=True,
                   tickfont=dict(size=10)),
    )


def fig_formation_chart(df, symbol, formations):
    """Formasyonları grafik üzerinde işaretle."""
    import plotly.graph_objects as go
    dates = [d.strftime('%Y-%m-%d') for d in df.index]
    closes = df['Close'].values
    volumes = df['Volume'].values

    fig = go.Figure()

    # Ana fiyat çizgisi
    fig.add_trace(go.Scatter(
        x=dates, y=closes, name=symbol,
        line=dict(color='#1A56DB', width=2),
        hovertemplate='%{x}: %{y:.2f} ₺<extra></extra>'
    ))

    # Formasyon renkleri
    cat_colors = {
        'klasik': '#E3A008',
        'trend': '#0E9F6E',
        'elliott': '#9061F9',
        'harmonik': '#E02424',
        'wyckoff': '#1A56DB',
        'psi_consensus': '#0D9488',
    }

    dir_colors = {
        'bullish': '#0E9F6E',
        'bearish': '#E02424',
        'neutral': '#888',
    }

    for i, f in enumerate(formations[:8]):  # Max 8 formasyon göster
        c_color = cat_colors.get(f.category, '#888')
        d_color = dir_colors.get(f.direction, '#888')

        if not f.key_points:
            continue

        # Key point'leri grafik üzerinde işaretle
        kp_valid = [(idx, price, label)
                    for idx, price, label in f.key_points
                    if 0 <= idx < len(dates)]

        if len(kp_valid) >= 2:
            kp_x = [dates[idx] for idx, _, _ in kp_valid]
            kp_y = [price for _, price, _ in kp_valid]
            kp_labels = [label for _, _, label in kp_valid]

            # Formasyon çizgisi
            fig.add_trace(go.Scatter(
                x=kp_x, y=kp_y,
                mode='lines+markers+text',
                name=f"{f.name} (%{f.confidence:.0f})",
                line=dict(color=c_color, width=1.5, dash='dot'),
                marker=dict(size=8, color=c_color, symbol='circle'),
                text=kp_labels,
                textposition='top center',
                textfont=dict(size=9, color=c_color),
                hovertemplate=f"{f.name}: %{{y:.2f}}<extra></extra>",
                showlegend=True
            ))

        # Hedef çizgisi
        if f.target and kp_valid:
            last_idx = kp_valid[-1][0]
            if last_idx < len(dates) - 1:
                future_idx = min(last_idx + len(dates)//5, len(dates)-1)
                fig.add_trace(go.Scatter(
                    x=[dates[last_idx], dates[future_idx]],
                    y=[f.target, f.target],
                    mode='lines',
                    name=f"Hedef: {f.target:.2f}",
                    line=dict(color=d_color, width=1, dash='dash'),
                    showlegend=False,
                    hovertemplate=f"Hedef: {f.target:.2f}<extra></extra>"
                ))

    layout = base_layout(420, f'<b>{symbol}</b> — Formasyon Analizi')
    layout['xaxis'] = dict(type='date', tickformat='%b %Y', tickangle=-30,
                           gridcolor='rgba(0,0,0,0.05)', tickfont=dict(size=10))
    layout['yaxis'] = dict(gridcolor='rgba(0,0,0,0.05)', ticksuffix=' ₺',
                           tickfont=dict(size=10))
    layout['legend'] = dict(orientation='v', x=1.01, y=1, font=dict(size=9),
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='#E5E9F0', borderwidth=1)
    layout['margin'] = dict(l=10, r=180, t=45, b=10)
    fig.update_layout(**layout)
    return fig

def fig_main_chart(df, symbol, sel_start=None, sel_end=None):
    dates = [d.strftime('%Y-%m-%d') for d in df.index]
    closes = df['Close'].values
    volumes = df['Volume'].values

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Scatter(x=dates, y=closes, name=symbol,
        line=dict(color='#1A56DB', width=2),
        hovertemplate='%{x}: %{y:.2f} ₺<extra></extra>'), row=1, col=1)

    if sel_start and sel_end:
        mask = (df.index >= pd.Timestamp(sel_start)) & (df.index <= pd.Timestamp(sel_end))
        seg_df = df[mask]
        if len(seg_df) > 0:
            seg_dates = [d.strftime('%Y-%m-%d') for d in seg_df.index]
            fig.add_trace(go.Scatter(x=seg_dates, y=seg_df['Close'].values,
                name='Şablon', line=dict(color='#E3A008', width=3.5),
                hovertemplate='%{x}: %{y:.2f} ₺<extra>Şablon</extra>'), row=1, col=1)
            fig.add_vrect(x0=seg_dates[0], x1=seg_dates[-1],
                fillcolor='rgba(227,160,8,0.07)', line_width=0,
                annotation_text='Şablon', annotation_position='top left',
                annotation_font_color='#E3A008', annotation_font_size=10)

    # Hacim
    vol_colors = ['rgba(14,159,110,0.5)' if closes[i] >= closes[i-1]
                  else 'rgba(224,36,36,0.5)' for i in range(len(closes))]
    fig.add_trace(go.Bar(x=dates, y=volumes, name='Hacim',
        marker_color=vol_colors, showlegend=False,
        hovertemplate='%{y:,.0f}<extra>Hacim</extra>'), row=2, col=1)

    layout = base_layout(400, f'<b>{symbol}</b> — Şablon Seçimi')
    layout['xaxis2'] = dict(type='date', tickformat='%b %Y', tickangle=-30, tickfont=dict(size=10))
    layout['yaxis'] = dict(gridcolor='rgba(0,0,0,0.05)', ticksuffix=' ₺', tickfont=dict(size=10))
    layout['yaxis2'] = dict(gridcolor='rgba(0,0,0,0.05)', tickfont=dict(size=9))
    fig.update_layout(**layout)
    return fig

def fig_history(result, symbol):
    closes = result['all_closes']
    dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in result['all_dates']]
    ms, me = result['ms'], result['me']
    fut = result['future_closes']
    fut_pct = result['fut_pct']
    c_fut = '#0E9F6E' if fut_pct >= 0 else '#E02424'
    icon = '▲' if fut_pct >= 0 else '▼'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=closes, name=result['ticker'],
        line=dict(color='rgba(100,130,180,0.4)', width=1.5),
        hovertemplate='%{x}: %{y:.2f} ₺<extra></extra>'))
    fig.add_trace(go.Scatter(x=dates[ms:me], y=closes[ms:me],
        name=f'Eşleşen Bölge (%{result["similarity"]})',
        line=dict(color='#E3A008', width=3.5),
        hovertemplate='%{x}: %{y:.2f} ₺<extra>Eşleşen</extra>'))

    if len(fut) > 1:
        fut_dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in result['future_dates']]
        fig.add_trace(go.Scatter(x=fut_dates, y=fut,
            name=f'Sonraki Hareket ({fut_pct:+.1f}%)',
            line=dict(color=c_fut, width=2.5, dash='dot'),
            hovertemplate='%{x}: %{y:.2f} ₺<extra>Sonrası</extra>'))

    if ms < me and me < len(dates):
        fig.add_vrect(x0=dates[ms], x1=dates[me-1],
            fillcolor='rgba(227,160,8,0.08)', line_width=0,
            annotation_text=f'Eşleşme\n{result["start_date"]}',
            annotation_position='top left',
            annotation_font_color='#E3A008', annotation_font_size=9)
        if len(fut) > 1 and me < len(dates):
            end_idx = min(me + len(fut) - 1, len(dates)-1)
            fig.add_vrect(x0=dates[me], x1=dates[end_idx],
                fillcolor=f'{"rgba(14,159,110,0.06)" if fut_pct>=0 else "rgba(224,36,36,0.06)"}',
                line_width=0,
                annotation_text=f'{icon} {fut_pct:+.1f}%',
                annotation_position='top right',
                annotation_font_color=c_fut, annotation_font_size=9)

    layout = base_layout(340, f'<b>{result["ticker"]}</b> — Tarihsel Konum | '
                              f'Eşleşme: <b>{result["start_date"]} → {result["end_date"]}</b>')
    layout['xaxis'] = dict(type='date', tickformat='%b %Y', tickangle=-30,
                           gridcolor='rgba(0,0,0,0.05)', tickfont=dict(size=10))
    layout['yaxis'] = dict(gridcolor='rgba(0,0,0,0.05)', ticksuffix=' ₺', tickfont=dict(size=10))
    fig.update_layout(**layout)
    return fig

def fig_normalize(template_prices, results, symbol):
    tpl_z = zscore(np.array(template_prices, dtype=float))
    n = len(tpl_z)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n)), y=tpl_z,
        name=f'{symbol} (Şablon)', line=dict(color='#1A1A2E', width=3),
        hovertemplate='Şablon: %{y:.2f}<extra></extra>'))

    for i, r in enumerate(results):
        c = COLORS[i % len(COLORS)]
        seg_z = zscore(r['match_closes'])
        fig.add_trace(go.Scatter(x=list(range(n)), y=seg_z,
            name=f"{r['ticker']} (%{r['similarity']})",
            line=dict(color=c, width=1.8, dash='dot'), opacity=0.85,
            hovertemplate=f"{r['ticker']}: %{{y:.2f}}<extra></extra>"))
        if len(r['future_closes']) > 2:
            fut = r['future_closes']
            last = float(seg_z[-1])
            fut_z = zscore(fut)
            fut_s = [last + v * 0.35 for v in fut_z]
            x_fut = list(range(n, n+len(fut_s)))
            fig.add_trace(go.Scatter(x=x_fut, y=fut_s,
                name=f"{r['ticker']} sonrası ({r['fut_pct']:+.1f}%)",
                line=dict(color=c, width=1.5, dash='longdash'), opacity=0.5,
                hovertemplate=f"{r['ticker']} sonrası: %{{y:.2f}}<extra></extra>"))

    fig.add_vline(x=n-0.5, line_dash='dash', line_color='rgba(0,0,0,0.2)',
                  line_width=2, annotation_text='← Geçmiş | Tahmin →',
                  annotation_font_color='#555', annotation_font_size=10)
    fig.add_vrect(x0=0, x1=n-1, fillcolor='rgba(227,160,8,0.04)', line_width=0)

    layout = base_layout(380, 'Normalize Karşılaştırma — Eşleşen Bölgeler + Sonraki Hareketler')
    layout['xaxis']['title'] = 'Gün'
    layout['yaxis']['title'] = 'Z-Score'
    fig.update_layout(**layout)
    return fig


def fig_segment_map(seg_map: np.ndarray, ticker: str, symbol: str):
    """OPT-4: Segment bazlı benzerlik haritası grafiği."""
    import plotly.graph_objects as go
    n = len(seg_map)
    labels = [f"Seg {i+1}" for i in range(n)]
    colors = ['#0E9F6E' if s >= 70 else '#E3A008' if s >= 50 else '#E02424'
              for s in seg_map]
    fig = go.Figure(go.Bar(
        x=labels, y=seg_map,
        marker_color=colors,
        text=[f"%{s:.0f}" for s in seg_map],
        textposition='outside',
        hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
    ))
    fig.add_hline(y=70, line_dash='dash', line_color='rgba(14,159,110,0.5)',
                  annotation_text='İyi Eşleşme (70)', annotation_font_size=10)
    fig.add_hline(y=50, line_dash='dot', line_color='rgba(227,160,8,0.5)',
                  annotation_text='Orta (50)', annotation_font_size=10)
    layout = base_layout(240,
        f'Segment Benzerlik Haritası — {symbol} vs {ticker}')
    layout['yaxis']['range'] = [0, 110]
    layout['yaxis']['title'] = 'Benzerlik %'
    layout['showlegend'] = False
    fig.update_layout(**layout)
    return fig

def fig_compare(result, template_prices, symbol):
    tpl_z = zscore(np.array(template_prices, dtype=float))
    seg_z = zscore(result['match_closes'])
    n = len(tpl_z)
    fut_pct = result['fut_pct']
    c_fut = '#0E9F6E' if fut_pct >= 0 else '#E02424'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n)), y=tpl_z,
        name=f'{symbol} (Şablon)', line=dict(color='#1A56DB', width=2.5),
        hovertemplate='Şablon: %{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=list(range(n)), y=seg_z,
        name=f"{result['ticker']} — Eşleşen (%{result['similarity']})",
        line=dict(color='#E3A008', width=2, dash='dot'),
        hovertemplate=f"{result['ticker']}: %{{y:.2f}}<extra></extra>"))

    if len(result['future_closes']) > 2:
        fut = result['future_closes']
        last = float(seg_z[-1])
        fut_z = zscore(fut)
        fut_s = [last + v * 0.35 for v in fut_z]
        x_fut = list(range(n, n+len(fut_s)))
        fig.add_trace(go.Scatter(x=x_fut, y=fut_s,
            name=f'Sonraki Hareket ({fut_pct:+.1f}%)',
            line=dict(color=c_fut, width=2, dash='longdash'),
            hovertemplate=f'Sonrası: %{{y:.2f}}<extra></extra>'))
        fig.add_vline(x=n-0.5, line_dash='dash', line_color='rgba(0,0,0,0.15)',
                      line_width=1.5,
                      annotation_text=f'{"▲" if fut_pct>=0 else "▼"} {fut_pct:+.1f}%',
                      annotation_font_color=c_fut, annotation_font_size=11)

    layout = base_layout(280, f'Şablon Uyumu — {symbol} vs {result["ticker"]}')
    layout['xaxis']['title'] = 'Gün'
    layout['yaxis']['title'] = 'Z-Score'
    fig.update_layout(**layout)
    return fig

def fig_breakdown_radar(breakdown, ticker):
    """Benzerlik boyutlarını radar grafiğiyle göster"""
    labels = ['Fiyat (DTW)', 'Fiyat (Pearson)', 'Getiri', 'Hacim', 'Momentum', 'Formasyon']
    values = [
        breakdown.get('fiyat_dtw', 0),
        breakdown.get('fiyat_pearson', 0),
        breakdown.get('getiri', 0),
        breakdown.get('hacim', 0),
        breakdown.get('momentum', 0),
        breakdown.get('formasyon', 0),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(26,86,219,0.15)',
        line=dict(color='#1A56DB', width=2),
        name=ticker
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        showlegend=False,
        height=280,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f'{ticker} — Benzerlik Profili', font=dict(size=12, color='#1A1A2E'))
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: ANA UYGULAMA
# ══════════════════════════════════════════════════════════════════════════════

def render_telegram_setup():
    """Telegram bildirim kurulum rehberi ve test mesajı gönderme."""
    st.markdown("## 🔔 Telegram Bildirimleri")
    st.caption(
        "GitHub Actions her gün BIST kapanışından sonra otomatik tarama yapar "
        "ve sonuçları bu Telegram botuna gönderir. Streamlit kapalı olsa bile çalışır."
    )
    st.divider()

    st.markdown("### 1️⃣ Telegram Bot Oluştur")
    st.markdown("""
    1. Telegram'da **@BotFather** ile sohbet açın
    2. `/newbot` komutunu gönderin
    3. Bot için bir isim ve kullanıcı adı belirleyin (örn: `bist_pattern_bot`)
    4. BotFather size bir **token** verecek — örnek: `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`
    5. Bu token'ı kopyalayın, sonraki adımda kullanacaksınız
    """)

    st.markdown("### 2️⃣ Chat ID'nizi Öğrenin")
    st.markdown("""
    1. Oluşturduğunuz bota Telegram'dan bir mesaj gönderin (örn: "merhaba")
    2. Tarayıcıda şu adresi açın (TOKEN yerine kendi token'ınızı yazın):
       ```
       https://api.telegram.org/botTOKEN/getUpdates
       ```
    3. Çıkan JSON içinde `"chat":{"id": 123456789, ...}` kısmındaki sayıyı bulun
    4. Bu sayı sizin **Chat ID**'niz
    """)

    st.markdown("### 3️⃣ GitHub Secrets'a Ekle")
    st.markdown("""
    1. GitHub reponuzda **Settings → Secrets and variables → Actions** sekmesine gidin
    2. **"New repository secret"** ile iki secret ekleyin:
       - İsim: `TELEGRAM_BOT_TOKEN` — Değer: BotFather'dan aldığınız token
       - İsim: `TELEGRAM_CHAT_ID` — Değer: 2. adımda bulduğunuz chat ID
    3. Kaydedin
    """)

    st.markdown("### 4️⃣ Otomasyonu Aktif Et")
    st.markdown("""
    `daily_scan.py` ve `.github/workflows/daily_scan.yml` dosyaları reponuza zaten eklendi.

    - Varsayılan çalışma saati: **Hafta içi 18:30 (İstanbul saati)** — BIST kapanışından sonra
    - Saat değiştirmek isterseniz `.github/workflows/daily_scan.yml` dosyasındaki
      `cron: '30 15 * * 1-5'` satırını düzenleyin (UTC saatine göre yazılır)
    - **Actions** sekmesinden **"Run workflow"** ile istediğiniz an manuel de tetikleyebilirsiniz
    """)

    st.divider()
    st.markdown("### 🧪 Bağlantıyı Test Et")
    st.caption(
        "Token ve Chat ID'yi burada girip test mesajı gönderebilirsiniz. "
        "Bu sadece test amaçlıdır — gerçek otomasyon GitHub Actions üzerinden çalışır, "
        "buradaki bilgiler kaydedilmez."
    )

    tc1, tc2 = st.columns(2)
    test_token = tc1.text_input("Bot Token (test için)", type="password")
    test_chat_id = tc2.text_input("Chat ID (test için)")

    if st.button("📤 Test Mesajı Gönder", type="primary"):
        if not test_token or not test_chat_id:
            st.warning("Token ve Chat ID girin.")
        else:
            try:
                import requests
                url = f"https://api.telegram.org/bot{test_token}/sendMessage"
                payload = {
                    "chat_id": test_chat_id,
                    "text": "✅ BIST Pattern Matcher botu başarıyla bağlandı! "
                            "Günlük taramalar bu sohbete gelecek.",
                }
                resp = requests.post(url, json=payload, timeout=15)
                if resp.status_code == 200:
                    st.success("✅ Test mesajı gönderildi! Telegram'ı kontrol edin.")
                else:
                    st.error(f"❌ Hata: {resp.status_code} — {resp.text}")
            except Exception as e:
                st.error(f"❌ Bağlantı hatası: {e}")

    st.divider()
    st.markdown("### ⚙️ Otomasyon Parametreleri")
    st.info(
        "`daily_scan.py` dosyasında şu parametreler kullanılıyor (Grid Search "
        "ile 2 bağımsız testte doğrulanmış):\n\n"
        "- **Min BIST-PSI:** 65 (Güncel Grid Search — son dönem: Sharpe 6.73, %67 kazanç oranı)\n"
        "- **Güven Bandı:** %50-65 (Grid Search: %69 kazanç oranı, +31.2% toplam getiri)\n"
        "- **Kapsam:** BIST 100 (GitHub Actions süre limiti için)\n"
        "- **Şablon Uzunlukları:** 20 ve 30 gün (30 gün, 40 günden kategorik olarak üstün çıktı)\n\n"
        "Bu değerleri değiştirmek isterseniz `daily_scan.py` dosyasının başındaki "
        "`MIN_SIM`, `MIN_CONFIDENCE`, `MAX_CONFIDENCE`, `SCAN_SCOPE` değişkenlerini düzenleyin."
    )


def main():
    # Veritabanını ilklendir
    try:
        db_utils.init_db()
    except Exception as e:
        print(f"⚠️ Veritabanı ilklendirilirken hata: {e}")

    _pages = ["🔍 Pattern Matcher", "🔭 Fırsat Tarayıcı",
              "📈 Sinyal Performansı",
              "📚 Şablon Kütüphanesi", "📊 Backtesting",
              "🗳️ Geri Bildirim",
              "🔔 Telegram Bildirimleri"]



    _goto = st.session_state.pop('_goto_page', None)
    if _goto and _goto in _pages:
        st.session_state['current_page'] = _goto
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = _pages[0]

    current_idx = _pages.index(st.session_state['current_page'])

    def _on_page_change_mobile():
        st.session_state['current_page'] = st.session_state['page_nav_mobile']

    def _on_page_change_sidebar():
        st.session_state['current_page'] = st.session_state['page_nav_sidebar']

    # Mobil cihazlarda sidebar görünmeyebilir — üstte de yedek navigasyon göster
    with st.expander("📌 Sayfa Seç (sidebar görünmüyorsa buradan geçin)", expanded=False):
        st.radio("Navigasyon", _pages, index=current_idx,
                 label_visibility="collapsed", key="page_nav_mobile",
                 on_change=_on_page_change_mobile)

    # Navigasyon — sidebar (masaüstünde birincil)
    with st.sidebar:
        st.markdown("### 📌 Sayfa")
        st.radio("Navigasyon", _pages, index=current_idx,
                 label_visibility="collapsed", key="page_nav_sidebar",
                 on_change=_on_page_change_sidebar)
        st.divider()


    page = st.session_state['current_page']

    if page == "🔔 Telegram Bildirimleri":
        render_telegram_setup()
        return

    if page == "🗳️ Geri Bildirim":
        render_feedback_summary_page()
        return


    if page == "📊 Backtesting":
        render_backtest(
            fetch_batch_fn  = fetch_batch,
            find_patterns_fn= find_patterns,
            all_bist_lists  = {'bist30': BIST30, 'bist100': BIST100, 'all': ALL_BIST}
        )
        return

    if page == "📚 Şablon Kütüphanesi":
        render_library(
            fetch_ticker_fn=fetch_ticker,
            find_patterns_fn=find_patterns,
            calc_consensus_fn=calc_consensus,
            fetch_batch_fn=fetch_batch,
            all_bist_lists={'bist30': BIST30, 'bist100': BIST100, 'all': ALL_BIST}
        )
        return

    if page == "🔭 Fırsat Tarayıcı":
        from scanner import render_scanner

        def _get_data(tickers, period="2y"):
            return fetch_batch(tickers, period)

        bist_lists = {
            'bist30': BIST30,
            'bist100': BIST100,
            'all': ALL_BIST
        }
        render_scanner(_get_data, bist_lists)
        return

    if page == "📈 Sinyal Performansı":
        render_performance_dashboard()
        return


    # Kütüphaneden yükleme isteği geldi mi?
    lib_action = st.session_state.get('library_action')
    if lib_action and lib_action.get('action') == 'load':
        t = lib_action['template']
        st.session_state['df']     = pd.DataFrame(
            {'Close': t['prices'], 'Volume': t['volumes']},
            index=pd.date_range(end=pd.Timestamp(t['end_date']),
                                periods=len(t['prices']), freq='B')
        )
        st.session_state['symbol'] = t['symbol']
        st.session_state['_load_start'] = t['start_date']
        st.session_state['_load_end']   = t['end_date']
        st.session_state['library_action'] = None
        st.info(f"📚 **{t['name']}** kütüphaneden yüklendi. Tarihler otomatik ayarlandı.")

    st.markdown("## 📊 BIST Pattern Matcher")
    st.caption("Çok boyutlu hisse senedi şablon eşleştirme — fiyat + hacim + momentum + formasyon")
    st.divider()

    # ── ADIM 1 ──
    st.markdown("### 1️⃣ Hisse ve Dönem")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        symbol = st.text_input("Hisse kodu", placeholder="THYAO, EREGL, ASELS...",
                                key="symbol_input").strip().upper()
    with c2:
        period = st.selectbox("Dönem", ["6mo","1y","2y"],
            format_func=lambda x: {"6mo":"6 Ay","1y":"1 Yıl","2y":"2 Yıl"}[x], index=1)
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        load = st.button("📥 Yükle", type="primary", use_container_width=True)

    # Hızlı seçim
    samples = ["THYAO","EREGL","ASELS","GARAN","BIMAS","KCHOL","SASA","TOASO","TUPRS","AKBNK"]
    scols = st.columns(len(samples))
    for i, s in enumerate(samples):
        if scols[i].button(s, key=f"q_{s}", use_container_width=True):
            st.session_state["symbol_input"] = s
            st.rerun()

    if load and symbol:
        with st.spinner(f"{symbol} yükleniyor..."):
            df = fetch_ticker(symbol, period)
        if df is None:
            st.error(f"'{symbol}' için veri alınamadı. Hisse az işlem görüyor olabilir, dönem kısaltın veya hisse kodunu kontrol edin.")

            return
        st.session_state.update({"df": df, "symbol": symbol,
                                  "matches": None, "selected": None})

    df = st.session_state.get("df")
    sym = st.session_state.get("symbol", "")
    if df is None:
        st.info("Bir hisse seçin ve 'Yükle' butonuna basın.")
        return

    st.divider()

    # ── ADIM 2 ──
    st.markdown("### 2️⃣ Şablon Aralığı")
    st.caption(
        "💡 Grid Search ile doğrulandı: **30 günlük** şablonlar 20 günlüklere göre "
        "kategorik olarak daha iyi sonuç veriyor (Sharpe 6.9 vs 3.5)."
    )
    date_list = [d.date() for d in df.index]
    mid = len(date_list) // 2

    # Kütüphaneden yüklenen tarihler varsa kullan
    _ls = st.session_state.pop('_load_start', None)
    _le = st.session_state.pop('_load_end', None)
    try:
        _default_start = pd.Timestamp(_ls).date() if _ls else date_list[max(0,mid-15)]
        _default_end   = pd.Timestamp(_le).date() if _le else date_list[min(len(date_list)-1,mid+15)]
        _default_start = max(date_list[0], min(_default_start, date_list[-2]))
        _default_end   = max(date_list[1], min(_default_end,   date_list[-1]))
    except Exception:
        _default_start = date_list[max(0,mid-15)]
        _default_end   = date_list[min(len(date_list)-1,mid+15)]

    c_s, c_e = st.columns(2)
    sel_start = c_s.date_input("📍 Başlangıç", value=_default_start,
                               min_value=date_list[0], max_value=date_list[-2],
                               key="sel_start")
    sel_end = c_e.date_input("🏁 Bitiş", value=_default_end,
                             min_value=date_list[1], max_value=date_list[-1],
                             key="sel_end")

    if sel_start >= sel_end:
        st.warning("Başlangıç bitiş tarihinden önce olmalı.")
        return

    sel_start_ts = pd.Timestamp(sel_start)
    sel_end_ts = pd.Timestamp(sel_end)
    seg_df = df.loc[sel_start_ts:sel_end_ts]
    if len(seg_df) < 5:
        st.warning("En az 5 günlük aralık seçin.")
        return

    # ── Şablon Kalite Kontrolü ──────────────────────────────────────────────
    n_days = len(seg_df)
    quality_warnings = []

    # 1. Uzunluk kontrolü
    if n_days > 90:
        quality_warnings.append(
            f"⚠️ **Şablon çok uzun ({n_days} gün).** 90+ günlük şablonlarda genel piyasa "
            f"trendi hisseye özgü pattern'i baskılayabilir. Sonuçlar tüm hisselerde "
            f"aynı tarihe yığılıyorsa bu büyük ihtimalle BIST geneli bir hareket "
            f"yakalandığı içindir — özgün bir pattern değildir. "
            f"**20-60 gün arası** daha güvenilir sonuç verir."
        )
    elif n_days > 60:
        quality_warnings.append(
            f"💡 Şablon {n_days} gün — orta-uzun vadeli. Sonuçları değerlendirirken "
            f"genel piyasa etkisini göz önünde bulundurun."
        )

    # 2. BIST100 endeksiyle korelasyon kontrolü
    index_corr = None
    try:
        xu100_raw = yf.download("XU100.IS", period=period,
                                auto_adjust=True, progress=False, threads=False, session=session)
        if xu100_raw is not None and not xu100_raw.empty:
            if isinstance(xu100_raw.columns, pd.MultiIndex):
                xu100_raw.columns = xu100_raw.columns.get_level_values(0)
            xu100_raw.index = pd.to_datetime(xu100_raw.index)
            idx_seg = xu100_raw.loc[sel_start_ts:sel_end_ts]
            if len(idx_seg) >= 5:
                min_len = min(len(seg_df), len(idx_seg))
                stock_rets = daily_returns(seg_df['Close'].values[-min_len:])
                idx_rets   = daily_returns(idx_seg['Close'].values[-min_len:])
                m = min(len(stock_rets), len(idx_rets))
                if m >= 4 and np.std(stock_rets[-m:]) > 1e-9 and np.std(idx_rets[-m:]) > 1e-9:
                    index_corr = float(np.corrcoef(stock_rets[-m:], idx_rets[-m:])[0, 1])
    except Exception:
        index_corr = None

    if index_corr is not None and index_corr > 0.75:
        quality_warnings.append(
            f"⚠️ **Bu şablon BIST100 endeksiyle %{index_corr*100:.0f} korelasyonlu.** "
            f"Seçtiğiniz hareket hisseye özgü olmaktan çok genel piyasa hareketini "
            f"yansıtıyor olabilir. Tarama sonuçlarında çok farklı karakterde hisseler "
            f"aynı tarihte 'benzer' çıkıyorsa bu yüzdendir. Daha kısa veya hisseye özgü "
            f"bir kırılma/dönüş bölgesi seçmeyi deneyin."
        )

    if quality_warnings:
        with st.container():
            for w in quality_warnings:
                st.warning(w)

    st.plotly_chart(fig_main_chart(df, sym, sel_start_ts, sel_end_ts),
                    use_container_width=True)

    last_date = df.index[-1].strftime('%d.%m.%Y')

    # Şablon istatistikleri — grafik hemen altında hesaplanmalı
    seg_closes = seg_df['Close'].values
    seg_vols = seg_df['Volume'].values

    # Kalite özet satırı
    if index_corr is not None:
        corr_color = '#E02424' if index_corr > 0.75 else ('#E3A008' if index_corr > 0.5 else '#0E9F6E')
        corr_label = 'Yüksek (piyasa geneli)' if index_corr > 0.75 else (
            'Orta' if index_corr > 0.5 else 'Düşük (hisseye özgü)')
        st.markdown(
            f"<div style='font-size:13px;color:#888'>🔗 BIST100 Korelasyonu: "
            f"<span style='color:{corr_color};font-weight:600'>%{index_corr*100:.0f} — {corr_label}</span></div>",
            unsafe_allow_html=True
        )

    # Caption + Kaydet butonu yan yana
    _cap_col, _save_col = st.columns([4, 1])
    _cap_col.caption(f"📅 Son veri: **{last_date}** — {len(df)} gün")
    if _save_col.button("💾 Şablonu Kaydet", use_container_width=True,
                        help="Seçili tarih aralığını kütüphaneye kaydet"):
        try:
            _fmts = scan_all_formations(seg_closes, seg_vols, 40)
            _fmt_names = [f.name for f in _fmts[:3]]
        except Exception:
            _fmt_names = []
        st.session_state['saving_template'] = {
            'symbol': sym,
            'start_date': str(sel_start),
            'end_date': str(sel_end),
            'prices': seg_closes,
            'volumes': seg_vols,
            'regime': "",
            'formations': _fmt_names,
        }
        st.rerun()
    seg_rets = daily_returns(seg_closes)
    pct = (seg_closes[-1] - seg_closes[0]) / seg_closes[0] * 100
    rsi_last = calc_rsi(seg_closes)[-1]
    _, _, macd_hist = calc_macd(seg_closes)
    fmts = detect_formations(seg_closes, seg_vols)
    top_fmt = max(fmts, key=fmts.get)
    top_fmt_score = fmts[top_fmt]

    # Piyasa rejimi tespiti
    with st.spinner("Piyasa rejimi analiz ediliyor..."):
        regime = detect_regime(seg_closes, seg_vols)
    reg_icons = {
        'trend_bull':'📈','trend_bear':'📉',
        'sideways':'↔️','high_vol':'⚡','low_vol':'😴'
    }
    reg_icon = reg_icons.get(regime.name, '📊')
    st.markdown(f"""
    <div style='background:#F0F7FF;border:1px solid #BFDBFE;border-radius:8px;
                padding:10px 16px;margin-bottom:12px;display:flex;gap:20px;flex-wrap:wrap'>
        <span style='font-size:12px;color:#1A56DB;font-weight:600'>
            {reg_icon} PİYASA REJİMİ: {regime.describe()}
        </span>
        <span style='font-size:11px;color:#555'>ADX: {regime.adx}</span>
        <span style='font-size:11px;color:#555'>ATR: %{regime.atr_pct}</span>
        <span style='font-size:11px;color:#555'>BB Genişlik: %{regime.bb_width}</span>
        <span style='font-size:11px;color:#888;font-style:italic'>
            BIST-PSI bu rejim için ağırlıkları otomatik ayarlıyor
        </span>
    </div>
    """, unsafe_allow_html=True)

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Uzunluk", f"{len(seg_df)} gün")
    m2.metric("Değişim", f"{pct:+.1f}%")
    m3.metric("Ort. Günlük", f"{seg_rets.mean()*100:+.2f}%")
    m4.metric("RSI", f"{rsi_last:.0f}")
    m5.metric("MACD", f"{'↑' if macd_hist[-1]>0 else '↓'} {macd_hist[-1]:.3f}")

    # Formasyon analizi
    with st.spinner("Formasyonlar taranıyor..."):
        seg_formations = scan_all_formations(seg_closes, seg_vols, min_confidence=40)
    fmt_summary = formation_summary_score(seg_formations)
    dom = fmt_summary['dominant_signal']
    dom_icon = '📈' if dom=='bullish' else ('📉' if dom=='bearish' else '↔️')
    m6.metric("Formasyon Sinyali", f"{dom_icon} {dom.title()}")

    if seg_formations:
        with st.expander(f"🔍 {len(seg_formations)} Formasyon Tespit Edildi — Detay"):
            # Grafik
            st.plotly_chart(fig_formation_chart(
                df.loc[sel_start_ts:sel_end_ts], sym, seg_formations),
                use_container_width=True)

            # Formasyon listesi
            cat_icons = {'klasik':'📊','trend':'📐','elliott':'🌊','harmonik':'🎯','wyckoff':'🏗️','psi_consensus':'⭐'}
            dir_colors_html = {'bullish':'#0E9F6E','bearish':'#E02424','neutral':'#888'}
            for f in seg_formations:
                icon = cat_icons.get(f.category, '📌')
                dc = dir_colors_html.get(f.direction, '#888')
                status_badge = {'active':'🟢 Aktif','completed':'⚪ Tamamlandı','forming':'🟡 Oluşuyor'}.get(f.status,'')
                st.markdown(f"""
                <div style='background:#FAFAFA;border:1px solid #E5E9F0;border-radius:8px;
                            padding:10px 14px;margin:6px 0'>
                    <div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap'>
                        <span style='font-size:16px'>{icon}</span>
                        <span style='font-weight:600;font-size:14px'>{f.name}</span>
                        <span style='background:{dc}22;color:{dc};padding:2px 8px;
                                     border-radius:4px;font-size:11px'>{f.direction.upper()}</span>
                        <span style='font-size:11px;color:#888'>{status_badge}</span>
                        <span style='margin-left:auto;font-size:13px;font-weight:600;
                                     color:#1A56DB'>%{f.confidence:.0f} güven</span>
                    </div>
                    <div style='font-size:12px;color:#555;margin-top:6px'>{f.description}</div>
                    <div style='display:flex;gap:16px;margin-top:4px;font-size:11px;color:#888'>
                        {'<span>🎯 Hedef: <b>' + str(f.target) + ' ₺</b></span>' if f.target else ''}
                        {'<span>🛡️ Stop: <b>' + str(f.stop) + ' ₺</b></span>' if f.stop else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ── ADIM 3 ──
    st.markdown("### 3️⃣ Tarama")
    c_scope, c_sim, c_btn = st.columns([2,1,1])
    with c_scope:
        scope = st.radio("Kapsam", ["BIST 30","BIST 100","Tüm BIST"], horizontal=True)
    with c_sim:
        min_sim = st.slider("Min. Benzerlik %", 55, 85, 65, 1,
                     help="Güncel Grid Search (son dönem): PSI 65 optimal (Sharpe 6.73, %67 kazanç)")
    with c_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        scan = st.button("🔍 Tara", type="primary", use_container_width=True)

    # Ağırlık açıklaması
    with st.expander("ℹ️ Algoritma — Nasıl Eşleştiriyor?"):
        st.markdown("""
        **6 boyutlu kompozit skor** (toplam %100):

        | Boyut | Ağırlık | Ne Ölçüyor |
        |-------|---------|-----------|
        | Fiyat Şekli (DTW) | %28 | Dynamic Time Warping ile fiyat hareketi benzerliği |
        | Fiyat Şekli (Pearson) | %12 | Korelasyon bazlı fiyat uyumu |
        | Günlük Getiri Dağılımı | %20 | Günlük % hareketlerin ort, std, çarpıklık benzerliği |
        | Hacim Profili | %15 | Hacim dağılımı + hacim trendi uyumu |
        | Momentum (RSI+MACD) | %15 | RSI seviyesi ve MACD sinyal yönü benzerliği |
        | Formasyonlar | %10 | Double Top/Bottom, H&S, Trend kanalı, Breakout uyumu |
        """)

    if st.session_state.get('saving_template'):
        st.divider()
        st.markdown("#### 💾 Şablonu Kaydet")
        s = st.session_state['saving_template']
        sc1, sc2 = st.columns(2)
        with sc1:
            tpl_name = st.text_input("Şablon adı",
                value=f"{s['symbol']} — {s['start_date']} / {s['end_date']}",
                key="tpl_name_input")
            tpl_notes = st.text_area("Notlar (isteğe bağlı)", key="tpl_notes_input", height=80)
        with sc2:
            tpl_tags_str = st.text_input("Etiketler (virgülle ayırın)",
                placeholder="yükseliş, breakout, banka...", key="tpl_tags_input")
        sc_b1, sc_b2 = st.columns(2)
        with sc_b1:
            if st.button("✅ Kaydet", type="primary", key="confirm_save_tpl"):
                tags = [tag.strip() for tag in tpl_tags_str.split(',') if tag.strip()]
                matches_now = st.session_state.get('matches')
                consensus_now = None
                if matches_now:
                    try:
                        consensus_now = calc_consensus(matches_now, float(s['prices'][-1]))
                    except Exception:
                        pass
                save_template(
                    symbol=s['symbol'],
                    start_date=s['start_date'],
                    end_date=s['end_date'],
                    prices=np.array(s['prices']),
                    volumes=np.array(s['volumes']),
                    name=tpl_name,
                    notes=tpl_notes,
                    tags=tags,
                    scan_results=matches_now,
                    consensus=consensus_now,
                    regime=s.get('regime', ''),
                    formations=s.get('formations', []),
                )
                st.session_state['saving_template'] = None
                st.success("✅ Şablon kaydedildi! Şablon Kütüphanesi sayfasından erişebilirsiniz.")
                st.rerun()
        with sc_b2:
            if st.button("❌ İptal", key="cancel_save_tpl"):
                st.session_state['saving_template'] = None
                st.rerun()

    if scan:
        scan_list = {"BIST 30": BIST30, "BIST 100": BIST100, "Tüm BIST": ALL_BIST}[scope]
        scan_list = [t for t in scan_list if t != sym]

        prog = st.progress(0, text=f"📥 {len(scan_list)} hisse indiriliyor...")
        with st.spinner(""):
            all_data = fetch_batch(scan_list, period="2y")
        prog.progress(50, text=f"🔍 {len(all_data)} hisse taranıyor ({scope})...")
        matches = find_patterns(seg_closes, seg_vols, all_data,
                                top_n=5, min_sim=min_sim)
        prog.progress(100, text="✅ Tamamlandı!")
        import time; time.sleep(0.3); prog.empty()

        # BIST-PSI detaylarını da kaydet
        psi_engine = BISTPSI()
        psi_details = {}
        for r in matches:
            try:
                n = min(len(seg_closes), len(r['match_closes']))
                _, psi_result = psi_engine.compute(
                    seg_closes[-n:], seg_vols[-n:],
                    r['match_closes'][-n:], r['match_volumes'][-n:]
                )
                psi_details[r['ticker']] = psi_result
            except Exception:
                pass
        st.session_state['psi_details'] = psi_details
        st.session_state.update({"matches": matches,
                                  "template_closes": seg_closes,
                                  "template_volumes": seg_vols,
                                  "selected": None})
        # Açık şablon kaydı varsa konsensüsü güncelle
        from template_library import get_library, update_scan_history
        lib = get_library()
        if lib:
            last_t = lib[-1]
            if (last_t['symbol'] == sym and
                last_t['start_date'] == str(sel_start)):
                try:
                    cons = calc_consensus(matches, float(seg_closes[-1]))
                    update_scan_history(last_t['id'], matches, cons or {})
                except Exception:
                    pass
        st.rerun()

    # ── SONUÇLAR ──
    matches = st.session_state.get("matches")
    template_closes = st.session_state.get("template_closes")
    if matches is None:
        return

    st.divider()
    if len(matches) == 0:
        st.warning(f"**%{min_sim}** üzeri benzerlik bulunamadı. Eşiği düşürün veya farklı aralık deneyin.")
        return

    st.markdown(f"### 📊 En Benzer {len(matches)} Hisse")
    st.caption("Bir hisseye tıklayarak detay görün — tarihsel konum, normalize karşılaştırma, benzerlik profili.")
    st.caption(
        "⚙️ Parametreler Grid Search ile doğrulanmış (Sharpe 6.92, %69 kazanç): "
        "PSI ~65 · Window 30 gün · Güven %50-65 bandı"
    )

    # ── KONSENSÜS PANELİ ──
    current_price = float(df['Close'].iloc[-1])

    # Eşleşme tarih çeşitliliği kontrolü
    unique_periods = len(set(m['start_date'][3:] for m in matches)) if matches else 0
    if matches and unique_periods <= 1 and len(matches) >= 3:
        st.warning(
            "⚠️ **Tüm eşleşmeler aynı döneme ait** — bu büyük ihtimalle hisseye özgü "
            "bir pattern değil, genel piyasa hareketi yakalandığı anlamına gelir. "
            "Şablon uzunluğunu kısaltmayı (20-60 gün) veya farklı bir tarih aralığı "
            "seçmeyi deneyin."
        )
    elif matches and unique_periods >= 3:
        st.success(f"✅ Eşleşmeler **{unique_periods} farklı dönemden** geliyor — çeşitlilik iyi.")

    consensus = calc_consensus(matches, current_price)
    if consensus:
        direction = consensus['direction']
        conf = consensus['confidence']
        wpct = consensus['weighted_pct']
        c_dir = '#0E9F6E' if direction == 'YÜKSELİŞ' else ('#E02424' if direction == 'DÜŞÜŞ' else '#E3A008')
        icon_dir = '📈' if direction == 'YÜKSELİŞ' else ('📉' if direction == 'DÜŞÜŞ' else '↔️')

        # Güven barı
        band_label = consensus.get('band_label', '')
        in_optimal = consensus.get('in_optimal_band', False)
        conf_color = '#0E9F6E' if in_optimal else ('#E3A008' if conf >= 45 else '#E02424')
        border_color = c_dir if direction != 'KARARSIZ' else '#E5E9F0'

        st.markdown(f"""
        <div style='background:#FFFFFF;border:1.5px solid {border_color};border-radius:12px;
                    padding:20px 24px;margin:12px 0 20px'>
            <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px'>
                <div>
                    <div style='font-size:11px;color:#888;letter-spacing:1px;margin-bottom:4px'>KONSENSÜS YÖN</div>
                    <div style='font-size:32px;font-weight:800;color:{c_dir}'>{icon_dir} {direction}</div>
                    <div style='font-size:13px;color:#555;margin-top:4px'>
                        {consensus["agreement_ratio"]} hisse aynı yönde &nbsp;|&nbsp;
                        ↑{consensus["up_count"]} Yükseliş &nbsp; ↓{consensus["down_count"]} Düşüş
                    </div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:11px;color:#888;letter-spacing:1px;margin-bottom:4px'>BEKLENEN HAREKET</div>
                    <div style='font-size:28px;font-weight:700;color:{c_dir}'>{wpct:+.1f}%</div>
                    <div style='font-size:12px;color:#888'>{current_price:.2f} ₺ → {consensus["target_low"]:.2f} / {consensus["target_high"]:.2f} ₺</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:11px;color:#888;letter-spacing:1px;margin-bottom:4px'>GÜVEN SKORU</div>
                    <div style='font-size:28px;font-weight:700;color:{conf_color}'>%{conf:.0f}</div>
                    <div style='font-size:11px;font-weight:600;color:{conf_color}'>{band_label}</div>
                    <div style='font-size:9px;color:#aaa;margin-top:2px'>Optimal bant: %50-65</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:11px;color:#888;letter-spacing:1px;margin-bottom:4px'>HEDEF ARALIK</div>
                    <div style='font-size:14px;font-weight:600;color:#0E9F6E'>↑ {consensus["target_pct_high"]:+.1f}%</div>
                    <div style='font-size:14px;font-weight:600;color:#E02424'>↓ {consensus["target_pct_low"]:.1f}%</div>
                    <div style='font-size:10px;color:#aaa;margin-top:2px'>Dağılım: ±{consensus["dispersion"]:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Portföye ekle butonu — şablonu seçtiğiniz ana hisse için
        if direction == "YÜKSELİŞ":
            pcol1, pcol2 = st.columns([3, 1])
            with pcol2:
                render_add_to_portfolio_button(
                    ticker=sym,
                    current_price=current_price,
                    source="Pattern Matcher",
                    signal_score=consensus.get('avg_similarity'),
                    confidence=conf,
                    expected_pct=wpct,
                    key_suffix="pm"
                )

        # Ağırlık tablosu
        with st.expander("📊 Bireysel Ağırlıklar ve Katkılar"):
            rows = []
            for item in consensus['individual']:
                yön = "📈 Yükseliş" if item['pct'] > 0 else "📉 Düşüş"
                rows.append({
                    "Hisse": item['ticker'],
                    "Benzerlik": f"%{item['sim']}",
                    "Ağırlık": f"%{item['weight']}",
                    "Beklenen Hareket": f"{item['pct']:+.1f}%",
                    "Yön": yön
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Konsensüs fan grafiği
        st.plotly_chart(fig_consensus_chart(consensus, matches, template_closes, sym),
                        use_container_width=True)
    st.divider()

    # Grid Search optimal bandı uygula: PSI 65+ (doğrulanmış optimal: ~70)
    matches_filtered = [r for r in matches if 65 <= r.get('similarity', 0)]
    matches = sorted(matches,
                     key=lambda r: r['similarity'],
                     reverse=True)

    # Kartlar
    card_cols = st.columns(len(matches))
    for i, r in enumerate(matches):
        c = COLORS[i % len(COLORS)]
        is_sel = st.session_state.get("selected") == r['ticker']
        fut_pct = r['fut_pct']
        c_fut = '#0E9F6E' if fut_pct >= 0 else '#E02424'
        icon = '▲' if fut_pct >= 0 else '▼'
        bd = r['breakdown']

        with card_cols[i]:
            bg = '#EFF6FF' if is_sel else '#FFFFFF'
            border = f'2px solid {c}' if is_sel else '1.5px solid #E5E9F0'
            psi_details = st.session_state.get('psi_details', {})
            psi = psi_details.get(r['ticker'])
            psi_line = ""
            if psi:
                reg_short = {'trend_bull':'↗ Trend','trend_bear':'↘ Trend',
                             'sideways':'↔ Yatay','high_vol':'⚡ Vol','low_vol':'😴 Dar'}.get(psi.regime.name,'')
                psi_line = f"<div style='font-size:9px;color:#1A56DB;margin-top:3px'>BIST-PSI: {psi.score:.0f} | {reg_short} | Mah: {psi.mahalanobis_sim:.0f}</div>"
            st.markdown(f"""
            <div style='background:{bg};border:{border};border-radius:10px;
                        padding:14px 10px;text-align:center'>
                <div style='font-size:16px;font-weight:700;color:#1A1A2E'>{r['ticker']}</div>
                <div style='font-size:10px;color:#888;margin:2px 0'>
                    {r['start_date']} → {r['end_date']}
                </div>
                <div style='margin:8px 0'>
                    <div style='font-size:10px;color:#888'>BIST-PSI SKORU</div>
                    <div style='font-size:26px;font-weight:700;color:{c}'>%{r['similarity']}</div>
                    {psi_line}
                </div>
                <div style='font-size:10px;color:#888;text-align:left;padding:0 4px'>
                    📐 Fiyat: %{bd.get('fiyat_dtw',0):.0f} &nbsp;
                    📊 Getiri: %{bd.get('getiri',0):.0f}<br>
                    📦 Hacim: %{bd.get('hacim',0):.0f} &nbsp;
                    ⚡ Mom: %{bd.get('momentum',0):.0f}<br>
                    🧬 Karakter: %{r.get('char_score',50):.0f} &nbsp;
                    🔮 Gelecek: %{r.get('fut_compat',50):.0f}<br>
                    🔗 Korelasyon: %{r.get('corr_score',50):.0f} &nbsp;
                    {"✅ Rejim Eşleşti" if r.get('regime_match') else "⚠️ Farklı Rejim"}
                </div>
                <div style='margin:8px 0'>
                    <div style='font-size:10px;color:#888'>SONRASI</div>
                    <div style='font-size:20px;font-weight:700;color:{c_fut}'>{icon} {fut_pct:+.1f}%</div>
                </div>
                <div style='font-size:10px;color:#aaa'>
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

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📅 Tarihsel Konum",
                "🔍 Şablon Uyumu",
                "📈 Tüm Eşleşmeler",
                "🎯 Benzerlik Profili",
                "🧮 BIST-PSI Detay"
            ])

            with tab1:
                st.plotly_chart(fig_history(sel, sym), use_container_width=True)
                st.caption(f"**{selected}** hissesinin tüm geçmişi. Sarı = eşleşen dönem, noktalı = sonraki hareket.")

            with tab2:
                st.plotly_chart(fig_compare(sel, template_closes, sym), use_container_width=True)
                st.caption("Z-score normalize fiyat karşılaştırması. Şekil ne kadar örtüşüyor?")
                # Segment haritası
                if 'seg_map' in sel and sel['seg_map'] is not None:
                    st.plotly_chart(fig_segment_map(sel['seg_map'], sel['ticker'], sym),
                                    use_container_width=True)
                    avg_seg = float(np.mean(sel['seg_map']))
                    weak = sum(1 for s in sel['seg_map'] if s < 50)
                    st.caption(f"Ort. segment benzerliği: **%{avg_seg:.0f}** — "
                               f"{weak} zayıf segment ({'var, dikkatli ol' if weak > 2 else 'az, iyi eşleşme'})")

            with tab3:
                st.plotly_chart(fig_normalize(template_closes, matches, sym), use_container_width=True)
                st.caption("Tüm eşleşmeler üst üste. Dikey çizgi sonrası geçmişteki 'sonraki hareket'.")

            with tab4:
                col_r, col_s = st.columns([1, 1])
                with col_r:
                    st.plotly_chart(fig_breakdown_radar(sel['breakdown'], selected),
                                    use_container_width=True)
                with col_s:
                    st.markdown("#### 📋 Boyut Detayları")
                    bd = sel['breakdown']
                    for k, v in bd.items():
                        label = {'fiyat_dtw':'Fiyat Şekli (DTW)','fiyat_pearson':'Fiyat (Pearson)',
                                 'getiri':'Günlük Getiri','hacim':'Hacim Profili',
                                 'momentum':'Momentum','formasyon':'Formasyon'}.get(k, k)
                        bar = int(v / 5)
                        st.markdown(f"**{label}:** %{v}  \n{'█'*bar}{'░'*(20-bar)}")
                    st.divider()
                    st.metric("Genel Benzerlik", f"%{sel['similarity']}")
                    s1,s2,s3 = st.columns(3)
                    s1.metric("Sonraki", f"{sel['fut_pct']:+.1f}%")
                    s2.metric("Maks ↑", f"+{sel['fut_max']:.1f}%")
                    s3.metric("Maks ↓", f"{sel['fut_min']:.1f}%")

                    st.divider()
                    st.markdown("##### 🗳️ Bu Eşleşme Hakkında Görüşünüz")
                    st.caption(
                        "Bu eşleşme gerçekten anlamlı görünüyor mu? Geri bildiriminiz "
                        "'🗳️ Geri Bildirim' sayfasında biriktirilir ve hangi analiz "
                        "boyutlarının güvenilir olduğunu görmenize yardımcı olur."
                    )
                    render_feedback_buttons(
                        symbol=sym,
                        ticker=selected,
                        similarity=sel['similarity'],
                        breakdown=sel.get('breakdown', {}),
                        fut_pct=sel['fut_pct'],
                        char_score=sel.get('char_score'),
                        fut_compat=sel.get('fut_compat'),
                        corr_score=sel.get('corr_score'),
                        regime_match=sel.get('regime_match'),
                        source_page="Pattern Matcher",
                        key_suffix="pm_tab4"
                    )
            with tab5:
                psi_details = st.session_state.get('psi_details', {})
                psi = psi_details.get(selected)
                if psi:
                    st.markdown(f"#### 🧮 BIST-PSI v2 — {selected}")
                    # Özet
                    p1,p2,p3,p4 = st.columns(4)
                    p1.metric("BIST-PSI Skoru", f"{psi.score:.1f}")
                    p2.metric("Mahalanobis", f"{psi.mahalanobis_sim:.1f}")
                    p3.metric("Ensemble Oy", f"%{psi.ensemble_detail['vote_ratio']}")
                    p4.metric("Güven Bandı", f"{psi.confidence_band[0]:.0f}-{psi.confidence_band[1]:.0f}")

                    # Rejim
                    reg_icons2 = {'trend_bull':'📈','trend_bear':'📉','sideways':'↔️','high_vol':'⚡','low_vol':'😴'}
                    st.markdown(f"""
                    <div style='background:#F0F7FF;border:1px solid #BFDBFE;border-radius:8px;padding:10px 16px;margin:10px 0'>
                        <b>Piyasa Rejimi:</b> {reg_icons2.get(psi.regime.name,'')} {psi.regime.describe()} &nbsp;|&nbsp;
                        ADX: {psi.regime.adx} &nbsp;|&nbsp; ATR: %{psi.regime.atr_pct} &nbsp;|&nbsp; BB: %{psi.regime.bb_width}
                    </div>
                    """, unsafe_allow_html=True)

                    # Ağırlıklar ve boyut skorları
                    st.markdown("##### Boyut Skorları ve Adaptif Ağırlıklar")
                    dim_names = {'dtw':'Fiyat DTW','pearson':'Fiyat Pearson',
                                 'returns':'Getiri Dağılımı','volume':'Hacim',
                                 'momentum':'Momentum','formation':'Formasyon'}
                    dim_icons = {'dtw':'📐','pearson':'📊','returns':'💹',
                                 'volume':'📦','momentum':'⚡','formation':'🔷'}
                    for dim, score in psi.dim_scores.items():
                        w = psi.weights.get(dim, 0)
                        label = dim_names.get(dim, dim)
                        icon2 = dim_icons.get(dim, '•')
                        bar = int(score / 5)
                        w_bar = int(w / 5)
                        color = '#0E9F6E' if score >= 70 else ('#E3A008' if score >= 55 else '#E02424')
                        st.markdown(f"""
                        <div style='background:#FAFAFA;border:1px solid #E5E9F0;border-radius:6px;
                                    padding:8px 12px;margin:4px 0;display:flex;align-items:center;gap:12px'>
                            <span style='width:20px'>{icon2}</span>
                            <span style='width:140px;font-size:12px;font-weight:500'>{label}</span>
                            <span style='color:{color};font-size:14px;font-weight:700;width:45px'>%{score:.0f}</span>
                            <span style='font-family:monospace;font-size:11px;color:#888'>
                                {'█'*bar}{'░'*(20-bar)}
                            </span>
                            <span style='font-size:10px;color:#1A56DB;margin-left:8px'>Ağırlık: %{w:.0f}</span>
                            <span style='font-family:monospace;font-size:9px;color:#BFDBFE'>
                                {'■'*w_bar}{'□'*(20-w_bar)}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Ensemble oy detayı
                    st.markdown("##### Ensemble Oylama")
                    ev = psi.ensemble_detail
                    st.markdown(f"""
                    <div style='background:#FAFAFA;border:1px solid #E5E9F0;border-radius:8px;padding:12px 16px'>
                        <b>Oy Sonucu:</b> {ev['yes_votes']}/{ev['total_votes']} boyut "Benzer" oyu kullandı
                        (%{ev['vote_ratio']}) &nbsp;|&nbsp;
                        <b>Consensus Bonus:</b> +%{ev['consensus_bonus']}
                    </div>
                    """, unsafe_allow_html=True)
                    vote_rows = []
                    for dim, v in ev['votes'].items():
                        vote_rows.append({
                            "Boyut": dim_names.get(dim, dim),
                            "Skor": f"%{v['score']:.0f}",
                            "Oy": "✅ Benzer" if v['vote'] else "❌ Farklı",
                            "Ağırlık": f"%{v['weight']:.0f}",
                            "Katkı": f"%{v['weighted_contribution']:.1f}"
                        })
                    st.dataframe(pd.DataFrame(vote_rows),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("BIST-PSI detayı için yeniden tarama yapın.")

    else:
        st.divider()
        if matches and template_closes is not None:
            st.markdown("#### 📈 Normalize Karşılaştırma")
            st.plotly_chart(fig_normalize(template_closes, matches, sym),
                            use_container_width=True)

if __name__ == "__main__":
    main()
