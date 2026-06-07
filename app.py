import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

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
                         auto_adjust=True, progress=False, threads=False)
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
                          auto_adjust=True, group_by='ticker', progress=False)
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
    """
    Formasyon skoru hesapla:
    - Double Top/Bottom
    - Head & Shoulders
    - Trend kanalı (ascending/descending)
    - Breakout (hacim artışıyla fiyat kırılması)
    Sonuç: dict of formation scores (0-1)
    """
    prices = np.array(prices, dtype=float)
    volumes = np.array(volumes, dtype=float)
    n = len(prices)
    scores = {
        'double_top': 0.0,
        'double_bottom': 0.0,
        'head_shoulders': 0.0,
        'ascending_channel': 0.0,
        'descending_channel': 0.0,
        'breakout_up': 0.0,
        'breakout_down': 0.0,
    }
    if n < 10:
        return scores

    # Yerel tepe ve dipler bul
    def local_extrema(arr, order=3):
        peaks, troughs = [], []
        for i in range(order, len(arr)-order):
            window = arr[i-order:i+order+1]
            if arr[i] == window.max() and arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                peaks.append(i)
            if arr[i] == window.min() and arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                troughs.append(i)
        return peaks, troughs

    peaks, troughs = local_extrema(prices, order=max(2, n//8))

    # Double Top: 2 tepe yakın seviyede
    if len(peaks) >= 2:
        for i in range(len(peaks)-1):
            p1, p2 = prices[peaks[i]], prices[peaks[i+1]]
            diff = abs(p1-p2) / (max(p1,p2)+1e-9)
            if diff < 0.03:  # %3 tolerans
                scores['double_top'] = max(scores['double_top'], 1 - diff/0.03)

    # Double Bottom: 2 dip yakın seviyede
    if len(troughs) >= 2:
        for i in range(len(troughs)-1):
            t1, t2 = prices[troughs[i]], prices[troughs[i+1]]
            diff = abs(t1-t2) / (max(t1,t2)+1e-9)
            if diff < 0.03:
                scores['double_bottom'] = max(scores['double_bottom'], 1 - diff/0.03)

    # Head & Shoulders: 3 tepe, ortadaki yüksek
    if len(peaks) >= 3:
        for i in range(len(peaks)-2):
            l, h, r = prices[peaks[i]], prices[peaks[i+1]], prices[peaks[i+2]]
            if h > l and h > r:
                sym = 1 - abs(l-r)/(h+1e-9)
                height_ratio = min(l,r)/h
                if height_ratio > 0.85 and sym > 0.7:
                    scores['head_shoulders'] = max(scores['head_shoulders'], sym * height_ratio)

    # Trend kanalı — linear regression
    x = np.arange(n)
    slope, intercept = np.polyfit(x, prices, 1)
    residuals = prices - (slope * x + intercept)
    r2 = 1 - residuals.var() / (prices.var() + 1e-9)

    if r2 > 0.6:
        norm_slope = slope / (prices.mean() + 1e-9)
        if norm_slope > 0.001:
            scores['ascending_channel'] = min(1.0, r2 * (norm_slope * 100))
        elif norm_slope < -0.001:
            scores['descending_channel'] = min(1.0, r2 * abs(norm_slope * 100))

    # Breakout: son %20'de hacim spike + fiyat kırılması
    split = int(n * 0.8)
    if split > 0 and len(volumes) == n:
        base_vol = volumes[:split].mean()
        recent_vol = volumes[split:].mean()
        base_price = prices[:split].max()
        recent_price = prices[split:].max()
        vol_ratio = recent_vol / (base_vol + 1e-9)
        if vol_ratio > 1.5 and recent_price > base_price * 1.02:
            scores['breakout_up'] = min(1.0, (vol_ratio - 1) * 0.5)
        elif vol_ratio > 1.5 and prices[split:].min() < prices[:split].min() * 0.98:
            scores['breakout_down'] = min(1.0, (vol_ratio - 1) * 0.5)

    return scores

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

def find_patterns(template_prices, template_volumes, all_data,
                  top_n=5, min_sim=65, future_mult=1.5):
    """
    Çok boyutlu pattern matching.
    Template özelliklerini hesapla, tüm hisselerde sliding window ile ara.
    """
    tpl_prices = np.array(template_prices, dtype=float)
    tpl_volumes = np.array(template_volumes, dtype=float)
    n = len(tpl_prices)
    fut_win = min(int(n * future_mult), 90)

    # Template özelliklerini hesapla
    feat_tpl = extract_features(tpl_prices, tpl_volumes)

    results = []

    for ticker, df in all_data.items():
        closes = df['Close'].values.astype(float)
        volumes = df['Volume'].values.astype(float)
        dates = df.index

        if len(closes) < n + fut_win + 10:
            continue

        max_i = len(closes) - n - fut_win
        step = max(1, n // 5)

        # Kaba tarama — sadece fiyat DTW kullan (hızlı)
        best_score, best_i = -1, 0
        for i in range(0, max_i, step):
            w_prices = closes[i:i+n]
            w_z = zscore(w_prices)
            quick_s = dtw_similarity(feat_tpl['price_z'], w_z) * 100
            if quick_s > best_score:
                best_score, best_i = quick_s, i

        # İnce tarama — en iyi bölge etrafında tam kompozit skor
        refine_range = range(max(0, best_i - step), min(max_i+1, best_i + step + 1))
        best_full_score, best_breakdown = -1, {}
        best_i_final = best_i

        for i in refine_range:
            w_prices = closes[i:i+n]
            w_volumes = volumes[i:i+n]
            feat_win = extract_features(w_prices, w_volumes)
            full_s, breakdown = composite_score(feat_tpl, feat_win)
            if full_s > best_full_score:
                best_full_score = full_s
                best_breakdown = breakdown
                best_i_final = i

        if best_full_score < min_sim:
            continue

        ms, me = best_i_final, best_i_final + n
        match_closes = closes[ms:me]
        match_volumes = volumes[ms:me]
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
            'similarity': best_full_score,
            'breakdown': best_breakdown,
            'ms': ms, 'me': me,
            'match_closes': match_closes,
            'match_volumes': match_volumes,
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

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3B: KONSENSÜS ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════

def calc_consensus(matches, current_price):
    """
    Ağırlıklı konsensüs analizi.
    Benzerlik skoru ağırlık olarak kullanılır.
    
    Döndürür:
    - direction: "YÜKSELİŞ" / "DÜŞÜŞ" / "KARARSIZ"
    - confidence: 0-100 güven skoru
    - weighted_pct: ağırlıklı ortalama beklenen değişim %
    - target_low / target_high: hedef fiyat aralığı
    - agreement_ratio: aynı yönde kaç tanesi (örn 4/5)
    - dispersion: sonuçların dağılımı (düşükse tutarlı, yüksekse karmaşık)
    """
    if not matches:
        return None

    weights = np.array([r['similarity'] for r in matches], dtype=float)
    weights = weights / weights.sum()  # normalize

    pcts = np.array([r['fut_pct'] for r in matches], dtype=float)
    maxs = np.array([r['fut_max'] for r in matches], dtype=float)
    mins = np.array([r['fut_min'] for r in matches], dtype=float)

    # Ağırlıklı ortalama değişim
    weighted_pct = float(np.dot(weights, pcts))

    # Ağırlıklı hedef aralık
    weighted_max = float(np.dot(weights, maxs))
    weighted_min = float(np.dot(weights, mins))
    target_high = current_price * (1 + weighted_max / 100)
    target_low = current_price * (1 + weighted_min / 100)

    # Yön oylaması (ağırlıklı)
    up_weight = sum(w for w, p in zip(weights, pcts) if p > 0)
    down_weight = sum(w for w, p in zip(weights, pcts) if p <= 0)
    up_count = sum(1 for p in pcts if p > 0)
    down_count = sum(1 for p in pcts if p <= 0)

    if up_weight > 0.6:
        direction = "YÜKSELİŞ"
    elif down_weight > 0.6:
        direction = "DÜŞÜŞ"
    else:
        direction = "KARARSIZ"

    # Güven skoru:
    # - Yön konsensüsü (tek yönde yüksek ağırlık)
    # - Hareket büyüklüğü tutarlılığı (std düşükse güven yüksek)
    direction_conf = max(up_weight, down_weight) * 100  # 50-100 arası
    dispersion = float(np.std(pcts))  # düşükse tutarlı
    dispersion_penalty = min(40, dispersion * 2)  # yüksek dağılım ceza
    avg_similarity = float(np.dot(weights, [r['similarity'] for r in matches]))
    similarity_bonus = (avg_similarity - 65) / 35 * 20  # 65-100 arası → 0-20 bonus

    confidence = max(0, min(100, direction_conf - dispersion_penalty + similarity_bonus))

    agreement_ratio = f"{max(up_count,down_count)}/{len(matches)}"

    return {
        'direction': direction,
        'confidence': round(confidence, 1),
        'weighted_pct': round(weighted_pct, 2),
        'target_high': round(target_high, 2),
        'target_low': round(target_low, 2),
        'target_pct_high': round(weighted_max, 2),
        'target_pct_low': round(weighted_min, 2),
        'agreement_ratio': agreement_ratio,
        'up_count': up_count,
        'down_count': down_count,
        'dispersion': round(dispersion, 2),
        'avg_similarity': round(avg_similarity, 1),
        'individual': [{'ticker': r['ticker'], 'pct': r['fut_pct'],
                        'weight': round(float(w)*100, 1), 'sim': r['similarity']}
                       for r, w in zip(matches, weights)]
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

def main():
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
    date_list = [d.date() for d in df.index]
    mid = len(date_list) // 2
    c_s, c_e = st.columns(2)
    sel_start = c_s.date_input("📍 Başlangıç", value=date_list[max(0,mid-20)],
                               min_value=date_list[0], max_value=date_list[-2],
                               key="sel_start")
    sel_end = c_e.date_input("🏁 Bitiş", value=date_list[min(len(date_list)-1,mid+20)],
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

    st.plotly_chart(fig_main_chart(df, sym, sel_start_ts, sel_end_ts),
                    use_container_width=True)

    last_date = df.index[-1].strftime('%d.%m.%Y')
    st.caption(f"📅 Son veri: **{last_date}** — {len(df)} gün")

    # Şablon istatistikleri
    seg_closes = seg_df['Close'].values
    seg_vols = seg_df['Volume'].values
    seg_rets = daily_returns(seg_closes)
    pct = (seg_closes[-1] - seg_closes[0]) / seg_closes[0] * 100
    rsi_last = calc_rsi(seg_closes)[-1]
    _, _, macd_hist = calc_macd(seg_closes)
    fmts = detect_formations(seg_closes, seg_vols)
    top_fmt = max(fmts, key=fmts.get)
    top_fmt_score = fmts[top_fmt]

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Uzunluk", f"{len(seg_df)} gün")
    m2.metric("Değişim", f"{pct:+.1f}%")
    m3.metric("Ort. Günlük", f"{seg_rets.mean()*100:+.2f}%")
    m4.metric("RSI", f"{rsi_last:.0f}")
    m5.metric("MACD", f"{'↑' if macd_hist[-1]>0 else '↓'} {macd_hist[-1]:.3f}")
    m6.metric("Formasyon", f"{top_fmt.replace('_',' ').title()}" if top_fmt_score > 0.3 else "—")

    st.divider()

    # ── ADIM 3 ──
    st.markdown("### 3️⃣ Tarama")
    c_scope, c_sim, c_btn = st.columns([2,1,1])
    with c_scope:
        scope = st.radio("Kapsam", ["BIST 30","BIST 100","Tüm BIST"], horizontal=True)
    with c_sim:
        min_sim = st.slider("Min. Benzerlik %", 55, 85, 65, 1)
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

        st.session_state.update({"matches": matches,
                                  "template_closes": seg_closes,
                                  "template_volumes": seg_vols,
                                  "selected": None})
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

    # ── KONSENSÜS PANELİ ──
    current_price = float(df['Close'].iloc[-1])
    consensus = calc_consensus(matches, current_price)
    if consensus:
        direction = consensus['direction']
        conf = consensus['confidence']
        wpct = consensus['weighted_pct']
        c_dir = '#0E9F6E' if direction == 'YÜKSELİŞ' else ('#E02424' if direction == 'DÜŞÜŞ' else '#E3A008')
        icon_dir = '📈' if direction == 'YÜKSELİŞ' else ('📉' if direction == 'DÜŞÜŞ' else '↔️')

        # Güven barı
        conf_bar = int(conf / 5)
        conf_color = '#0E9F6E' if conf >= 65 else ('#E3A008' if conf >= 45 else '#E02424')
        conf_label = 'Yüksek Güven' if conf >= 65 else ('Orta Güven' if conf >= 45 else 'Düşük Güven')

        st.markdown(f"""
        <div style='background:#FFFFFF;border:1.5px solid {c_dir};border-radius:12px;
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
                    <div style='font-size:11px;color:#888;letter-spacing:1px;margin-bottom:4px'>BEKLENEN HAREKETTr</div>
                    <div style='font-size:28px;font-weight:700;color:{c_dir}'>{wpct:+.1f}%</div>
                    <div style='font-size:12px;color:#888'>{current_price:.2f} ₺ → {consensus["target_low"]:.2f} / {consensus["target_high"]:.2f} ₺</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:11px;color:#888;letter-spacing:1px;margin-bottom:4px'>GÜVEN SKORU</div>
                    <div style='font-size:28px;font-weight:700;color:{conf_color}'>%{conf:.0f}</div>
                    <div style='font-size:11px;color:{conf_color}'>{conf_label}</div>
                    <div style='font-size:10px;color:#aaa;margin-top:2px'>{'█'*conf_bar}{'░'*(20-conf_bar)}</div>
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
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Konsensüs fan grafiği
        st.plotly_chart(fig_consensus_chart(consensus, matches, template_closes, sym),
                        use_container_width=True)
    st.divider()

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
            st.markdown(f"""
            <div style='background:{bg};border:{border};border-radius:10px;
                        padding:14px 10px;text-align:center'>
                <div style='font-size:16px;font-weight:700;color:#1A1A2E'>{r['ticker']}</div>
                <div style='font-size:10px;color:#888;margin:2px 0'>
                    {r['start_date']} → {r['end_date']}
                </div>
                <div style='margin:8px 0'>
                    <div style='font-size:10px;color:#888'>GENEL BENZERLİK</div>
                    <div style='font-size:26px;font-weight:700;color:{c}'>%{r['similarity']}</div>
                </div>
                <div style='font-size:10px;color:#888;text-align:left;padding:0 4px'>
                    📐 Fiyat: %{bd.get('fiyat_dtw',0):.0f} &nbsp;
                    📊 Getiri: %{bd.get('getiri',0):.0f}<br>
                    📦 Hacim: %{bd.get('hacim',0):.0f} &nbsp;
                    ⚡ Mom: %{bd.get('momentum',0):.0f}
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

            tab1, tab2, tab3, tab4 = st.tabs([
                "📅 Tarihsel Konum",
                "🔍 Şablon Uyumu",
                "📈 Tüm Eşleşmeler",
                "🎯 Benzerlik Profili"
            ])

            with tab1:
                st.plotly_chart(fig_history(sel, sym), use_container_width=True)
                st.caption(f"**{selected}** hissesinin tüm geçmişi. Sarı = eşleşen dönem, noktalı = sonraki hareket.")

            with tab2:
                st.plotly_chart(fig_compare(sel, template_closes, sym), use_container_width=True)
                st.caption("Z-score normalize fiyat karşılaştırması. Şekil ne kadar örtüşüyor?")

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
    else:
        st.divider()
        if matches and template_closes is not None:
            st.markdown("#### 📈 Normalize Karşılaştırma")
            st.plotly_chart(fig_normalize(template_closes, matches, sym),
                            use_container_width=True)

if __name__ == "__main__":
    main()
