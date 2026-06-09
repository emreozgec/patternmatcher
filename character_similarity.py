"""
character_similarity.py — Hisse Karakter Benzerliği

Her hissenin davranış profilini çıkarır:
- Volatilite karakteri
- Beta (BIST ile korelasyon)
- Ortalama günlük hareket
- Sektör/büyüklük grubu
- Trend/yatay eğilim
- Tipik düşüş/yükseliş oranı (asymmetry)

İki hissenin karakter benzerliğini 0-100 arasında skorlar.
"""

import numpy as np
from typing import Dict, Optional, Tuple


# ── Hisse karakter profili ─────────────────────────────────────────────────────

def build_character_profile(prices: np.ndarray,
                             volumes: np.ndarray,
                             market_prices: Optional[np.ndarray] = None) -> Dict:
    """
    Hissenin davranış profilini çıkar.
    prices: tüm mevcut fiyat serisi (uzun dönem — en az 60 gün)
    volumes: hacim serisi
    market_prices: BIST endeks fiyatları (beta hesabı için, opsiyonel)
    """
    prices  = np.array(prices,  dtype=float)
    volumes = np.array(volumes, dtype=float)
    n = len(prices)

    if n < 10:
        return _empty_profile()

    # Günlük getiriler
    rets = np.diff(prices) / (np.abs(prices[:-1]) + 1e-9)

    # 1. Volatilite karakteri
    vol_daily  = float(np.std(rets))                          # Günlük vol
    vol_weekly = float(np.std([                               # Haftalık vol
        sum(rets[i:i+5]) for i in range(0, len(rets)-4, 5)
    ])) if len(rets) >= 10 else vol_daily * np.sqrt(5)

    # 2. Hareket asimetrisi (yükseliş gün / düşüş gün oranı)
    up_days   = float(np.mean(rets > 0))
    down_days = float(np.mean(rets < 0))
    asymmetry = up_days / (down_days + 1e-9)                 # >1 bullish eğilim

    # 3. Ortalama kazanç / kayıp büyüklüğü
    up_rets   = rets[rets > 0]
    down_rets = rets[rets < 0]
    avg_gain  = float(np.mean(up_rets))   if len(up_rets)   > 0 else 0.0
    avg_loss  = float(np.mean(down_rets)) if len(down_rets) > 0 else 0.0

    # 4. Trend eğilimi (uzun dönem)
    x = np.arange(n)
    slope, _ = np.polyfit(x, prices, 1)
    norm_slope = float(slope / (prices.mean() + 1e-9) * 252)  # Yıllık normalize

    # 5. Ortalama hacim trendi
    if len(volumes) >= 20:
        vol_trend = float(np.polyfit(np.arange(len(volumes)),
                                     volumes / (volumes.mean() + 1e-9), 1)[0])
    else:
        vol_trend = 0.0

    # 6. Max drawdown
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / (peak + 1e-9)
        if dd > max_dd:
            max_dd = dd

    # 7. Beta (varsa market verisi)
    beta = 1.0
    if market_prices is not None and len(market_prices) >= n:
        mkt = np.array(market_prices[-n:], dtype=float)
        mkt_rets = np.diff(mkt) / (np.abs(mkt[:-1]) + 1e-9)
        min_len = min(len(rets), len(mkt_rets))
        if min_len >= 10:
            cov   = np.cov(rets[-min_len:], mkt_rets[-min_len:])[0, 1]
            mkt_v = np.var(mkt_rets[-min_len:])
            beta  = float(cov / (mkt_v + 1e-9))
            beta  = max(-3.0, min(3.0, beta))

    # 8. Volatilite rejimi geçmişi (son dönem vs uzun dönem)
    if n >= 40:
        recent_vol = float(np.std(rets[-20:]))
        long_vol   = float(np.std(rets))
        vol_regime_ratio = recent_vol / (long_vol + 1e-9)
    else:
        vol_regime_ratio = 1.0

    # 9. Momentum skoru (son 20 gün vs öncesi)
    if n >= 40:
        recent_ret  = float(prices[-1] / prices[-20] - 1)
        earlier_ret = float(prices[-20] / prices[-40] - 1)
        momentum    = recent_ret - earlier_ret
    else:
        momentum = float(prices[-1] / prices[0] - 1)

    # 10. Likidite skoru (normalize hacim düzgünlüğü)
    if len(volumes) >= 10:
        vol_cv = float(np.std(volumes) / (np.mean(volumes) + 1e-9))
        liquidity = max(0.0, 1.0 - min(1.0, vol_cv))
    else:
        liquidity = 0.5

    return {
        'vol_daily':       vol_daily,
        'vol_weekly':      vol_weekly,
        'asymmetry':       asymmetry,
        'avg_gain':        avg_gain,
        'avg_loss':        avg_loss,
        'norm_slope':      norm_slope,
        'vol_trend':       vol_trend,
        'max_drawdown':    max_dd,
        'beta':            beta,
        'vol_regime_ratio': vol_regime_ratio,
        'momentum':        momentum,
        'liquidity':       liquidity,
        'up_days':         up_days,
        'n':               n,
    }


def _empty_profile() -> Dict:
    return {
        'vol_daily': 0.02, 'vol_weekly': 0.04, 'asymmetry': 1.0,
        'avg_gain': 0.01, 'avg_loss': -0.01, 'norm_slope': 0.0,
        'vol_trend': 0.0, 'max_drawdown': 0.1, 'beta': 1.0,
        'vol_regime_ratio': 1.0, 'momentum': 0.0, 'liquidity': 0.5,
        'up_days': 0.5, 'n': 0,
    }


def character_similarity(prof1: Dict, prof2: Dict) -> Tuple[float, Dict]:
    """
    İki hisse profili arasındaki karakter benzerliğini hesapla.
    Returns: (score 0-100, breakdown dict)
    """

    def ratio_sim(a: float, b: float, scale: float = 1.0) -> float:
        """İki skaler değerin benzerliği (0-1)."""
        return float(np.exp(-abs(a - b) / (scale + 1e-9)))

    # 1. Volatilite benzerliği (en önemli)
    s_vol = ratio_sim(prof1['vol_daily'], prof2['vol_daily'],
                      scale=(prof1['vol_daily'] + prof2['vol_daily']) / 2 + 1e-9)

    # 2. Asimetri benzerliği (yükseliş/düşüş eğilimi)
    s_asym = ratio_sim(prof1['asymmetry'], prof2['asymmetry'], scale=0.5)

    # 3. Kazanç/kayıp profili
    s_gain = ratio_sim(prof1['avg_gain'], prof2['avg_gain'],
                       scale=max(abs(prof1['avg_gain']), abs(prof2['avg_gain'])) + 1e-9)
    s_loss = ratio_sim(prof1['avg_loss'], prof2['avg_loss'],
                       scale=max(abs(prof1['avg_loss']), abs(prof2['avg_loss'])) + 1e-9)
    s_gl = (s_gain + s_loss) / 2

    # 4. Trend eğilimi benzerliği
    s_trend = ratio_sim(prof1['norm_slope'], prof2['norm_slope'], scale=0.5)

    # 5. Beta benzerliği
    s_beta = ratio_sim(prof1['beta'], prof2['beta'], scale=0.5)

    # 6. Max drawdown benzerliği
    s_dd = ratio_sim(prof1['max_drawdown'], prof2['max_drawdown'],
                     scale=max(prof1['max_drawdown'], prof2['max_drawdown']) + 1e-9)

    # 7. Momentum benzerliği
    s_mom = ratio_sim(prof1['momentum'], prof2['momentum'], scale=0.1)

    # Ağırlıklı kompozit
    # Volatilite ve asimetri en kritik
    score = (
        0.28 * s_vol   +
        0.18 * s_asym  +
        0.15 * s_gl    +
        0.13 * s_trend +
        0.12 * s_beta  +
        0.08 * s_dd    +
        0.06 * s_mom
    )

    breakdown = {
        'volatilite':  round(s_vol   * 100, 1),
        'asimetri':    round(s_asym  * 100, 1),
        'k_k_profil':  round(s_gl    * 100, 1),
        'trend':       round(s_trend * 100, 1),
        'beta':        round(s_beta  * 100, 1),
        'drawdown':    round(s_dd    * 100, 1),
        'momentum':    round(s_mom   * 100, 1),
    }

    return round(score * 100, 1), breakdown


# ── Sonraki hareket uyumluluk skoru ───────────────────────────────────────────

def future_behavior_compatibility(
    template_prices:  np.ndarray,
    match_future:     np.ndarray,
    candidate_profile: Dict,
    template_profile:  Dict,
) -> float:
    """
    Eşleşen dönemin sonraki hareketinin şablon hissesiyle uyumlu olup olmadığını ölç.

    Mantık:
    - Sonraki hareketin volatilitesi şablon hissesinin tipik volatilitesiyle uyumlu mu?
    - Sonraki hareketin yönü şablon hissesinin trend eğilimiyle tutarlı mı?
    - Sonraki hareketin büyüklüğü makul mü?

    Returns: 0-100 uyumluluk skoru
    """
    if len(match_future) < 3:
        return 50.0

    match_future = np.array(match_future, dtype=float)

    # Sonraki hareketin günlük getirileri
    fut_rets = np.diff(match_future) / (np.abs(match_future[:-1]) + 1e-9)
    fut_vol  = float(np.std(fut_rets))
    fut_ret  = float((match_future[-1] - match_future[0]) / (match_future[0] + 1e-9))
    fut_up   = float(np.mean(fut_rets > 0))

    # 1. Volatilite uyumluluğu
    tpl_vol = template_profile.get('vol_daily', 0.02)
    vol_compat = float(np.exp(-abs(fut_vol - tpl_vol) / (tpl_vol + 1e-9)))

    # 2. Hareket büyüklüğü uyumluluğu
    # Şablon hissesinin tipik kazanç/kayıp büyüklüğüyle kıyasla
    n_days = len(match_future)
    expected_move = tpl_vol * np.sqrt(n_days)
    actual_move   = abs(fut_ret)
    size_compat   = float(np.exp(-abs(actual_move - expected_move) / (expected_move + 0.05)))

    # 3. Yön uyumluluğu
    # Şablon hissesinin asimetrisiyle uyumlu mu?
    tpl_asym = template_profile.get('asymmetry', 1.0)
    # asimetri > 1 → bullish eğilim → yükseliş daha uyumlu
    if tpl_asym > 1.1:
        dir_compat = fut_up  # Yükseliş günleri fazlaysa uyumlu
    elif tpl_asym < 0.9:
        dir_compat = 1 - fut_up  # Düşüş günleri fazlaysa uyumlu
    else:
        dir_compat = 0.5  # Nötr

    score = 0.40 * vol_compat + 0.35 * size_compat + 0.25 * dir_compat
    return round(min(100.0, score * 100), 1)


# ── Korelasyon filtresi ────────────────────────────────────────────────────────

def historical_correlation(prices1: np.ndarray,
                            prices2: np.ndarray,
                            window: int = 60) -> float:
    """
    İki hissenin tarihsel korelasyonu (son `window` gün).
    Returns: -1 ile 1 arası korelasyon
    """
    p1 = np.array(prices1, dtype=float)
    p2 = np.array(prices2, dtype=float)

    n = min(len(p1), len(p2), window)
    if n < 10:
        return 0.0

    p1 = p1[-n:]
    p2 = p2[-n:]

    r1 = np.diff(p1) / (np.abs(p1[:-1]) + 1e-9)
    r2 = np.diff(p2) / (np.abs(p2[:-1]) + 1e-9)

    if np.std(r1) < 1e-9 or np.std(r2) < 1e-9:
        return 0.0

    return float(np.corrcoef(r1, r2)[0, 1])


def correlation_score(corr: float) -> float:
    """Korelasyonu 0-100 benzerlik skoruna çevir."""
    # Yüksek pozitif korelasyon → yüksek skor
    # Negatif korelasyon → düşük skor
    return round((corr + 1) / 2 * 100, 1)
