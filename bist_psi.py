"""
bist_psi.py — BIST Pattern Similarity Index v2

Üç katmanlı hibrit algoritma:
  1. Piyasa Rejimi Tespiti   → ADX, ATR, Bollinger Width, Trend Slope
  2. Adaptif Ağırlıklandırma → Rejime göre 6 boyutun ağırlığı dinamik değişir
  3. Mahalanobis Mesafesi    → Boyutlar arası korelasyonu hesaba katar
  4. Ensemble Oylama         → Her boyut bağımsız oy, çoğunluk kararı
  5. BIST-PSI Kompozit Skor  → 0-100 arası, kalibre edilmiş

Kullanım:
    from bist_psi import BISTPSI
    psi = BISTPSI()
    score, detail = psi.compute(template_prices, template_volumes,
                                window_prices, window_volumes)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: PİYASA REJİMİ TESPİTİ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketRegime:
    name: str           # trend_bull / trend_bear / sideways / high_vol / low_vol
    adx: float          # 0-100 trend gücü
    atr_pct: float      # ATR / fiyat (volatilite)
    bb_width: float     # Bollinger Band genişliği
    trend_slope: float  # normalize slope
    weights: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        icons = {
            'trend_bull':  '📈 Yükseliş Trendi',
            'trend_bear':  '📉 Düşüş Trendi',
            'sideways':    '↔️ Yatay Piyasa',
            'high_vol':    '⚡ Yüksek Volatilite',
            'low_vol':     '😴 Düşük Volatilite / Sıkışma',
        }
        return icons.get(self.name, self.name)


# Rejime göre ağırlık tabloları
# Her rejim için 6 boyut ağırlıkları (toplam = 1.0)
REGIME_WEIGHTS = {
    'trend_bull': {
        'dtw':        0.32,   # Trend şekli kritik
        'pearson':    0.13,
        'returns':    0.25,   # Günlük getiri büyüklüğü önemli
        'volume':     0.10,
        'momentum':   0.15,   # Momentum trendi onaylamalı
        'formation':  0.05,
    },
    'trend_bear': {
        'dtw':        0.30,
        'pearson':    0.12,
        'returns':    0.28,   # Düşüşte getiri dağılımı çok önemli
        'volume':     0.12,   # Panik satışta hacim artar
        'momentum':   0.13,
        'formation':  0.05,
    },
    'sideways': {
        'dtw':        0.20,   # Şekil önemli ama trend yok
        'pearson':    0.10,
        'returns':    0.15,
        'volume':     0.28,   # Yatay piyasada hacim dağılımı kritik
        'momentum':   0.12,
        'formation':  0.15,   # Formasyonlar yatay piyasada daha güvenilir
    },
    'high_vol': {
        'dtw':        0.22,
        'pearson':    0.10,
        'returns':    0.35,   # Yüksek vol'da getiri dağılımı dominant
        'volume':     0.18,
        'momentum':   0.12,
        'formation':  0.03,   # Yüksek vol'da formasyonlar bozulabilir
    },
    'low_vol': {
        'dtw':        0.28,
        'pearson':    0.12,
        'returns':    0.12,
        'volume':     0.22,   # Düşük vol'da hacim değişimi sinyal
        'momentum':   0.10,
        'formation':  0.16,   # Sıkışmadan çıkış formasyon sinyali
    },
}


def calc_adx(prices: np.ndarray, n: int = 14) -> float:
    """Average Directional Index — trend gücü (0-100)."""
    prices = np.array(prices, dtype=float)
    if len(prices) < n + 2:
        return 20.0
    high = prices  # Basitleştirme: sadece close kullanıyoruz
    low = prices
    close = prices

    tr_list = []
    for i in range(1, len(prices)):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i] - close[i-1]))
        tr_list.append(tr)

    dm_plus = [max(prices[i] - prices[i-1], 0) for i in range(1, len(prices))]
    dm_minus = [max(prices[i-1] - prices[i], 0) for i in range(1, len(prices))]

    def smooth(arr, n):
        if len(arr) < n:
            return np.mean(arr)
        result = np.mean(arr[:n])
        for i in range(n, len(arr)):
            result = result - result/n + arr[i]
        return result

    atr = smooth(tr_list, n)
    dmp = smooth(dm_plus, n)
    dmm = smooth(dm_minus, n)

    if atr < 1e-9:
        return 20.0

    di_plus = 100 * dmp / atr
    di_minus = 100 * dmm / atr
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
    return min(100.0, float(dx))


def calc_atr_pct(prices: np.ndarray, n: int = 14) -> float:
    """ATR / ortalama fiyat — normalize volatilite."""
    prices = np.array(prices, dtype=float)
    if len(prices) < n + 1:
        return 0.02
    tr_list = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    atr = np.mean(tr_list[-n:])
    return float(atr / (prices.mean() + 1e-9))


def calc_bb_width(prices: np.ndarray, n: int = 20) -> float:
    """Bollinger Band genişliği — (üst-alt)/orta."""
    prices = np.array(prices, dtype=float)
    if len(prices) < n:
        return 0.04
    sma = prices[-n:].mean()
    std = prices[-n:].std()
    return float(2 * std / (sma + 1e-9))


def detect_regime(prices: np.ndarray, volumes: np.ndarray) -> MarketRegime:
    """
    Piyasa rejimini tespit et ve uygun ağırlıkları seç.
    
    Rejim mantığı:
    - ADX > 25 + pozitif slope → trend_bull
    - ADX > 25 + negatif slope → trend_bear
    - ATR% > 0.04 → high_vol
    - BB Width < 0.03 → low_vol
    - Diğer → sideways
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)
    
    adx = calc_adx(prices)
    atr_pct = calc_atr_pct(prices)
    bb_width = calc_bb_width(prices)
    
    # Trend slope
    x = np.arange(n)
    slope = np.polyfit(x, prices, 1)[0]
    norm_slope = slope / (prices.mean() + 1e-9)

    # Rejim kararı
    if adx > 28 and norm_slope > 0.001:
        regime_name = 'trend_bull'
    elif adx > 28 and norm_slope < -0.001:
        regime_name = 'trend_bear'
    elif atr_pct > 0.035:
        regime_name = 'high_vol'
    elif bb_width < 0.025:
        regime_name = 'low_vol'
    else:
        regime_name = 'sideways'

    weights = REGIME_WEIGHTS[regime_name].copy()

    return MarketRegime(
        name=regime_name,
        adx=round(adx, 1),
        atr_pct=round(atr_pct * 100, 2),
        bb_width=round(bb_width * 100, 2),
        trend_slope=round(norm_slope * 1000, 3),
        weights=weights
    )

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: ÖZELLİK ÇIKARIMI
# ══════════════════════════════════════════════════════════════════════════════

def zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=float)
    mu, sigma = arr.mean(), arr.std()
    return (arr - mu) / (sigma + 1e-9)

def daily_returns(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=float)
    if len(arr) < 2:
        return np.zeros(1)
    return np.diff(arr) / (np.abs(arr[:-1]) + 1e-9)

def calc_rsi_arr(prices: np.ndarray, n: int = 14) -> np.ndarray:
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
        ag = (ag*(n-1) + gains[i]) / n
        al = (al*(n-1) + losses[i]) / n
        rsi[i+1] = 100 - 100 / (1 + ag / (al + 1e-9))
    return rsi

def calc_macd_hist(prices: np.ndarray) -> np.ndarray:
    prices = np.array(prices, dtype=float)
    def ema(x, n):
        k = 2/(n+1)
        e = [x[0]]
        for v in x[1:]: e.append(v*k + e[-1]*(1-k))
        return np.array(e)
    if len(prices) < 26:
        return np.zeros(len(prices))
    macd = ema(prices, 12) - ema(prices, 26)
    return macd - ema(macd, 9)

def volume_profile(volumes: np.ndarray, prices: np.ndarray, n_bins: int = 5) -> np.ndarray:
    volumes = np.array(volumes, dtype=float)
    prices = np.array(prices, dtype=float)
    p_min, p_max = prices.min(), prices.max()
    if p_max == p_min or volumes.sum() < 1e-9:
        return np.zeros(n_bins)
    bins = np.linspace(p_min, p_max, n_bins + 1)
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (prices >= bins[i]) & (prices < bins[i+1])
        profile[i] = volumes[mask].sum()
    total = profile.sum()
    return profile / (total + 1e-9)


def extract_features(prices: np.ndarray, volumes: np.ndarray) -> Dict:
    prices = np.array(prices, dtype=float)
    volumes = np.array(volumes, dtype=float)
    rets = daily_returns(prices)
    rsi = calc_rsi_arr(prices)
    macd_h = calc_macd_hist(prices)
    vol_norm = volumes / (volumes.mean() + 1e-9)

    return {
        'price_z':       zscore(prices),
        'returns':       rets,
        'ret_mean':      float(rets.mean()),
        'ret_std':       float(rets.std()),
        'ret_skew':      float(_skew(rets)),
        'rsi':           rsi,
        'rsi_end':       float(rsi[-1]),
        'rsi_mean':      float(rsi.mean()),
        'macd_hist':     macd_h,
        'macd_dir':      1.0 if macd_h[-1] > 0 else -1.0,
        'volume_z':      zscore(vol_norm),
        'volume_trend':  float(np.polyfit(np.arange(len(prices)), vol_norm, 1)[0]),
        'vol_profile':   volume_profile(volumes, prices),
    }

def _skew(arr: np.ndarray) -> float:
    arr = np.array(arr, dtype=float)
    n = len(arr)
    if n < 3:
        return 0.0
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-9:
        return 0.0
    return float(np.mean(((arr - mu) / sigma) ** 3))

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: MAHALANOBİS MESAFESİ
# ══════════════════════════════════════════════════════════════════════════════

def mahalanobis_sim(vec1: np.ndarray, vec2: np.ndarray,
                    cov_matrix: Optional[np.ndarray] = None) -> float:
    """
    İki özellik vektörü arasında Mahalanobis mesafesi hesapla.
    Kovaryans matrisi yoksa kimlik matrisi kullanılır (= Öklid mesafesi).
    
    Mesafe → 0-1 benzerlik skoru:
    sim = exp(-distance / scale)
    """
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    diff = v1 - v2

    if cov_matrix is not None:
        try:
            cov_inv = np.linalg.inv(cov_matrix + np.eye(len(cov_matrix)) * 1e-6)
            dist = float(np.sqrt(diff @ cov_inv @ diff))
        except np.linalg.LinAlgError:
            dist = float(np.sqrt(np.dot(diff, diff)))
    else:
        dist = float(np.sqrt(np.dot(diff, diff)))

    # Mesafeyi 0-1 benzerliğe çevir
    scale = np.sqrt(len(v1))  # Boyut sayısına göre ölçekle
    sim = float(np.exp(-dist / (scale + 1e-9)))
    return max(0.0, min(1.0, sim))


def build_feature_vector(feat: Dict) -> np.ndarray:
    """
    Tüm özelliklerden sabit boyutlu tek bir vektör oluştur.
    Mahalanobis için gerekli.
    """
    # Her özellikten skaler temsil al
    price_stats = [
        feat['price_z'].mean(),
        feat['price_z'].std(),
        float(np.polyfit(np.arange(len(feat['price_z'])), feat['price_z'], 1)[0]),
    ]
    ret_stats = [
        feat['ret_mean'] * 100,
        feat['ret_std'] * 100,
        feat['ret_skew'],
    ]
    rsi_stats = [
        feat['rsi_end'] / 100,
        feat['rsi_mean'] / 100,
    ]
    macd_stats = [
        feat['macd_dir'],
        feat['macd_hist'][-1] / (abs(feat['macd_hist']).mean() + 1e-9),
    ]
    vol_stats = [
        feat['volume_trend'],
        feat['volume_z'].mean(),
    ]
    vol_profile = feat['vol_profile'].tolist()  # 5 bin

    return np.array(price_stats + ret_stats + rsi_stats +
                    macd_stats + vol_stats + vol_profile, dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: ENSEMBLE OYLAMA
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_vote(dim_scores: Dict[str, float],
                  weights: Dict[str, float],
                  threshold: float = 0.60) -> Tuple[float, Dict]:
    """
    Her boyut bağımsız oy kullanır.
    
    Oylama mantığı:
    - Skor > threshold → "Benzer" oyu (1)
    - Skor <= threshold → "Farklı" oyu (0)
    
    Ağırlıklı oy oranı hesaplanır.
    Sonuç: (ensemble_skor, oy_detayı)
    """
    votes = {}
    weighted_yes = 0.0
    total_weight = 0.0

    for dim, score in dim_scores.items():
        w = weights.get(dim, 1/len(dim_scores))
        vote = 1 if score >= threshold else 0
        votes[dim] = {
            'score': round(score * 100, 1),
            'vote': vote,
            'weight': round(w * 100, 1),
            'weighted_contribution': round(score * w * 100, 1)
        }
        weighted_yes += score * w
        total_weight += w

    ensemble_score = weighted_yes / (total_weight + 1e-9)
    yes_votes = sum(1 for v in votes.values() if v['vote'] == 1)
    total_votes = len(votes)
    vote_ratio = yes_votes / (total_votes + 1e-9)

    # Consensus bonus: tüm boyutlar aynı yönde oy kullanıyorsa bonus
    consensus_bonus = 0.05 if vote_ratio == 1.0 else (0.02 if vote_ratio >= 0.8 else 0.0)

    final_score = min(1.0, ensemble_score + consensus_bonus)

    return final_score, {
        'votes': votes,
        'yes_votes': yes_votes,
        'total_votes': total_votes,
        'vote_ratio': round(vote_ratio * 100, 1),
        'consensus_bonus': round(consensus_bonus * 100, 1),
    }

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: DTW HESAPLAMA
# ══════════════════════════════════════════════════════════════════════════════

def dtw_sim(s1: np.ndarray, s2: np.ndarray, band: Optional[int] = None) -> float:
    n = len(s1)
    if n == 0:
        return 0.0
    band = band or max(2, n // 8)
    dtw = np.full((n+1, n+1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n+1):
        j0 = max(1, i - band)
        j1 = min(n, i + band) + 1
        for j in range(j0, j1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    dist = dtw[n, n] / n
    return max(0.0, float(np.exp(-dist * 1.5)))


def pearson_sim(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 3:
        return 0.5
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.5
    r = float(np.corrcoef(a, b)[0, 1])
    return (r + 1) / 2

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: BIST-PSI ANA SINIFI
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PSIResult:
    """BIST-PSI hesaplama sonucu."""
    score: float                    # 0-100 final skor
    regime: MarketRegime            # Tespit edilen piyasa rejimi
    dim_scores: Dict[str, float]    # Her boyutun skoru (0-100)
    weights: Dict[str, float]       # Kullanılan ağırlıklar
    mahalanobis_sim: float          # Mahalanobis benzerliği (0-100)
    ensemble_detail: Dict           # Ensemble oy detayı
    weighted_score: float           # Ağırlıklı boyut skoru (0-100)
    confidence_band: Tuple[float, float]  # (alt sınır, üst sınır)

    def summary(self) -> str:
        level = ("Yüksek" if self.score >= 75 else
                 "İyi" if self.score >= 65 else
                 "Orta" if self.score >= 55 else "Düşük")
        return (f"BIST-PSI: {self.score:.1f} ({level}) | "
                f"Rejim: {self.regime.describe()} | "
                f"Ensemble: {self.ensemble_detail['vote_ratio']}% oy")


class BISTPSI:
    """
    BIST Pattern Similarity Index — Ana sınıf.
    
    Üç katman:
    1. Rejim bazlı adaptif ağırlıklar
    2. Mahalanobis mesafesi
    3. Ensemble oylama
    """

    def __init__(self, ensemble_threshold: float = 0.60):
        self.ensemble_threshold = ensemble_threshold
        self._covariance_cache: Optional[np.ndarray] = None

    def update_covariance(self, feature_vectors: List[np.ndarray]) -> None:
        """
        Referans veri setinden kovaryans matrisi hesapla.
        Ne kadar çok örnek → daha iyi Mahalanobis.
        """
        if len(feature_vectors) < 3:
            self._covariance_cache = None
            return
        matrix = np.vstack(feature_vectors)
        cov = np.cov(matrix.T)
        # Sayısal kararlılık için küçük değer ekle
        cov += np.eye(cov.shape[0]) * 1e-6
        self._covariance_cache = cov

    def compute(self,
                tpl_prices: np.ndarray,
                tpl_volumes: np.ndarray,
                win_prices: np.ndarray,
                win_volumes: np.ndarray) -> Tuple[float, PSIResult]:
        """
        İki fiyat/hacim serisi arasında BIST-PSI hesapla.
        
        Returns:
            (score, PSIResult)
        """
        tpl_prices = np.array(tpl_prices, dtype=float)
        win_prices = np.array(win_prices, dtype=float)
        tpl_vols = np.array(tpl_volumes, dtype=float)
        win_vols = np.array(win_volumes, dtype=float)

        # Boyut eşitle
        n = min(len(tpl_prices), len(win_prices))
        tpl_prices = tpl_prices[-n:]
        win_prices = win_prices[-n:]
        tpl_vols = tpl_vols[-n:]
        win_vols = win_vols[-n:]

        # 1. Piyasa rejimi (şablon üzerinde)
        regime = detect_regime(tpl_prices, tpl_vols)

        # 2. Özellik çıkarımı
        feat_tpl = extract_features(tpl_prices, tpl_vols)
        feat_win = extract_features(win_prices, win_vols)

        # 3. Boyut bazlı benzerlik skorları
        # DTW
        s_dtw = dtw_sim(feat_tpl['price_z'], feat_win['price_z'])

        # Pearson
        s_pearson = pearson_sim(feat_tpl['price_z'], feat_win['price_z'])

        # Getiri dağılımı
        min_r = min(len(feat_tpl['returns']), len(feat_win['returns']))
        s_ret_corr = pearson_sim(feat_tpl['returns'][:min_r], feat_win['returns'][:min_r])
        s_ret_mean = 1 - min(1.0, abs(feat_tpl['ret_mean']-feat_win['ret_mean']) / (abs(feat_tpl['ret_mean'])+0.001))
        s_ret_std  = 1 - min(1.0, abs(feat_tpl['ret_std']-feat_win['ret_std']) / (feat_tpl['ret_std']+0.001))
        s_ret_skew = 1 - min(1.0, abs(feat_tpl['ret_skew']-feat_win['ret_skew']) / (abs(feat_tpl['ret_skew'])+0.5))
        s_returns  = 0.50*s_ret_corr + 0.20*s_ret_mean + 0.20*s_ret_std + 0.10*s_ret_skew

        # Hacim
        s_vol_profile = pearson_sim(feat_tpl['vol_profile'], feat_win['vol_profile'])
        s_vol_trend = 1 - min(1.0, abs(feat_tpl['volume_trend']-feat_win['volume_trend']))
        s_vol_shape = pearson_sim(feat_tpl['volume_z'], feat_win['volume_z'])
        s_volume = 0.40*s_vol_profile + 0.30*s_vol_trend + 0.30*s_vol_shape

        # Momentum (RSI + MACD)
        min_rsi = min(len(feat_tpl['rsi']), len(feat_win['rsi']))
        s_rsi_shape = pearson_sim(feat_tpl['rsi'][:min_rsi], feat_win['rsi'][:min_rsi])
        s_rsi_level = 1 - min(1.0, abs(feat_tpl['rsi_end']-feat_win['rsi_end']) / 50.0)
        min_macd = min(len(feat_tpl['macd_hist']), len(feat_win['macd_hist']))
        s_macd = pearson_sim(feat_tpl['macd_hist'][:min_macd], feat_win['macd_hist'][:min_macd])
        s_macd_dir = 1.0 if feat_tpl['macd_dir'] == feat_win['macd_dir'] else 0.0
        s_momentum = 0.30*s_rsi_shape + 0.20*s_rsi_level + 0.30*s_macd + 0.20*s_macd_dir

        # Formasyon (basit fiyat yapısı üzerinden)
        price_struct_tpl = self._price_structure(tpl_prices)
        price_struct_win = self._price_structure(win_prices)
        s_formation = 1 - min(1.0, np.mean(np.abs(
            np.array(price_struct_tpl) - np.array(price_struct_win)
        )))

        dim_scores = {
            'dtw':       s_dtw,
            'pearson':   s_pearson,
            'returns':   s_returns,
            'volume':    s_volume,
            'momentum':  s_momentum,
            'formation': s_formation,
        }

        # 4. Adaptif ağırlıklı skor
        weights = regime.weights
        weighted_score = sum(dim_scores[k] * weights[k] for k in dim_scores)

        # 5. Mahalanobis benzerliği
        vec_tpl = build_feature_vector(feat_tpl)
        vec_win = build_feature_vector(feat_win)
        mah_sim = mahalanobis_sim(vec_tpl, vec_win, self._covariance_cache)

        # 6. Ensemble oylama
        ensemble_score, ensemble_detail = ensemble_vote(
            dim_scores, weights, self.ensemble_threshold
        )

        # 7. BIST-PSI Final Skor
        # Ağırlıklı kombinasyon:
        # %50 adaptif ağırlıklı skor
        # %25 Mahalanobis benzerliği
        # %25 Ensemble skoru
        final = (0.50 * weighted_score +
                 0.25 * mah_sim +
                 0.25 * ensemble_score)

        final_score = round(final * 100, 1)

        # Güven bandı: ±1 sigma sapma beklentisi
        vote_ratio = ensemble_detail['vote_ratio'] / 100
        uncertainty = (1 - vote_ratio) * 8  # Düşük consensus → geniş bant
        lower = max(0, final_score - uncertainty)
        upper = min(100, final_score + uncertainty)

        result = PSIResult(
            score=final_score,
            regime=regime,
            dim_scores={k: round(v*100, 1) for k, v in dim_scores.items()},
            weights={k: round(v*100, 1) for k, v in weights.items()},
            mahalanobis_sim=round(mah_sim * 100, 1),
            ensemble_detail=ensemble_detail,
            weighted_score=round(weighted_score * 100, 1),
            confidence_band=(round(lower, 1), round(upper, 1))
        )

        return final_score, result

    def _price_structure(self, prices: np.ndarray) -> List[float]:
        """Fiyat yapısı özellikleri — formasyon boyutu için."""
        n = len(prices)
        if n < 4:
            return [0.5, 0.5, 0.5, 0.5]
        q = np.array_split(prices, 4)
        return [float(s.mean() / prices.mean()) for s in q]

    def batch_compute(self,
                      tpl_prices: np.ndarray,
                      tpl_volumes: np.ndarray,
                      candidates: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      min_score: float = 65.0) -> List[Tuple[str, float, PSIResult]]:
        """
        Birden fazla aday ile toplu BIST-PSI hesabı.
        min_score altındaki adayları filtrele.
        Sonucu skora göre sıralı döndür.
        
        candidates: {ticker: (prices, volumes)}
        """
        # Önce kovaryans matrisini güncelle (daha iyi Mahalanobis için)
        feat_vectors = []
        for prices, volumes in list(candidates.values())[:50]:  # Max 50 örnek
            try:
                n = min(len(prices), len(tpl_prices))
                feat = extract_features(prices[-n:], volumes[-n:])
                feat_vectors.append(build_feature_vector(feat))
            except Exception:
                pass
        if len(feat_vectors) >= 3:
            self.update_covariance(feat_vectors)

        results = []
        for ticker, (prices, volumes) in candidates.items():
            try:
                score, detail = self.compute(tpl_prices, tpl_volumes, prices, volumes)
                if score >= min_score:
                    results.append((ticker, score, detail))
            except Exception:
                pass

        results.sort(key=lambda x: x[1], reverse=True)
        return results
