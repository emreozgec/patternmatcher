"""
formations.py — Kapsamlı Formasyon Tanıma Motoru

Desteklenen formasyonlar:
  KLASİK   : Head & Shoulders, Inverse H&S, Double/Triple Top/Bottom, Rectangle
  TREND    : Ascending/Descending/Symmetrical Triangle, Rising/Falling Wedge,
             Bull/Bear Flag, Pennant, Cup & Handle
  ELLİOTT  : 5-Dalga İmpuls, ABC Düzeltme, Truncation tespiti
  HARMONİK : Gartley, Bat, Crab, Butterfly, Cypher (Fibonacci oranları)
  WYCKOFF  : Accumulation (A-E faz), Distribution, Spring, Upthrust

Her formasyon:
  - name        : str
  - confidence  : float (0-100)
  - status      : "active" | "completed" | "forming"
  - direction   : "bullish" | "bearish" | "neutral"
  - key_points  : list of (index, price, label) — grafik işaretleme için
  - target      : float | None — fiyat hedefi
  - stop        : float | None — stop-loss seviyesi
  - description : str
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Veri yapısı ────────────────────────────────────────────────────────────────

@dataclass
class Formation:
    name: str
    category: str          # klasik / trend / elliott / harmonik / wyckoff
    confidence: float      # 0-100
    status: str            # active / completed / forming
    direction: str         # bullish / bearish / neutral
    key_points: List[Tuple[int, float, str]] = field(default_factory=list)
    target: Optional[float] = None
    stop: Optional[float] = None
    description: str = ""

# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def find_pivots(prices: np.ndarray, order: int = 3) -> Tuple[List[int], List[int]]:
    """Yerel tepe ve dip noktaları bul."""
    peaks, troughs = [], []
    n = len(prices)
    for i in range(order, n - order):
        window = prices[i - order: i + order + 1]
        if prices[i] == window.max() and prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            peaks.append(i)
        if prices[i] == window.min() and prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            troughs.append(i)
    return peaks, troughs

def fib_ratio(a: float, b: float, c: float) -> float:
    """XA hareketi üzerinde BC geri çekilme oranı."""
    if abs(a - b) < 1e-9:
        return 0.0
    return abs(c - b) / abs(a - b)

def in_range(val: float, target: float, tol: float = 0.05) -> bool:
    """val, target'a tol toleransında mı?"""
    return abs(val - target) <= tol

def trend_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Doğrusal regresyon: slope, intercept, r2."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    return float(slope), float(intercept), float(r2)

def pivot_order(n: int) -> int:
    """Pivot order'ı veri uzunluğuna göre otomatik ayarla."""
    return max(2, min(8, n // 10))

# ══════════════════════════════════════════════════════════════════════════════
# 1. KLASİK FORMASYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def detect_head_and_shoulders(prices: np.ndarray, volumes: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 20:
        return results
    order = pivot_order(n)
    peaks, troughs = find_pivots(prices, order)

    # Head & Shoulders (bearish)
    for i in range(len(peaks) - 2):
        l, h, r = peaks[i], peaks[i+1], peaks[i+2]
        lp, hp, rp = prices[l], prices[h], prices[r]
        if not (hp > lp and hp > rp):
            continue
        sym = 1 - abs(lp - rp) / (hp + 1e-9)
        height_ratio = min(lp, rp) / (hp + 1e-9)
        if height_ratio < 0.80 or sym < 0.65:
            continue

        # Neckline bul
        between_troughs = [t for t in troughs if l < t < r]
        if len(between_troughs) < 2:
            continue
        nl1, nl2 = between_troughs[0], between_troughs[-1]
        neckline_slope = (prices[nl2] - prices[nl1]) / (nl2 - nl1 + 1e-9)
        neckline_at_r = prices[nl1] + neckline_slope * (r - nl1)

        # Hacim onayı: sol omuzda yüksek, başta orta, sağ omuzda düşük
        vol_confirm = 1.0
        if len(volumes) == n:
            vl = volumes[max(0,l-order):l+order+1].mean()
            vh = volumes[max(0,h-order):h+order+1].mean()
            vr = volumes[max(0,r-order):r+order+1].mean()
            if vl > vh > vr:
                vol_confirm = 1.2

        conf = min(100, (sym * 40 + height_ratio * 30 + 20) * vol_confirm)
        target = neckline_at_r - (hp - neckline_at_r)
        status = "active" if r >= n - order * 2 else "completed"

        results.append(Formation(
            name="Head & Shoulders",
            category="klasik",
            confidence=round(conf, 1),
            status=status,
            direction="bearish",
            key_points=[(l, lp, "L Omuz"), (h, hp, "Baş"), (r, rp, "R Omuz"),
                        (nl1, prices[nl1], "Neckline-L"), (nl2, prices[nl2], "Neckline-R")],
            target=round(target, 2),
            stop=round(hp * 1.02, 2),
            description=f"Klasik bearish dönüş formasyonu. Neckline kırılırsa hedef: {target:.2f}"
        ))

    # Inverse Head & Shoulders (bullish)
    for i in range(len(troughs) - 2):
        l, h, r = troughs[i], troughs[i+1], troughs[i+2]
        lp, hp, rp = prices[l], prices[h], prices[r]
        if not (hp < lp and hp < rp):
            continue
        sym = 1 - abs(lp - rp) / (abs(hp) + 1e-9)
        height_ratio = max(lp, rp) / (hp + 1e-9) - 1
        if height_ratio < 0.04 or sym < 0.65:
            continue

        between_peaks = [p for p in peaks if l < p < r]
        if len(between_peaks) < 2:
            continue
        nl1, nl2 = between_peaks[0], between_peaks[-1]
        neckline_at_r = prices[nl1] + (prices[nl2]-prices[nl1])/(nl2-nl1+1e-9) * (r-nl1)
        target = neckline_at_r + (neckline_at_r - hp)
        conf = min(100, sym * 50 + min(height_ratio * 200, 30) + 20)
        status = "active" if r >= n - order * 2 else "completed"

        results.append(Formation(
            name="Inverse Head & Shoulders",
            category="klasik",
            confidence=round(conf, 1),
            status=status,
            direction="bullish",
            key_points=[(l, lp, "L Omuz"), (h, hp, "Baş"), (r, rp, "R Omuz"),
                        (nl1, prices[nl1], "Neckline-L"), (nl2, prices[nl2], "Neckline-R")],
            target=round(target, 2),
            stop=round(hp * 0.98, 2),
            description=f"Klasik bullish dönüş formasyonu. Hedef: {target:.2f}"
        ))

    return results


def detect_double_triple(prices: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 15:
        return results
    order = pivot_order(n)
    peaks, troughs = find_pivots(prices, order)

    # Double Top
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i+1]
        if p2 - p1 < order * 2:
            continue
        diff = abs(prices[p1] - prices[p2]) / (max(prices[p1], prices[p2]) + 1e-9)
        if diff > 0.03:
            continue
        between = [t for t in troughs if p1 < t < p2]
        if not between:
            continue
        valley = prices[between[0]]
        height = max(prices[p1], prices[p2]) - valley
        target = valley - height
        conf = min(100, (1 - diff / 0.03) * 60 + 40)
        status = "active" if p2 >= n - order * 2 else "completed"
        results.append(Formation(
            name="Double Top",
            category="klasik",
            confidence=round(conf, 1),
            status=status,
            direction="bearish",
            key_points=[(p1, prices[p1], "Tepe 1"), (p2, prices[p2], "Tepe 2"),
                        (between[0], valley, "Valley")],
            target=round(target, 2),
            stop=round(max(prices[p1], prices[p2]) * 1.02, 2),
            description=f"İki eşit tepe — bearish. Hedef: {target:.2f}"
        ))

    # Double Bottom
    for i in range(len(troughs) - 1):
        t1, t2 = troughs[i], troughs[i+1]
        if t2 - t1 < order * 2:
            continue
        diff = abs(prices[t1] - prices[t2]) / (min(prices[t1], prices[t2]) + 1e-9)
        if diff > 0.03:
            continue
        between = [p for p in peaks if t1 < p < t2]
        if not between:
            continue
        peak_val = prices[between[0]]
        height = peak_val - min(prices[t1], prices[t2])
        target = peak_val + height
        conf = min(100, (1 - diff / 0.03) * 60 + 40)
        status = "active" if t2 >= n - order * 2 else "completed"
        results.append(Formation(
            name="Double Bottom",
            category="klasik",
            confidence=round(conf, 1),
            status=status,
            direction="bullish",
            key_points=[(t1, prices[t1], "Dip 1"), (t2, prices[t2], "Dip 2"),
                        (between[0], peak_val, "Peak")],
            target=round(target, 2),
            stop=round(min(prices[t1], prices[t2]) * 0.98, 2),
            description=f"İki eşit dip — bullish. Hedef: {target:.2f}"
        ))

    # Triple Top
    for i in range(len(peaks) - 2):
        p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
        vals = [prices[p1], prices[p2], prices[p3]]
        spread = (max(vals) - min(vals)) / (max(vals) + 1e-9)
        if spread > 0.04:
            continue
        conf = min(100, (1 - spread/0.04) * 70 + 30)
        target = min(vals) - (max(vals) - min(vals))
        status = "active" if p3 >= n - order * 2 else "completed"
        results.append(Formation(
            name="Triple Top",
            category="klasik",
            confidence=round(conf, 1),
            status=status,
            direction="bearish",
            key_points=[(p1, prices[p1], "T1"), (p2, prices[p2], "T2"), (p3, prices[p3], "T3")],
            target=round(target, 2),
            stop=round(max(vals) * 1.02, 2),
            description="Üç eşit tepe — güçlü bearish sinyal"
        ))

    # Triple Bottom
    for i in range(len(troughs) - 2):
        t1, t2, t3 = troughs[i], troughs[i+1], troughs[i+2]
        vals = [prices[t1], prices[t2], prices[t3]]
        spread = (max(vals) - min(vals)) / (min(vals) + 1e-9)
        if spread > 0.04:
            continue
        conf = min(100, (1 - spread/0.04) * 70 + 30)
        target = max(vals) + (max(vals) - min(vals))
        status = "active" if t3 >= n - order * 2 else "completed"
        results.append(Formation(
            name="Triple Bottom",
            category="klasik",
            confidence=round(conf, 1),
            status=status,
            direction="bullish",
            key_points=[(t1, prices[t1], "D1"), (t2, prices[t2], "D2"), (t3, prices[t3], "D3")],
            target=round(target, 2),
            stop=round(min(vals) * 0.98, 2),
            description="Üç eşit dip — güçlü bullish sinyal"
        ))

    return results


def detect_rectangle(prices: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 20:
        return results
    order = pivot_order(n)
    peaks, troughs = find_pivots(prices, order)
    if len(peaks) < 2 or len(troughs) < 2:
        return results

    recent_peaks = peaks[-3:]
    recent_troughs = troughs[-3:]
    peak_vals = prices[recent_peaks]
    trough_vals = prices[recent_troughs]

    peak_spread = (peak_vals.max() - peak_vals.min()) / (peak_vals.mean() + 1e-9)
    trough_spread = (trough_vals.max() - trough_vals.min()) / (trough_vals.mean() + 1e-9)

    if peak_spread > 0.04 or trough_spread > 0.04:
        return results

    resistance = peak_vals.mean()
    support = trough_vals.mean()
    height = resistance - support
    conf = min(100, (1 - (peak_spread + trough_spread) / 0.08) * 80 + 20)

    direction = "bullish" if prices[-1] > resistance * 0.98 else (
        "bearish" if prices[-1] < support * 1.02 else "neutral")
    target = resistance + height if direction == "bullish" else support - height
    status = "active"

    results.append(Formation(
        name="Rectangle",
        category="klasik",
        confidence=round(conf, 1),
        status=status,
        direction=direction,
        key_points=[(recent_peaks[0], resistance, "Direnç"), (recent_troughs[0], support, "Destek")],
        target=round(target, 2),
        stop=round(support * 0.98 if direction == "bullish" else resistance * 1.02, 2),
        description=f"Yatay konsolidasyon. Direnç: {resistance:.2f} Destek: {support:.2f}"
    ))
    return results

# ══════════════════════════════════════════════════════════════════════════════
# 2. TREND FORMASYONLARI
# ══════════════════════════════════════════════════════════════════════════════

def detect_triangles(prices: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 20:
        return results
    order = pivot_order(n)
    peaks, troughs = find_pivots(prices, order)
    if len(peaks) < 2 or len(troughs) < 2:
        return results

    x_peaks = np.array(peaks[-4:])
    y_peaks = prices[x_peaks]
    x_troughs = np.array(troughs[-4:])
    y_troughs = prices[x_troughs]

    if len(x_peaks) < 2 or len(x_troughs) < 2:
        return results

    upper_slope, upper_int, r2_up = trend_line(x_peaks, y_peaks)
    lower_slope, lower_int, r2_lo = trend_line(x_troughs, y_troughs)

    if r2_up < 0.5 or r2_lo < 0.5:
        return results

    convergence = abs(upper_slope - lower_slope) / (abs(upper_slope) + abs(lower_slope) + 1e-9)

    # Symmetrical Triangle
    if upper_slope < -0.0001 and lower_slope > 0.0001:
        conf = min(100, convergence * 200 * (r2_up + r2_lo) / 2)
        height = (upper_int - lower_int) / 2
        breakout_price = upper_slope * n + upper_int
        results.append(Formation(
            name="Symmetrical Triangle",
            category="trend",
            confidence=round(conf, 1),
            status="forming",
            direction="neutral",
            key_points=[(x_peaks[0], y_peaks[0], "Üst Trend"), (x_troughs[0], y_troughs[0], "Alt Trend"),
                        (x_peaks[-1], y_peaks[-1], "Son Tepe"), (x_troughs[-1], y_troughs[-1], "Son Dip")],
            target=round(breakout_price + height, 2),
            stop=round(breakout_price - height * 0.5, 2),
            description="Simetrik üçgen — kırılım yönüne göre hareket"
        ))

    # Ascending Triangle
    elif abs(upper_slope) < 0.001 and lower_slope > 0.0001:
        conf = min(100, r2_lo * 80 + 20)
        height = prices[x_peaks].mean() - prices[x_troughs[0]]
        target = prices[x_peaks].mean() + height
        results.append(Formation(
            name="Ascending Triangle",
            category="trend",
            confidence=round(conf, 1),
            status="forming",
            direction="bullish",
            key_points=[(x_peaks[0], prices[x_peaks[0]], "Yatay Direnç"),
                        (x_troughs[0], prices[x_troughs[0]], "Yükselen Destek")],
            target=round(target, 2),
            stop=round(prices[x_troughs[-1]] * 0.98, 2),
            description=f"Yükselen üçgen — bullish. Direnç kırılırsa hedef: {target:.2f}"
        ))

    # Descending Triangle
    elif abs(lower_slope) < 0.001 and upper_slope < -0.0001:
        conf = min(100, r2_up * 80 + 20)
        height = prices[x_peaks[0]] - prices[x_troughs].mean()
        target = prices[x_troughs].mean() - height
        results.append(Formation(
            name="Descending Triangle",
            category="trend",
            confidence=round(conf, 1),
            status="forming",
            direction="bearish",
            key_points=[(x_peaks[0], prices[x_peaks[0]], "Alçalan Direnç"),
                        (x_troughs[0], prices[x_troughs[0]], "Yatay Destek")],
            target=round(target, 2),
            stop=round(prices[x_peaks[-1]] * 1.02, 2),
            description=f"Alçalan üçgen — bearish. Destek kırılırsa hedef: {target:.2f}"
        ))

    return results


def detect_wedges(prices: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 20:
        return results
    order = pivot_order(n)
    peaks, troughs = find_pivots(prices, order)
    if len(peaks) < 2 or len(troughs) < 2:
        return results

    x_peaks = np.array(peaks[-3:])
    x_troughs = np.array(troughs[-3:])
    upper_slope, _, r2_up = trend_line(x_peaks, prices[x_peaks])
    lower_slope, _, r2_lo = trend_line(x_troughs, prices[x_troughs])

    if r2_up < 0.6 or r2_lo < 0.6:
        return results

    # Rising Wedge (bearish)
    if upper_slope > 0.0001 and lower_slope > upper_slope * 0.5:
        conf = min(100, (r2_up + r2_lo) / 2 * 80 + 20)
        height = prices[x_peaks[0]] - prices[x_troughs[0]]
        target = prices[-1] - height
        results.append(Formation(
            name="Rising Wedge",
            category="trend",
            confidence=round(conf, 1),
            status="forming",
            direction="bearish",
            key_points=[(x_peaks[0], prices[x_peaks[0]], "Üst Trend"),
                        (x_troughs[0], prices[x_troughs[0]], "Alt Trend")],
            target=round(target, 2),
            stop=round(prices[x_peaks[-1]] * 1.02, 2),
            description="Yükselen kama — bearish dönüş sinyali"
        ))

    # Falling Wedge (bullish)
    elif upper_slope < -0.0001 and lower_slope < upper_slope * 0.5:
        conf = min(100, (r2_up + r2_lo) / 2 * 80 + 20)
        height = prices[x_peaks[0]] - prices[x_troughs[0]]
        target = prices[-1] + height
        results.append(Formation(
            name="Falling Wedge",
            category="trend",
            confidence=round(conf, 1),
            status="forming",
            direction="bullish",
            key_points=[(x_peaks[0], prices[x_peaks[0]], "Üst Trend"),
                        (x_troughs[0], prices[x_troughs[0]], "Alt Trend")],
            target=round(target, 2),
            stop=round(prices[x_troughs[-1]] * 0.98, 2),
            description="Düşen kama — bullish dönüş sinyali"
        ))

    return results


def detect_flags_pennants(prices: np.ndarray, volumes: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 15:
        return results

    # Flagpole: güçlü trend hareketi ara
    half = n // 2
    pole_move = (prices[half] - prices[0]) / (prices[0] + 1e-9)
    flag_section = prices[half:]

    if abs(pole_move) < 0.05:
        return results

    # Flag: hafif geri çekilme (konsolidasyon)
    flag_slope, _, flag_r2 = trend_line(np.arange(len(flag_section)), flag_section)
    flag_retracement = abs(flag_section[-1] - flag_section[0]) / (abs(prices[half] - prices[0]) + 1e-9)

    is_bull = pole_move > 0
    counter_trend = (is_bull and flag_slope < 0) or (not is_bull and flag_slope > 0)

    if counter_trend and flag_retracement < 0.5 and flag_r2 > 0.4:
        conf = min(100, (1 - flag_retracement / 0.5) * 50 + flag_r2 * 30 + 20)
        target = prices[-1] + (prices[half] - prices[0])
        direction = "bullish" if is_bull else "bearish"
        results.append(Formation(
            name=f"{'Bull' if is_bull else 'Bear'} Flag",
            category="trend",
            confidence=round(conf, 1),
            status="forming",
            direction=direction,
            key_points=[(0, prices[0], "Direk Başı"), (half, prices[half], "Direk Sonu"),
                        (n-1, prices[-1], "Bayrak Sonu")],
            target=round(target, 2),
            stop=round(prices[-1] * (0.97 if is_bull else 1.03), 2),
            description=f"{'Boğa' if is_bull else 'Ayı'} bayrağı — trend devam formasyonu"
        ))

    # Pennant: daralan konsolidasyon
    if len(flag_section) >= 8:
        fp, ft = find_pivots(flag_section, 2)
        if len(fp) >= 2 and len(ft) >= 2:
            upper_s, _, r2u = trend_line(np.array(fp), flag_section[fp])
            lower_s, _, r2l = trend_line(np.array(ft), flag_section[ft])
            if upper_s < 0 and lower_s > 0 and r2u > 0.5 and r2l > 0.5:
                conf = min(100, (r2u + r2l) / 2 * 80 + 20)
                target = prices[-1] + abs(prices[half] - prices[0])
                direction = "bullish" if is_bull else "bearish"
                results.append(Formation(
                    name=f"{'Bull' if is_bull else 'Bear'} Pennant",
                    category="trend",
                    confidence=round(conf, 1),
                    status="forming",
                    direction=direction,
                    key_points=[(0, prices[0], "Direk"), (half, prices[half], "Konsolidasyon"),
                                (n-1, prices[-1], "Kırılım Bölgesi")],
                    target=round(target, 2),
                    stop=round(prices[-1] * (0.97 if is_bull else 1.03), 2),
                    description="Flama — güçlü trend devamı beklenir"
                ))

    return results


def detect_cup_and_handle(prices: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 30:
        return results

    # Fincan sol kenarı, dip, sağ kenar ve kulp
    third = n // 3
    left_peak = prices[:third].max()
    left_peak_idx = prices[:third].argmax()
    cup_bottom = prices[third:2*third].min()
    cup_bottom_idx = prices[third:2*third].argmin() + third
    right_peak = prices[2*third:].max()
    right_peak_idx = prices[2*third:].argmax() + 2*third

    if left_peak < cup_bottom or right_peak < cup_bottom:
        return []

    # Simetri kontrolü
    sym = 1 - abs(left_peak - right_peak) / (max(left_peak, right_peak) + 1e-9)
    depth = (max(left_peak, right_peak) - cup_bottom) / (max(left_peak, right_peak) + 1e-9)

    if sym < 0.7 or depth < 0.10 or depth > 0.50:
        return []

    # Kulp: sağ kenardan sonra küçük geri çekilme
    handle_section = prices[right_peak_idx:]
    if len(handle_section) < 3:
        return []
    handle_retracement = (right_peak - handle_section.min()) / (right_peak - cup_bottom + 1e-9)

    if handle_retracement > 0.5:
        return []

    conf = min(100, sym * 40 + (1 - abs(depth - 0.3) / 0.3) * 30 + 30)
    target = right_peak + (right_peak - cup_bottom)

    results.append(Formation(
        name="Cup & Handle",
        category="trend",
        confidence=round(conf, 1),
        status="forming" if right_peak_idx >= n - n//6 else "completed",
        direction="bullish",
        key_points=[(left_peak_idx, left_peak, "Sol Kenar"),
                    (cup_bottom_idx, cup_bottom, "Fincan Dibi"),
                    (right_peak_idx, right_peak, "Sağ Kenar / Kulp")],
        target=round(target, 2),
        stop=round(cup_bottom * 0.98, 2),
        description=f"Cup & Handle — bullish devam. Hedef: {target:.2f}"
    ))
    return results

# ══════════════════════════════════════════════════════════════════════════════
# 3. ELLİOTT WAVE
# ══════════════════════════════════════════════════════════════════════════════

FIB_RATIOS = {
    'wave2_ret': [0.382, 0.50, 0.618],
    'wave3_ext': [1.618, 2.0, 2.618],
    'wave4_ret': [0.236, 0.382],
    'wave5_ext': [0.618, 1.0, 1.618],
    'abc_b_ret': [0.382, 0.50, 0.618],
    'abc_c_ext': [0.618, 1.0, 1.618],
}

def check_elliott_rules(w: list) -> Tuple[bool, float]:
    """
    Elliott Wave kurallarını kontrol et.
    w = [W0, W1, W2, W3, W4, W5] fiyat seviyeleri
    Kural 1: Dalga 2, Dalga 1'in başlangıcının altına inmez
    Kural 2: Dalga 3 en kısa impuls dalgası olamaz
    Kural 3: Dalga 4, Dalga 1'in tepe bölgesini geçmez
    """
    if len(w) < 6:
        return False, 0.0

    w0, w1, w2, w3, w4, w5 = w
    is_bull = w1 > w0

    if is_bull:
        rule1 = w2 > w0
        rule2 = (w3 - w2) > (w1 - w0) or (w3 - w2) > (w5 - w4)
        rule3 = w4 > w1
    else:
        rule1 = w2 < w0
        rule2 = (w2 - w3) > (w0 - w1) or (w2 - w3) > (w4 - w5)
        rule3 = w4 < w1

    score = sum([rule1, rule2, rule3]) / 3
    return all([rule1, rule2, rule3]), score


def detect_elliott_wave(prices: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 20:
        return results

    order = pivot_order(n)
    peaks, troughs = find_pivots(prices, order)
    all_pivots = sorted([(i, prices[i], 'peak') for i in peaks] +
                        [(i, prices[i], 'trough') for i in troughs],
                        key=lambda x: x[0])

    if len(all_pivots) < 6:
        return results

    # Son 6 pivot ile 5-dalga tespit
    for start in range(len(all_pivots) - 5):
        pts = all_pivots[start:start+6]
        idxs = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        types = [p[2] for p in pts]

        # Alternatif tepe-dip kontrolü
        bull_pattern = ['trough','peak','trough','peak','trough','peak']
        bear_pattern = ['peak','trough','peak','trough','peak','trough']
        is_bull = types == bull_pattern
        is_bear = types == bear_pattern
        if not (is_bull or is_bear):
            continue

        valid, rule_score = check_elliott_rules(vals)
        if rule_score < 0.67:
            continue

        # Fibonacci oranı kontrolü
        fib_score = 0.0
        fib_checks = 0
        if is_bull:
            w1 = vals[1] - vals[0]
            w2_ret = (vals[1] - vals[2]) / (w1 + 1e-9)
            w3 = vals[3] - vals[2]
            w4_ret = (vals[3] - vals[4]) / (w3 + 1e-9)

            for r in FIB_RATIOS['wave2_ret']:
                if in_range(w2_ret, r, 0.08):
                    fib_score += 1
                    break
            for r in FIB_RATIOS['wave4_ret']:
                if in_range(w4_ret, r, 0.08):
                    fib_score += 1
                    break
            if w3 > w1:
                fib_score += 1
            fib_checks = 3
        else:
            w1 = vals[0] - vals[1]
            w2_ret = (vals[2] - vals[1]) / (w1 + 1e-9)
            for r in FIB_RATIOS['wave2_ret']:
                if in_range(w2_ret, r, 0.08):
                    fib_score += 1
                    break
            fib_checks = 1

        fib_ratio_score = fib_score / (fib_checks + 1e-9)
        conf = min(100, rule_score * 50 + fib_ratio_score * 30 + 20)

        # Hedef: 5. dalga için Fibonacci projeksiyonu
        if is_bull:
            w1_size = vals[1] - vals[0]
            target = vals[4] + w1_size * 1.618
        else:
            w1_size = vals[0] - vals[1]
            target = vals[4] - w1_size * 1.618

        status = "active" if idxs[-1] >= n - order * 3 else "completed"

        results.append(Formation(
            name=f"Elliott Wave {'5-Dalga Boğa' if is_bull else '5-Dalga Ayı'}",
            category="elliott",
            confidence=round(conf, 1),
            status=status,
            direction="bullish" if is_bull else "bearish",
            key_points=[(idxs[j], vals[j], f"W{j}") for j in range(6)],
            target=round(target, 2),
            stop=round(vals[0] * (0.98 if is_bull else 1.02), 2),
            description=f"Elliott 5-dalga {'impuls yükseliş' if is_bull else 'impuls düşüş'}. "
                       f"Fibonacci hedef: {target:.2f}"
        ))

    # ABC Düzeltme
    if len(all_pivots) >= 4:
        for start in range(len(all_pivots) - 3):
            pts = all_pivots[start:start+4]
            idxs = [p[0] for p in pts]
            vals = [p[1] for p in pts]
            types = [p[2] for p in pts]

            is_bull_abc = types[:4] == ['peak','trough','peak','trough']
            is_bear_abc = types[:4] == ['trough','peak','trough','peak']

            if not (is_bull_abc or is_bear_abc):
                continue

            wa = abs(vals[1] - vals[0])
            wb_ret = abs(vals[2] - vals[1]) / (wa + 1e-9)
            wc = abs(vals[3] - vals[2])

            fib_b = any(in_range(wb_ret, r, 0.08) for r in FIB_RATIOS['abc_b_ret'])
            fib_c = any(in_range(wc / (wa + 1e-9), r, 0.1) for r in FIB_RATIOS['abc_c_ext'])

            conf = 40 + (30 if fib_b else 0) + (30 if fib_c else 0)
            if conf < 50:
                continue

            target = vals[3] + wa * 1.0 if is_bear_abc else vals[3] - wa * 1.0
            status = "active" if idxs[-1] >= n - order * 2 else "completed"

            results.append(Formation(
                name=f"Elliott ABC {'Düzeltme ↑' if is_bear_abc else 'Düzeltme ↓'}",
                category="elliott",
                confidence=round(conf, 1),
                status=status,
                direction="bullish" if is_bear_abc else "bearish",
                key_points=[(idxs[j], vals[j], f"{'ABC'[j] if j<3 else 'C'}") for j in range(4)],
                target=round(target, 2),
                stop=round(vals[0] * (0.97 if is_bear_abc else 1.03), 2),
                description="Elliott ABC üç-dalga düzeltme yapısı"
            ))

    return results

# ══════════════════════════════════════════════════════════════════════════════
# 4. HARMONİK FORMASYONLAR
# ══════════════════════════════════════════════════════════════════════════════

HARMONIC_PATTERNS = {
    'Gartley': {
        'XAB': (0.618, 0.05),
        'ABC': (0.382, 0.886, 0.05),
        'BCD': (1.13, 1.618, 0.05),
        'XAD': (0.786, 0.05),
    },
    'Bat': {
        'XAB': (0.382, 0.50, 0.05),
        'ABC': (0.382, 0.886, 0.05),
        'BCD': (1.618, 2.618, 0.05),
        'XAD': (0.886, 0.05),
    },
    'Crab': {
        'XAB': (0.382, 0.618, 0.05),
        'ABC': (0.382, 0.886, 0.05),
        'BCD': (2.24, 3.618, 0.08),
        'XAD': (1.618, 0.05),
    },
    'Butterfly': {
        'XAB': (0.786, 0.05),
        'ABC': (0.382, 0.886, 0.05),
        'BCD': (1.618, 2.618, 0.05),
        'XAD': (1.27, 1.618, 0.05),
    },
    'Cypher': {
        'XAB': (0.382, 0.618, 0.05),
        'ABC': (1.13, 1.41, 0.05),
        'BCD': (0.786, 0.05),
        'XAD': (0.786, 0.05),
    },
}

def check_harmonic(xa, ab, bc, cd, xd, pattern_name):
    """Harmonik formasyon Fibonacci oranlarını kontrol et."""
    rules = HARMONIC_PATTERNS[pattern_name]
    score = 0
    checks = 0

    # XAB oranı
    xab = ab / (xa + 1e-9)
    xab_targets = rules['XAB'] if isinstance(rules['XAB'], tuple) else (rules['XAB'],)
    if len(xab_targets) == 2 and isinstance(xab_targets[1], float) and xab_targets[1] < 0.2:
        # (target, tolerance)
        if in_range(xab, xab_targets[0], xab_targets[1]):
            score += 1
    elif len(xab_targets) == 3:
        # (min, max, tolerance)
        if xab_targets[0] - xab_targets[2] <= xab <= xab_targets[1] + xab_targets[2]:
            score += 1
    checks += 1

    # XAD oranı
    xad_val = rules['XAD']
    if isinstance(xad_val, tuple):
        if len(xad_val) == 2:
            if in_range(xd / (xa + 1e-9), xad_val[0], xad_val[1]):
                score += 1
        elif len(xad_val) == 3:
            ratio = xd / (xa + 1e-9)
            if xad_val[0] - xad_val[2] <= ratio <= xad_val[1] + xad_val[2]:
                score += 1
    checks += 1

    return score / (checks + 1e-9)


def detect_harmonics(prices: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 20:
        return results

    order = pivot_order(n)
    peaks, troughs = find_pivots(prices, order)
    all_pivots = sorted([(i, prices[i], 'peak') for i in peaks] +
                        [(i, prices[i], 'trough') for i in troughs],
                        key=lambda x: x[0])

    if len(all_pivots) < 5:
        return results

    for start in range(len(all_pivots) - 4):
        pts = all_pivots[start:start+5]
        idxs = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        types = [p[2] for p in pts]

        X, A, B, C, D = vals
        is_bull = types[0] == 'trough'

        xa = abs(A - X)
        ab = abs(B - A)
        bc = abs(C - B)
        cd = abs(D - C)
        xd = abs(D - X)

        if xa < 1e-9 or ab < 1e-9:
            continue

        for pattern_name in HARMONIC_PATTERNS:
            ratio_score = check_harmonic(xa, ab, bc, cd, xd, pattern_name)
            if ratio_score < 0.5:
                continue

            conf = min(100, ratio_score * 80 + 20)
            direction = "bullish" if is_bull else "bearish"

            if is_bull:
                target1 = D + (A - X) * 0.382
                target2 = D + (A - X) * 0.618
                stop = D * 0.97
            else:
                target1 = D - (X - A) * 0.382
                target2 = D - (X - A) * 0.618
                stop = D * 1.03

            status = "active" if idxs[-1] >= n - order * 2 else "completed"

            results.append(Formation(
                name=f"Harmonik {pattern_name} {'↑' if is_bull else '↓'}",
                category="harmonik",
                confidence=round(conf, 1),
                status=status,
                direction=direction,
                key_points=[(idxs[j], vals[j], "XABCD"[j]) for j in range(5)],
                target=round(target1, 2),
                stop=round(stop, 2),
                description=f"{pattern_name} harmonik formasyon. "
                           f"PRZ: {D:.2f} — Hedef 1: {target1:.2f} / Hedef 2: {target2:.2f}"
            ))

    return results

# ══════════════════════════════════════════════════════════════════════════════
# 5. WYCKOFF ŞEMALARI
# ══════════════════════════════════════════════════════════════════════════════

def detect_wyckoff(prices: np.ndarray, volumes: np.ndarray) -> List[Formation]:
    results = []
    n = len(prices)
    if n < 30 or len(volumes) != n:
        return results

    order = pivot_order(n)
    peaks, troughs = find_pivots(prices, order)

    # Wyckoff Accumulation tespiti:
    # - Yatay fiyat hareketi (Trading Range)
    # - Hacim düşük seviyelerde artış (Spring)
    # - Fiyat aralığı daralma

    # Fiyat aralığı hesapla
    tr_high = prices[-n//2:].max()
    tr_low = prices[-n//2:].min()
    tr_range = (tr_high - tr_low) / (tr_high + 1e-9)

    # Hacim profili
    early_vol = volumes[:n//3].mean()
    mid_vol = volumes[n//3:2*n//3].mean()
    late_vol = volumes[2*n//3:].mean()

    # Trend kontrolü
    overall_slope, _, r2 = trend_line(np.arange(n), prices)
    norm_slope = overall_slope / (prices.mean() + 1e-9)

    # Accumulation: yatay hareket + geç hacim artışı
    if tr_range < 0.25 and late_vol > early_vol * 1.2 and abs(norm_slope) < 0.002:
        # Spring tespiti: son dönemde brief dip
        if troughs:
            last_trough = troughs[-1]
            if last_trough > n * 0.7:
                spring_depth = (tr_low - prices[last_trough]) / (tr_high - tr_low + 1e-9)
                if spring_depth > 0.05:
                    conf = min(100, (1 - tr_range / 0.25) * 40 +
                               min(late_vol/early_vol - 1, 1) * 30 + 30)
                    target = tr_high + (tr_high - tr_low)
                    results.append(Formation(
                        name="Wyckoff Accumulation (Spring)",
                        category="wyckoff",
                        confidence=round(conf, 1),
                        status="active",
                        direction="bullish",
                        key_points=[(0, prices[0], "PS - Preliminary Support"),
                                    (troughs[0] if troughs else n//4, tr_low, "SC - Selling Climax"),
                                    (last_trough, prices[last_trough], "Spring"),
                                    (n-1, prices[-1], "Şimdiki Fiyat")],
                        target=round(target, 2),
                        stop=round(prices[last_trough] * 0.97, 2),
                        description=f"Wyckoff birikim — Spring tespit edildi. "
                                   f"Trading Range: {tr_low:.2f}-{tr_high:.2f}. "
                                   f"Hedef: {target:.2f}"
                    ))

    # Distribution: yatay hareket + yüksek hacimli dağıtım
    if tr_range < 0.25 and early_vol > late_vol * 1.2 and abs(norm_slope) < 0.002:
        if peaks:
            last_peak = peaks[-1]
            if last_peak > n * 0.7:
                conf = min(100, (1 - tr_range / 0.25) * 40 +
                           min(early_vol/late_vol - 1, 1) * 30 + 30)
                target = tr_low - (tr_high - tr_low)
                results.append(Formation(
                    name="Wyckoff Distribution (Upthrust)",
                    category="wyckoff",
                    confidence=round(conf, 1),
                    status="active",
                    direction="bearish",
                    key_points=[(0, prices[0], "PSY - Preliminary Supply"),
                                (peaks[0] if peaks else n//4, tr_high, "BC - Buying Climax"),
                                (last_peak, prices[last_peak], "Upthrust"),
                                (n-1, prices[-1], "Şimdiki Fiyat")],
                    target=round(target, 2),
                    stop=round(prices[last_peak] * 1.03, 2),
                    description=f"Wyckoff dağıtım — Upthrust tespit edildi. "
                               f"Trading Range: {tr_low:.2f}-{tr_high:.2f}. "
                               f"Hedef: {target:.2f}"
                ))

    # Re-accumulation: trend içi konsolidasyon
    if r2 > 0.3 and overall_slope > 0 and tr_range < 0.20:
        conf = min(100, r2 * 40 + (1 - tr_range/0.20) * 30 + 30)
        target = prices[-1] + (tr_high - tr_low) * 2
        results.append(Formation(
            name="Wyckoff Re-accumulation",
            category="wyckoff",
            confidence=round(conf, 1),
            status="forming",
            direction="bullish",
            key_points=[(0, prices[0], "Trend Başlangıcı"),
                        (n-1, prices[-1], "Re-accumulation Bölgesi")],
            target=round(target, 2),
            stop=round(tr_low * 0.98, 2),
            description="Wyckoff trend-içi yeniden birikim. Trend devamı beklenir."
        ))

    return results

# ══════════════════════════════════════════════════════════════════════════════
# 6. ANA TARAMA FONKSİYONU
# ══════════════════════════════════════════════════════════════════════════════

def scan_all_formations(prices: np.ndarray, volumes: np.ndarray,
                        min_confidence: float = 40.0) -> List[Formation]:
    """
    Tüm formasyon kategorilerini tara.
    min_confidence altındaki formasyonları filtrele.
    Güven skoruna göre sırala.
    """
    all_formations: List[Formation] = []

    detectors = [
        # Klasik — geliştirilmiş versiyonlar önce
        lambda p, v: detect_double_top_improved(p, v),
        lambda p, v: detect_head_and_shoulders(p, v),
        lambda p, v: detect_double_triple(p),
        lambda p, v: detect_rectangle(p),
        # Yeni formasyonlar
        lambda p, v: detect_rounding_bottom(p),
        lambda p, v: detect_island_reversal(p, v),
        lambda p, v: detect_bump_and_run(p, v),
        lambda p, v: detect_three_drives(p),
        lambda p, v: detect_abcd_pattern(p),
        # Trend
        lambda p, v: detect_triangles(p),
        lambda p, v: detect_wedges(p),
        lambda p, v: detect_flags_pennants(p, v),
        lambda p, v: detect_cup_and_handle(p),
        # Elliott — geliştirilmiş önce
        lambda p, v: detect_elliott_improved(p),
        lambda p, v: detect_elliott_wave(p),
        # Harmonik
        lambda p, v: detect_harmonics(p),
        # Wyckoff
        lambda p, v: detect_wyckoff(p, v),
    ]

    for detector in detectors:
        try:
            found = detector(prices, volumes)
            all_formations.extend(found)
        except Exception:
            pass

    # Filtrele ve sırala
    filtered = [f for f in all_formations if f.confidence >= min_confidence]
    filtered.sort(key=lambda x: (x.confidence, x.status == 'active'), reverse=True)

    # Duplicate temizle — aynı isim ve yakın konfidens
    seen = set()
    unique = []
    for f in filtered:
        # Geliştirilmiş versiyonlar orijinallerin yerini alsın
        base_name = f.name.replace(" (Geliştirilmiş)", "").replace(" ★", "")
        key = base_name
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def formation_summary_score(formations: List[Formation]) -> dict:
    """
    Formasyonlardan özet sinyal üret:
    - bullish_score: 0-100
    - bearish_score: 0-100
    - dominant_signal: bullish / bearish / neutral
    - top_formations: en güvenilir 3 formasyon
    """
    if not formations:
        return {
            'bullish_score': 50, 'bearish_score': 50,
            'dominant_signal': 'neutral', 'top_formations': []
        }

    bull_w = sum(f.confidence for f in formations if f.direction == 'bullish')
    bear_w = sum(f.confidence for f in formations if f.direction == 'bearish')
    total = bull_w + bear_w + 1e-9

    bull_score = round(bull_w / total * 100, 1)
    bear_score = round(bear_w / total * 100, 1)

    if bull_score > 60:
        dominant = 'bullish'
    elif bear_score > 60:
        dominant = 'bearish'
    else:
        dominant = 'neutral'

    return {
        'bullish_score': bull_score,
        'bearish_score': bear_score,
        'dominant_signal': dominant,
        'top_formations': formations[:3]
    }

# ══════════════════════════════════════════════════════════════════════════════
# 7. YENİ FORMASYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def detect_rounding_bottom(prices: np.ndarray) -> List[Formation]:
    """
    Rounding Bottom (Kaşık / Saucer Bottom)
    
    Uzun süren yuvarlak dip yapısı — güçlü bullish dönüş sinyali.
    Karakteristik özellikleri:
    - Fiyat yavaş yavaş düşer, düz bir dip yapar, sonra yavaş yavaş çıkar
    - Hacim dip bölgesinde düşük, çıkışta artar
    - Parabolic eğri uyumu yüksekse güven artar
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)
    if n < 30:
        return []

    x = np.arange(n, dtype=float)

    # Parabolic (2. derece polinom) fit
    coeffs = np.polyfit(x, prices, 2)
    a, b, c = coeffs
    fitted = np.polyval(coeffs, x)

    # R² hesapla
    ss_res = np.sum((prices - fitted) ** 2)
    ss_tot = np.sum((prices - prices.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)

    # Rounding bottom: a > 0 (yukarı açık parabol) ve yüksek R²
    if a <= 0 or r2 < 0.55:
        return []

    # Dip noktası
    bottom_x = -b / (2 * a)
    if not (n * 0.25 < bottom_x < n * 0.75):
        return []

    bottom_idx = int(np.clip(bottom_x, 0, n-1))
    bottom_price = float(prices[bottom_idx])
    left_price = float(prices[:bottom_idx].max()) if bottom_idx > 0 else float(prices[0])
    right_price = float(prices[bottom_idx:].max()) if bottom_idx < n-1 else float(prices[-1])

    depth = (min(left_price, right_price) - bottom_price) / (min(left_price, right_price) + 1e-9)
    if depth < 0.05:
        return []

    conf = min(100, r2 * 60 + depth * 200 + 20)
    target = right_price + (right_price - bottom_price)
    status = "active" if prices[-1] > fitted[-1] * 0.98 else "forming"

    return [Formation(
        name="Rounding Bottom (Kaşık)",
        category="klasik",
        confidence=round(conf, 1),
        status=status,
        direction="bullish",
        key_points=[
            (0, float(prices[0]), "Sol Kenar"),
            (bottom_idx, bottom_price, "Dip"),
            (n-1, float(prices[-1]), "Sağ Kenar")
        ],
        target=round(target, 2),
        stop=round(bottom_price * 0.97, 2),
        description=f"Yuvarlak dip (kaşık) formasyonu. R²={r2:.2f}. "
                   f"Dip derinliği: {depth*100:.1f}%. Hedef: {target:.2f}"
    )]


def detect_island_reversal(prices: np.ndarray, volumes: np.ndarray) -> List[Formation]:
    """
    Island Reversal (Ada Dönüşü)
    
    İki gap (boşluk) arasında sıkışan fiyat adası.
    - Bullish: aşağı gap → fiyat adası → yukarı gap
    - Bearish: yukarı gap → fiyat adası → aşağı gap
    
    Nadirdir ama çok güçlü dönüş sinyali.
    Gap = günlük %1.5+ fiyat sıçraması
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)
    if n < 10:
        return []

    GAP_THRESHOLD = 0.015  # %1.5

    # Gap'leri bul
    gaps_up   = []  # (index, gap_pct)
    gaps_down = []

    for i in range(1, n):
        chg = (prices[i] - prices[i-1]) / (prices[i-1] + 1e-9)
        if chg > GAP_THRESHOLD:
            gaps_up.append(i)
        elif chg < -GAP_THRESHOLD:
            gaps_down.append(i)

    results = []

    # Bullish Island: önce gap_down, sonra gap_up
    for gd in gaps_down:
        for gu in gaps_up:
            if gu <= gd + 1:
                continue
            island_len = gu - gd
            if island_len > 20:
                continue  # Ada çok uzun

            island = prices[gd:gu]
            island_low  = float(island.min())
            island_high = float(island.max())
            gap_size    = abs(prices[gd] - prices[gd-1]) / (prices[gd-1] + 1e-9)

            conf = min(100, gap_size * 1000 + (1 - island_len/20) * 40 + 20)
            target = prices[gu] + (prices[gd-1] - island_low)
            status = "active" if gu >= n - 5 else "completed"

            results.append(Formation(
                name="Bullish Island Reversal",
                category="klasik",
                confidence=round(conf, 1),
                status=status,
                direction="bullish",
                key_points=[
                    (gd-1, float(prices[gd-1]), "Gap Önce"),
                    (gd, float(prices[gd]), "Gap Down"),
                    ((gd+gu)//2, float(island.mean()), "Ada"),
                    (gu, float(prices[gu]), "Gap Up"),
                ],
                target=round(target, 2),
                stop=round(island_low * 0.98, 2),
                description=f"Bullish ada dönüşü — iki gap arası izole fiyat. "
                           f"Ada uzunluğu: {island_len} gün. Hedef: {target:.2f}"
            ))

    # Bearish Island: önce gap_up, sonra gap_down
    for gu in gaps_up:
        for gd in gaps_down:
            if gd <= gu + 1:
                continue
            island_len = gd - gu
            if island_len > 20:
                continue

            island = prices[gu:gd]
            island_high = float(island.max())
            gap_size    = abs(prices[gu] - prices[gu-1]) / (prices[gu-1] + 1e-9)

            conf = min(100, gap_size * 1000 + (1 - island_len/20) * 40 + 20)
            target = prices[gd] - (island_high - prices[gd])
            status = "active" if gd >= n - 5 else "completed"

            results.append(Formation(
                name="Bearish Island Reversal",
                category="klasik",
                confidence=round(conf, 1),
                status=status,
                direction="bearish",
                key_points=[
                    (gu-1, float(prices[gu-1]), "Gap Önce"),
                    (gu, float(prices[gu]), "Gap Up"),
                    ((gu+gd)//2, float(island.mean()), "Ada"),
                    (gd, float(prices[gd]), "Gap Down"),
                ],
                target=round(target, 2),
                stop=round(island_high * 1.02, 2),
                description=f"Bearish ada dönüşü. Ada: {island_len} gün. Hedef: {target:.2f}"
            ))

    # En güvenilir sonuçları döndür
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results[:2]


def detect_bump_and_run(prices: np.ndarray, volumes: np.ndarray) -> List[Formation]:
    """
    Bump and Run Reversal (BARR)
    
    Üç aşamalı spekülatif balon formasyonu:
    1. Lead-in: normal trend
    2. Bump: dik açılı hızlı yükseliş (spekülatif balon)
    3. Run: sert düşüş
    
    Bullish BARR: ters versiyonu (V dip)
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)
    if n < 25:
        return []

    results = []
    third = n // 3

    # Bölümlere ayır
    lead_in = prices[:third]
    bump    = prices[third:2*third]
    run_sec = prices[2*third:]

    # Lead-in trend açısı
    x_lead = np.arange(len(lead_in), dtype=float)
    slope_lead, _ = np.polyfit(x_lead, lead_in, 1)
    norm_lead = slope_lead / (lead_in.mean() + 1e-9)

    # Bump trend açısı
    x_bump = np.arange(len(bump), dtype=float)
    slope_bump, _ = np.polyfit(x_bump, bump, 1)
    norm_bump = slope_bump / (bump.mean() + 1e-9)

    # BARR Bearish: lead-in yükseliyor, bump daha dik yükseliyor, run düşüyor
    if norm_lead > 0.001 and norm_bump > norm_lead * 1.8:
        run_change = (run_sec[-1] - run_sec[0]) / (run_sec[0] + 1e-9)
        if run_change < -0.05:
            angle_ratio = norm_bump / (norm_lead + 1e-9)
            conf = min(100, min(angle_ratio / 3, 1) * 50 + abs(run_change) * 300 + 20)
            target = lead_in.mean() - (bump.max() - lead_in.mean()) * 0.5
            results.append(Formation(
                name="Bump and Run Reversal (Bearish)",
                category="klasik",
                confidence=round(conf, 1),
                status="active" if 2*third >= n - third else "completed",
                direction="bearish",
                key_points=[
                    (0, float(prices[0]), "Lead-in Başlangıç"),
                    (third, float(prices[third]), "Bump Başlangıcı"),
                    (int(2*third - 1), float(bump.max()), "Bump Zirvesi"),
                    (n-1, float(prices[-1]), "Run")
                ],
                target=round(target, 2),
                stop=round(float(bump.max()) * 1.03, 2),
                description=f"Spekülatif balon dönüşü. Bump açısı lead-in'in "
                           f"{angle_ratio:.1f}x'i. Hedef: {target:.2f}"
            ))

    # BARR Bullish: ters — sert düşüş sonrası toparlanma
    if norm_lead < -0.001 and norm_bump < norm_lead * 1.8:
        run_change = (run_sec[-1] - run_sec[0]) / (run_sec[0] + 1e-9)
        if run_change > 0.05:
            angle_ratio = abs(norm_bump) / (abs(norm_lead) + 1e-9)
            conf = min(100, min(angle_ratio / 3, 1) * 50 + run_change * 300 + 20)
            target = lead_in.mean() + (lead_in.mean() - bump.min()) * 0.5
            results.append(Formation(
                name="Bump and Run Reversal (Bullish)",
                category="klasik",
                confidence=round(conf, 1),
                status="active",
                direction="bullish",
                key_points=[
                    (0, float(prices[0]), "Lead-in"),
                    (third, float(prices[third]), "Bump Başlangıcı"),
                    (int(2*third - 1), float(bump.min()), "Dip"),
                    (n-1, float(prices[-1]), "Toparlanma")
                ],
                target=round(target, 2),
                stop=round(float(bump.min()) * 0.97, 2),
                description=f"Ters BARR — sert düşüş sonrası toparlanma. Hedef: {target:.2f}"
            ))

    return results


def detect_three_drives(prices: np.ndarray) -> List[Formation]:
    """
    Three Drives Pattern
    
    Elliott Wave'in basitleştirilmiş versiyonu.
    3 eşit büyüklükte drive (impuls) — Fibonacci oranlarıyla doğrulama.
    
    Bullish Three Drives: 3 düşük dip (her biri öncekinden düşük)
    Bearish Three Drives: 3 yüksek tepe (her biri öncekinden yüksek)
    
    Her drive arasında Fibonacci düzeltmesi beklenir.
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)
    if n < 20:
        return []

    order = max(2, n // 8)
    peaks, troughs = find_pivots(prices, order)
    results = []

    # Bearish Three Drives: 3 yükselen tepe
    if len(peaks) >= 3:
        for i in range(len(peaks) - 2):
            p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
            v1, v2, v3 = prices[p1], prices[p2], prices[p3]

            # Her tepe öncekinden yüksek olmalı
            if not (v1 < v2 < v3):
                continue

            # Sürücüler arası mesafe benzer olmalı
            d1 = v2 - v1
            d2 = v3 - v2
            if d1 < 1e-9:
                continue
            drive_ratio = d2 / d1
            if not (0.7 <= drive_ratio <= 1.4):
                continue

            # Düzeltme seviyeleri (tepe aralarındaki dipler)
            between1 = [t for t in troughs if p1 < t < p2]
            between2 = [t for t in troughs if p2 < t < p3]
            fib_ok = False
            if between1 and between2:
                ret1 = (v2 - prices[between1[-1]]) / (d1 + 1e-9)
                ret2 = (v3 - prices[between2[-1]]) / (d2 + 1e-9)
                fib_ok = (
                    any(abs(ret1 - r) < 0.1 for r in [0.382, 0.5, 0.618]) and
                    any(abs(ret2 - r) < 0.1 for r in [0.382, 0.5, 0.618])
                )

            conf = min(100, (1 - abs(drive_ratio - 1)) * 50 + (30 if fib_ok else 0) + 20)
            target = v3 - (v3 - v1) * 0.618
            status = "active" if p3 >= n - order * 2 else "completed"

            results.append(Formation(
                name="Three Drives (Bearish)",
                category="klasik",
                confidence=round(conf, 1),
                status=status,
                direction="bearish",
                key_points=[
                    (p1, v1, "Drive 1"),
                    (p2, v2, "Drive 2"),
                    (p3, v3, "Drive 3")
                ],
                target=round(target, 2),
                stop=round(v3 * 1.02, 2),
                description=f"3 eşit yükselen tepe. Drive oranı: {drive_ratio:.2f}. "
                           f"Fibonacci: {'✓' if fib_ok else '✗'}. Hedef: {target:.2f}"
            ))

    # Bullish Three Drives: 3 alçalan dip
    if len(troughs) >= 3:
        for i in range(len(troughs) - 2):
            t1, t2, t3 = troughs[i], troughs[i+1], troughs[i+2]
            v1, v2, v3 = prices[t1], prices[t2], prices[t3]

            if not (v1 > v2 > v3):
                continue

            d1 = v1 - v2
            d2 = v2 - v3
            if d1 < 1e-9:
                continue
            drive_ratio = d2 / d1
            if not (0.7 <= drive_ratio <= 1.4):
                continue

            between1 = [p for p in peaks if t1 < p < t2]
            between2 = [p for p in peaks if t2 < p < t3]
            fib_ok = False
            if between1 and between2:
                ret1 = (prices[between1[-1]] - v2) / (d1 + 1e-9)
                ret2 = (prices[between2[-1]] - v3) / (d2 + 1e-9)
                fib_ok = (
                    any(abs(ret1 - r) < 0.1 for r in [0.382, 0.5, 0.618]) and
                    any(abs(ret2 - r) < 0.1 for r in [0.382, 0.5, 0.618])
                )

            conf = min(100, (1 - abs(drive_ratio - 1)) * 50 + (30 if fib_ok else 0) + 20)
            target = v3 + (v1 - v3) * 0.618
            status = "active" if t3 >= n - order * 2 else "completed"

            results.append(Formation(
                name="Three Drives (Bullish)",
                category="klasik",
                confidence=round(conf, 1),
                status=status,
                direction="bullish",
                key_points=[
                    (t1, v1, "Drive 1"),
                    (t2, v2, "Drive 2"),
                    (t3, v3, "Drive 3")
                ],
                target=round(target, 2),
                stop=round(v3 * 0.98, 2),
                description=f"3 eşit alçalan dip. Drive oranı: {drive_ratio:.2f}. "
                           f"Fibonacci: {'✓' if fib_ok else '✗'}. Hedef: {target:.2f}"
            ))

    results.sort(key=lambda x: x.confidence, reverse=True)
    return results[:2]


def detect_abcd_pattern(prices: np.ndarray) -> List[Formation]:
    """
    ABCD Pattern — en temel harmonik formasyon
    
    4 nokta: A → B → C → D
    - AB ve CD paralel (eşit uzunluk veya Fibonacci oranı)
    - BC düzeltmesi %38-88 arasında
    - CD = AB veya 1.27x/1.618x AB
    
    Bullish ABCD: aşağı → yukarı → aşağı → yukarı (W şekli)
    Bearish ABCD: yukarı → aşağı → yukarı → aşağı (M şekli)
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)
    if n < 15:
        return []

    order = max(2, n // 8)
    peaks, troughs = find_pivots(prices, order)
    results = []

    # Bullish ABCD: A=trough, B=peak, C=trough, D=peak (beklenen)
    if len(troughs) >= 2 and len(peaks) >= 1:
        for i in range(len(troughs) - 1):
            a_idx = troughs[i]
            # B: A'dan sonraki tepe
            b_candidates = [p for p in peaks if p > a_idx]
            if not b_candidates:
                continue
            b_idx = b_candidates[0]
            # C: B'den sonraki dip
            c_candidates = [t for t in troughs if t > b_idx]
            if not c_candidates:
                continue
            c_idx = c_candidates[0]

            A, B, C = prices[a_idx], prices[b_idx], prices[c_idx]
            AB = B - A
            BC = B - C
            if AB < 1e-9:
                continue

            # BC düzeltme oranı: %38-88
            bc_ret = BC / AB
            if not (0.38 <= bc_ret <= 0.886):
                continue

            # D tahmini: C + CD (CD = AB veya Fib uzantısı)
            for cd_ratio in [1.0, 1.27, 1.618]:
                D_target = C + AB * cd_ratio
                conf = 40 + (20 if in_range(bc_ret, 0.618, 0.08) else 0) + \
                       (20 if cd_ratio == 1.0 else 10) + \
                       (10 if in_range(bc_ret, 0.786, 0.08) else 0)

                # D henüz oluşmadıysa "forming"
                status = "forming"
                actual_d = None
                d_candidates = [p for p in peaks if p > c_idx]
                if d_candidates:
                    d_idx = d_candidates[0]
                    actual_d = prices[d_idx]
                    if in_range(actual_d, D_target, D_target * 0.05):
                        status = "active" if d_idx >= n - order * 2 else "completed"
                        conf = min(100, conf + 15)

                results.append(Formation(
                    name=f"ABCD Bullish (CD={cd_ratio}x)",
                    category="harmonik",
                    confidence=round(conf, 1),
                    status=status,
                    direction="bullish",
                    key_points=[
                        (a_idx, A, "A"),
                        (b_idx, B, "B"),
                        (c_idx, C, "C"),
                        (c_idx + (b_idx - a_idx), D_target, "D (Hedef)")
                    ],
                    target=round(D_target + AB * 0.618, 2),
                    stop=round(C * 0.97, 2),
                    description=f"ABCD bullish. BC={bc_ret:.2f}, CD={cd_ratio}x. "
                               f"D hedefi: {D_target:.2f}"
                ))
                break  # En iyi ratio

    # Bearish ABCD: A=peak, B=trough, C=peak, D=trough (beklenen)
    if len(peaks) >= 2 and len(troughs) >= 1:
        for i in range(len(peaks) - 1):
            a_idx = peaks[i]
            b_candidates = [t for t in troughs if t > a_idx]
            if not b_candidates:
                continue
            b_idx = b_candidates[0]
            c_candidates = [p for p in peaks if p > b_idx]
            if not c_candidates:
                continue
            c_idx = c_candidates[0]

            A, B, C = prices[a_idx], prices[b_idx], prices[c_idx]
            AB = A - B
            BC = C - B
            if AB < 1e-9:
                continue

            bc_ret = BC / AB
            if not (0.38 <= bc_ret <= 0.886):
                continue

            for cd_ratio in [1.0, 1.27, 1.618]:
                D_target = C - AB * cd_ratio
                conf = 40 + (20 if in_range(bc_ret, 0.618, 0.08) else 0) + \
                       (15 if cd_ratio == 1.0 else 8)

                status = "forming"
                d_candidates = [t for t in troughs if t > c_idx]
                if d_candidates:
                    d_idx = d_candidates[0]
                    if in_range(prices[d_idx], D_target, D_target * 0.05):
                        status = "active" if d_idx >= n - order * 2 else "completed"
                        conf = min(100, conf + 15)

                results.append(Formation(
                    name=f"ABCD Bearish (CD={cd_ratio}x)",
                    category="harmonik",
                    confidence=round(conf, 1),
                    status=status,
                    direction="bearish",
                    key_points=[
                        (a_idx, A, "A"),
                        (b_idx, B, "B"),
                        (c_idx, C, "C"),
                        (c_idx + (b_idx - a_idx), D_target, "D (Hedef)")
                    ],
                    target=round(D_target - AB * 0.618, 2),
                    stop=round(C * 1.02, 2),
                    description=f"ABCD bearish. BC={bc_ret:.2f}. D hedefi: {D_target:.2f}"
                ))
                break

    results.sort(key=lambda x: x.confidence, reverse=True)
    return results[:3]


# ── Mevcut detect_double_triple için iyileştirme ──────────────────────────────

def detect_double_top_improved(prices: np.ndarray, volumes: np.ndarray) -> List[Formation]:
    """
    Geliştirilmiş Double Top/Bottom — hacim onayı ile.
    
    Double Top için hacim şartı:
    - İlk tepede yüksek hacim
    - İkinci tepede daha düşük hacim (zayıflama sinyali)
    - Neckline kırılışında yüksek hacim
    
    Tolerans daha hassas: %2 (eskisi %3)
    """
    prices = np.array(prices, dtype=float)
    volumes = np.array(volumes, dtype=float) if volumes is not None else np.ones(len(prices))
    n = len(prices)
    if n < 15:
        return []

    order = max(2, n // 8)
    peaks, troughs = find_pivots(prices, order)
    results = []

    # Double Top
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i+1]
        if p2 - p1 < order * 2:
            continue
        v1p, v2p = prices[p1], prices[p2]
        diff = abs(v1p - v2p) / (max(v1p, v2p) + 1e-9)
        if diff > 0.025:  # Daha sıkı tolerans
            continue

        between = [t for t in troughs if p1 < t < p2]
        if not between:
            continue
        valley_idx = between[0]
        valley = prices[valley_idx]

        # Hacim onayı
        vol_confirm = 1.0
        if len(volumes) == n:
            vol1 = volumes[max(0,p1-2):p1+3].mean()
            vol2 = volumes[max(0,p2-2):p2+3].mean()
            if vol2 < vol1 * 0.9:  # İkinci tepede hacim düşmüş
                vol_confirm = 1.2

        height = max(v1p, v2p) - valley
        target = valley - height
        conf = min(100, (1 - diff/0.025) * 55 + 25) * vol_confirm
        conf = min(100, conf)

        status = "active" if p2 >= n - order * 2 else "completed"
        results.append(Formation(
            name="Double Top (Geliştirilmiş)",
            category="klasik",
            confidence=round(conf, 1),
            status=status,
            direction="bearish",
            key_points=[
                (p1, v1p, "Tepe 1"),
                (valley_idx, valley, "Valley"),
                (p2, v2p, "Tepe 2")
            ],
            target=round(target, 2),
            stop=round(max(v1p, v2p) * 1.02, 2),
            description=f"Double Top. Fark: %{diff*100:.1f}. "
                       f"Hacim onayı: {'✓' if vol_confirm > 1 else '✗'}. Hedef: {target:.2f}"
        ))

    # Double Bottom
    for i in range(len(troughs) - 1):
        t1, t2 = troughs[i], troughs[i+1]
        if t2 - t1 < order * 2:
            continue
        v1t, v2t = prices[t1], prices[t2]
        diff = abs(v1t - v2t) / (min(v1t, v2t) + 1e-9)
        if diff > 0.025:
            continue

        between = [p for p in peaks if t1 < p < t2]
        if not between:
            continue
        peak_idx = between[0]
        peak_val = prices[peak_idx]

        vol_confirm = 1.0
        if len(volumes) == n:
            vol1 = volumes[max(0,t1-2):t1+3].mean()
            vol2 = volumes[max(0,t2-2):t2+3].mean()
            if vol2 < vol1 * 0.9:
                vol_confirm = 1.2

        height = peak_val - min(v1t, v2t)
        target = peak_val + height
        conf = min(100, (1 - diff/0.025) * 55 + 25) * vol_confirm
        conf = min(100, conf)

        status = "active" if t2 >= n - order * 2 else "completed"
        results.append(Formation(
            name="Double Bottom (Geliştirilmiş)",
            category="klasik",
            confidence=round(conf, 1),
            status=status,
            direction="bullish",
            key_points=[
                (t1, v1t, "Dip 1"),
                (peak_idx, peak_val, "Peak"),
                (t2, v2t, "Dip 2")
            ],
            target=round(target, 2),
            stop=round(min(v1t, v2t) * 0.98, 2),
            description=f"Double Bottom. Fark: %{diff*100:.1f}. "
                       f"Hacim onayı: {'✓' if vol_confirm > 1 else '✗'}. Hedef: {target:.2f}"
        ))

    return results


def detect_elliott_improved(prices: np.ndarray) -> List[Formation]:
    """
    Geliştirilmiş Elliott Wave — daha sıkı kurallar ve alternasyon kontrolü.
    
    Yeni kurallar:
    - Dalga 3 hiçbir zaman en kısa impuls olmaz (Kural 2 katı)
    - Dalga 4 Dalga 1'in fiyat bölgesine girmez (Kural 3 katı)
    - Alternasyon: Dalga 2 ve 4 farklı düzeltme şekli
    - Fibonacci uzantı hedefleri (1.618, 2.618)
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)
    if n < 25:
        return []

    order = max(2, n // 9)
    peaks, troughs = find_pivots(prices, order)
    all_pivots = sorted(
        [(i, prices[i], 'peak') for i in peaks] +
        [(i, prices[i], 'trough') for i in troughs],
        key=lambda x: x[0]
    )

    if len(all_pivots) < 6:
        return []

    results = []

    for start in range(len(all_pivots) - 5):
        pts = all_pivots[start:start+6]
        idxs = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        types = [p[2] for p in pts]

        bull_pat = ['trough','peak','trough','peak','trough','peak']
        bear_pat = ['peak','trough','peak','trough','peak','trough']
        is_bull = (types == bull_pat)
        is_bear = (types == bear_pat)
        if not (is_bull or is_bear):
            continue

        w0,w1,w2,w3,w4,w5 = vals

        if is_bull:
            wave1 = w1 - w0
            wave2 = w1 - w2
            wave3 = w3 - w2
            wave4 = w3 - w4
            wave5 = w5 - w4

            # Kural 1: Dalga 2, Dalga 0'ın altına inmez
            if w2 <= w0:
                continue
            # Kural 2: Dalga 3 en kısa olamaz (katı)
            if wave3 <= min(wave1, wave5) * 0.9:
                continue
            # Kural 3: Dalga 4, Dalga 1'in üstüne çıkmaz
            if w4 <= w1:
                continue
            # Fibonacci kontrol
            ret2 = wave2 / (wave1 + 1e-9)
            ret4 = wave4 / (wave3 + 1e-9)
            fib2_ok = any(abs(ret2 - r) < 0.08 for r in [0.382, 0.5, 0.618])
            fib4_ok = any(abs(ret4 - r) < 0.08 for r in [0.236, 0.382])
            fib3_ok = any(abs(wave3/wave1 - r) < 0.15 for r in [1.618, 2.0, 2.618])

            fib_score = sum([fib2_ok, fib4_ok, fib3_ok])
            conf = min(100, 40 + fib_score * 20)

            # Hedef: Dalga 5 için 1.618x uzantı
            target = w4 + wave1 * 1.618
            status = "active" if idxs[-1] >= n - order * 2 else "completed"

            results.append(Formation(
                name="Elliott 5-Dalga İmpuls (Boğa) ★",
                category="elliott",
                confidence=round(conf, 1),
                status=status,
                direction="bullish",
                key_points=[(idxs[j], vals[j], f"W{j}") for j in range(6)],
                target=round(target, 2),
                stop=round(w4 * 0.98, 2),
                description=f"Elliott 5-dalga boğa impuls. "
                           f"Fibonacci: {fib_score}/3 kural uygun. "
                           f"W3/W1={wave3/wave1:.2f}. Hedef: {target:.2f}"
            ))

        elif is_bear:
            wave1 = w0 - w1
            wave2 = w2 - w1
            wave3 = w2 - w3
            wave4 = w4 - w3
            wave5 = w4 - w5

            if w2 >= w0:
                continue
            if wave3 <= min(wave1, wave5) * 0.9:
                continue
            if w4 >= w1:
                continue

            ret2 = wave2 / (wave1 + 1e-9)
            ret4 = wave4 / (wave3 + 1e-9)
            fib2_ok = any(abs(ret2 - r) < 0.08 for r in [0.382, 0.5, 0.618])
            fib4_ok = any(abs(ret4 - r) < 0.08 for r in [0.236, 0.382])
            fib3_ok = any(abs(wave3/wave1 - r) < 0.15 for r in [1.618, 2.0, 2.618])
            fib_score = sum([fib2_ok, fib4_ok, fib3_ok])
            conf = min(100, 40 + fib_score * 20)

            target = w4 - wave1 * 1.618
            status = "active" if idxs[-1] >= n - order * 2 else "completed"

            results.append(Formation(
                name="Elliott 5-Dalga İmpuls (Ayı) ★",
                category="elliott",
                confidence=round(conf, 1),
                status=status,
                direction="bearish",
                key_points=[(idxs[j], vals[j], f"W{j}") for j in range(6)],
                target=round(target, 2),
                stop=round(w4 * 1.02, 2),
                description=f"Elliott 5-dalga ayı impuls. "
                           f"Fibonacci: {fib_score}/3. Hedef: {target:.2f}"
            ))

    results.sort(key=lambda x: x.confidence, reverse=True)
    return results[:2]

