"""
rise_reason.py — BIST Teknik Yükseliş Nedenleri Analiz Modülü
Hisselerin son dönem fiyat ve hacim hareketlerini inceleyerek
yükseliş veya akümülasyon arkasındaki teknik nedenleri tespit eder.
"""

import numpy as np
import pandas as pd

def calc_rsi(prices, n=14):
    prices = np.array(prices, dtype=float)
    if len(prices) < n + 1:
        return np.full(len(prices), 50.0)
    
    deltas = np.diff(prices)
    seed = deltas[:n]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    
    rs = up / (down + 1e-9)
    rsi = np.zeros_like(prices)
    rsi[:n] = 100.0 - 100.0 / (1.0 + rs)
    
    for i in range(n, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            up_val = delta
            down_val = 0.0
        else:
            up_val = 0.0
            down_val = -delta
        up = (up * (n - 1) + up_val) / n
        down = (down * (n - 1) + down_val) / n
        rs = up / (down + 1e-9)
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi

def analyze_rise_reasons(df, window=20):
    """
    Hissenin son dönemdeki fiyat ve hacim hareketlerini analiz ederek
    yükseliş/akümülasyon sebeplerini tespit eder.
    """
    reasons = []
    if df is None or len(df) < min(40, window + 10):
        return ["📊 Teknik Düzeltme"]

    closes = df['Close'].values.astype(float)
    volumes = df['Volume'].values.astype(float) if 'Volume' in df.columns else np.zeros(len(closes))
    
    # Son window günkü hareket
    tpl_prices = closes[-window:]
    
    # 1. Hacimli Kırılım / Hacim Patlaması
    if len(volumes) >= window + 20:
        base_vol_avg = np.mean(volumes[-(window+20):-window])
        recent_vol_avg = np.mean(volumes[-5:])
        vol_ratio = recent_vol_avg / (base_vol_avg + 1e-9)
        
        # Son 5 günde fiyat artışı var mı?
        price_change_5d = (closes[-1] - closes[-5]) / (closes[-5] + 1e-9) * 100
        
        if vol_ratio >= 2.0 and price_change_5d > 2.0:
            reasons.append("🔥 Hacimli Kırılım")
        elif vol_ratio >= 1.5 and price_change_5d > 1.0:
            reasons.append("📈 Hacim Desteği")

    # 2. Sıkışma / Daralma (Bollinger Band Squeeze veya Yatay Konsolidasyon)
    p_max = np.max(tpl_prices)
    p_min = np.min(tpl_prices)
    p_range = (p_max - p_min) / (p_min + 1e-9) * 100
    if p_range < 12.0:
        reasons.append("💎 Akümülasyon (Sıkışma)")

    # 3. Direnç/Zirve Kırılımı
    if len(closes) >= 60:
        max_60d_before = np.max(closes[-60:-2])
        if closes[-1] > max_60d_before and closes[-2] <= max_60d_before:
            reasons.append("🚀 Direnç Kırılımı")
        elif closes[-1] > np.percentile(closes[-60:], 95):
            reasons.append("🎯 Zirve Bölgesi")

    # 4. Aşırı Satımdan Tepki (RSI Dönüşü)
    rsi_vals = calc_rsi(closes)
    if len(rsi_vals) >= 10:
        recent_rsi = rsi_vals[-10:]
        if np.any(recent_rsi < 35) and rsi_vals[-1] > 45 and closes[-1] > closes[-3]:
            reasons.append("🛡️ Dip Dönüşü (RSI)")

    # 5. Boşluklu Yükseliş (Gap Up)
    if 'Open' in df.columns and 'High' in df.columns:
        opens = df['Open'].values.astype(float)
        highs = df['High'].values.astype(float)
        gaps = 0
        for idx in range(-5, 0):
            if idx >= -len(df) + 1:
                if opens[idx] > highs[idx-1] * 1.01:
                    gaps += 1
        if gaps > 0:
            reasons.append("⚡ Boşluklu Yükseliş")

    # 6. Güçlü Momentum
    if len(closes) >= 50:
        ma_20 = np.mean(closes[-20:])
        ma_50 = np.mean(closes[-50:])
        if closes[-1] > ma_20 and ma_20 > ma_50:
            reasons.append("💪 Güçlü Trend")

    if not reasons:
        reasons.append("📊 Teknik Düzeltme")

    # Benzersiz ve en fazla 3 neden döndür
    seen = set()
    unique_reasons = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique_reasons.append(r)
            
    return unique_reasons[:3]
