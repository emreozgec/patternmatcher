"""
daily_scan.py — Günlük Otomatik Tarama + Telegram Bildirimi

GitHub Actions tarafından her gün belirli saatte çalıştırılır.
Streamlit'ten BAĞIMSIZ çalışır (Streamlit Cloud sürekli açık kalmadığı için).

Akış:
1. BIST hisselerinin verisini çek (yfinance)
2. scanner.py'deki scan_single_ticker mantığını kullanarak tara
3. Backtesting'den gelen optimal parametreleri uygula (PSI 80+, Güven 55-68)
4. Bulunan fırsatları Telegram'a gönder

Ortam değişkenleri (.env veya GitHub Secrets üzerinden):
    TELEGRAM_BOT_TOKEN : BotFather'dan alınan token
    TELEGRAM_CHAT_ID   : Bildirim gönderilecek chat/kanal ID'si
"""

import os
import sys
# Mock numba to prevent its binary C-extensions from importing and causing a Segmentation Fault in Streamlit Cloud.
sys.modules['numba'] = None

import json
import time
import html
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

# Yerel ortamda .env dosyasından token'ları yüklemek için
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Windows ve diğer konsollarda emoji yazdırma hatalarını önlemek için stdout kodlamasını ayarla
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# yfinance indirmelerinin GitHub Actions ve yerel çalışmalarda engellenmesini önlemek için User-Agent tanımlı session oluştur
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# ── Proje modüllerini import edebilmek için path ekle ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanner import scan_single_ticker

# ══════════════════════════════════════════════════════════════════════════════
# AYARLAR — Backtesting'den gelen optimal parametreler
# ══════════════════════════════════════════════════════════════════════════════

MIN_SIM        = 80    # PSI 80+ optimal bant (backtesting: %61 kazanç)
MIN_CONFIDENCE = 55    # Güven 55-65 bandı optimal (backtesting: %66 kazanç)
MAX_CONFIDENCE = 68    # Anti-consensus filtresi
SCAN_SCOPE     = "BIST100"   # BIST30 / BIST100 / ALL — GitHub Actions süresi için BIST100 önerilir
WINDOWS        = [40, 60, 90]    # Şablon uzunlukları

# ── Tekrar bildirim önleme ──────────────────────────────────────────────────
# Aynı hisse+vade için, bu kadar saat içinde tekrar sinyal geldiyse tekrar
# Telegram'a gönderilmez. Tarama her 2 saatte bir çalıştığı için varsayılan
# 8 saat, bir sinyalin aynı gün içinde en fazla ~2 kez bildirilmesini sağlar.
DEDUP_HOURS = float(os.environ.get("DEDUP_HOURS", "8"))

# Gönderim geçmişinin tutulduğu dosya (workflow bu dosyayı commit'leyerek kalıcı hale getirir)
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sent_signals.json")

# ── BIST listeleri (app.py ile aynı) ───────────────────────────────────────────

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


def fetch_ticker(symbol, period="2y"):
    """app.py'deki fetch_ticker ile aynı mantık."""
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
    except Exception:
        return None


def fetch_batch(tickers, period="2y"):
    """app.py'deki fetch_batch ile aynı mantık."""
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
            except Exception:
                pass
    except Exception:
        pass
    return results


def fetch_index_closes(period="2y"):
    """BIST100 endeks verisini çek — genel piyasa kontrolü için."""
    try:
        today = datetime.today().strftime('%Y-%m-%d')
        raw = yf.download("XU100.IS", period=period, end=today,
                          auto_adjust=True, progress=False, threads=False, session=session)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            return raw['Close'].values.astype(float)
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# TEKRAR BİLDİRİM ÖNLEME (DEDUP)
# ══════════════════════════════════════════════════════════════════════════════

def load_sent_history() -> dict:
    """
    Daha önce gönderilmiş sinyallerin geçmişini yükle.
    Format: {"TICKER_WINDOW": "2026-07-01T09:00:00"}
    """
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Geçmiş dosyası okunamadı, sıfırdan başlanıyor: {e}")
        return {}


def save_sent_history(history: dict) -> None:
    """Gönderim geçmişini diske yaz (workflow bunu daha sonra commit'ler)."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    # Çok eskimiş kayıtları temizle (dosya sonsuza kadar büyümesin) — 30 günden eski sil
    cutoff = datetime.now() - timedelta(days=30)
    cleaned = {}
    for key, ts_str in history.items():
        try:
            if datetime.fromisoformat(ts_str) >= cutoff:
                cleaned[key] = ts_str
        except Exception:
            continue
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Geçmiş dosyası yazılamadı: {e}")


def filter_new_signals(results_by_window: dict, history: dict) -> dict:
    """
    Son DEDUP_HOURS saat içinde aynı hisse+vade için sinyal gönderilmişse
    o sonucu listeden çıkar. Böylece aynı fırsat art arda spam olarak gelmez.
    """
    now = datetime.now()
    filtered = {w: [] for w in results_by_window}
    skipped = 0

    for window, results in results_by_window.items():
        for r in results:
            key = f"{r['ticker']}_{window}"
            last_sent = history.get(key)
            if last_sent:
                try:
                    elapsed_hours = (now - datetime.fromisoformat(last_sent)).total_seconds() / 3600
                    if elapsed_hours < DEDUP_HOURS:
                        skipped += 1
                        continue  # Yakın zamanda zaten gönderilmiş, atla
                except Exception:
                    pass
            filtered[window].append(r)

    if skipped:
        print(f"🔁 {skipped} sinyal son {DEDUP_HOURS:.0f} saat içinde zaten gönderildiği için atlandı.")

    return filtered


def update_history_with_sent(results_by_window: dict, history: dict) -> dict:
    """Az önce gönderilen sonuçları geçmişe zaman damgasıyla ekle."""
    now_str = datetime.now().isoformat(timespec="seconds")
    for window, results in results_by_window.items():
        for r in results:
            key = f"{r['ticker']}_{window}"
            history[key] = now_str
    return history


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM BİLDİRİMİ (MODÜLDEN YÜKLENİR)
# ══════════════════════════════════════════════════════════════════════════════

from telegram_utils import send_telegram_message, format_results_message


# ══════════════════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def run_daily_scan():
    print(f"🚀 Günlük tarama başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    scope_map = {"BIST30": BIST30, "BIST100": BIST100}
    tickers = scope_map.get(SCAN_SCOPE, BIST100)
    print(f"📥 {len(tickers)} hisse ({SCAN_SCOPE}) için veri indiriliyor...")

    all_data = fetch_batch(tickers, period="2y")
    print(f"✅ {len(all_data)} hisse verisi alındı.")

    index_closes = fetch_index_closes(period="2y")
    print(f"📊 Endeks verisi: {'alındı' if index_closes is not None else 'alınamadı'}")

    # SQLite Veritabanı ve Açık Pozisyon Güncellemesi
    try:
        import db_utils
        print("🔄 SQLite veritabanı ilklendiriliyor ve açık pozisyonlar güncelleniyor...")
        db_utils.init_db()
        db_utils.update_signal_statuses(all_data)
    except Exception as e:
        print(f"⚠️ Veritabanı güncelleme hatası: {e}")

    results_by_window = {w: [] for w in WINDOWS}

    import concurrent.futures

    def _process_ticker_daily(ticker):
        df = all_data.get(ticker)
        if df is None or len(df) < 10:
            return None
        t_res_90 = None
        t_res_120 = None
        
        # 90G
        try:
            r90 = scan_single_ticker(ticker, df, all_data,
                                     window=90, fut_window=135,
                                     min_sim=MIN_SIM, index_closes=index_closes)
            if r90 and MIN_CONFIDENCE <= r90['confidence'] <= MAX_CONFIDENCE and r90.get('weighted_pct', 0.0) >= 5.0:
                t_res_90 = r90
        except Exception:
            pass
            
        # 120G
        try:
            r120 = scan_single_ticker(ticker, df, all_data,
                                      window=120, fut_window=180,
                                      min_sim=MIN_SIM, index_closes=index_closes)
            if r120 and MIN_CONFIDENCE <= r120['confidence'] <= MAX_CONFIDENCE and r120.get('weighted_pct', 0.0) >= 5.0:
                t_res_120 = r120
        except Exception:
            pass
            
        return t_res_90, t_res_120

    tickers_list = list(all_data.keys())
    total_tickers = len(tickers_list)
    print(f"⚡ {total_tickers} hisse paralel olarak taranıyor (8 threads)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_ticker_daily, t): t for t in tickers_list}
        for idx, fut in enumerate(concurrent.futures.as_completed(futures)):
            if (idx + 1) % 20 == 0 or (idx + 1) == total_tickers:
                print(f"   ... {idx+1}/{total_tickers} tarandı")
            try:
                res = fut.result()
                if res:
                    r90, r120 = res
                    if r90:
                        results_by_window[90].append(r90)
                    if r120:
                        results_by_window[120].append(r120)
            except Exception as e:
                print(f"⚠️ {futures[fut]} taranırken hata: {e}")

    for w in results_by_window:
        results_by_window[w].sort(
            key=lambda x: x['confidence'] * 0.5 + x['avg_sim'] * 0.3 + x['weighted_pct'] * 0.2,
            reverse=True
        )

    total = sum(len(v) for v in results_by_window.values())
    print(f"🎯 Toplam {total} fırsat bulundu.")

    # SQLite veritabanına kaydet (Yeni)
    try:
        import db_utils
        today_str = datetime.today().strftime('%Y-%m-%d')
        for window, results in results_by_window.items():
            for r in results:
                stop_val = round(r['current_price'] * (1 - r['stop_pct'] / 100), 2)
                db_utils.save_signal(
                    ticker=r['ticker'],
                    window=window,
                    signal_date=today_str,
                    entry_price=r['current_price'],
                    target_price=r['target'],
                    weighted_pct=r['weighted_pct'],
                    confidence=r['confidence'],
                    avg_sim=r['avg_sim'],
                    source='daily_scan',
                    expected_days=r['expected_days'],
                    stop_price=stop_val,
                    ml_prob=r.get('ml_prob')
                )




    except Exception as e:
        print(f"⚠️ Veritabanına kaydederken hata: {e}")



    # ── Tekrar bildirim önleme: son DEDUP_HOURS saatte gönderilenleri çıkar ──
    history = load_sent_history()
    new_results_by_window = filter_new_signals(results_by_window, history)
    new_total = sum(len(v) for v in new_results_by_window.values())
    print(f"🆕 {new_total} fırsat yeni (son {DEDUP_HOURS:.0f} saatte gönderilmemiş).")

    if new_total == 0 and total > 0:
        print("ℹ️ Tüm sinyaller zaten yakın zamanda gönderilmişti, Telegram mesajı atlanıyor.")
        return

    messages = format_results_message(new_results_by_window, SCAN_SCOPE)

    print(f"📤 {len(messages)} mesaj gönderiliyor...")
    for i, msg in enumerate(messages):
        success = send_telegram_message(msg)
        print(f"   Mesaj {i+1}/{len(messages)}: {'✅' if success else '❌'}")
        if i < len(messages) - 1:
            time.sleep(1)  # Telegram rate limit için kısa bekleme

    # ── Gönderilenleri geçmişe kaydet (workflow bu dosyayı commit'leyecek) ──
    history = update_history_with_sent(new_results_by_window, history)
    save_sent_history(history)

    print("✅ Günlük tarama tamamlandı.")


if __name__ == "__main__":
    run_daily_scan()
