import os
import sqlite3
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

# yfinance indirmelerinin engellenmesini önlemek için User-Agent tanımlı session oluştur
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "pattern_matcher.db")

def init_db():
    """Veritabanını ve tabloları oluşturur."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        window INTEGER NOT NULL,
        signal_date TEXT NOT NULL,
        entry_price REAL NOT NULL,
        target_price REAL NOT NULL,
        weighted_pct REAL,
        confidence REAL,
        avg_sim REAL,
        status TEXT DEFAULT 'OPEN', -- 'OPEN', 'WIN', 'LOSS', 'EXPIRED'
        close_date TEXT,
        close_price REAL,
        pct_change REAL,
        source TEXT, -- 'daily_scan', 'manual_scan'
        expected_days INTEGER, -- Hedefe ulaşma tahmini süresi (Gün)
        UNIQUE(ticker, window, signal_date)
    )
    """)
    
    # Mevcut veritabanlarına kolon eklemek için (schema migration)
    try:
        cursor.execute("ALTER TABLE signals ADD COLUMN expected_days INTEGER")
    except sqlite3.OperationalError:
        pass
        
    # Performans için indeksler
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(signal_date)")
    
    conn.commit()
    conn.close()

def save_signal(ticker, window, signal_date, entry_price, target_price, weighted_pct, confidence, avg_sim, source, expected_days=None):
    """Yeni bir sinyali veritabanına kaydeder (mükerrer kayıtları önler)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO signals 
        (ticker, window, signal_date, entry_price, target_price, weighted_pct, confidence, avg_sim, status, source, expected_days)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
        """, (ticker, window, signal_date, entry_price, target_price, weighted_pct, confidence, avg_sim, source, expected_days))
        conn.commit()
    except Exception as e:
        print(f"⚠️ Sinyal kaydedilirken hata: {e}")
    finally:
        conn.close()


def get_open_signals():
    """Açık pozisyondaki (OPEN) sinyalleri getirir."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, ticker, window, signal_date, entry_price, target_price FROM signals WHERE status = 'OPEN'")
    rows = cursor.fetchall()
    conn.close()
    
    signals = []
    for r in rows:
        signals.append({
            'id': r[0],
            'ticker': r[1],
            'window': r[2],
            'signal_date': r[3],
            'entry_price': r[4],
            'target_price': r[5]
        })
    return signals

def update_signal_statuses(all_data=None):
    """
    Açık sinyalleri son fiyat verileriyle kontrol edip günceller.
    - Hedefe ulaştıysa: WIN (close_price = target_price)
    - Stop olduysa: LOSS (stop-loss = %5, close_price = entry_price * 0.95)
    - 20 günlük şablon için 30 işlem günü, 40 günlük şablon için 60 işlem günü
      sonunda hedefe ulaşamadıysa: EXPIRED (close_price = son kapanış fiyatı)
    """
    open_signals = get_open_signals()
    if not open_signals:
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Benzersiz ticker'ları grupla (API çağrısını azaltmak için)
    tickers_to_update = list(set(s['ticker'] for s in open_signals))
    
    # Eğer önceden çekilmiş veri yoksa yfinance ile geçmiş verileri indir
    ticker_dfs = {}
    if all_data:
        ticker_dfs = {t: df for t, df in all_data.items()}
        
    for ticker in tickers_to_update:
        if ticker not in ticker_dfs:
            try:
                # En fazla 90 gün öncesinden bugüne kadar olan verileri çek
                t_sym = ticker if ticker.endswith(".IS") else ticker + ".IS"
                df = yf.download(t_sym, period="3mo", progress=False, auto_adjust=True, session=session)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.index = pd.to_datetime(df.index)
                    ticker_dfs[ticker] = df
            except Exception as e:
                print(f"⚠️ {ticker} verisi güncellenirken hata: {e}")
                
    for sig in open_signals:
        ticker = sig['ticker']
        df = ticker_dfs.get(ticker)
        if df is None or df.empty:
            continue
            
        entry_price = sig['entry_price']
        target_price = sig['target_price']
        stop_price = entry_price * 0.95 # %5 stop-loss
        
        # Sinyal tarihinden sonraki verileri filtrele
        sig_dt = pd.to_datetime(sig['signal_date'])
        future_df = df[df.index > sig_dt]
        if future_df.empty:
            continue
            
        # Vadeye göre maksimum işlem günü
        max_days = 30 if sig['window'] == 20 else 60
        eval_df = future_df.head(max_days)
        
        status = 'OPEN'
        close_price = None
        close_date = None
        pct_change = None
        
        for date, row in eval_df.iterrows():
            high = float(row['High'])
            low = float(row['Low'])
            close = float(row['Close'])
            
            # Stop loss tetiklendi mi?
            if low <= stop_price:
                status = 'LOSS'
                close_price = stop_price
                close_date = date.strftime('%Y-%m-%d')
                pct_change = -5.0
                break
                
            # Hedef fiyat görüldü mü?
            if high >= target_price:
                status = 'WIN'
                close_price = target_price
                close_date = date.strftime('%Y-%m-%d')
                pct_change = ((target_price - entry_price) / entry_price) * 100
                break
                
        # Eğer vade dolduysa ve hala OPEN ise son günün kapanışıyla kapat
        if status == 'OPEN' and len(eval_df) >= max_days:
            last_row = eval_df.iloc[-1]
            last_date = eval_df.index[-1]
            last_close = float(last_row['Close'])
            
            close_price = last_close
            close_date = last_date.strftime('%Y-%m-%d')
            pct_change = ((last_close - entry_price) / entry_price) * 100
            
            if last_close >= target_price:
                status = 'WIN'
            elif last_close <= stop_price:
                status = 'LOSS'
            else:
                status = 'EXPIRED'
                
        if status != 'OPEN':
            cursor.execute("""
            UPDATE signals 
            SET status = ?, close_date = ?, close_price = ?, pct_change = ?
            WHERE id = ?
            """, (status, close_date, close_price, pct_change, sig['id']))
            print(f"🔄 Pozisyon Kapatıldı: {ticker} ({sig['window']}G) -> Sonuç: {status} | Getiri: {pct_change:+.2f}%")
            
    conn.commit()
    conn.close()

def get_all_signals():
    """Tüm sinyalleri DataFrame olarak döndürür."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY signal_date DESC, id DESC", conn)
    conn.close()
    return df

def get_performance_metrics():
    """Performans istatistiklerini hesaplar."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Genel özet
    cursor.execute("SELECT COUNT(*) FROM signals")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'OPEN'")
    open_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE status != 'OPEN'")
    closed_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'WIN'")
    win_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(pct_change) FROM signals WHERE status != 'OPEN'")
    avg_return = cursor.fetchone()[0] or 0.0
    
    # Pencere (Vade) bazlı win-rate
    cursor.execute("""
    SELECT window, 
           COUNT(*),
           SUM(CASE WHEN status = 'WIN' THEN 1 ELSE 0 END),
           AVG(pct_change)
    FROM signals 
    WHERE status != 'OPEN'
    GROUP BY window
    """)
    window_stats = cursor.fetchall()
    
    conn.close()
    
    win_rate = (win_count / closed_count * 100) if closed_count > 0 else 0.0
    
    return {
        'total': total,
        'open': open_count,
        'closed': closed_count,
        'win_count': win_count,
        'win_rate': round(win_rate, 2),
        'avg_return': round(avg_return, 2),
        'window_stats': window_stats
    }
