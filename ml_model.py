import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier

# Standard stream encoding fix for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


# Model kaydetme yolu
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meta_model.pkl")

FEATURE_NAMES = [
    "window", "tpl_change", "tpl_rsi", "weighted_pct", "confidence", "avg_sim",
    "unique_periods", "dispersion", "index_corr", "relative_volume", "expected_days", "regime_val"
]

def get_regime_value(regime_str: str) -> int:
    """Regime etiketini sayısal değere çevirir."""
    regime = str(regime_str).lower()
    if "bullish" in regime and "trend" in regime:
        return 4
    elif "bullish" in regime:
        return 3
    elif "bearish" in regime and "trend" in regime:
        return 0
    elif "bearish" in regime:
        return 2
    return 1 # ranging / default

def extract_features_from_dict(r: dict) -> list:
    """Taramadan gelen tek bir sinyal sözlüğünden özellik vektörü çıkarır."""
    regime_val = get_regime_value(r.get('regime', 'Ranging'))
    index_corr = r.get('index_corr', 0.0)
    if index_corr is None:
        index_corr = 0.0
        
    return [
        float(r.get('window', 20)),
        float(r.get('tpl_change', 0.0)),
        float(r.get('tpl_rsi', 50.0)),
        float(r.get('weighted_pct', 0.0)),
        float(r.get('confidence', 50.0)),
        float(r.get('avg_sim', 70.0)),
        float(r.get('unique_periods', 1)),
        float(r.get('dispersion', 5.0)),
        float(index_corr),
        float(r.get('relative_volume', 1.0)),
        float(r.get('expected_days', 15)),
        float(regime_val)
    ]

def train_and_save_model(all_data: Dict[str, pd.DataFrame], index_closes: np.ndarray = None) -> dict:
    """
    Geçmiş 2 yıllık verileri tarayarak sinyal örnekleri üretir,
    Random Forest modeli eğitir ve kaydeder.
    """
    from backtesting import generate_historical_signals
    from scanner import scan_single_ticker
    
    print("🤖 Model eğitimi için geçmiş sinyaller toplanıyor...")
    
    # 1. Aşama: 20 ve 40 günlük şablonlar için geçmiş sinyalleri üret
    all_signals = []
    
    # generate_historical_signals fonksiyonuna göndermek için find_patterns_fn referansı al
    from app import find_patterns
    
    for win in [90, 120]:
        try:
            sigs = generate_historical_signals(
                all_data=all_data,
                find_patterns_fn=find_patterns,
                window=win,
                min_psi=70,  # Eğitim kümesi için biraz daha geniş tutuyoruz
                min_confidence=45,
                step_days=10, # Hızlı çalışması için adım sayısını 10 yapıyoruz
                max_signals=300
            )
            all_signals.extend(sigs)
        except Exception as e:
            print(f"⚠️ {win} günlük geçmiş sinyal üretiminde hata: {e}")
            
    if not all_signals:
        return {"status": "error", "message": "Hiç geçmiş sinyal üretilemedi. Eğitim veri kümesi boş."}
        
    print(f"📊 Toplam {len(all_signals)} adet geçmiş aday sinyal üretildi. Özellikler çıkarılıyor...")
    
    X_list = []
    y_list = []
    
    # 2. Aşama: Her bir sinyalin özelliklerini çıkar ve etiketle (WIN/LOSS)
    for sig in all_signals:
        ticker = sig.ticker
        df = all_data.get(ticker)
        if df is None or df.empty:
            continue
            
        # Sinyal tarihinin DataFrame'deki yerini bul
        # entry_date formatı 'YYYY-MM-DD'
        try:
            entry_dt = pd.to_datetime(sig.entry_date)
            # Sinyal anına kadar olan veriyi kes (look-ahead bias önleme)
            df_slice = df[df.index <= entry_dt]
            if len(df_slice) < sig.window + 10:
                continue
                
            # Sinyal tarihindeki endeks verisi
            idx_closes_slice = None
            if index_closes is not None:
                # Endeks dizisindeki ilgili alt diziyi kes
                # df index ile eşleştirme
                idx_closes_slice = index_closes[:len(df_slice)]
                
            # scanner.scan_single_ticker'ı çağır
            r = scan_single_ticker(
                ticker=ticker,
                df=df_slice,
                all_data={k: v[v.index <= entry_dt] for k, v in all_data.items() if k != ticker},
                window=sig.window,
                fut_window=int(sig.window * 1.5),
                min_sim=70,
                index_closes=idx_closes_slice
            )
            
            if not r:
                continue
                
            # Gelecekteki fiyat hareketine bakarak etiketleme (Labeling)
            # max_days vade süresi
            max_days = int(sig.window * 1.5)
            future_df = df[df.index > entry_dt].head(max_days)
            if future_df.empty:
                continue
                
            entry_price = sig.entry_price
            # Hedef ve stop seviyeleri
            target_pct = float(r.get('weighted_max', 10.0))
            target_price = entry_price * (1 + target_pct / 100)
            stop_price = entry_price * 0.95 # %5 stop-loss
            
            label = 0 # varsayılan kayıp (veya vade sonu)
            for _, row in future_df.iterrows():
                high = float(row['High'])
                low = float(row['Low'])
                
                # Stop loss tetiklendi mi?
                if low <= stop_price:
                    label = 0
                    break
                # Hedefe ulaştı mı?
                if high >= target_price:
                    label = 1
                    break
                    
            # Özellik vektörünü ekle
            feat_vec = extract_features_from_dict(r)
            X_list.append(feat_vec)
            y_list.append(label)
            
        except Exception as e:
            # Hata veren tekil kayıtları atla
            continue
            
    if len(X_list) < 10:
        return {"status": "error", "message": f"Yetersiz veri kümesi ({len(X_list)} adet). Model eğitilemez."}
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Sınıf dağılımını kontrol et
    win_rate = (y.sum() / len(y)) * 100
    print(f"📉 Veri kümesi boyutu: {len(X)} | Kazanma Oranı (Win Rate): %{win_rate:.1f}")
    
    # 3. Aşama: Random Forest Classifier eğit
    # Veriyi zaman serisi mantığıyla böl (overfitting kontrolü için son %20'yi test yap)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Model
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, min_samples_leaf=2)
    model.fit(X_train, y_train)
    
    # Metrikleri hesapla
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test) if len(X_test) > 0 else 0.0
    
    # Test setindeki Precision (Hassasiyet) - Çok önemlidir çünkü hedefe gidecek olanları doğru bilmeliyiz
    test_precision = 0.0
    if len(X_test) > 0:
        preds = model.predict(X_test)
        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        if (tp + fp) > 0:
            test_precision = tp / (tp + fp)
            
    # Feature Importances
    importances = model.feature_importances_
    feat_imp = {name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)}
    
    # Modeli tüm veri kümesiyle yeniden eğit ve kaydet
    final_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, min_samples_leaf=2)
    final_model.fit(X, y)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
        
    stats = {
        "status": "success",
        "dataset_size": len(X),
        "win_rate": float(win_rate),
        "train_accuracy": float(train_acc * 100),
        "test_accuracy": float(test_acc * 100),
        "test_precision": float(test_precision * 100),
        "feature_importances": feat_imp
    }
    
    import json
    stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meta_model_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        
    return stats


def get_ml_win_probability(r: dict) -> float:
    """Eğitilmiş meta-modeli yükleyerek sinyalin kazanma olasılığını döndürür."""
    if not os.path.exists(MODEL_PATH):
        return None
        
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            
        feat_vec = np.array([extract_features_from_dict(r)])
        # Kazanma olasılığı (class 1 probability)
        prob = model.predict_proba(feat_vec)[0, 1]
        return float(prob * 100)
    except Exception as e:
        print(f"⚠️ ML modeli tahmini hatası: {e}")
        return None


def render_ml_model_page(fetch_batch_fn, fetch_ticker_fn, bist100_list: List[str]):
    import streamlit as st
    import plotly.graph_objects as go
    import json
    
    st.markdown("## 🤖 ML Meta-Model Yönetimi")
    st.caption(
        "Sinyal performansını artırmak amacıyla Scikit-Learn tabanlı Random Forest modelini yönetin. "
        "Model, geçmiş sinyallerin özelliklerini ve kazanma/kaybetme etiketlerini kullanarak "
        "her yeni sinyalin başarı olasılığını tahmin eder."
    )
    st.divider()
    
    # İstatistik dosyasını oku
    stats = None
    stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meta_model_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception:
            pass
            
    # Model eğitimi alanı
    c_info, c_btn = st.columns([3, 1])
    with c_info:
        st.markdown("### 🔄 Modeli Yeniden Eğit")
        st.caption("Son 2 yıllık BIST 100 hisse verilerini indirip geçmiş sinyalleri çıkararak modeli sıfırdan eğitir.")
    with c_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Eğitimi Başlat", use_container_width=True, type="primary"):
            with st.spinner("BIST 100 verileri indiriliyor (2 yıllık)..."):
                try:
                    all_data = fetch_batch_fn(bist100_list, period="2y")
                    idx_df = fetch_ticker_fn("XU100.IS", period="2y")
                    index_closes = idx_df['Close'].values.astype(float) if idx_df is not None and not idx_df.empty else None
                    
                    st.toast("Veri indirildi, model eğitimi başlatılıyor...")
                    res = train_and_save_model(all_data, index_closes)
                    if res.get('status') == 'success':
                        st.success("Model başarıyla eğitildi ve meta_model.pkl olarak kaydedildi!")
                        st.rerun()
                    else:
                        st.error(f"Eğitim hatası: {res.get('message')}")
                except Exception as e:
                    st.error(f"Eğitim sırasında beklenmeyen hata: {e}")
                    
    st.divider()
    
    if stats is None:
        st.info("💡 Eğitilmiş model istatistikleri bulunamadı. Lütfen yukarıdaki 'Eğitimi Başlat' butonuyla modeli ilk kez eğitin.")
        return
        
    # Başarı metrikleri
    st.markdown("### 📊 Model Performans Özeti")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Örnek Sinyal", f"{stats['dataset_size']} adet")
    m2.metric("Tarihsel Win Rate", f"%{stats['win_rate']:.1f}")
    m3.metric("Test Doğruluğu (Accuracy)", f"%{stats['test_accuracy']:.1f}")
    m4.metric("Test Hassasiyeti (Precision)", f"%{stats['test_precision']:.1f}", 
              help="Modelin 'Kazanacak' dediği sinyallerden yüzde kaçinin gerçekten kazandığı. Bu oran ne kadar yüksekse o kadar iyidir.")
              
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Importance Grafiği
    st.markdown("### 🧠 Hangi Parametreler En Önemli?")
    st.caption("Modelin formasyonun başarısına karar verirken indikatörlere ve özelliklere verdiği ağırlıkların dağılımı.")
    
    feat_imp = stats.get('feature_importances', {})
    if feat_imp:
        # Türkçe etiketler
        labels_map = {
            "window": "Şablon Vadesi (Gün)",
            "tpl_change": "Şablon Değişim %",
            "tpl_rsi": "Şablon RSI Değeri",
            "weighted_pct": "Beklenen Getiri %",
            "confidence": "Konsensüs Güveni %",
            "avg_sim": "Ortalama Benzerlik (PSI)",
            "unique_periods": "Dönem Çeşitliliği",
            "dispersion": "Fikir Ayrılığı (Std)",
            "index_corr": "Endeks Korelasyonu",
            "relative_volume": "Göreli Hacim",
            "expected_days": "Tahmini Hedef Süresi",
            "regime_val": "Piyasa Rejimi (Trend/Range)"
        }
        
        sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=False)
        y_labels = [labels_map.get(k, k) for k, _ in sorted_imp]
        x_values = [v * 100 for _, v in sorted_imp]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_values,
            y=y_labels,
            orientation='h',
            marker=dict(color='#1A56DB'),
            hovertemplate='Ağırlık: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=10),
            height=400,
            xaxis=dict(title="Önem Derecesi (%)", gridcolor='#E5E9F0'),
            yaxis=dict(gridcolor='#E5E9F0'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

