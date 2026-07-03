import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import db_utils
from datetime import datetime

def render_performance_dashboard():
    st.markdown("## 📈 Sinyal Performansı ve Başarı Takibi")
    st.caption(
        "Sistem tarafından üretilen tüm sinyallerin SQLite veritabanındaki geçmişini, "
        "başarı oranlarını ve geriye dönük getirilerini izleyin."
    )
    st.divider()

    # Durum güncelleme butonu (Manuel yenileme)
    c_title, c_refresh = st.columns([4, 1])
    with c_refresh:
        if st.button("🔄 Pozisyon Durumlarını Güncelle", key="refresh_db_btn", use_container_width=True, type="primary"):
            with st.spinner("Güncel fiyatlar indiriliyor ve sinyaller kontrol ediliyor..."):
                try:
                    db_utils.update_signal_statuses()
                    st.success("Pozisyonlar güncellendi!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Güncelleme hatası: {e}")

    # Metrikleri veritabanından çek
    try:
        metrics = db_utils.get_performance_metrics()
        df_all = db_utils.get_all_signals()
    except Exception as e:
        st.error(f"Veritabanı bağlantı hatası: {e}")
        return

    # ── Metrik Kartları ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Sinyal", f"{metrics['total']} adet")
    m2.metric("Açık Pozisyonlar", f"{metrics['open']} adet")
    
    # Win Rate için delta/renk
    win_rate = metrics['win_rate']
    win_rate_color = "green" if win_rate >= 55 else ("orange" if win_rate >= 45 else "red")
    m3.markdown(f"""
    <div style='background:#FFFFFF; border:1px solid #E5E9F0; border-radius:8px; padding:12px'>
        <div style='font-size:14px; color:#6B7280; font-weight:500'>Başarı Oranı (Win Rate)</div>
        <div style='font-size:28px; font-weight:700; color:{win_rate_color}'>%{win_rate:.1f}</div>
        <div style='font-size:12px; color:#9CA3AF'>Hedefe ulaşan / Kapanan</div>
    </div>
    """, unsafe_allow_html=True)
    
    avg_ret = metrics['avg_return']
    avg_ret_color = "#0E9F6E" if avg_ret > 0 else ("#EF5350" if avg_ret < 0 else "#6B7280")
    m4.markdown(f"""
    <div style='background:#FFFFFF; border:1px solid #E5E9F0; border-radius:8px; padding:12px'>
        <div style='font-size:14px; color:#6B7280; font-weight:500'>Ortalama Sinyal Getirisi</div>
        <div style='font-size:28px; font-weight:700; color:{avg_ret_color}'>{avg_ret:+.2f}%</div>
        <div style='font-size:12px; color:#9CA3AF'>Pozisyon başına ortalama</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df_all.empty:
        st.info("💡 Henüz veritabanında sinyal kaydı bulunmuyor. Fırsat Tarayıcı veya daily_scan.py çalıştıktan sonra sinyaller burada listelenecektir.")
        return

    # ── Grafikler Bölümü ────────────────────────────────────────────────────────
    st.markdown("### 📊 Performans Analiz Grafikleri")
    g1, g2 = st.columns(2)

    # 1. Kümülatif Getiri Grafiği (Zaman İçinde)
    df_closed = df_all[df_all['status'] != 'OPEN'].copy()
    if not df_closed.empty:
        # Tarihe göre sırala
        df_closed['close_date'] = pd.to_datetime(df_closed['close_date'])
        df_closed = df_closed.sort_values('close_date')
        df_closed['cum_return'] = df_closed['pct_change'].cumsum()

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df_closed['close_date'],
            y=df_closed['cum_return'],
            mode='lines+markers',
            name='Kümülatif Getiri %',
            line=dict(color='#1A56DB', width=3),
            marker=dict(size=6, color='#1A56DB'),
            hovertemplate='Tarih: %{x}<br>Küm. Getiri: %{y:.2f}%<extra></extra>'
        ))
        fig_cum.update_layout(
            title='Sinyallerin Zaman İçindeki Kümülatif Getirisi (%)',
            xaxis_title='Kapanış Tarihi',
            yaxis_title='Küm. Getiri %',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(gridcolor='#E5E9F0'),
            xaxis=dict(gridcolor='#E5E9F0')
        )
        g1.plotly_chart(fig_cum, use_container_width=True)
    else:
        g1.info("Kümülatif getiri grafiği için kapanmış işlem olması gerekmektedir.")

    # 2. Getiri Dağılımı (Histogram)
    if not df_closed.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_closed['pct_change'],
            nbinsx=15,
            marker_color='#0E9F6E',
            opacity=0.85,
            hovertemplate='Getiri Aralığı: %{x}<br>Sayı: %{y}<extra></extra>'
        ))
        fig_hist.update_layout(
            title='İşlemlerin Getiri Dağılımı (%)',
            xaxis_title='Getiri %',
            yaxis_title='İşlem Sayısı',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(gridcolor='#E5E9F0'),
            xaxis=dict(gridcolor='#E5E9F0')
        )
        g2.plotly_chart(fig_hist, use_container_width=True)
    else:
        g2.info("Getiri dağılım grafiği için kapanmış işlem olması gerekmektedir.")

    st.divider()

    # ── Sinyal Detay Tabloları ──────────────────────────────────────────────────
    st.markdown("### 📋 Sinyal Detayları")
    
    tab_open, tab_closed = st.tabs(["🔓 Açık Pozisyonlar / Fırsatlar", "🔒 Kapanmış Pozisyonlar / Geçmiş"])

    with tab_open:
        df_open = df_all[df_all['status'] == 'OPEN'].copy()
        if not df_open.empty:
            # Potansiyel getiri sütunu ekle
            df_open['pot_return'] = ((df_open['target_price'] - df_open['entry_price']) / df_open['entry_price']) * 100
            
            # Kolay okunabilmesi için formatla
            # expected_days yoksa NaN veya 0 gelebilir, dolgu yapalım
            if 'expected_days' not in df_open.columns:
                df_open['expected_days'] = 0
            df_open['expected_days'] = df_open['expected_days'].fillna(0).astype(int)
            
            df_show_open = df_open[['ticker', 'window', 'signal_date', 'entry_price', 'target_price', 'pot_return', 'expected_days', 'confidence', 'avg_sim', 'source']].copy()
            df_show_open.columns = ['Hisse', 'Şablon Vadesi', 'Sinyal Tarihi', 'Giriş Fiyatı ₺', 'Hedef Fiyatı ₺', 'Potansiyel Getiri %', 'Tahmini Ulaşma (Gün)', 'Güven %', 'PSI Benzerlik', 'Kaynak']
            
            st.dataframe(
                df_show_open.style.format({
                    'Giriş Fiyatı ₺': '{:.2f}',
                    'Hedef Fiyatı ₺': '{:.2f}',
                    'Potansiyel Getiri %': '{:+.2f}%',
                    'Güven %': '%{:.1f}',
                    'PSI Benzerlik': '{:.1f}',
                    'Tahmini Ulaşma (Gün)': '{} gün'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Açık pozisyon bulunmuyor.")

    with tab_closed:
        if not df_closed.empty:
            # Sinyal ile kapanış arasındaki takvim günü farkını hesapla
            df_closed['actual_days'] = (pd.to_datetime(df_closed['close_date']) - pd.to_datetime(df_closed['signal_date'])).dt.days
            if 'expected_days' not in df_closed.columns:
                df_closed['expected_days'] = 0
            df_closed['expected_days'] = df_closed['expected_days'].fillna(0).astype(int)
            df_closed['actual_days'] = df_closed['actual_days'].fillna(0).astype(int)

            df_show_closed = df_closed[['ticker', 'window', 'signal_date', 'close_date', 'entry_price', 'close_price', 'status', 'pct_change', 'expected_days', 'actual_days', 'source']].copy()
            df_show_closed.columns = ['Hisse', 'Şablon Vadesi', 'Sinyal Tarihi', 'Kapanış Tarihi', 'Giriş Fiyatı ₺', 'Kapanış Fiyatı ₺', 'Durum', 'Getiri %', 'Tahmini Ulaşma (Gün)', 'Gerçekleşen Süre (Gün)', 'Kaynak']
            
            # Renklendirme fonksiyonu
            def _color_status(val):
                if val == 'WIN':
                    return 'color: #0E9F6E; font-weight: bold;'
                elif val == 'LOSS':
                    return 'color: #EF5350; font-weight: bold;'
                return 'color: #6B7280;'

            st.dataframe(
                df_show_closed.style.format({
                    'Giriş Fiyatı ₺': '{:.2f}',
                    'Kapanış Fiyatı ₺': '{:.2f}',
                    'Getiri %': '{:+.2f}%',
                    'Tahmini Ulaşma (Gün)': '{} gün',
                    'Gerçekleşen Süre (Gün)': '{} gün'
                }).map(_color_status, subset=['Durum']),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("Kapanmış pozisyon geçmişi bulunmuyor.")
