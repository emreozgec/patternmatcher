import os
import html
import requests
from datetime import datetime

def send_telegram_message(text: str) -> bool:
    """Telegram bot üzerinden mesaj gönder."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("⚠️ TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID ayarlanmamış.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return True
        print(f"⚠️ Telegram hatası: {resp.status_code} — {resp.text}")
        return False
    except Exception as e:
        print(f"⚠️ Telegram bağlantı hatası: {e}")
        return False

def format_results_message(results_by_window: dict, scope: str) -> list:
    """
    Tarama sonuçlarını Telegram mesaj(lar)ı haline getir.
    Telegram mesaj limiti ~4096 karakter olduğu için gerekirse bölünür.
    """
    messages = []
    header = f"<b>🔍 BIST Pattern Matcher Fırsat Raporu ({scope})</b>\n"
    header += f"📅 Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
    header += "──────────────────\n"
    current_msg = header

    for window, results in results_by_window.items():
        if not results:
            continue

        section = f"\n📈 <b>{window} Günlük Şablon</b> — {len(results)} fırsat\n\n"
        if len(current_msg) + len(section) > 3800:
            messages.append(current_msg)
            current_msg = header

        current_msg += section

        for r in results[:10]:  # Mesaj başına max 10 hisse
            ticker_esc = html.escape(r['ticker'])
            stop_price = r['current_price'] * 0.95
            ml_prob_str = f" | 🤖 ML: %{r['ml_prob']:.0f}" if r.get('ml_prob') is not None else ""
            line = (
                f"🏢 <b>{ticker_esc}</b> — {r['current_price']:.2f} ₺\n"
                f"   📈 Beklenen: {r['weighted_pct']:+.1f}% → "
                f"🎯 Hedef: {r['target']:.2f} ₺\n"
                f"   🛑 Stop-Loss: {stop_price:.2f} ₺ (5%)\n"
                f"   🔒 Güven: %{r['confidence']:.0f} | Vade: ~{r.get('expected_days', 0)} gün{ml_prob_str}\n"
                f"   PSI: {r['avg_sim']:.0f} | Oy: {r['up_count']}/{r['total_matches']}\n"
            )

            if r.get('relative_volume', 1.0) > 1.5:
                line += "   🔥 Hacimli Kırılım Sinyali\n"
            if not r.get('index_trend_bullish', True):
                line += "   ⚠️ Endeks Negatif Trendde\n"
            if r.get('index_penalty_applied'):
                line += "   ⚠️ Piyasa geneli hareket olabilir\n"
            if r.get('formations'):
                fmts_esc = html.escape(', '.join(r['formations'][:2]))
                line += f"   🔷 {fmts_esc}\n"
            line += "\n"


            if len(current_msg) + len(line) > 3800:
                messages.append(current_msg)
                current_msg = header + section

            current_msg += line

    if current_msg.strip() != header.strip():
        messages.append(current_msg)

    if messages:
        # Feragatnameyi son mesaja ekle
        messages[-1] += (
            "\n⚠️ <i>Bu bir yatırım tavsiyesi değildir. "
            "Karar vermeden önce kendi analizinizi yapın.</i>"
        )

    return messages
