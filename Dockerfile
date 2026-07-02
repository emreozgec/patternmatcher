FROM python:3.11-slim

# Numba ve yfinance için derleme bağımlılıklarını kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Veri dizini için klasör oluştur (SQLite persistence için)
RUN mkdir -p data

# Streamlit portunu dışa aç
EXPOSE 8501

# Uygulamayı başlat
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
