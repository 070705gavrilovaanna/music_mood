# MusicMood RAG

DJ-агент на базе 10 000 треков Spotify с гибридным поиском и ReAct-архитектурой.

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

## Возможности

- ReAct-агент с 3 инструментами (explain_mood, search_tracks, get_track_info)
- Гибридный поиск: семантический (FAISS + all-MiniLM-L6-v2) + фильтрация по метаданным
- 12 аудио-фич Spotify для точного подбора
- Понимание русского языка
- Прямые ссылки на Spotify для каждого трека
- LLM: Qwen3-Next 80B через OpenRouter


## Старт

### 1. Клонирование

```bash
git clone https://github.com/YOUR_USERNAME/music-mood-rag.git
cd music-mood-rag
```

### 2. Виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Зависимости

```bash
pip install -r requirements.txt
```


### 4. API-ключи

Создайте `.env`:

```env
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 5. Запуск

```bash
python app.py
```

Откройте http://localhost:7860

## Примеры запросов

- "Подбери грустную музыку для дождливого вечера"
- "Энергичные треки для тренировки"
- "Спокойный джаз для работы фоном без слов"
- "Расскажи про трек Blinding Lights"

## Технологии

- LangGraph + LangChain (ReAct-паттерн)
- FAISS + sentence-transformers (all-MiniLM-L6-v2)
- Qwen3-Next 80B через OpenRouter API
- Gradio 4.44
- Spotify Tracks Dataset (10 000 треков)

## Структура проекта

```
.
├── app.py              # Агент, инструменты, UI
├── requirements.txt    # Зависимости
├── dataset.csv         # Датасет Spotify
├── .env                # API-ключи (не коммитится)
└── screenshots/        # Скриншоты
```
