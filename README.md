# Рекомендательная система музыки: ReAct-агент с FAISS и фильтрацией по аудио-признакам

Агент на базе 10 000 треков Spotify с гибридным поиском и ReAct-архитектурой.

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)


## Возможности

- ReAct-агент с 3 инструментами (explain_mood, search_tracks, get_track_info)
- Гибридный поиск: семантический (FAISS + all-MiniLM-L6-v2) + фильтрация по метаданным
- 12 аудио-фич Spotify для точного подбора
- Понимание русского языка
- Прямые ссылки на Spotify для каждого трека
- Автоматическая оценка качества рекомендаций
- LLM: GPT-OSS 120B через OpenRouter

## 1. Клонирование

```bash
git clone https://github.com/YOUR_USERNAME/music-mood-rag.git
cd music-mood-rag
```

## 2. Виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

## 3. Зависимости

```bash
pip install -r requirements.txt
```

## 4. API-ключи

Создайте `.env`:

```env
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

## 5. Запуск

```bash
python app.py
```

Откройте http://localhost:7860

## Примеры запросов

- "Подбери грустную музыку для дождливого вечера"
- "Энергичные треки для тренировки"
- "Спокойный джаз для работы фоном без слов"
- "Расскажи про трек Blinding Lights"

## Как работает

1. Пользователь вводит запрос на русском языке
2. Агент анализирует запрос с помощью explain_mood и определяет нужные аудио-фичи (valence, energy, tempo и др.)
3. Выполняется семантический поиск по 10 000 трекам через FAISS
4. Результаты фильтруются по аудио-параметрам (грустный/весёлый, быстрый/медленный, акустика/инструментал и т.д.)
5. Агент формирует ответ с 3-5 треками, объяснением и ссылками на Spotify

## Оценка качества

Для каждого запроса автоматически считаются метрики:
- Релевантность - насколько треки соответствуют запросу
- Разнообразие - насколько треки отличаются друг от друга
- Покрытие жанров - сколько уникальных жанров в рекомендациях
- Новизна - насколько треки непопулярны

Метрики сохраняются в `evaluation_history.json` для анализа.

## Технологии

- LangGraph + LangChain (ReAct-паттерн)
- FAISS + sentence-transformers (all-MiniLM-L6-v2)
- GPT-OSS 120B через OpenRouter API
- Gradio 4.44
- Spotify Tracks Dataset (10 000 треков)

## Структура проекта

```
.
├── app.py              # агент, инструменты, UI
├── evaluation.py       # система оценки качества
├── requirements.txt    # зависимости
├── dataset.csv         # датасет Spotify
├── .env                # API-ключи
└── screenshots/        # скриншоты
```
