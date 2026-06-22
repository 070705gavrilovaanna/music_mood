import os
import gradio as gr
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from huggingface_hub import login
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
print("starting app")

hf_token=os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)

df = pd.read_csv('dataset.csv', index_col=0)
df = df.sort_values('popularity', ascending=False).head(10000).reset_index(drop=True)

print("dataset loaded, rows:", len(df))

GENRE_DESC = {
    'rock': 'рок, гитары, драйвовая', 'alternative': 'альтернативный рок',
    'alt-rock': 'альтернативный рок, современный', 'punk': 'панк, быстрая, агрессивная',
    'punk-rock': 'панк-рок, бунтарская', 'grunge': 'гранж, тяжёлый, 90-е',
    'hard-rock': 'хард-рок, тяжёлые гитары', 'psych-rock': 'психоделический рок',
    'rock-n-roll': 'рок-н-ролл, классический', 'rockabilly': 'рокабилли, ретро',
    'british': 'британский рок', 'garage': 'гаражный рок, сырой',
    'goth': 'готик-рок, мрачный', 'metal': 'метал, тяжёлая, агрессивная',
    'black-metal': 'блэк-метал, экстремальный', 'death-metal': 'дэт-метал, брутальный',
    'heavy-metal': 'хэви-метал, классический', 'metalcore': 'металкор, современный',
    'grindcore': 'грайндкор, очень быстрый', 'hardcore': 'хардкор, агрессивный',
    'pop': 'поп, мелодичная, mainstream', 'indie-pop': 'инди-поп, лёгкая',
    'synth-pop': 'синти-поп, 80-е', 'power-pop': 'пауэр-поп, мелодичная',
    'cantopop': 'канто-поп, китайская', 'j-pop': 'J-pop, японская',
    'k-pop': 'K-pop, корейская, танцевальная', 'mandopop': 'мандопоп, китайская',
    'swedish': 'шведская поп', 'french': 'французская поп, шансон',
    'german': 'немецкая поп', 'spanish': 'испанская поп',
    'turkish': 'турецкая поп', 'iranian': 'иранская поп',
    'indian': 'индийская поп, Болливуд', 'hip-hop': 'хип-хоп, рэп, бит',
    'r-n-b': 'ритм-н-блюз, соул, чувственная', 'soul': 'соул, вокальная',
    'funk': 'фанк, грув, танцевальная', 'gospel': 'госпел, духовная, хор',
    'edm': 'электронная, танцевальная, клубная', 'electro': 'электро, синтезаторы',
    'electronic': 'электронная музыка', 'house': 'хаус, танцевальная',
    'deep-house': 'дип-хаус, расслабленный', 'chicago-house': 'чикаго-хаус',
    'detroit-techno': 'детройт-техно, индустриальный', 'techno': 'техно, механический бит',
    'minimal-techno': 'минимал-техно', 'trance': 'транс, эйфорическая',
    'progressive-house': 'прогрессив-хаус', 'dubstep': 'дабстеп, тяжёлый бас',
    'drum-and-bass': 'драм-н-бэйс, быстрый бит', 'breakbeat': 'брейкбит, ломаный бит',
    'idm': 'IDM, интеллектуальная электронная', 'club': 'клубная, танцевальная',
    'dance': 'танцевальная, поп-дэнс', 'disco': 'диско, 70-е',
    'hardstyle': 'хардстайл, жёсткий бит', 'jazz': 'джаз, импровизация, саксофон',
    'blues': 'блюз, гитара, эмоциональная', 'classical': 'классическая, оркестровая',
    'opera': 'опера, театральная', 'piano': 'фортепиано, инструментальная',
    'new-age': 'нью-эйдж, медитативная', 'ambient': 'эмбиент, атмосферная, фоновая',
    'country': 'кантри, акустическая', 'folk': 'фолк, акустическая',
    'bluegrass': 'блюграсс, банджо', 'honky-tonk': 'хонки-тонк, классический кантри',
    'singer-songwriter': 'авторская песня', 'songwriter': 'авторская песня',
    'latin': 'латиноамериканская, ритмичная', 'latino': 'латино, испанская',
    'reggaeton': 'реггетон, латинский хип-хоп', 'salsa': 'сальса, кубинская',
    'samba': 'самба, бразильская', 'brazil': 'бразильская',
    'forro': 'форро, бразильская народная', 'sertanejo': 'сертанежу, бразильский кантри',
    'pagode': 'пагоди, бразильская', 'mpb': 'MPB, бразильская поп',
    'tango': 'танго, аргентинская', 'reggae': 'регги, ямайская, расслабленная',
    'dancehall': 'дэнсхолл, ямайская, энергичная', 'dub': 'даб, ямайская',
    'indie': 'инди, альтернативная, атмосферная', 'emo': 'эмо, эмоциональная',
    'industrial': 'индастриал, экспериментальная', 'trip-hop': 'трип-хоп, атмосферный',
    'acoustic': 'акустическая, гитара, спокойная', 'chill': 'чилл, расслабленная',
    'sleep': 'для сна, медитативная', 'study': 'для учёбы, фоновая',
    'romance': 'романтическая, нежная', 'sad': 'грустная, меланхоличная',
    'happy': 'весёлая, позитивная', 'children': 'детская, простая',
    'kids': 'для детей', 'disney': 'диснеевская, из мультфильмов',
    'show-tunes': 'мюзиклы, театральная', 'world-music': 'мировая, этническая',
    'malay': 'малайская', 'j-dance': 'J-dance, японская танцевальная',
    'j-idol': 'J-idol, японские айдолы', 'j-rock': 'J-rock, японский рок',
    'afrobeat': 'афробит, африканская', 'anime': 'аниме',
    'comedy': 'комедийная', 'party': 'для вечеринки',
    'guitar': 'гитарная, инструментальная', 'groove': 'грув, ритмичная'
}

def make_desc(row):
    g = GENRE_DESC.get(row['track_genre'], row['track_genre'])
    mood = 'весёлая' if row['valence'] > 0.6 else 'грустная' if row['valence'] < 0.4 else 'нейтральная'
    energy = 'энергичная' if row['energy'] > 0.6 else 'спокойная' if row['energy'] < 0.4 else 'умеренная'
    extra = []
    if row['acousticness'] > 0.6:
        extra.append('акустическая')
    if row['instrumentalness'] > 0.5:
        extra.append('без вокала')
    if row['explicit']:
        extra.append('explicit')
    if row['liveness'] > 0.5:
        extra.append('live')
    extra_str = f' ({", ".join(extra)})' if extra else ''
    return f"{row['track_name']} — {row['artists']}. Жанр: {row['track_genre']} ({g}). {mood}, {energy}. {row['tempo']:.0f} BPM. Popularity: {row['popularity']}.{extra_str}"


docs = [Document(
    page_content=make_desc(r),
    metadata={
        'track_id': r['track_id'], 'track_name': r['track_name'],
        'artists': r['artists'], 'album_name': r['album_name'],
        'genre': r['track_genre'], 'popularity': int(r['popularity']),
        'duration_ms': int(r['duration_ms']),
        'valence': float(r['valence']), 'energy': float(r['energy']),
        'danceability': float(r['danceability']),
        'acousticness': float(r['acousticness']),
        'instrumentalness': float(r['instrumentalness']),
        'speechiness': float(r['speechiness']),
        'liveness': float(r['liveness']),
        'tempo': float(r['tempo']), 'loudness': float(r['loudness']),
        'explicit': bool(r['explicit']),
        'key': int(r['key']), 'mode': int(r['mode']),
        'time_signature': int(r['time_signature'])
    }
) for _, r in df.iterrows()]

print("documents created:", len(docs))

emb = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
VECTORSTORE = FAISS.from_documents(docs, emb)

print("vector store created")


GENRE_MAP = {
    'рок': ['rock', 'alternative', 'alt-rock', 'punk', 'grunge', 'hard-rock', 'psych-rock', 'rock-n-roll', 'garage', 'goth', 'british', 'punk-rock', 'rockabilly'],
    'метал': ['metal', 'black-metal', 'death-metal', 'heavy-metal', 'metalcore', 'grindcore', 'hardcore'],
    'панк': ['punk', 'punk-rock'],
    'поп': ['pop', 'indie-pop', 'synth-pop', 'power-pop'],
    'k-pop': ['k-pop'], 'j-pop': ['j-pop'], 'j-rock': ['j-rock'],
    'электрон': ['edm', 'electro', 'electronic', 'house', 'techno', 'trance', 'dubstep', 'drum-and-bass', 'breakbeat', 'idm', 'hardstyle', 'club', 'dance', 'disco', 'progressive-house', 'deep-house', 'chicago-house', 'detroit-techno', 'minimal-techno'],
    'хаус': ['house', 'deep-house', 'chicago-house', 'progressive-house'],
    'техно': ['techno', 'minimal-techno', 'detroit-techno'],
    'хип-хоп': ['hip-hop'], 'рэп': ['hip-hop'],
    'джаз': ['jazz'], 'блюз': ['blues'],
    'классик': ['classical', 'opera', 'piano'],
    'кантри': ['country', 'bluegrass', 'honky-tonk'],
    'фолк': ['folk', 'acoustic', 'singer-songwriter', 'songwriter'],
    'латин': ['latin', 'latino', 'reggaeton', 'salsa', 'samba', 'brazil', 'forro', 'sertanejo', 'pagode', 'mpb', 'tango'],
    'регги': ['reggae', 'dancehall', 'dub'],
    'инди': ['indie', 'indie-pop', 'alternative'],
    'r&b': ['r-n-b', 'soul', 'funk'], 'соул': ['soul'], 'фанк': ['funk'],
    'чилл': ['chill', 'ambient', 'sleep', 'study', 'new-age'],
    'диско': ['disco'],
    'танц': ['dance', 'edm', 'house', 'disco', 'club', 'reggaeton', 'salsa', 'samba'],
    'аниме': ['anime'],
    'детск': ['children', 'kids', 'disney'],
    'этно': ['world-music', 'malay', 'indian', 'turkish', 'iranian', 'afrobeat'],
    'акустик': ['acoustic', 'singer-songwriter', 'guitar', 'folk', 'country', 'bluegrass']
}


def analyze_query(q):
    ql = q.lower()
    f = {}
    if any(w in ql for w in ['грустн', 'печаль', 'тоск', 'меланхол', 'дождь', 'депресс', 'sad']):
        f['max_valence'] = 0.35
    elif any(w in ql for w in ['весёл', 'радост', 'счаст', 'позитив', 'happy']):
        f['min_valence'] = 0.65
    if any(w in ql for w in ['спокойн', 'расслаб', 'тих', 'медитац', 'фон', 'chill', 'study', 'sleep']):
        f['max_energy'] = 0.35
    elif any(w in ql for w in ['энергичн', 'драйв', 'тренировк', 'актив', 'workout']):
        f['min_energy'] = 0.7
    if any(w in ql for w in ['танц', 'dance', 'вечеринк', 'party', 'клуб']):
        f['min_danceability'] = 0.7
    if any(w in ql for w in ['акустик', 'гитар', 'unplugged']):
        f['min_acousticness'] = 0.6
    if any(w in ql for w in ['без слов', 'инструментал', 'для концентрац', 'только музыка']):
        f['min_instrumentalness'] = 0.5
    if any(w in ql for w in ['без мата', 'для детей', 'чистое', 'clean', 'без неценз']):
        f['explicit'] = False
    if any(w in ql for w in ['быстр', 'fast', 'бег', '120 bpm', '130 bpm', '140 bpm']):
        f['min_tempo'] = 120
    elif any(w in ql for w in ['медлен', 'slow', '80 bpm', '90 bpm', 'баллад']):
        f['max_tempo'] = 90
    if any(w in ql for w in ['без рэпа', 'без речи', 'только пение']):
        f['max_speechiness'] = 0.1
    for kw, genres in GENRE_MAP.items():
        if kw in ql:
            f['genres'] = genres
            break
    return f


@tool
def search_tracks(query: str) -> str:
    """поиск треков по запросу с фильтрами по настроению, энергии, жанру."""
    docs = VECTORSTORE.similarity_search(query, k=50)
    f = analyze_query(query)
    filtered = []
    for d in docs:
        m = d.metadata
        if 'min_valence' in f and m['valence'] < f['min_valence']:
            continue
        if 'max_valence' in f and m['valence'] > f['max_valence']:
            continue
        if 'min_energy' in f and m['energy'] < f['min_energy']:
            continue
        if 'max_energy' in f and m['energy'] > f['max_energy']:
            continue
        if 'min_danceability' in f and m['danceability'] < f['min_danceability']:
            continue
        if 'min_acousticness' in f and m['acousticness'] < f['min_acousticness']:
            continue
        if 'min_instrumentalness' in f and m['instrumentalness'] < f['min_instrumentalness']:
            continue
        if 'min_liveness' in f and m['liveness'] < f['min_liveness']:
            continue
        if 'max_speechiness' in f and m['speechiness'] > f['max_speechiness']:
            continue
        if 'min_tempo' in f and m['tempo'] < f['min_tempo']:
            continue
        if 'max_tempo' in f and m['tempo'] > f['max_tempo']:
            continue
        if 'explicit' in f and m['explicit'] != f['explicit']:
            continue
        if 'genres' in f and m['genre'] not in f['genres']:
            continue
        filtered.append(d)
        if len(filtered) >= 5:
            break
    if not filtered:
        return 'Треки не найдены. Попробуй изменить запрос.'
    out = []
    for i, d in enumerate(filtered, 1):
        m = d.metadata
        out.append(f"{i}. {d.page_content}\n   Spotify: https://open.spotify.com/track/{m['track_id']}")
    return '\n\n'.join(out)


@tool
def get_track_info(track_name: str) -> str:
    """получить детали конкретного трека по названию или артисту."""
    docs = VECTORSTORE.similarity_search(track_name, k=3)
    if not docs:
        return 'Трек не найден'
    out = []
    for d in docs:
        m = d.metadata
        out.append(f"Название: {m['track_name']}\nАртист: {m['artists']}\nАльбом: {m['album_name']}\n"
                   f"Жанр: {m['genre']}\nДлительность: {m['duration_ms']//1000//60}:{(m['duration_ms']//1000)%60:02d}\n"
                   f"Темп: {m['tempo']:.0f} BPM\nНастроение (valence): {m['valence']:.2f}\n"
                   f"Энергия: {m['energy']:.2f}\nТанцевальность: {m['danceability']:.2f}\n"
                   f"Акустичность: {m['acousticness']:.2f}\nПопулярность: {m['popularity']}\n"
                   f"Spotify: https://open.spotify.com/track/{m['track_id']}")
    return '\n---\n'.join(out)


@tool
def explain_mood(query: str) -> str:
    """объяснить, какие аудио-фичи подходят под запрос пользователя."""
    filters = analyze_query(query)
    parts = []
    if 'min_valence' in filters:
        parts.append(f"настроение весёлое (valence > {filters['min_valence']})")
    if 'max_valence' in filters:
        parts.append(f"настроение грустное (valence < {filters['max_valence']})")
    if 'min_energy' in filters:
        parts.append(f"энергичная (energy > {filters['min_energy']})")
    if 'max_energy' in filters:
        parts.append(f"спокойная (energy < {filters['max_energy']})")
    if 'min_danceability' in filters:
        parts.append(f"танцевальная (danceability > {filters['min_danceability']})")
    if 'min_acousticness' in filters:
        parts.append(f"акустическая (acousticness > {filters['min_acousticness']})")
    if 'min_instrumentalness' in filters:
        parts.append(f"без вокала (instrumentalness > {filters['min_instrumentalness']})")
    if 'min_tempo' in filters:
        parts.append(f"быстрый темп (tempo > {filters['min_tempo']} BPM)")
    if 'max_tempo' in filters:
        parts.append(f"медленный темп (tempo < {filters['max_tempo']} BPM)")
    if 'explicit' in filters:
        parts.append("без нецензурной лексики")
    if 'genres' in filters:
        parts.append(f"жанры: {', '.join(filters['genres'])}")
    if not parts:
        return 'Запрос общий, без специфических фильтров. Будет использован семантический поиск.'
    return 'Под запрос подходят треки со следующими характеристиками:\n- ' + '\n- '.join(parts)


api_key = os.environ.get('OPENROUTER_API_KEY')
if not api_key:
    raise ValueError('OPENROUTER_API_KEY не установлен')
print("OPENROUTER_API_KEY found")

LLM = ChatOpenAI(
    model='openai/gpt-oss-120b:free',
    openai_api_key=api_key,
    openai_api_base='https://openrouter.ai/api/v1',
    temperature=0.3
)

SYSTEM_PROMPT = """Ты DJ-ассистент для подбора музыки.

ПРАВИЛА:
1. Всегда начинай с инструмента explain_mood, чтобы понять запрос
2. Затем используй search_tracks для подбора треков
3. Если нужно уточнить детали трека - используй get_track_info
4. В финальном ответе укажи 3-5 треков с объяснением, почему они подходят, и ссылками на Spotify
5. Отвечай на русском языке, живо, как настоящий DJ"""

prompt=ChatPromptTemplate.from_messages([('system',SYSTEM_PROMPT), MessagesPlaceholder(variable_name='messages')])

TOOLS = [search_tracks, get_track_info, explain_mood]
agent = create_react_agent(LLM, TOOLS, prompt=prompt)
print("аgent created")


def chat(message, history):
    if not message.strip():
        return 'Введи запрос!'
    try:
        result = agent.invoke({'messages':[('user',message)]})
        return result['messages'][-1].content
    except Exception as e:
        return f'Ошибка: {e}'


with gr.Blocks(title='MusicMood RAG') as demo:
    gr.Markdown('# MusicMood RAG\nDJ-агент на базе 20K треков Spotify. ReAct-агент с 3 инструментами.')
    gr.ChatInterface(
        fn=chat,
        examples=[
            'Подбери грустную музыку для дождливого вечера',
            'Что послушать для тренировки в зале?',
            'Нужен спокойный джаз для работы фоном без слов',
            'Весёлые поп-треки для вечеринки без мата',
            'Расскажи про трек Blinding Lights и подбери похожие',
            'Акустическая гитара для учёбы, медленный темп'
        ]
    )

print("launching Gradio...")
demo.launch(server_name='0.0.0.0', server_port=7860, share=False)