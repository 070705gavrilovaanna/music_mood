import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger=logging.getLogger(__name__)


class RelevanceEvaluator:
    '''Оценивает релевантность рекомендаций запросу'''

    def __init__(self, df:pd.DataFrame):
        self.df=df
        self.semantic_model=SentenceTransformer('all-MiniLM-L6-v2')

        # правила для запросов
        self.query_rules={
            'грустн': {'valence': (0.0, 0.4), 'energy': (0.0, 0.6)},
            'печаль': {'valence': (0.0, 0.4), 'energy': (0.0, 0.6)},
            'весел': {'valence': (0.7, 1.0), 'energy': (0.5, 1.0)},
            'радост': {'valence': (0.7, 1.0)},
            'энергичн': {'energy': (0.7, 1.0), 'danceability': (0.6, 1.0)},
            'спокойн': {'energy': (0.0, 0.35), 'danceability': (0.0, 0.4)},
            'танц': {'danceability': (0.7, 1.0)},
            'акустик': {'acousticness': (0.7, 1.0)},
            'без слов': {'instrumentalness': (0.7, 1.0)},
            'инструментал': {'instrumentalness': (0.7, 1.0)},
            'дожд': {'valence': (0.0, 0.4), 'acousticness': (0.3, 1.0)},
            'тренировк': {'energy': (0.7, 1.0), 'tempo': (120, 200)},
            'workout': {'energy': (0.7, 1.0), 'tempo': (120, 200)},
            'быстр': {'tempo': (120, 200)},
            'медлен': {'tempo': (0, 90)},
            'джаз': {'acousticness': (0.5, 1.0), 'instrumentalness': (0.3, 1.0)},
            'рок': {'energy': (0.6, 1.0), 'acousticness': (0.0, 0.4)},
            'поп': {'danceability': (0.5, 1.0), 'energy': (0.4, 0.8)},   
        }

        self._embedding_cache={}
        self.features=['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'tempo']
        self.scaler=StandardScaler()
        self._fit_scaler()

    def _fit_scaler(self):
        feature_matrix=self.df[self.features].values
        self.scaler.fit(feature_matrix)

    def _extract_rules(self, query: str) -> Dict:
        query_lower=query.lower()
        rules={}
        for keyword, feature_rules in self.query_rules.items():
            if keyword in query_lower:
                for feature, (min_val, max_val) in feature_rules.items():
                    if feature not in rules:
                        rules[feature]= {'min': min_val, 'max': max_val}
                    else:
                        rules[feature]['min']=min(rules[feature]['min'], min_val)
                        rules[feature]['max']=max(rules[feature]['max'], max_val)
        return rules
    
    def _score_track_by_rules(self, track: Dict, rules:Dict) -> float:
        if not rules:
            return 0.5
        
        score=0
        total_weight=0

        for feature, range_vals in rules.items():
            if feature in track and track[feature] is not None:
                value=track[feature]
                min_val=range_vals['min']
                max_val=range_vals['max']

                if min_val<=value<=max_val:
                    center=(min_val + max_val)/2
                    if max_val > min_val:
                        distance_to_center = abs(value - center)
                        max_distance = (max_val - min_val)/2
                        feature_score = 1 - (distance_to_center/max_distance)
                    else:
                        feature_score=1
                else:
                    if value < min_val:
                        penalty = (min_val - value) / (min_val + 0.001) if min_val>0 else 1
                    else:
                        penalty = (value - max_val) / (1 - max_val + 0.001) if max_val < 1 else 1
                    feature_score = max(0, 1 - penalty * 2)
                
                score += feature_score
                total_weight += 1
        
        return score/total_weight if total_weight > 0 else 0.5
    
    def _semantic_similarity(self, query: str, track_text: str) -> float:
        if query not in self._embedding_cache:
            self._embedding_cache[query] = self.semantic_model.encode([query])
        if track_text not in self._embedding_cache:
            self._embedding_cache[track_text] = self.semantic_model.encode([track_text])
        
        query_embed=self._embedding_cache[query]
        track_embed = self._embedding_cache[track_text]

        return cosine_similarity(query_embed, track_embed)[0][0]

    def evaluate_recommendations(self, query: str, recomendations: List[Dict]) -> Dict:
        if not recomendations:
            return {
                'avg_relevance': 0.0,
                'good_rate': 0.0, 
                'num_tracks': 0,
                'interpretation': 'Нет рекомендаций'
            }
        rules=self._extract_rules(query)

        feature_scores=[]
        semantic_scores=[]


        for track in recomendations:
            feature_score = self._score_track_by_rules(track, rules)
            feature_scores.append(feature_score)

            track_text=f"{track.get('track_name', '')} {track.get('artists', '')} {track.get('genre', '')}"
            sem_score = self._semantic_similarity(query, track_text) if track_text.strip() else 0.5
            semantic_scores.append(sem_score)

        combined_scores=[0.6*f + 0.4*s for f,s in zip(feature_scores, semantic_scores)]

        avg_relevance=np.mean(combined_scores)
        good_count = sum(1 for s in combined_scores if s > 0.6)
        good_rate = good_count / len(combined_scores)

        if avg_relevance>0.8 and good_count==len(combined_scores):
            interpretation = 'Все рекомендации релевантны'
        elif avg_relevance>0.6 and good_rate>=0.7:
            interpretation = 'Большинство рекомендаций релевантны'
        elif avg_relevance>0.4:
            interpretation = 'Есть нерелевантные треки'
        else:
            interpretation = 'Рекомендации не соответствуют запросу'

        return {
            'avg_relevance': avg_relevance,
            'good_rate': good_rate,
            'good_count': good_count,
            'total_tracks': len(recomendations),
            'interpretation': interpretation,
            'rules_used': rules
        }


class MusicMetrics:
    '''Метрики рекомендаций: diversity, coverage, novelty'''

    def __init__(self, df: pd.DataFrame):
        self.df=df
        self.total_genres=df['track_genre'].nunique()
        self.max_pop=df['popularity'].max()
        self.min_pop=df['popularity'].min()
        self.features=['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'tempo']
        self.scaler=StandardScaler()
        self.scaler.fit(df[self.features].values)

    def diversity(self, recommendations: List[Dict]) -> float:
        if len(recommendations) < 2:
            return 0.0

        vectors = []
        for rec in recommendations:
            vec=[rec.get(f,0) for f in self.features]
            vectors.append(np.array(vec))

        vectors=np.array(vectors)
        vectors=self.scaler.transform(vectors)

        distances=[]
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                sim=cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                distances.append(1-sim)

        return np.mean(distances) if distances else 0.0

    def coverage(self, recommendations: List[Dict]) -> float:
        if not recommendations:
            return 0.0
        unique_genres=len(set(rec.get('genre', 'unknown') for rec in recommendations))

        return unique_genres / self.total_genres if self.total_genres>0 else 0.0

    def novelty(self, recommendations: List[Dict]) -> float:
        if not recommendations:
            return 0.0
        novelty_scores=[]
        for rec in recommendations:
            pop=rec.get('popularity',0)
            if self.max_pop > self.min_pop:
                normalized_pop = (pop - self.min_pop) / (self.max_pop - self.min_pop)
                novelty_scores.append(1 - normalized_pop)
            else:
                novelty_scores.append(0)
        return np.mean(novelty_scores)

    def calculate_all(self, recommendations: List[Dict]) -> Dict:
        return {
            'diversity': self.diversity(recommendations),
            'coverage': self.coverage(recommendations),
            'novelty': self.novelty(recommendations),
            'num_tracks': len(recommendations)
        }


class MetricsLogger:
    '''Логирование метрик в JSON файл'''

    def __init__(self, log_file='evaluation_history.json'):
        self.log_file=log_file
        self.history=self._load_history()

    def _load_history(self) -> List[Dict]:
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_history(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def log(self, query: str, relevance: Dict, metrics: Dict):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query': str(query),
            'avg_relevance': float(relevance.get('avg_relevance', 0)),
            'good_rate': float(relevance.get('good_rate', 0)),
            'interpretation': str(relevance.get('interpretation', '')),
            'diversity': float(metrics.get('diversity', 0)),
            'coverage': float(metrics.get('coverage', 0)),
            'novelty': float(metrics.get('novelty', 0)),
            'num_tracks': int(metrics.get('num_tracks', 0))
        }
        self.history.append(entry)
        self._save_history()

        if relevance.get('avg_relevance', 0) < 0.4:
            logger.warning(f"Низкая релевантность: {query} ({relevance['avg_relevance']:.2f})")

    def get_stats(self) -> Dict:
        if not self.history:
            return {'total_queries':0}

        avg_rel=np.mean([h['avg_relevance'] for h in self.history])
        good_queries = sum(1 for h in self.history if h.get('good_rate', 0)>0.7)

        return {
            'total_queries': len(self.history),
            'avg_relevance': avg_rel,
            'good_queries': good_queries,
            'good_rate': good_queries / len(self.history) if self.history else 0
        }





