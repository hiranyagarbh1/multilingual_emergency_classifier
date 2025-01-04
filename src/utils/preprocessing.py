# src/utils/preprocessing.py

import re
from typing import List, Dict
from transformers import pipeline

class TextPreprocessor:
    def __init__(self):
        self.language_classifier = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection"
        )
        
        self.keywords = {
            'non_emergency_keywords': [
                'general', 'inquiry', 'information', 'schedule', 'routine',
                'normal', 'regular', 'question', 'about', 'services'
            ],
            'urgent_keywords': [
                'help', 'asap', 'urgent', 'immediate', 'emergency',
                'ambulance', 'police', 'medical', 'assistance'
            ],
            'critical_keywords': [
                'fire', 'accident', 'dying', 'death', 'critical',
                'fatal', 'crash', 'burning', 'bleeding', 'stroke'
            ]
        }

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s!?.,]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def detect_language(self, text: str) -> Dict:
        result = self.language_classifier(text)[0]
        return {
            'language': result['label'],
            'confidence': result['score']
        }

    def count_keywords(self, text: str) -> Dict[str, int]:
        text_lower = text.lower()
        words = set(text_lower.split())
        
        counts = {}
        for category, keywords in self.keywords.items():
            count = sum(1 for keyword in keywords if keyword in words)
            counts[category] = count
        
        return counts

    def prepare_batch(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            cleaned_text = self.clean_text(text)
            lang_info = self.detect_language(text)
            keyword_counts = self.count_keywords(cleaned_text)
            
            # Calculate additional metrics
            metrics = {
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'word_count': len(cleaned_text.split())
            }
            
            result = {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'language': lang_info['language'],
                'language_confidence': lang_info['confidence'],
                **keyword_counts,
                **metrics
            }
            
            results.append(result)
        
        return results