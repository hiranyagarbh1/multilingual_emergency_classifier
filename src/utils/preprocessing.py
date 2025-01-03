from transformers import pipeline
import re
from typing import List, Dict
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        self.language_classifier = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection"
        )
        self.emergency_keywords = self._create_emergency_keywords()
    
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s!?.,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def detect_language(self, text: str) -> Dict:
        result = self.language_classifier(text)[0]
        return {
            'language': result['label'],
            'confidence': result['score']
        }
    
    def _create_emergency_keywords(self) -> Dict[str, List[str]]:
        return {
            'en': ['emergency', 'help', 'urgent', 'critical', 'danger'],
            'es': ['emergencia', 'ayuda', 'urgente', 'crítico', 'peligro'],
            'fr': ['urgence', 'aide', 'urgent', 'critique', 'danger']
        }
    
    def analyze_emergency_indicators(self, text: str) -> Dict:
        text = text.lower()
        lang_info = self.detect_language(text)
        lang = lang_info['language']
        
        keywords = self.emergency_keywords.get(lang, self.emergency_keywords['en'])
        keyword_count = sum(1 for word in keywords if word in text)
        
        return {
            'language': lang,
            'language_confidence': lang_info['confidence'],
            'emergency_keyword_count': keyword_count,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text)
        }
    
    def prepare_batch(self, texts: List[str]) -> List[Dict]:
        return [
            {
                'original_text': text,
                'cleaned_text': self.clean_text(text),
                **self.analyze_emergency_indicators(text)
            }
            for text in texts
        ]