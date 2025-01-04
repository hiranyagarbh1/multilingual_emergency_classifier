from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union
import re

class EmergencyClassifier:
    """
    A multilingual emergency message classifier supporting English, French, and Spanish.
    Combines transformer-based classification with pattern matching for robust emergency detection.
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 3,
        device: str = None
    ):
        """
        Initialize the emergency classifier with model and multilingual patterns.
        
        Args:
            model_name: Name of the pre-trained model to use
            num_labels: Number of classification categories
            device: Computation device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # Classification categories
        self.categories = {
            0: "NON_EMERGENCY",
            1: "URGENT",
            2: "CRITICAL"
        }
        
        # Multilingual emergency patterns
        self.patterns = {
            'critical': [
                # English patterns
                r'\b(fire|burning|explosion|crash|accident)\b',
                r'\b(dying|death|fatal|stroke|heart attack)\b',
                r'\b(not breathing|unconscious|bleeding)\b',
                r'\b(emergency|critical|life.?threatening)\b',
                
                # French patterns
                r'\b(incendie|feu|explosion|accident grave)\b',
                r'\b(mort|mourant|inconscient|ne respire)\b',
                r'\b(effondré|blessé grave|sang|hémorragie)\b',
                r'\b(au secours|à l\'aide|urgence vitale)\b',
                r'\b(danger|critique|vie en danger)\b',
                
                # Spanish patterns
                r'\b(fuego|incendio|explosión|accidente grave)\b',
                r'\b(muriendo|muerte|fatal|inconsciente)\b',
                r'\b(no respira|sangrado|hemorragia)\b',
                r'\b(crítico|vida en peligro|grave)\b'
            ],
            'urgent': [
                # English patterns
                r'\b(help|urgent|asap|immediately)\b',
                r'\b(medical|ambulance|police|assistance)\b',
                r'\b(hurt|injured|wound|pain)\b',
                
                # French patterns
                r'\b(aide|urgence|urgent|immédiat)\b',
                r'\b(médical|ambulance|police|assistance)\b',
                r'\b(coincé|bloqué|blessé|douleur)\b',
                r'\b(besoin|intervention|secours)\b',
                
                # Spanish patterns
                r'\b(ayuda|urgencia|urgente|inmediato)\b',
                r'\b(médico|ambulancia|policía|asistencia)\b',
                r'\b(auxilio|socorro|emergencia)\b',
                r'\b(herido|dolor|necesito)\b'
            ],
            'non_emergency': [
                # English patterns
                r'\b(schedule|routine|regular|general)\b',
                r'\b(inquiry|question|information|about)\b',
                r'\b(appointment|checkup|consultation)\b',
                
                # French patterns
                r'\b(rendez-vous|routine|régulier|général)\b',
                r'\b(consultation|information|horaire|question)\b',
                r'\b(renouvellement|ordinaire|normal)\b',
                
                # Spanish patterns
                r'\b(cita|rutina|regular|general)\b',
                r'\b(consulta|información|pregunta|horario)\b',
                r'\b(normal|ordinario|reservar)\b'
            ]
        }
        
        # Classification thresholds
        self.thresholds = {
            "CRITICAL": 0.45,
            "URGENT": 0.35
        }

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text to handle special characters in French and Spanish.
        """
        # Normalize accented characters
        replacements = {
            'é': 'e', 'è': 'e', 'ê': 'e',
            'à': 'a', 'â': 'a',
            'ù': 'u', 'û': 'u',
            'î': 'i', 'ï': 'i',
            'ô': 'o',
            'ç': 'c'
        }
        
        processed = text.lower()
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        return processed

    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive text analysis for emergency indicators.
        """
        # Process text while preserving original
        processed_text = self._preprocess_text(text)
        text_lower = text.lower()
        
        # Pattern matching on both original and processed text
        pattern_matches = {
            category: sum(1 for pattern in patterns 
                        if re.search(pattern, text_lower) or 
                           re.search(pattern, processed_text))
            for category, patterns in self.patterns.items()
        }
        
        # Text characteristics analysis
        characteristics = {
            'exclamation_marks': text.count('!'),
            'caps_words': sum(1 for word in text.split() if word.isupper()),
            'word_count': len(text.split()),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'emergency_indicators': pattern_matches['critical'] + pattern_matches['urgent']
        }
        
        return {**pattern_matches, **characteristics}

    def adjust_probabilities(self, base_probs: np.ndarray, indicators: Dict) -> np.ndarray:
        """
        Adjust classification probabilities based on text analysis.
        """
        adjusted = base_probs.copy()
        
        # Critical patterns boost
        if indicators['critical'] > 0:
            adjusted[2] *= (1.8 + 0.3 * indicators['critical'])
            adjusted[0] *= 0.2
        
        # Urgent patterns boost
        if indicators['urgent'] > 0:
            adjusted[1] *= (1.5 + 0.2 * indicators['urgent'])
            adjusted[0] *= 0.3
        
        # Non-emergency patterns boost
        if indicators['non_emergency'] > 0:
            adjusted[0] *= (1.4 + 0.1 * indicators['non_emergency'])
            adjusted[1:] *= 0.3
        
        # Additional signal boosters
        if indicators['exclamation_marks'] > 0:
            adjusted[1:] *= (1.2 + 0.1 * indicators['exclamation_marks'])
        
        if indicators['caps_ratio'] > 0.3:
            adjusted[1:] *= 1.2
        
        # Enhanced emergency emphasis
        if indicators['emergency_indicators'] > 1:
            adjusted[0] *= 0.5
        
        # Normalize probabilities
        return adjusted / adjusted.sum()

    def predict(self, text: str) -> Dict:
        """
        Predict emergency severity with detailed analysis.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing classification results and analysis
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get base predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            base_probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        # Analyze text and adjust probabilities
        indicators = self.analyze_text(text)
        adjusted_probs = self.adjust_probabilities(base_probs, indicators)
        
        # Determine final classification
        if (indicators['critical'] >= 2 or 
            (indicators['critical'] >= 1 and indicators['exclamation_marks'] >= 1)):
            pred_idx = 2  # Force CRITICAL
        elif (indicators['urgent'] >= 2 or 
              (indicators['urgent'] >= 1 and indicators['exclamation_marks'] >= 1)):
            pred_idx = 1  # Force URGENT
        elif indicators['non_emergency'] >= 1:
            pred_idx = 0  # Force NON_EMERGENCY
        else:
            pred_idx = adjusted_probs.argmax()
        
        # Prepare detailed response
        return {
            'text': text,
            'severity': self.categories[pred_idx],
            'confidence': float(adjusted_probs[pred_idx]),
            'probabilities': {
                self.categories[i]: float(p) 
                for i, p in enumerate(adjusted_probs)
            },
            'analysis': indicators
        }

    def save_model(self, path: str) -> None:
        """Save the model and tokenizer to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str) -> None:
        """Load the model and tokenizer from disk."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)