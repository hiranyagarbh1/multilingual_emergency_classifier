from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import torch.nn.functional as F
from typing import List, Dict, Union
import numpy as np

class EmergencyClassifier:
    """A multilingual classifier for emergency messages."""
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 3,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = num_labels
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # Define classification labels
        self.labels = {
            0: "NON_EMERGENCY",
            1: "URGENT",
            2: "CRITICAL"
        }
    
    def preprocess_text(self, texts: Union[str, List[str]]) -> Dict:
        """Tokenize and prepare text input for the model."""
        if isinstance(texts, str):
            texts = [texts]
            
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_probabilities: bool = False
    ) -> Dict:
        """
        Predict emergency severity of input text(s).
        
        Args:
            texts: Input text or list of texts
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with predictions and optional probabilities
        """
        self.model.eval()
        inputs = self.preprocess_text(texts)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        # Convert to numpy for easier handling
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'text': texts[idx] if isinstance(texts, list) else texts,
                'severity': self.labels[pred],
                'severity_code': int(pred)
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    label: float(p)
                    for label, p in zip(self.labels.values(), prob)
                }
                result['confidence'] = float(prob[pred])
            
            results.append(result)
        
        return results[0] if isinstance(texts, str) else results
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        training_args=None,
        **kwargs
    ):
        """
        Fine-tune the model on emergency message data.
        
        Args:
            train_dataset: Dataset for training
            eval_dataset: Dataset for evaluation
            training_args: TrainingArguments instance
            **kwargs: Additional arguments for training
        """
        if training_args is None:
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
                **kwargs
            )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        return trainer.train()
    
    def save_model(self, path: str):
        """Save the model and tokenizer to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str):
        """Load the model and tokenizer from disk."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)