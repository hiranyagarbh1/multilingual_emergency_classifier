from models.classifier import EmergencyClassifier
from utils.preprocessing import TextPreprocessor

def test_system():
    classifier = EmergencyClassifier()
    preprocessor = TextPreprocessor()
    
    test_messages = [
        "Help! There's a fire in my building!",
        "¡Ayuda! ¡Necesito una ambulancia inmediatamente!",
        "Just a normal message, no emergency.",
        "URGENT! Medical assistance needed!"
    ]
    
    print("\n=== Emergency Message Analysis Demo ===\n")
    
    for message in test_messages:
        # Analyze text
        processed = preprocessor.prepare_batch([message])[0]
        # Get classification
        classification = classifier.predict(message, return_probabilities=True)
        
        print(f"\nOriginal Message: {message}")
        print(f"Language: {processed['language']} (confidence: {processed['language_confidence']:.2f})")
        print(f"Emergency Keywords: {processed['emergency_keyword_count']}")
        print(f"Classification: {classification['severity']}")
        print(f"Confidence: {classification['confidence']:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    test_system()