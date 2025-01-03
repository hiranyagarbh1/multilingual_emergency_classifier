from models.classifier import EmergencyClassifier

def test_classifier():
    # Initialize classifier
    classifier = EmergencyClassifier()
    
    # Test messages
    test_messages = [
        "Help! There's a fire in my building!",
        "This is a normal message, no emergency here.",
        "URGENT: Medical assistance needed immediately!"
    ]
    
    print("\n=== Emergency Message Classification Demo ===\n")
    
    # Test single message
    print("Testing single message classification:")
    result = classifier.predict(test_messages[0], return_probabilities=True)
    print(f"\nMessage: {result['text']}")
    print(f"Classification: {result['severity']}")
    print(f"Confidence Scores:")
    for label, prob in result['probabilities'].items():
        print(f"  - {label}: {prob:.2f}")
    
    # Test batch classification
    print("\nTesting batch classification:")
    results = classifier.predict(test_messages, return_probabilities=True)
    
    for result in results:
        print(f"\nMessage: {result['text']}")
        print(f"Classification: {result['severity']}")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    test_classifier()