# Multilingual Emergency Response Classifier

An advanced NLP system for classifying and prioritizing emergency messages across multiple languages.

## Project Structure
```
multilingual_emergency_classifier/
├── src/
│   ├── data/          # Data handling utilities
│   ├── models/        # ML models
│   ├── utils/         # Helper functions
│   ├── api/           # FastAPI backend
│   └── web/          # Streamlit frontend
├── tests/             # Unit tests
├── notebooks/         # Jupyter notebooks
├── docs/             # Documentation
└── data/             # Dataset storage
    ├── raw/          # Original data
    └── processed/    # Processed data
```

## Features
- Multi-language support (English, Spanish, French)
- Emergency severity classification
- Response template generation
- Real-time processing capability
- API and Web interface

## Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. Run the demo:
   ```bash
   python src/demo.py
   ```

2. Start the API:
   ```bash
   uvicorn src.api.app:app --reload
   ```

3. Launch web interface:
   ```bash
   streamlit run src/web/app.py
   ```
