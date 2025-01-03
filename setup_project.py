import os

def create_project_structure():
    # Define the directory structure
    directories = [
        'src/data',
        'src/models',
        'src/utils',
        'src/api',
        'src/web',
        'tests',
        'notebooks',
        'docs',
        'data/raw',
        'data/processed'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py in Python package directories
        if directory.startswith('src/'):
            init_file = os.path.join(directory, '__init__.py')
            with open(init_file, 'w') as f:
                pass
    
    # Create initial files
    files = {
        'README.md': '''# Multilingual Emergency Response Classifier

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

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
''',
        'requirements.txt': '''# Core ML
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0

# API and Web
fastapi>=0.68.0
uvicorn>=0.15.0
streamlit>=1.2.0
python-dotenv>=0.19.0

# Visualization
plotly>=5.3.0

# Testing
pytest>=6.2.5

# Development
black>=22.3.0
isort>=5.10.1
flake8>=4.0.1
''',
        '.gitignore': '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environments
.env
.venv
env/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/processed/*
!data/processed/.gitkeep
data/raw/*
!data/raw/.gitkeep
models/*.pkl
*.log

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints/
''',
        'src/models/__init__.py': '',
        'src/utils/__init__.py': '',
        'src/api/__init__.py': '',
        'src/web/__init__.py': '',
    }
    
    # Create files
    for file_path, content in files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.lstrip())
        
    # Create .gitkeep files in empty directories
    for directory in ['data/raw', 'data/processed']:
        gitkeep_file = os.path.join(directory, '.gitkeep')
        with open(gitkeep_file, 'w') as f:
            pass
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()