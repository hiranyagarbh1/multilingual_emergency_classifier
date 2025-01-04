import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.models.classifier import EmergencyClassifier
import pandas as pd

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Emergency Message Classifier",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visual hierarchy
st.markdown("""
<style>
    .main > div {max-width: 1200px}
    .stTextArea textarea {
        font-size: 16px !important;
        border: 2px solid #e6e6e6;
    }
    .severity-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .critical {background-color: rgba(255, 75, 75, 0.1)}
    .urgent {background-color: rgba(255, 166, 0, 0.1)}
    .non-emergency {background-color: rgba(0, 204, 150, 0.1)}
    .metric-card {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_classifier():
    """Initialize and cache the classifier."""
    return EmergencyClassifier()

def create_confidence_chart(probabilities: dict) -> go.Figure:
    """Create an enhanced confidence score visualization."""
    colors = {
        'CRITICAL': '#ff4b4b',
        'URGENT': '#ffa600',
        'NON_EMERGENCY': '#00cc96'
    }
    
    fig = go.Figure([
        go.Bar(
            x=list(probabilities.values()),
            y=list(probabilities.keys()),
            orientation='h',
            marker_color=[colors[cat] for cat in probabilities.keys()],
            text=[f"{v:.1%}" for v in probabilities.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores",
        xaxis_title="Confidence",
        yaxis_title="",
        showlegend=False,
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(tickformat=".0%")
    )
    
    return fig

def render_severity_badge(severity: str) -> None:
    """Render a styled severity badge."""
    colors = {
        'CRITICAL': 'red',
        'URGENT': 'orange',
        'NON_EMERGENCY': 'green'
    }
    emojis = {
        'CRITICAL': '🔴',
        'URGENT': '🟡',
        'NON_EMERGENCY': '🟢'
    }
    
    st.markdown(
        f"""
        <div class='severity-card {severity.lower()}'>
            <h2>{emojis[severity]} Severity: {severity}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_analysis_metrics(analysis: dict) -> None:
    """Display detailed analysis metrics in an organized layout."""
    st.subheader("Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Emergency Indicators")
        indicators = {
            "Critical Patterns": analysis['critical'],
            "Urgent Patterns": analysis['urgent'],
            "Non-Emergency Patterns": analysis['non_emergency']
        }
        for label, value in indicators.items():
            st.metric(label, value)
    
    with col2:
        st.markdown("##### Text Characteristics")
        metrics = {
            "Exclamation Marks": analysis['exclamation_marks'],
            "ALL CAPS Words": analysis['caps_words'],
            "CAPS Ratio": f"{analysis['caps_ratio']:.1%}"
        }
        for label, value in metrics.items():
            st.metric(label, value)

def initialize_session_state():
    """Initialize or reset session state variables."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'message' not in st.session_state:
        st.session_state.message = ""
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load classifier
    classifier = initialize_classifier()
    
    # Main interface
    st.title("🚨 Emergency Message Classifier")
    st.markdown("### Analyze and classify emergency messages with detailed explanation")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        message = st.text_area(
            "Enter your message:",
            value=st.session_state.message,
            height=150,
            key="message_input",
            placeholder="Type or paste your message here..."
        )
        
        analyze_button = st.button("🔍 Analyze Message", type="primary")
    
    with col2:
        st.info("""
        **Example Messages:**
        
        🔴 Critical:
        - "Help! Fire in building B!"
        - "Car accident with injuries on Main St"
        
        🟡 Urgent:
        - "Need medical assistance ASAP"
        - "Stuck in elevator, need help!"
        
        🟢 Non-Emergency:
        - "Schedule routine maintenance"
        - "General inquiry about services"
        """)
    
    # Analysis section
    if analyze_button and message:
        try:
            # Get prediction
            result = classifier.predict(message)
            
            # Store in session state
            st.session_state.last_analysis = result
            if len(st.session_state.history) >= 5:
                st.session_state.history.pop(0)
            st.session_state.history.append(result)
            
            # Display results
            st.markdown("---")
            
            # Severity and confidence
            render_severity_badge(result['severity'])
            
            # Confidence scores visualization
            fig = create_confidence_chart(result['probabilities'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis
            display_analysis_metrics(result['analysis'])
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    
    # Sidebar with history and controls
    with st.sidebar:
        st.header("Analysis History")
        if st.button("Clear History", type="secondary"):
            st.session_state.history = []
            st.session_state.last_analysis = None
            st.session_state.message = ""
            st.rerun()
        
        if st.session_state.history:
            for past_analysis in reversed(st.session_state.history):
                with st.expander(f"{past_analysis['text'][:50]}..."):
                    st.write(f"Severity: {past_analysis['severity']}")
                    st.write(f"Confidence: {past_analysis['confidence']:.1%}")
                    st.write("Key Indicators:")
                    st.write(f"- Critical Patterns: {past_analysis['analysis']['critical']}")
                    st.write(f"- Urgent Patterns: {past_analysis['analysis']['urgent']}")

if __name__ == "__main__":
    main()