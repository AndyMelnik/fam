#!/bin/bash
# FAM - Fuel Analytics Module
# Run script

set -e

echo "🚀 Starting FAM - Fuel Analytics Module..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Run Streamlit
echo "🌐 Starting Streamlit server..."
streamlit run app.py --server.port 8501 --server.headless true



