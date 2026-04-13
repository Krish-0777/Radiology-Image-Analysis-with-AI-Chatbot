#!/bin/bash
# check_gemini_setup.sh - Verify Gemini API Key is properly configured

echo "🔍 Checking Gemini API Setup..."
echo "================================"

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY is not set"
    echo ""
    echo "Setup instructions:"
    echo "1. Get your API key at: https://ai.google.dev/"
    echo "2. Set the environment variable:"
    echo ""
    echo "   export GEMINI_API_KEY='your_actual_api_key_here'"
    echo ""
    echo "3. Or create a .env file in the app folder with:"
    echo "   GEMINI_API_KEY=your_actual_api_key_here"
    echo ""
    echo "4. Then run the app:"
    echo "   streamlit run app.py"
else
    echo "✅ GEMINI_API_KEY is set"
    echo "   Key length: ${#GEMINI_API_KEY} characters"
    echo "   (This is good - key is configured)"
fi

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 is installed"
else
    echo "❌ Python 3 not found"
fi

# Check if streamlit is installed
python3 -c "import streamlit" 2>/dev/null && echo "✅ Streamlit is installed" || echo "❌ Streamlit not installed - run: pip install streamlit"

# Check if google-genai is installed
python3 -c "from google import genai" 2>/dev/null && echo "✅ Google GenAI is installed" || echo "❌ Google GenAI not installed - run: pip install google-genai"

echo ""
echo "================================"
echo "Setup complete! Run the app:"
echo "  streamlit run app.py"
