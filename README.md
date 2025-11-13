# AI Financial Advisor

An intelligent portfolio management system with OCR, live market data, and AI-powered insights.

## Features
- 📊 Portfolio tracking with live prices
- 🔍 OCR for screenshot-based portfolio upload
- 🤖 AI financial advisor using RAG
- 📰 News sentiment analysis
- 📈 Market indices tracking

## Setup

1. Clone repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file with your API keys:
```env
HF_API_KEY=your_huggingface_key
NEWSAPI_KEY=your_newsapi_key
FMP_API_KEY=your_fmp_key
```

4. Run the app:
```bash
streamlit run chatbot.py
```

## API Keys Required
- Hugging Face: https://huggingface.co/settings/tokens
- NewsAPI: https://newsapi.org/register
- Financial Modeling Prep: https://site.financialmodelingprep.com/developer/docs

## Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for AWS deployment instructions.
