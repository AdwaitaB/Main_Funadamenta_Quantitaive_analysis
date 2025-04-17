import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
import requests
import numpy as np
import pandas as pd
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import pytz
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'dashboard'

# ---------------- Constants & Config -------------------
COMPANY_DATA = {
    "Tesla": {
        "Description": "Tesla, Inc. is an American electric vehicle and clean energy company.",
        "Industry": "Automotive",
        "Revenue": 71.0,  # in billions
        "Operating Income": 8.9,  # in billions
        "Net Income": 5.5,  # in billions
        "EPS": 2.0,
        "Other Metrics": {
            "Market Cap": "800 billion USD",
            "Employees": "70,000",
            "CEO": "Elon Musk",
            "Founded": "2003"
        }
    },
    # ... (other company data)
}

TOP_COMPANIES = {
    "AAPL": {"name": "Apple", "sector": "Technology"},
    "TSLA": {"name": "Tesla", "sector": "Automotive"},
    "AMZN": {"name": "Amazon", "sector": "E-commerce"},
    "GOOGL": {"name": "Google", "sector": "Technology"},
    "MSFT": {"name": "Microsoft", "sector": "Technology"},
    "META": {"name": "Meta", "sector": "Technology"},
    "BRK-B": {"name": "Berkshire Hathaway", "sector": "Conglomerate"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
    "V": {"name": "Visa", "sector": "Financial Services"},
    "WMT": {"name": "Walmart", "sector": "Retail"}
}

# ---------------- Page Config -------------------
st.set_page_config(
    page_title="Quant-Eye Pro | Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom Styles -------------------
st.markdown("""
    <style>
    :root {
        --primary: #2563eb;
        --secondary: #1e40af;
        --accent: #3b82f6;
        --background: #f8fafc;
        --card: #ffffff;
        --text: #1e293b;
        --text-light: #64748b;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--background);
        color: var(--text);
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1.5rem;
        border-radius: 0 0 1rem 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .ticker-tape {
        background: linear-gradient(90deg, #e0f2fe, #dbeafe);
        color: #1e3a8a;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        overflow-x: auto;
        white-space: nowrap;
    }
    
    .ticker-item {
        display: inline-block;
        margin-right: 2rem;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .positive {
        color: var(--success);
    }
    
    .negative {
        color: var(--danger);
    }
    
    .card {
        background: var(--card);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .nav-item {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.25rem;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .nav-item:hover {
        background: rgba(37, 99, 235, 0.1);
    }
    
    .nav-item.active {
        background: rgba(37, 99, 235, 0.2);
        font-weight: 600;
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--text-light);
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Helper Functions -------------------
def get_current_time():
    return datetime.now(pytz.timezone('America/New_York')).strftime("%Y-%m-%d %H:%M:%S")

@st.cache_data(ttl=300)
def get_stock_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            data[ticker] = {
                "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "change": info.get("regularMarketChange", 0),
                "change_pct": info.get("regularMarketChangePercent", 0),
                "volume": info.get("regularMarketVolume", 0),
                "market_cap": info.get("marketCap", 0)
            }
        except:
            continue
    return data

def format_currency(value):
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    return f"${value:,.2f}"

def create_ticker_tape(stock_data):
    tape_html = '<div class="ticker-tape">'
    for ticker, data in stock_data.items():
        change_class = "positive" if data["change"] >= 0 else "negative"
        tape_html += f"""
        <div class="ticker-item">
            <strong>{ticker}</strong>: {data['price']:.2f} 
            <span class="{change_class}">{data['change']:+.2f} ({data['change_pct']:+.2f}%)</span>
        </div>
        """
    tape_html += '</div>'
    return tape_html

# Document Analysis Functions
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF using pdfplumber"""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text if text.strip() else None
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def analyze_sentiment(text):
    """Perform sentiment analysis using both TextBlob and VADER"""
    analysis = {
        "textblob": TextBlob(text).sentiment,
        "vader": SentimentIntensityAnalyzer().polarity_scores(text)
    }
    return analysis

def extract_financial_metrics(text):
    """Use regex patterns to find financial metrics"""
    metrics = {
        "revenue": None,
        "net_income": None,
        "eps": None,
        "gross_profit": None,
        "ebitda": None,
        "free_cash_flow": None
    }
    
    # Improved regex patterns for financial metrics
    patterns = {
        "revenue": r'(?:Revenue|Sales|Income)\D*(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?[BM]?)',
        "net_income": r'(?:Net\sIncome|Profit|Net\sEarnings)\D*(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?[BM]?)',
        "eps": r'(?:EPS|Earnings\sPer\sShare)\D*(\$?\d+\.\d+)',
        "gross_profit": r'(?:Gross\sProfit)\D*(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?[BM]?)',
        "ebitda": r'(?:EBITDA)\D*(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?[BM]?)',
        "free_cash_flow": r'(?:Free\sCash\sFlow|FCF)\D*(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?[BM]?)'
    }
    
    for metric, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            metrics[metric] = matches[-1]  # Get most recent mention
    
    return metrics

def analyze_key_terms(text, n=10):
    """Extract key terms using TF-IDF"""
    stop_words = set(stopwords.words('english'))
    custom_stopwords = ['company', 'financial', 'report', 'year', 'fiscal', 'quarter']
    stop_words.update(custom_stopwords)
    
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    
    return feature_array[tfidf_sorting][:n]

# ---------------- Header -------------------
st.markdown(f"""
    <div class="header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 2rem;">Quant-Eye Pro</h1>
                <p style="margin: 0; opacity: 0.9;">AI-Powered Financial Intelligence Platform</p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 0.9rem;">{get_current_time()} ET</p>
                <p style="margin: 0; font-size: 0.9rem;">Market Open</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------- Sidebar Navigation -------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2721/2721295.png", width=60)
    st.markdown("<h3 style='margin-top: -10px;'>Quant-Eye Pro</h3>", unsafe_allow_html=True)
    
    # Navigation menu
    st.markdown("### Navigation")
    
    if st.button("📊 Dashboard", key="dashboard_btn", use_container_width=True):
        st.session_state.current_page = 'dashboard'
    
    if st.button("📈 Market Analysis", key="market_btn", use_container_width=True):
        st.session_state.current_page = 'market'
    
    if st.button("💰 Portfolio", key="portfolio_btn", use_container_width=True):
        st.session_state.current_page = 'portfolio'
    
    if st.button("🔍 Stock Screener", key="screener_btn", use_container_width=True):
        st.session_state.current_page = 'screener'
    
    if st.button("📄 Documents", key="documents_btn", use_container_width=True):
        st.session_state.current_page = 'documents'
    
    st.markdown("---")
    
    # Watchlist
    st.markdown("### 📌 My Watchlist")
    watchlist = st.multiselect(
        "Select stocks to watch",
        list(TOP_COMPANIES.keys()),
        ["AAPL", "TSLA", "GOOGL"],
        format_func=lambda x: f"{x} - {TOP_COMPANIES[x]['name']}",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.8rem; color: #64748b;">
            <p>Quant-Eye Pro v2.0</p>
            <p>© 2023 Quant-Eye Technologies</p>
        </div>
    """, unsafe_allow_html=True)

# ---------------- Page Content -------------------
if st.session_state.current_page == 'dashboard':
    st.markdown("<h2 style='margin-bottom: 1.5rem;'>📊 Dashboard Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="card">
                <h3 style="margin-top: 0; color: #000000; font-size: 0.9rem;">S&P 500</h3>
                <h2 style="margin: 0;">4,450.38</h2>
                <p style="margin: 0; color: #000000;">+1.25%</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="card">
                <h3 style="margin-top: 0; color: #000000; font-size: 0.9rem;">NASDAQ</h3>
                <h2 style="margin: 0;">13,678.42</h2>
                <p style="margin: 0; color:#000000;">+1.75%</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="card">
                <h3 style="margin-top: 0; color: #000000; font-size: 0.9rem;">DOW</h3>
                <h2 style="margin: 0;">34,500.67</h2>
                <p style="margin: 0; color: #000000;">-0.25%</p>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="card">
                <h3 style="margin-top: 0; color: #000000b; font-size: 0.9rem;">VIX</h3>
                <h2 style="margin: 0;">18.25</h2>
                <p style="margin: 0; color: #000000;">+2.15%</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Portfolio Performance Chart
    st.markdown("<h3 style='margin-top: 1.5rem;'>📈 Portfolio Performance</h3>", unsafe_allow_html=True)
    portfolio_data = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=90, freq="D"),
        "Value": np.cumsum(np.random.normal(1000, 200, 90))
    })
    fig = px.line(portfolio_data, x="Date", y="Value", 
                 title="Your Portfolio Value Over Time",
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.current_page == 'market':
    st.markdown("<h2 style='margin-bottom: 1.5rem;'>📈 Market Analysis</h2>", unsafe_allow_html=True)
    
    selected_ticker = st.selectbox(
        "Select a stock",
        list(TOP_COMPANIES.keys()),
        format_func=lambda x: f"{x} - {TOP_COMPANIES[x]['name']}",
        key="market_stock_select"
    )
    
    if selected_ticker:
        try:
            stock = yf.Ticker(selected_ticker)
            hist = stock.history(period="1mo")
            
            if not hist.empty:
                st.markdown(f"<h3>{TOP_COMPANIES[selected_ticker]['name']} ({selected_ticker})</h3>", unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name="Price"
                ))
                fig.update_layout(
                    title=f"{selected_ticker} Stock Price (1 Month)",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<h3 style='margin-top: 1.5rem;'>📊 Key Metrics</h3>", unsafe_allow_html=True)
                info = stock.info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Market Cap", format_currency(info.get('marketCap', 0)))
                with col2:
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                with col3:
                    st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
                with col4:
                    st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")

                # Additional Financial Metrics
                st.markdown("<h3 style='margin-top: 1.5rem;'>📉 Advanced Metrics</h3>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("EBITDA", format_currency(info.get('ebitda', 0)))
                with col2:
                    st.metric("ROE", f"{info.get('returnOnEquity', 'N/A')}%")
                with col3:
                    st.metric("Debt/Equity", info.get('debtToEquity', 'N/A'))
                with col4:
                    st.metric("Beta", info.get('beta', 'N/A'))

                # News Sentiment Analysis
                st.markdown("<h3 style='margin-top: 1.5rem;'>📰 News Sentiment</h3>", unsafe_allow_html=True)
                try:
                    news = stock.news
                    if news:
                        sentiments = []
                        for article in news[:5]:  # Analyze top 5 articles
                            analysis = SentimentIntensityAnalyzer().polarity_scores(article['title'])
                            sentiments.append(analysis['compound'])
                        
                        avg_sentiment = np.mean(sentiments)
                        sentiment_color = "#10b981" if avg_sentiment >= 0 else "#ef4444"
                        
                        st.markdown(f"""
                            <div class="card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <h3 style="margin: 0;">Average News Sentiment</h3>
                                        <p style="margin: 0; color: {sentiment_color}; font-size: 2rem; font-weight: bold;">
                                            {avg_sentiment:.2f}
                                        </p>
                                        <small>Range: -1 (Negative) to 1 (Positive)</small>
                                    </div>
                                    <div style="text-align: right;">
                                        <p style="margin: 0;">Latest Headline:</p>
                                        <p style="margin: 0; font-size: 0.9rem;">{news[0]['title']}</p>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not fetch news: {str(e)}")

        except Exception as e:
            st.error(f"Error loading data for {selected_ticker}: {str(e)}")

elif st.session_state.current_page == 'portfolio':
    st.markdown("<h2 style='margin-bottom: 1.5rem;'>💰 Portfolio Manager</h2>", unsafe_allow_html=True)
    
    # Portfolio Allocation
    st.markdown("<h3>📌 Your Holdings</h3>", unsafe_allow_html=True)
    allocation_data = pd.DataFrame({
        "Stock": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "Cash"],
        "Value": [45000, 38000, 28000, 22000, 15000, 20000],
        "Allocation %": [30, 25, 18, 15, 10, 12]
    })
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(
            allocation_data.style.format({
                "Value": "${:,.0f}",
                "Allocation %": "{:.0f}%"
            }),
            height=400,
            use_container_width=True
        )
    
    with col2:
        fig = px.pie(
            allocation_data,
            values="Value",
            names="Stock",
            title="Portfolio Allocation",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    # Performance Comparison
    st.markdown("<h3 style='margin-top: 1.5rem;'>📊 Benchmark Comparison</h3>", unsafe_allow_html=True)
    comparison_data = pd.DataFrame({
        "Metric": ["YTD Return", "1Y Return", "3Y CAGR", "Volatility", "Sharpe Ratio"],
        "Your Portfolio": [8.2, 12.5, 9.8, 15.4, 1.2],
        "S&P 500": [7.1, 10.3, 8.9, 12.1, 0.9]
    })
    st.dataframe(
        comparison_data.style.format({
            "Your Portfolio": "{:.1f}%",
            "S&P 500": "{:.1f}%"
        }).highlight_between(
            left=0.1, right=100,
            subset=["Your Portfolio", "S&P 500"],
            props="background-color: #e6f4ea;"
        ),
        use_container_width=True
    )

elif st.session_state.current_page == 'screener':
    st.markdown("<h2 style='margin-bottom: 1.5rem;'>🔍 Stock Screener</h2>", unsafe_allow_html=True)
    
    # Screener Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        market_cap = st.selectbox(
            "Market Cap",
            ["Any", "Mega (>$100B)", "Large ($10B-$100B)", "Mid ($2B-$10B)", "Small (<$2B)"]
        )
    with col2:
        pe_ratio = st.slider("P/E Ratio", 0.0, 50.0, (5.0, 25.0))
    with col3:
        sector = st.selectbox(
            "Sector",
            ["All"] + list(set([c["sector"] for c in TOP_COMPANIES.values()]))
        )
    
    if st.button("Run Screener", type="primary"):
        # Simulate screening results
        screened_stocks = [
            {"Ticker": "AAPL", "Name": "Apple", "Price": 185.32, "Change %": 1.25, "P/E": 28.5, "Market Cap": "2.8T", "Sector": "Technology"},
            {"Ticker": "MSFT", "Name": "Microsoft", "Price": 328.39, "Change %": 0.85, "P/E": 32.1, "Market Cap": "2.4T", "Sector": "Technology"},
            {"Ticker": "JNJ", "Name": "Johnson & Johnson", "Price": 152.76, "Change %": -0.42, "P/E": 15.2, "Market Cap": "398B", "Sector": "Healthcare"},
            {"Ticker": "JPM", "Name": "JPMorgan Chase", "Price": 142.18, "Change %": 1.12, "P/E": 10.8, "Market Cap": "412B", "Sector": "Financial"},
            {"Ticker": "WMT", "Name": "Walmart", "Price": 158.92, "Change %": 0.35, "P/E": 24.7, "Market Cap": "428B", "Sector": "Retail"}
        ]
        
        # Display results
        st.dataframe(
            pd.DataFrame(screened_stocks).style.format({
                "Price": "${:.2f}",
                "Change %": "{:.2f}%",
                "P/E": "{:.1f}"
            }).applymap(lambda x: "color: #10b981" if isinstance(x, (int, float)) and x > 0 else "color: #ef4444", 
                      subset=["Change %"]),
            height=500,
            use_container_width=True
        )

elif st.session_state.current_page == 'documents':
    st.markdown("<h2 style='margin-bottom: 1.5rem;'>📄 Document Analysis</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload financial documents (PDF, Excel, CSV)",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        with st.spinner("Analyzing document..."):
            if uploaded_file.type == "application/pdf":
                # Process PDF Document
                text = extract_text_from_pdf(uploaded_file)
                
                if text:
                    # Display extracted text preview
                    with st.expander("View Extracted Text"):
                        st.text(text[:2000] + "..." if len(text) > 2000 else text)
                    
                    # Sentiment Analysis
                    sentiment = analyze_sentiment(text)
                    
                    # Financial Metrics Extraction
                    metrics = extract_financial_metrics(text)
                    
                    # Key Terms Analysis
                    key_terms = analyze_key_terms(text)
                    
                    # Display Results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Sentiment Cards
                        st.markdown("### 📊 Sentiment Analysis")
                        tb_polarity = sentiment['textblob'].polarity
                        vd_compound = sentiment['vader']['compound']
                        
                        # TextBlob Polarity
                        st.markdown(f"""
                            <div class="card" style="margin-bottom: 1rem;">
                                <h3 style="margin-top: 0; font-size: 1rem;">TextBlob Polarity</h3>
                                <div style="font-size: 1.5rem; color: {'#10b981' if tb_polarity >=0 else '#ef4444'}">
                                    {tb_polarity:.2f}
                                </div>
                                <small>Range: -1 (Negative) to 1 (Positive)</small>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # VADER Compound Score
                        st.markdown(f"""
                            <div class="card">
                                <h3 style="margin-top: 0; font-size: 1rem;">VADER Compound Score</h3>
                                <div style="font-size: 1.5rem; color: {'#10b981' if vd_compound >=0 else '#ef4444'}">
                                    {vd_compound:.2f}
                                </div>
                                <small>Range: -1 (Negative) to 1 (Positive)</small>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Financial Metrics
                        st.markdown("### 💰 Extracted Financial Metrics")
                        metrics_html = """
                            <div class="card">
                                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                        """
                        for metric, value in metrics.items():
                            if value:
                                metrics_html += f"""
                                    <div>
                                        <span style="color: #64748b; font-size: 0.9rem;">{metric.title().replace('_', ' ')}</span>
                                        <div style="font-size: 1.2rem; font-weight: 600;">{value}</div>
                                    </div>
                                """
                        metrics_html += "</div></div>"
                        st.markdown(metrics_html, unsafe_allow_html=True)
                        
                        # Key Terms Visualization
                        st.markdown("### 🔑 Key Terms")
                        if key_terms.any():
                            terms_df = pd.DataFrame({
                                "Term": key_terms,
                                "Importance": np.linspace(1, 0.5, len(key_terms))
                            })
                            fig = px.bar(terms_df, x="Importance", y="Term", orientation='h',
                                        color="Importance", color_continuous_scale='Blues')
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error("Failed to extract text from PDF. The document might be scanned or image-based.")
            
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
                # Process Spreadsheet Data
                try:
                    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                        df = pd.read_excel(uploaded_file)
                    else:
                        df = pd.read_csv(uploaded_file)
                    
                    st.markdown("### 📈 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.markdown("### 📊 Data Summary")
                    st.write(df.describe())
                    
                    # Auto-detect potential metrics
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        selected_col = st.selectbox("Select column to visualize", numeric_cols)
                        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing spreadsheet: {str(e)}")
            
            st.success("Analysis complete!")

# ---------------- Footer -------------------
st.markdown("""
    <div class="footer">
        <p>© 2023 Quant-Eye Technologies | All rights reserved</p>
    </div>
""", unsafe_allow_html=True)
