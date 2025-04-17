import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page Config
st.set_page_config(
    page_title="Financial Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for professional styling
st.markdown("""
    <style>
    
    /* Main background and text */
    body {
        background-color: #0a192f; /* Dark navy background */
        color: #ccd6f6; /* Light text for readability */
        font-family: 'Inter', sans-serif; /* Modern font */
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #112240; /* Darker sidebar */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }
    /* Buttons */
    .stButton>button {
        background-color: #64ffda; /* Teal accent */
        color: #0a192f; /* Dark text */
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button:hover {
        background-color: #52d1b2; /* Darker teal on hover */
        transform: translateY(-2px); /* Slight lift effect */
    }
    /* Input fields */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div,
    .stFileUploader>div>div>div {
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #233554; /* Dark border */
        background-color: #112240; /* Dark input background */
        color: #ccd6f6; /* Light text */
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>div:focus,
    .stFileUploader>div>div>div:focus {
        border-color: #64ffda; /* Teal border on focus */
        outline: none; /* Remove default outline */
    }
    /* Metrics */
    .stMetric {
        background-color: #112240; /* Dark background for metrics */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }
    .stMetric label {
        color: #64ffda; /* Teal for metric labels */
    }

    .li1{
    color:rgb(206, 13, 13); 
    }

    /* Plotly charts */
    .plotly-graph-div {
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #64ffda; /* Teal for headers */
        font-weight: 600; /* Semi-bold headers */
    }
    /* Lists */
    ul {
        padding-left: 20px; /* Indent list items */
    }
    /* Expanders */
    .stExpander {
        background-color: #112240; /* Dark background for expanders */
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }
    .stExpander .stMarkdown {
        color: #ccd6f6; /* Light text inside expanders */
    }
    /* Footer */
    .sidebar .stMarkdown {
        color: #8892b0; /* Light gray for footer text */
        font-size: 0.9em;
    }
    /* Custom animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stMarkdown, .stMetric, .stButton, .stTextInput, .stSelectbox, .stFileUploader {
        animation: fadeIn 0.6s ease-out;
    }
    
    </style>
    """, unsafe_allow_html=True)

    # Dynamic Header with Scrolling Stock Ticker
st.markdown("""
    <div style="background-color: #112240; padding: 10px; border-radius: 5px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6); overflow: hidden;">
        <marquee behavior="scroll" direction="left" style="color: #64ffda; font-weight: bold;">
            AAPL: $175.32 (+1.2%) | TSLA: $850.45 (-0.5%) | AMZN: $3,400.12 (+0.8%) | GOOGL: $2,800.50 (+0.3%) | MSFT: $299.99 (+0.7%) | META: $320.10 (-0.2%) | BRK-B: $450.00 (+0.1%) | JNJ: $165.75 (+0.4%) | V: $250.30 (+0.6%) | WMT: $145.60 (+0.9%)
        </marquee>
    </div>
    """, unsafe_allow_html=True)

# Welcome Page
def welcome_page():
    st.markdown("""<h1 style='text-align: center;'>Quant-Eye</h1>", unsafe_allow_html=True)
    <div style="background-image: url('https://engineering.jhu.edu/ams/wp-content/uploads/2021/06/pexels-pixabay-534216-1-1440x810.jpg'); 
                    background-size: cover; 
                    background-position: center; 
                    padding: 100px 20px; 
                    border-radius: 10px; 
                    text-align: center; 
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);">
        <div style="background-color: #112240; padding: 20px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);">
            <h3 style="color: #64ffda;">Our Goal</h3>
            <p>Whether you are an amateur college student or a long-term investor, we built this tool to make it simple, fast, and accessible for everyone—so you can focus on making smart decisions, not decoding data..</p>
            <h3 style="color: #64ffda;">Features</h3>
            <ul>
                <li>🔍 Company Search: Analyze financial data for any publicly traded company through their financial reports.</li>
                <li>📄 Document Upload: Upload and summarize financial documents (PDFs).</li>
                <li>📊 Top 10 Companies Analysis: Pre-analyzed data for top companies.</li>
                <li>🤖 AI-Powered Insights: Get buy/sell/hold recommendations on the basis of P.E. ratios.</li>
                <li>📈 Interactive Visualizations: Explore trends and metrics with beautiful charts.</li>
                <li>🔥 Interactive Heatmaps: Visualize metric deviations over time.</li>
                <li class="li1"> THIS IS NOT A STOCK PREDICTOR!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Predefined list of top 10 companies
Company_data = {
    "Tesla": {
        "Description": "Tesla, Inc. is an American electric vehicle and clean energy company.",
        "Industry": "Automotive",
        "Revenue": 7.1,  # in billions
        "Operating Income": 2024.0,  # in millions
        "Net Income": 2024.0,  # in millions
        "EPS": 2.0,  # Earnings Per Share
        "Other Metrics": {
            "Market Cap": "800 billion USD",
            "Employees": "70,000"
        }
    },
    "Google": {
        "Description": "Google LLC is an American multinational technology company specializing in Internet-related services and products.",
        "Industry": "Technology",
        "Revenue": 80539.0,  # in millions
        "Operating Income": 17415.0,  # in millions
        "Net Income": 15051.0,  # in millions
        "EPS": 1.17,  # Earnings Per Share
        "Other Metrics": {
            "Market Cap": "1.5 trillion USD",
            "Employees": "156,500"
        }
    },
    "Berkshire Hathaway": {
        "Description": "Berkshire Hathaway Inc. is an American multinational conglomerate holding company.",
        "Industry": "Conglomerate",
        "Revenue": 8941.0,  # in millions
        "Operating Income": 16598,  # in millions
        "Net Income": 26674,  # in millions
        "EPS": None,  # Earnings Per Share (not available)
        "Other Metrics": {
            "Market Cap": "600 billion USD",
            "Employees": "360,000"
        }
    },
    "Opendoor Technologies": {
        "Description": "Opendoor Technologies Inc. is a digital platform for residential real estate transactions.",
        "Industry": "Real Estate",
        "Revenue": 1377.0,  # in millions
        "Operating Income": 172,  # in millions
        "Net Income": None,  # (not available)
        "EPS": None,  # Earnings Per Share (not available)
        "Other Metrics": {
            "Market Cap": "8 billion USD",
            "Employees": "1,500"
        }
    }
}

TOP_10_COMPANIES = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "META": "Meta",
    "BRK-B": "Berkshire Hathaway",
    "OPEN": "Opendoor Technologies",
    "JNJ": "Johnson & Johnson",
    "V": "Visa",
    "WMT": "Walmart"
}

# Function to fetch financial data
def fetch_financial_data(ticker):
    try:
        company = yf.Ticker(ticker)
        info = company.info
        financials = company.financials
        balance_sheet = company.balance_sheet
        cash_flow = company.cashflow
        
        # Check if data is available
        if not info or financials.empty or balance_sheet.empty or cash_flow.empty:
            st.error(f"No data found for ticker: {ticker}. Please check the ticker symbol and try again.")
            return None, None, None, None
        
        return info, financials, balance_sheet, cash_flow
    except Exception as e:
        st.error(f"An error occurred while fetching data for {ticker}: {str(e)}")
        return None, None, None, None

# Function to calculate financial ratios
def calculate_ratios(financials, balance_sheet):
    ratios = {}
    try:
        # P/E Ratio
        if "Net Income" in financials.index and "Total Revenue" in financials.index:
            ratios["P/E Ratio"] = financials.loc["Net Income"].iloc[0] / financials.loc["Total Revenue"].iloc[0]
        else:
            ratios["P/E Ratio"] = "N/A"
        
        # Debt-to-Equity
        if "Total Liab" in balance_sheet.index and "Total Stockholder Equity" in balance_sheet.index:
            ratios["Debt-to-Equity"] = balance_sheet.loc["Total Liab"].iloc[0] / balance_sheet.loc["Total Stockholder Equity"].iloc[0]
        else:
            ratios["Debt-to-Equity"] = "N/A"
        
        # ROE
        if "Net Income" in financials.index and "Total Stockholder Equity" in balance_sheet.index:
            ratios["ROE"] = financials.loc["Net Income"].iloc[0] / balance_sheet.loc["Total Stockholder Equity"].iloc[0]
        else:
            ratios["ROE"] = "N/A"
    except Exception as e:
        st.error(f"Error calculating ratios: {str(e)}")
    return ratios

# Function to analyze sentiment
def analyze_sentiment(ticker):
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+news"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = [h.text for h in soup.find_all("h3")]
        sentiment_scores = [TextBlob(headline).sentiment.polarity for headline in headlines]
        avg_sentiment = np.mean(sentiment_scores)
        return avg_sentiment
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return 0.0

# Function to clean and preprocess text
def clean_text(text):
    # Remove extra spaces, special characters, and numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s.,]', '', text)  # Remove special characters except periods and commas
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.strip()

# Function to extract key phrases using TextBlob
def extract_key_phrases(text, num_phrases=5):
    blob = TextBlob(text)
    phrases = blob.noun_phrases  # Extract noun phrases
    return list(set(phrases))[:num_phrases]  # Return unique phrases

# Function to summarize text into bullet points
def summarize_text(text, num_sentences=5):
    try:
        # Clean the text
        text = clean_text(text)
        
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        
        # Remove stopwords and tokenize words
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Calculate TF-IDF scores for sentences
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        word_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Rank sentences based on their scores
        ranked_sentences = [sentence for _, sentence in sorted(zip(word_scores, sentences), reverse=True)]
        
        # Select the top N sentences as the summary
        summary = " ".join(ranked_sentences[:num_sentences])
        
        # Extract key phrases
        key_phrases = extract_key_phrases(summary)
        
        # Format summary into bullet points
        bullet_points = "\n".join([f"• {sentence}" for sentence in ranked_sentences[:num_sentences]])
        
        return bullet_points, key_phrases
    except Exception as e:
        st.error(f"Error summarizing text: {str(e)}")
        return text[:500], []  # Fallback to the first 500 characters

# Function to extract and summarize PDF data
def extract_and_summarize_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        
        # Summarize the text into bullet points and extract key phrases
        bullet_points, key_phrases = summarize_text(text)
        return text, bullet_points, key_phrases
    except Exception as e:
        st.error(f"Error extracting PDF data: {str(e)}")
        return "", "", []

# Function to generate AI recommendations
def generate_recommendation(ratios, sentiment):
    if ratios["P/E Ratio"] != "N/A" and ratios["P/E Ratio"] < 15 and sentiment > 0.2:
        return "Buy"
    elif ratios["P/E Ratio"] != "N/A" and (ratios["P/E Ratio"] > 25 or sentiment < -0.2):
        return "Sell"
    else:
        return "Hold"

# Function to display company profile from Company_data
def display_company_profile(company_name):
    if company_name in Company_data:
        company = Company_data[company_name]
        st.subheader(f"📊 Financial Data for {company_name}")
        st.markdown(f"**Description:** {company['Description']}")
        st.markdown(f"**Industry:** {company['Industry']}")
        
        # Display Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Revenue", f"${company['Revenue']} billion" if company['Revenue'] else "N/A")
        with col2:
            st.metric("Operating Income", f"${company['Operating Income']} million" if company['Operating Income'] else "N/A")
        with col3:
            st.metric("Net Income", f"${company['Net Income']} million" if company['Net Income'] else "N/A")
        
        # Display Other Metrics
        st.subheader("📈 Other Metrics")
        for key, value in company['Other Metrics'].items():
            st.markdown(f"**{key}:** {value}")
    else:
        st.error(f"Company '{company_name}' not found in the dataset.")


# Function to create financial visualizations
def create_visualizations(financials, balance_sheet):
    # Revenue Trend
    if "Total Revenue" in financials.index:
        revenue_data = financials.loc["Total Revenue"].reset_index()
        revenue_data.columns = ["Year", "Revenue"]
        fig_revenue = px.line(revenue_data, x="Year", y="Revenue", title="Revenue Trend", template="plotly_dark")
        fig_revenue.update_traces(line=dict(color="#2563eb", width=3))
    else:
        fig_revenue = None
    
    # Debt-to-Equity Trend
    if "Total Liab" in balance_sheet.index and "Total Stockholder Equity" in balance_sheet.index:
        debt_equity_data = balance_sheet.loc[["Total Liab", "Total Stockholder Equity"]].T.reset_index()
        debt_equity_data.columns = ["Year", "Total Liab", "Total Stockholder Equity"]
        fig_debt_equity = px.bar(debt_equity_data, x="Year", y=["Total Liab", "Total Stockholder Equity"], 
                                 title="Debt vs Equity", template="plotly_dark", barmode="group")
        fig_debt_equity.update_traces(marker_color=["#6c757d", "#2563eb"])
    else:
        fig_debt_equity = None
    
    return fig_revenue, fig_debt_equity

# Function to calculate metric deviations
def calculate_metric_deviations(financials):
    deviations = {}
    if "Total Revenue" in financials.index:
        revenue = financials.loc["Total Revenue"]
        revenue_deviations = revenue.pct_change().dropna()
        deviations["Revenue"] = revenue_deviations
    if "Net Income" in financials.index:
        net_income = financials.loc["Net Income"]
        net_income_deviations = net_income.pct_change().dropna()
        deviations["Net Income"] = net_income_deviations
    return deviations

# Function to create heatmap
def create_heatmap(deviations):
    if not deviations:
        return None
    
    # Create a DataFrame for the heatmap
    heatmap_data = pd.DataFrame(deviations)
    heatmap_data.index = heatmap_data.index.astype(str)  # Convert index to string for better visualization
    
    # Create the heatmap using go.Figure
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        colorbar=dict(title='Deviation (%)')
    ))
    
    # Update layout for the figure
    fig.update_layout(
        title="Metric Deviations Over Time",
        xaxis_title="Metrics",
        yaxis_title="Year",
        template="plotly_dark"
    )
    
    return fig

# Sidebar for Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose an Option", [
    "Welcome", "Company Search", "Document Upload", "Top 10 Companies Analysis"
])

# Welcome Page
if options == "Welcome":
    welcome_page()

# Company Search
elif options == "Company Search":
    st.header("🔍 Company Search")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):")
    if ticker:
        with st.spinner("Fetching data..."):
            info, financials, balance_sheet, cash_flow = fetch_financial_data(ticker)
        
        if info and not financials.empty and not balance_sheet.empty and not cash_flow.empty:
            st.subheader(f"📊 Financial Data for {info.get('longName', ticker)}")
            
            # Display Key Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
            with col2:
                st.metric("Revenue", f"${info.get('totalRevenue', 'N/A'):,}")
            with col3:
                st.metric("Net Income", f"${financials.loc['Net Income'].iloc[0]:,}" if "Net Income" in financials.index else "N/A")
            
            # Calculate Ratios
            ratios = calculate_ratios(financials, balance_sheet)
            st.subheader("📈 Financial Ratios")
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("P/E Ratio", f"{ratios.get('P/E Ratio', 'N/A'):.2f}" if isinstance(ratios.get('P/E Ratio'), (int, float)) else "N/A")
            with col5:
                st.metric("Debt-to-Equity", f"{ratios.get('Debt-to-Equity', 'N/A'):.2f}" if isinstance(ratios.get('Debt-to-Equity'), (int, float)) else "N/A")
            with col6:
                st.metric("ROE", f"{ratios.get('ROE', 'N/A'):.2f}" if isinstance(ratios.get('ROE'), (int, float)) else "N/A")
            
            # AI Recommendation
            sentiment = analyze_sentiment(ticker)
            recommendation = generate_recommendation(ratios, sentiment)
            st.subheader("🤖 AI Recommendation")
            st.success(f"Recommendation: {recommendation}")
            
            # Visualizations
            st.subheader("📊 Visualizations")
            fig_revenue, fig_debt_equity = create_visualizations(financials, balance_sheet)
            if fig_revenue:
                st.plotly_chart(fig_revenue, use_container_width=True)
            if fig_debt_equity:
                st.plotly_chart(fig_debt_equity, use_container_width=True)
            
            # Heatmap for Metric Deviations
            st.subheader(" Metric Deviations Over Time")
            deviations = calculate_metric_deviations(financials)
            heatmap_fig = create_heatmap(deviations)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)

# Document Upload
elif options == "Document Upload":
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Extracting and summarizing document..."):
            text, bullet_points, key_phrases = extract_and_summarize_pdf(uploaded_file)
        
        # Display Extracted Text in an Expandable Section
        with st.expander(" View Extracted Text"):
            st.text_area("Extracted Text", text, height=300)
        
        # Display Summary in an Expandable Section
        with st.expander(" View Summary (Bullet Points)"):
            st.markdown(bullet_points)
        
        # Display Key Phrases in an Expandable Section
        with st.expander("View Key Phrases"):
            st.markdown(", ".join([f"**{phrase}**" for phrase in key_phrases]))

# Top 10 Companies Analysis
# Update the Top 10 Companies Analysis section
elif options == "Top 10 Companies Analysis":
    st.header("📊 Top 10 Companies Analysis")
    
    selected_company = st.selectbox("Select a Company", list(TOP_10_COMPANIES.keys()), format_func=lambda x: f"{x} - {TOP_10_COMPANIES[x]}")
    
    if selected_company:
        company_name = TOP_10_COMPANIES[selected_company]
        
        # Display data from Company_data if available
        if company_name in Company_data:
            display_company_profile(company_name)
        
        # Fetch data from Yahoo Finance
        with st.spinner(f"Fetching data for {company_name}..."):
            info, financials, balance_sheet, cash_flow = fetch_financial_data(selected_company)
        
        if info and not financials.empty and not balance_sheet.empty and not cash_flow.empty:
            st.subheader(f"📊 Financial Data from Yahoo Finance for {info.get('longName', selected_company)}")
            
            # Display Key Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
            with col2:
                st.metric("Revenue", f"${info.get('totalRevenue', 'N/A'):,}")
            with col3:
                st.metric("Net Income", f"${financials.loc['Net Income'].iloc[0]:,}" if "Net Income" in financials.index else "N/A")
            
            # Calculate Ratios
            ratios = calculate_ratios(financials, balance_sheet)
            st.subheader(" Financial Ratios")
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("P/E Ratio", f"{ratios.get('P/E Ratio', 'N/A'):.2f}" if isinstance(ratios.get('P/E Ratio'), (int, float)) else "N/A")
            with col5:
                st.metric("Debt-to-Equity", f"{ratios.get('Debt-to-Equity', 'N/A'):.2f}" if isinstance(ratios.get('Debt-to-Equity'), (int, float)) else "N/A")
            with col6:
                st.metric("ROE", f"{ratios.get('ROE', 'N/A'):.2f}" if isinstance(ratios.get('ROE'), (int, float)) else "N/A")
            
            # AI Recommendation
            sentiment = analyze_sentiment(selected_company)
            recommendation = generate_recommendation(ratios, sentiment)
            st.subheader(" AI Recommendation")
            st.success(f"Recommendation: {recommendation}")
            
            # Visualizations
            st.subheader(" Visualizations")
            fig_revenue, fig_debt_equity = create_visualizations(financials, balance_sheet)
            if fig_revenue:
                st.plotly_chart(fig_revenue, use_container_width=True)
            if fig_debt_equity:
                st.plotly_chart(fig_debt_equity, use_container_width=True)
            
            # Heatmap for Metric Deviations
            st.subheader(" Metric Deviations Over Time")
            deviations = calculate_metric_deviations(financials)
            heatmap_fig = create_heatmap(deviations)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Adwaita And Gayatri AKA Team BeautifulSoup")
st.sidebar.markdown("Version 1.0")
