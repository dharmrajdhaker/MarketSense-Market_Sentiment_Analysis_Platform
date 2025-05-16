import logging
import pandas as pd
from transformers import pipeline
from utils.config import SENTIMENT_CATEGORIES, PROCESSED_DATA_DIR, NSE500_COMPANIES
import os
import re


logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        try:
            # Initialize sentiment analysis pipeline with FinBERT model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,  # Use CPU
                max_length=128,
                truncation=True
            )
            logger.info("FinBERT sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            raise

    def _extract_company_name(self, text):
        """Extract company name from text, ensuring it's in NSE500_COMPANIES list"""
        if not isinstance(text, str) or not text.strip():
            return "Unknown"
            
        text_lower = text.lower()
        
        # First try direct matches with NSE500 companies
        for company in NSE500_COMPANIES:
            company_lower = company.lower()
            # Use word boundaries to ensure we match complete words only
            pattern = r'(?<![a-zA-Z])' + re.escape(company_lower) + r'(?![a-zA-Z])'
            if re.search(pattern, text_lower):
                logger.debug(f"Found direct match for NSE500 company: {company}")
                return company
            
            # Check for company name with common suffixes, ensuring word boundaries
            suffix_pattern = r'(?<![a-zA-Z])' + re.escape(company_lower) + r'(?:\s+(?:limited|ltd|corporation|corp|inc|llc|pvt|private|public|company|co\.))?(?![a-zA-Z])'
            if re.search(suffix_pattern, text_lower):
                logger.debug(f"Found direct match with suffix for NSE500 company: {company}")
                return company
        
        # If no direct match, try to find company names using patterns
        patterns = [
            # Standard company formats with word boundaries
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+(?:Limited|Ltd|Corporation|Corp|Inc|LLC|Pvt|Private|Public|Company|Co\.))(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+(?:Bank|Financial|Insurance|Capital|Securities|Investments))(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+(?:Energy|Power|Electric|Infrastructure|Realty|Properties))(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+(?:Pharma|Healthcare|Biotech|Medical|Life Sciences))(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+(?:Tech|Technology|Software|Digital|Solutions|Systems))(?![a-zA-Z])',
            
            # IPO specific patterns with word boundaries
            r'(?<![a-zA-Z])IPO\s+of\s+([A-Z][a-zA-Z\s]+(?:Limited|Ltd|Corporation|Corp|Inc|LLC|Pvt|Private|Public|Company|Co\.))(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+IPO(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+to\s+launch\s+IPO(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+files\s+for\s+IPO(?![a-zA-Z])',
            
            # News specific patterns with word boundaries
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+reports(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+announces(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+declares(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+Q[1-4](?![a-zA-Z])',
            
            # Common company name patterns with word boundaries
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+Group(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+Holdings(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+Industries(?![a-zA-Z])',
            r'(?<![a-zA-Z])([A-Z][a-zA-Z\s]+)\s+Enterprises(?![a-zA-Z])'
        ]
        
        # Try to find company names using patterns and validate against NSE500
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                company = matches[0].strip()
                # Clean up the company name
                company = re.sub(r'\s+', ' ', company)  # Remove extra spaces
                company = company.strip()
                
                if len(company) > 2:  # Ensure it's not too short
                    # Validate against NSE500 companies with word boundaries
                    company_lower = company.lower()
                    for nse_company in NSE500_COMPANIES:
                        nse_company_lower = nse_company.lower()
                        # Use word boundaries for validation
                        pattern = r'(?<![a-zA-Z])' + re.escape(company_lower) + r'(?![a-zA-Z])'
                        if re.search(pattern, nse_company_lower):
                            logger.debug(f"Found pattern match for NSE500 company: {nse_company}")
                            return nse_company
        
        logger.debug("No NSE500 company match found")
        return "Unknown"

    def _generate_signal(self, sentiment, confidence):
        """Generate trading signal based on sentiment and confidence"""
        # FinBERT uses 'positive', 'negative', 'neutral' labels
        if sentiment == 'positive' and confidence > 0.7:
            return 'BUY'
        elif sentiment == 'negative' and confidence > 0.7:
            return 'SELL'
        else:
            return 'HOLD'

    def _summarize_reason(self, text):
        """Generate a short summary of the text"""
        # Remove URLs and special characters
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Take first 100 characters and add ellipsis if longer
        if len(text) > 100:
            return text[:100] + "..."
        return text

    def _analyze_text(self, text):
        try:
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'company_name': 'Unknown',
                    'reason': 'No text content',
                    'signal': 'HOLD'
                }
            
            # Clean and truncate text before analysis
            text = text.strip()
            if len(text) > 500:
                text = text[:500] + "..."
                
            result = self.sentiment_pipeline(text)[0]
            company_name = self._extract_company_name(text)
            reason = self._summarize_reason(text)
            
            # Map FinBERT labels to our format
            sentiment_map = {
                'positive': 'POS',
                'negative': 'NEG',
                'neutral': 'NEU'
            }
            sentiment = sentiment_map.get(result['label'], 'NEU')
            
            signal = self._generate_signal(result['label'], result['score'])
            
            return {
                'sentiment': sentiment,
                'confidence': result['score'],
                'company_name': company_name,
                'reason': reason,
                'signal': signal
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {
                'sentiment': 'NEU',
                'confidence': 0.0,
                'company_name': 'Unknown',
                'reason': 'Error in analysis',
                'signal': 'HOLD'
            }

    def analyze_sentiment_df(self, data):
        logger.info("Starting sentiment analysis...")
        
        # Convert data to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Ensure required columns exist
        if 'text' not in df.columns:
            df['text'] = ''
        if 'date' not in df.columns:
            df['date'] = pd.NaT
            
        df['text'] = df['text'].fillna('')
        
        # Clean and truncate text before analysis
        df['text'] = df['text'].apply(lambda x: x.strip()[:500] + "..." if len(x.strip()) > 500 else x.strip())
        
        # Analyze sentiment for each text
        df['analysis'] = df['text'].apply(self._analyze_text)
        
        # Extract all fields from analysis
        df['sentiment'] = df['analysis'].apply(lambda x: x['sentiment'])
        df['confidence'] = df['analysis'].apply(lambda x: x['confidence'])
        df['company_name'] = df['analysis'].apply(lambda x: x['company_name'])
        df['reason'] = df['analysis'].apply(lambda x: x['reason'])
        df['signal'] = df['analysis'].apply(lambda x: x['signal'])
        
        # Drop the intermediate column
        df = df.drop('analysis', axis=1)
        
        # Select and reorder columns
        df = df[['date', 'company_name', 'sentiment', 'reason', 'signal', 'confidence', 'text']]
        
        # Save processed data
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        output_file = f"{PROCESSED_DATA_DIR}/sentiment_analysis_results.csv"
        df.to_csv(output_file, index=False)
        
        # Generate sentiment statistics
        sentiment_stats = df['sentiment'].value_counts().to_dict()
        logger.info(f"Sentiment distribution: {sentiment_stats}")
        
        # Log signal distribution
        signal_stats = df['signal'].value_counts().to_dict()
        logger.info(f"Signal distribution: {signal_stats}")
        
        logger.info(f"Analysis results saved to: {output_file}")
        
        return df
