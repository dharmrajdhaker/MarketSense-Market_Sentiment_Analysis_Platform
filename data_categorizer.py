import logging
import pandas as pd
from utils.config import NSE500_COMPANIES, DATA_CATEGORIES, PROCESSED_DATA_DIR
import os
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataCategorizer:
    def __init__(self):
        logger.info("Initializing DataCategorizer...")
        logger.info(f"NSE500_COMPANIES type: {type(NSE500_COMPANIES)}")
        logger.info(f"NSE500_COMPANIES length: {len(NSE500_COMPANIES)}")
        if len(NSE500_COMPANIES) > 0:
            logger.info(f"First company entry type: {type(NSE500_COMPANIES[0])}")
            logger.info(f"First company entry: {NSE500_COMPANIES[0]}")
            logger.info(f"First company entry keys: {NSE500_COMPANIES[0].keys() if isinstance(NSE500_COMPANIES[0], dict) else 'Not a dict'}")
        
        self.companies = NSE500_COMPANIES
        logger.info(f"self.companies type after assignment: {type(self.companies)}")
        logger.info(f"self.companies length after assignment: {len(self.companies)}")
        if len(self.companies) > 0:
            logger.info(f"First self.companies entry type: {type(self.companies[0])}")
            logger.info(f"First self.companies entry: {self.companies[0]}")
            logger.info(f"First self.companies entry keys: {self.companies[0].keys() if isinstance(self.companies[0], dict) else 'Not a dict'}")
        
        self.categories = DATA_CATEGORIES
        self.company_timeline = defaultdict(list)
        
    def _identify_companies(self, text):
        """Identify NSE 500 companies mentioned in the text"""
        if not isinstance(text, str):
            return []
            
        mentioned_companies = []
        text_lower = text.lower()
        
        # Debug logging
        logger.debug(f"Processing text: {text[:100]}...")  # Log first 100 chars of text
        logger.debug(f"Number of companies to check: {len(self.companies)}")
        
        for company in self.companies:
            try:
                # Convert company name to lowercase for case-insensitive matching
                company_lower = company.lower()
                
                # Check if the company name is in the text
                if company_lower in text_lower:
                    # Extract the symbol (first word) from the company entry
                    symbol = company.split()[0]
                    mentioned_companies.append(symbol)
                    
            except Exception as e:
                logger.error(f"Error processing company entry: {company}")
                logger.error(f"Error type: {type(e)}, Error message: {str(e)}")
                continue
                    
        return list(set(mentioned_companies))  # Remove duplicates
        
    def _categorize_content(self, text):
        """Categorize the content based on keywords"""
        if not isinstance(text, str):
            return []
            
        content_categories = []
        text_lower = text.lower()
        
        # Define category keywords with more specific subcategories
        category_keywords = {
            "Company Performance": {
                "keywords": ["performance", "growth", "decline", "improvement", "downturn", "quarterly results", "annual results"],
                "subcategories": ["Revenue Growth", "Profitability", "Market Share", "Operational Efficiency"]
            },
            "Management Updates": {
                "keywords": ["management", "ceo", "director", "board", "leadership", "appointment", "resignation", "promotion"],
                "subcategories": ["Leadership Changes", "Strategic Decisions", "Corporate Governance"]
            },
            "Market Sentiment": {
                "keywords": ["market", "sentiment", "trend", "outlook", "forecast", "bullish", "bearish", "neutral"],
                "subcategories": ["Market Outlook", "Trading Signals", "Investor Sentiment"]
            },
            "Financial Results": {
                "keywords": ["financial", "results", "earnings", "revenue", "profit", "loss", "dividend", "balance sheet"],
                "subcategories": ["Quarterly Results", "Annual Results", "Dividend Announcements", "Financial Ratios"]
            },
            "Industry News": {
                "keywords": ["industry", "sector", "competition", "market share", "innovation", "technology", "disruption"],
                "subcategories": ["Industry Trends", "Competitive Analysis", "Market Position"]
            },
            "Regulatory Updates": {
                "keywords": ["regulatory", "compliance", "government", "policy", "approval", "license", "permit"],
                "subcategories": ["Regulatory Approvals", "Compliance Updates", "Policy Changes"]
            },
            "Technical Analysis": {
                "keywords": ["technical", "analysis", "chart", "pattern", "indicator", "support", "resistance"],
                "subcategories": ["Price Analysis", "Technical Indicators", "Trading Patterns"]
            },
            "Investment Recommendations": {
                "keywords": ["buy", "sell", "hold", "target", "recommendation", "upgrade", "downgrade"],
                "subcategories": ["Analyst Ratings", "Price Targets", "Investment Calls"]
            }
        }
        
        for category, details in category_keywords.items():
            if any(keyword in text_lower for keyword in details["keywords"]):
                # Add both main category and relevant subcategories
                content_categories.append({
                    "main_category": category,
                    "subcategories": details["subcategories"]
                })
                
        return content_categories

    def _update_company_timeline(self, company, date, categories, text):
        """Update the timeline for a specific company"""
        entry = {
            "date": date,
            "categories": categories,
            "text": text
        }
        self.company_timeline[company].append(entry)
        # Sort timeline by date
        self.company_timeline[company].sort(key=lambda x: x["date"])

    def get_company_developments(self, company, start_date=None, end_date=None):
        """Get developments for a specific company within a date range"""
        if company not in self.company_timeline:
            return []
            
        timeline = self.company_timeline[company]
        
        if start_date:
            timeline = [entry for entry in timeline if entry["date"] >= start_date]
        if end_date:
            timeline = [entry for entry in timeline if entry["date"] <= end_date]
            
        return timeline

    def process_data(self, data):
        """
        Process and categorize the data with enhanced tracking
        """
        logger.info("Starting data categorization...")
        logger.info(f"self.companies type at start of process_data: {type(self.companies)}")
        logger.info(f"self.companies length at start of process_data: {len(self.companies)}")
        if len(self.companies) > 0:
            logger.info(f"First self.companies entry type at start of process_data: {type(self.companies[0])}")
            logger.info(f"First self.companies entry at start of process_data: {self.companies[0]}")
            logger.info(f"First self.companies entry keys at start of process_data: {self.companies[0].keys() if isinstance(self.companies[0], dict) else 'Not a dict'}")
        
        try:
            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Ensure required columns exist
            required_columns = ['text', 'date']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Input data must contain a '{col}' column")
                    
            # Convert date column to datetime if it's not already
            df['date'] = pd.to_datetime(df['date'])
            
            # Ensure text column contains strings
            df['text'] = df['text'].astype(str)
                
            # Add empty columns if they don't exist
            if 'companies' not in df.columns:
                df['companies'] = None
            if 'categories' not in df.columns:
                df['categories'] = None
                
            # Process each row
            df['companies'] = df['text'].apply(self._identify_companies)
            df['categories'] = df['text'].apply(self._categorize_content)
            
            # Update company timelines
            for _, row in df.iterrows():
                for company in row['companies']:
                    self._update_company_timeline(
                        company=company,
                        date=row['date'],
                        categories=row['categories'],
                        text=row['text']
                    )
            
            # Save processed data
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            
            # Save main categorized data
            df.to_csv(f"{PROCESSED_DATA_DIR}/categorized_data.csv", index=False)
            
            # Save company-specific timelines
            for company, timeline in self.company_timeline.items():
                company_df = pd.DataFrame(timeline)
                company_file = f"{PROCESSED_DATA_DIR}/company_timelines/{company.replace(' ', '_')}_timeline.csv"
                os.makedirs(os.path.dirname(company_file), exist_ok=True)
                company_df.to_csv(company_file, index=False)
            
            logger.info("Data categorization and timeline tracking completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data categorization: {str(e)}")
            raise 