import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from data_collectors.telegram_collector import TelegramCollector
from data_processors.sentiment_analyzer import SentimentAnalyzer
from data_processors.data_categorizer import DataCategorizer
from visualization.dashboard import create_dashboard
from utils.config import (
    TELEGRAM_CHANNELS,
    NSE500_COMPANIES,
    START_DATE,
    END_DATE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize collector
        telegram_collector = TelegramCollector()
        
        # Initialize processors
        sentiment_analyzer = SentimentAnalyzer()
        data_categorizer = DataCategorizer()
        
        # Collect data
        logger.info("Starting data collection from Telegram channels...")
        telegram_data = telegram_collector.collect_data_sync(
            channels=TELEGRAM_CHANNELS,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # Process and analyze data
        logger.info("Processing collected data...")
        processed_data = data_categorizer.process_data(telegram_data)
        
        # Perform sentiment analysis
        logger.info("Performing sentiment analysis...")
        sentiment_data = sentiment_analyzer.analyze_sentiment_df(processed_data)
        
        # Create visualization dashboard
        logger.info("Generating visualization dashboard...")
        create_dashboard(sentiment_data)
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()