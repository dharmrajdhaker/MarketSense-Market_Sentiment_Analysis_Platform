# MarketSense - Market Sentiment Analysis Platform
<img width="1440" alt="Screenshot 2025-05-16 at 6 44 48 PM" src="https://github.com/user-attachments/assets/b9d44f24-2011-4055-8412-95600d723e65" />
<img width="1440" alt="Screenshot 2025-05-16 at 6 49 12 PM" src="https://github.com/user-attachments/assets/50f32590-a9a7-4780-a56b-2b7b1dd37d0c" />

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📊 Overview

MarketSense is a sophisticated market sentiment analysis platform that leverages Telegram channel data to provide insights into market trends and sentiment. The platform collects, processes, and analyzes data from various Telegram channels to generate actionable market intelligence.

## 🌟 Key Features

- **Real-time Data Collection**: Automated collection of market-related data from multiple Telegram channels
- **Advanced Sentiment Analysis**: Sophisticated sentiment analysis of collected data
- **Intelligent Data Categorization**: Smart categorization of market data
- **Technical Analysis Integration**:
  - Multiple technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
  - Customizable indicator parameters
  - Real-time indicator calculations
- **Advanced Charting System**:
  - Interactive price charts with multiple timeframes
  - Candlestick and line chart options
  - Volume analysis visualization
  - Customizable chart layouts and themes
- **Trading Signal Generation**:
  - Automated signal detection based on technical indicators
  - Sentiment-based signal confirmation
  - Customizable signal parameters
  - Signal strength indicators
- **Interactive Dashboard**: Dynamic visualization of sentiment trends and market insights
- **NSE500 Integration**: Coverage of NSE500 companies for comprehensive market analysis
- **Configurable Time Windows**: Flexible date range selection for analysis
- **Robust Error Handling**: Comprehensive logging and error management
- **Environment-based Configuration**: Secure configuration management using environment variables

## 🛠️ Technical Stack

- **Python 3.8+**: Core programming language
- **Data Collection**: Custom Telegram data collector
- **Data Processing**: 
  - Sentiment Analysis Engine
  - Data Categorization System
  - Technical Analysis Engine
- **Technical Analysis Libraries**:
  - TA-Lib for technical indicators
  - Pandas for data manipulation
  - NumPy for numerical computations
- **Charting and Visualization**:
  - Plotly for interactive charts
  - Custom charting components
- **Configuration**: Environment-based configuration management
- **Logging**: Comprehensive logging system

## 📋 Prerequisites

- Python 3.8 or higher
- Telegram API credentials
- Required Python packages (see Installation section)
- Access to specified Telegram channels

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/marketsense.git
cd marketsense
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```env
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
```

## ⚙️ Configuration

The project uses a configuration system that can be customized in `utils/config.py`:

- `TELEGRAM_CHANNELS`: List of Telegram channels to monitor
- `NSE500_COMPANIES`: List of NSE500 companies for analysis
- `START_DATE`: Analysis start date
- `END_DATE`: Analysis end date
- `TECHNICAL_INDICATORS`: List of technical indicators to use
- `SIGNAL_PARAMETERS`: Configuration for trading signal generation
- `CHART_SETTINGS`: Customization options for charts and visualizations

## 🎯 Usage

1. Ensure all environment variables are set correctly
2. Run the main script:
```bash
python main.py
```

The script will:
1. Collect data from specified Telegram channels
2. Process and categorize the collected data
3. Perform sentiment analysis
4. Generate an interactive dashboard

## 📊 Dashboard

The platform generates an interactive dashboard that provides:
- **Market Analysis**:
  - Sentiment trends over time
  - Company-specific sentiment analysis
  - Market trend visualizations
  - Key insights and metrics
- **Technical Analysis**:
  - Interactive price charts with multiple timeframes
  - Overlay of technical indicators
  - Volume analysis
  - Support and resistance levels
- **Trading Signals**:
  - Real-time signal generation
  - Signal strength indicators
  - Historical signal performance
  - Custom signal alerts
- **Customization Options**:
  - Adjustable chart layouts
  - Customizable indicator parameters
  - Multiple chart types (candlestick, line, area)
  - Theme selection

## 🔍 Project Structure

```
marketsense/
├── data_collectors/
│   └── telegram_collector.py
├── data_processors/
│   ├── sentiment_analyzer.py
│   └── data_categorizer.py
├── visualization/
│   └── dashboard.py
├── utils/
│   └── config.py
├── main.py
├── requirements.txt
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for informational purposes only and should not be considered as financial advice. Always do your own research before making any investment decisions.

## 📞 Support

For support, please open an issue in the GitHub repository or contact the development team.

---

Made with ❤️ by Dharmraj Dhaker 


<img width="1440" alt="Screenshot 2025-05-16 at 6 48 58 PM" src="https://github.com/user-attachments/assets/b3af7af3-c18e-4395-9e67-31eda9e6a11c" />
<img width="1438" alt="Screenshot 2025-05-16 at 6 47 38 PM" src="https://github.com/user-attachments/assets/680a7a62-bbe7-4c17-aecc-4b005281c1f9" />
<img width="1438" alt="Screenshot 2025-05-16 at 6 48 23 PM" src="https://github.com/user-attachments/assets/49b4b332-bcdb-4269-97b8-cd6c12d41d0d" />


*Note: This is a professional tool for market analysis. Always verify signals and do your own research before making investment decisions.* 
