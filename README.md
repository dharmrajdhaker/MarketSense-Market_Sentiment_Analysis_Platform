# Market Intelligence Dashboard

<img width="1440" alt="Screenshot 2025-05-12 at 6 02 41 PM" src="https://github.com/user-attachments/assets/b21ac11f-3f8d-4782-95bb-cff1fa222d8d" />

A sophisticated market intelligence platform that collects, analyzes, and visualizes market sentiment data from Telegram channels. This tool helps investors and analysts track market trends, company-specific insights, and trading signals through an interactive dashboard.

## 🌟 Features

### Data Collection
- Automated collection of market-related messages from Telegram channels
- Real-time data processing and sentiment analysis
- Support for multiple Telegram channels
- Historical data analysis capabilities

### Sentiment Analysis
- Advanced sentiment classification (Positive, Negative, Neutral)
- Trading signal generation (BUY, SELL, HOLD)
- Confidence scoring for each analysis
- Company-specific sentiment tracking

### Interactive Dashboard
- **Overview Tab**
  - Key market metrics and sentiment trends
  - Total mentions and average sentiment
  - Signal distribution analysis
  - Company mention frequency

- **Company Analysis Tab**
  - Detailed company-specific sentiment timeline
  - Sentiment distribution visualization
  - Recent updates and news tracking
  - Comprehensive company metrics
  - Trading signal analysis

- **Market Analysis Tab**
  - Market sentiment heatmap
  - Sector-wise analysis
  - Market metrics and trends
  - Top movers identification

- **Alerts & Insights Tab**
  - Real-time market alerts
  - Sentiment change notifications
  - Volume spike detection
  - Market insights and trend analysis

## 🛠️ Technical Stack

- **Backend**
  - Python 3.8+
  - Pandas for data manipulation
  - NLTK for natural language processing
  - Custom sentiment analysis algorithms

- **Frontend**
  - Dash/Plotly for interactive visualizations
  - Bootstrap for responsive design
  - Custom CSS for professional styling

- **Data Storage**
  - CSV-based data storage
  - Structured data organization
  - Efficient data processing pipeline

## 📋 Prerequisites

- Python 3.8 or higher
- Telegram API credentials
- Required Python packages (see `requirements.txt`)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/market-intelligence-dashboard.git
   cd market-intelligence-dashboard
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv

   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with:
   ```
   TELEGRAM_API_ID=your_api_id
   TELEGRAM_API_HASH=your_api_hash
   ```

## ⚙️ Configuration

1. Configure Telegram channels in `utils/config.py`:
   ```python
   TELEGRAM_CHANNELS = [
       'channel1',
       'channel2',
       # Add your channels here
   ]
   ```

2. Adjust analysis parameters in `utils/config.py`:
   ```python
   START_DATE = '2024-01-01'  # Customize start date
   END_DATE = '2024-12-31'    # Customize end date
   ```

## 🎮 Usage

1. Start the data collection:
   ```bash
   python DT_capital.py
   ```

2. Access the dashboard:
   - The dashboard will automatically start on a free port (default: 8054)
   - Open your browser and navigate to `http://localhost:8054`

3. Using the Dashboard:
   - Select date ranges for analysis
   - Choose specific companies for detailed analysis
   - Monitor market trends and signals
   - Set up alerts for important changes

## 📊 Dashboard Features

### Data Visualization
- Interactive charts and graphs
- Real-time updates
- Customizable date ranges
- Company-specific analysis
- Market-wide trends

### Analysis Tools
- Sentiment analysis
- Signal generation
- Volume analysis
- Trend detection
- Alert system

### Export Capabilities
- Save visualizations as HTML
- Export data for further analysis
- Generate reports

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any queries or support, please reach out to:
- Email: your.email@example.com
- Project Link: https://github.com/yourusername/market-intelligence-dashboard

## 🙏 Acknowledgments

- Telegram API for data collection
- Dash/Plotly for visualization capabilities
- Open-source community for various libraries and tools

---

Made with ❤️ by Dharmraj Dhaker

*Note: This is a professional tool for market analysis. Always verify signals and do your own research before making investment decisions.* 
