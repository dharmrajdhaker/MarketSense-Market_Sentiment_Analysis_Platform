import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from utils.config import VISUALIZATIONS_DIR, PROCESSED_DATA_DIR, get_company_symbol
import os
from datetime import datetime, timedelta
import numpy as np
from utils.nse_master_data import NSEMasterData
import talib

logger = logging.getLogger(__name__)

# Update the custom CSS with a modern color scheme and improved styling
CUSTOM_CSS = {
    'container': {
        'backgroundColor': '#f8f9fa',
        'minHeight': '100vh',
        'padding': '2rem 0'
    },
    'header': {
        'backgroundColor': '#ffffff',
        'padding': '1.5rem',
        'borderRadius': '0.5rem',
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.05)',
        'marginBottom': '2rem'
    },
    'card': {
        'backgroundColor': '#ffffff',
        'borderRadius': '0.5rem',
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.05)',
        'marginBottom': '1.5rem',
        'border': 'none',
        'transition': 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out'
    },
    'card:hover': {
        'transform': 'translateY(-2px)',
        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
    },
    'alert-card': {
        'backgroundColor': '#ffffff',
        'borderRadius': '0.5rem',
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.05)',
        'marginBottom': '1rem',
        'borderLeft': '4px solid',
        'transition': 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out'
    },
    'alert-card:hover': {
        'transform': 'translateX(5px)',
        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
    },
    'sentiment-indicator': {
        'padding': '0.25rem 0.75rem',
        'borderRadius': '1rem',
        'fontWeight': '600',
        'display': 'inline-block',
        'fontSize': '0.875rem'
    },
    'news-card': {
        'backgroundColor': '#ffffff',
        'borderRadius': '0.5rem',
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.05)',
        'marginBottom': '1.5rem',
        'padding': '1.25rem',
        'transition': 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out'
    },
    'news-card:hover': {
        'transform': 'translateY(-2px)',
        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
    },
    'tab-content': {
        'backgroundColor': '#ffffff',
        'borderRadius': '0.5rem',
        'padding': '1.5rem',
        'marginTop': '1rem'
    },
    'metric-card': {
        'backgroundColor': '#ffffff',
        'borderRadius': '0.5rem',
        'padding': '1.25rem',
        'textAlign': 'center',
        'border': 'none',
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.05)'
    },
    'metric-value': {
        'fontSize': '2rem',
        'fontWeight': '600',
        'color': '#2c3e50',
        'marginBottom': '0.5rem'
    },
    'metric-label': {
        'fontSize': '0.875rem',
        'color': '#6c757d',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px'
    }
}

# Update the color scheme for sentiment and signals
SENTIMENT_COLORS = {
    'POS': '#2ecc71',  # Modern green
    'NEU': '#3498db',  # Modern blue
    'NEG': '#e74c3c',  # Modern red
    'BUY': '#2ecc71',  # Modern green
    'SELL': '#e74c3c',  # Modern red
    'HOLD': '#3498db'   # Modern blue
}

def get_sentiment_color(sentiment, confidence):
    """Get color based on sentiment and confidence."""
    return SENTIMENT_COLORS.get(sentiment, '#95a5a6')  # Default to modern gray

def get_alert_severity(signal, confidence):
    """Calculate alert severity based on signal and confidence."""
    if signal in ['BUY', 'SELL'] and confidence > 0.8:
        return 'high'
    elif signal in ['BUY', 'SELL'] and confidence > 0.6:
        return 'medium'
    return 'low'

def format_datetime(dt_str):
    """Format datetime string to show both UTC and IST"""
    try:
        # First convert to pandas datetime if it's a string
        dt = pd.to_datetime(dt_str)
        
        # If the datetime is naive (no timezone), assume it's UTC
        if dt.tz is None:
            dt = dt.tz_localize('UTC')
        
        # Now convert to IST
        ist_dt = dt.tz_convert('Asia/Kolkata')
        
        # Format both times
        utc_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        ist_str = ist_dt.strftime('%Y-%m-%d %H:%M:%S IST')
        
        return f"{utc_str} ({ist_str})"
    except Exception as e:
        logger.error(f"Error formatting datetime {dt_str}: {str(e)}")
        return str(dt_str)

def create_price_sentiment_chart(price_data, sentiment_data, company):
    """Create a combined price and sentiment chart with volume and technical indicators."""
    from plotly.subplots import make_subplots
    import talib
    import numpy as np
    
    logger.info("Creating price-sentiment chart...")
    
    # Create copies to avoid SettingWithCopyWarning
    price_data = price_data.copy()
    sentiment_data = sentiment_data.copy()
    
    # Validate input data
    if price_data.empty:
        logger.error("Price data is empty")
        return go.Figure()
    
    if sentiment_data.empty:
        logger.warning("Sentiment data is empty")
    
    # Log data info
    logger.info(f"Price data shape: {price_data.shape}")
    logger.info(f"Sentiment data shape: {sentiment_data.shape}")
    
    # Ensure all timestamps are in UTC
    if price_data.index.tz is None:
        price_data.index = price_data.index.tz_localize('UTC')
        logger.info("Localized price data index to UTC")
    
    if sentiment_data['date'].dt.tz is None:
        sentiment_data['date'] = sentiment_data['date'].dt.tz_localize('UTC')
        logger.info("Localized sentiment data dates to UTC")
    
    # Create a function to format time in IST
    def format_time_ist(dt):
        if dt.tz is None:
            dt = dt.tz_localize('UTC')
        ist_time = dt.tz_convert('Asia/Kolkata')
        return ist_time.strftime('%Y-%m-%d %H:%M:%S IST')
    
    # Handle duplicate timestamps by aggregating data
    if not price_data.index.is_unique:
        logger.info("Found duplicate timestamps, aggregating data...")
        price_data = price_data.groupby(price_data.index).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    
    # Ensure price data columns are numeric and convert to float64 for technical indicators
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col not in price_data.columns:
            logger.error(f"Required column {col} not found in price data")
            return go.Figure()
        price_data[col] = pd.to_numeric(price_data[col], errors='coerce').astype('float64')
    
    # Drop any rows with NaN values after conversion
    price_data = price_data.dropna(subset=numeric_columns)
    
    if price_data.empty:
        logger.error("No valid price data after conversion")
        return go.Figure()
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{company} Price & News Signals (IST)', 'Volume', 'Technical Indicators')
    )
    
    try:
        # Calculate technical indicators using numpy arrays with explicit float64 type
        close_prices = price_data['Close'].values.astype(np.float64)
        high_prices = price_data['High'].values.astype(np.float64)
        low_prices = price_data['Low'].values.astype(np.float64)
        volume = price_data['Volume'].values.astype(np.float64)
        
        logger.info("Calculating technical indicators...")
        
        # Calculate RSI
        rsi = talib.RSI(close_prices, timeperiod=14)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)
        
        # Calculate Volume Moving Average
        volume_ma = talib.SMA(volume, timeperiod=20)
        
        logger.info("Technical indicators calculated successfully")
        
        # Create candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red',
                hoverinfo='all',
                xhoverformat='%Y-%m-%d %H:%M:%S IST',
                yhoverformat='.2f'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        for trace_data, name in [(upper, 'BB Upper'), (middle, 'BB Middle'), (lower, 'BB Lower')]:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=trace_data,
                    name=name,
                    line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                    showlegend=False,
                    hoverinfo='x+y',
                    xhoverformat='%Y-%m-%d %H:%M:%S IST',
                    yhoverformat='.2f'
                ),
                row=1, col=1
            )
        
        # Add Volume
        fig.add_trace(
            go.Bar(
                x=price_data.index,
                y=price_data['Volume'],
                name='Volume',
                marker_color='rgba(100, 100, 100, 0.5)',
                hoverinfo='x+y',
                xhoverformat='%Y-%m-%d %H:%M:%S IST',
                yhoverformat=',.0f'
            ),
            row=2, col=1
        )
        
        # Add Volume MA
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=volume_ma,
                name='Volume MA',
                line=dict(color='blue', width=1),
                hoverinfo='x+y',
                xhoverformat='%Y-%m-%d %H:%M:%S IST',
                yhoverformat=',.0f'
            ),
            row=2, col=1
        )
        
        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=rsi,
                name='RSI',
                line=dict(color='purple', width=1),
                hoverinfo='x+y',
                xhoverformat='%Y-%m-%d %H:%M:%S IST',
                yhoverformat='.2f'
            ),
            row=3, col=1
        )
        
        # Add MACD
        for trace_data, name, color in [(macd, 'MACD', 'blue'), (macd_signal, 'Signal', 'red')]:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=trace_data,
                    name=name,
                    line=dict(color=color, width=1),
                    hoverinfo='x+y',
                    xhoverformat='%Y-%m-%d %H:%M:%S IST',
                    yhoverformat='.2f'
                ),
                row=3, col=1
            )
        
        # Add MACD Histogram
        fig.add_trace(
            go.Bar(
                x=price_data.index,
                y=macd_hist,
                name='MACD Histogram',
                marker_color=['red' if val < 0 else 'green' for val in macd_hist],
                opacity=0.5,
                hoverinfo='x+y',
                xhoverformat='%Y-%m-%d %H:%M:%S IST',
                yhoverformat='.2f'
            ),
            row=3, col=1
        )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Add sentiment signals if available
        if not sentiment_data.empty:
            logger.info("Adding sentiment signals to chart...")
            buy_signals = sentiment_data[sentiment_data['signal'] == 'BUY']
            sell_signals = sentiment_data[sentiment_data['signal'] == 'SELL']
            
            logger.info(f"Found {len(buy_signals)} buy signals and {len(sell_signals)} sell signals")
            
            # Add buy signals
            if not buy_signals.empty:
                for _, signal in buy_signals.iterrows():
                    # Find the closest candle to the signal time
                    time_diffs = np.array([abs((t - signal['date']).total_seconds()) for t in price_data.index])
                    closest_idx = np.argmin(time_diffs)
                    price = price_data['Close'].iloc[closest_idx]
                    
                    # Add buy signal marker
                    fig.add_trace(
                        go.Scatter(
                            x=[signal['date']],
                            y=[price],
                            mode='markers+text',
                            marker=dict(
                                symbol='triangle-up',
                                size=20,
                                color='green',
                                line=dict(width=2, color='darkgreen'),
                                opacity=0.8
                            ),
                            text=[signal['reason']],
                            textposition="top center",
                            name='Buy Signal',
                            hoverinfo='text',
                            hovertext=f"Buy Signal<br>Time: {format_time_ist(signal['date'])}<br>Price: ₹{price:.2f}<br>Reason: {signal['reason']}<br>Confidence: {signal['confidence']:.0%}"
                        ),
                        row=1, col=1
                    )
            
            # Add sell signals
            if not sell_signals.empty:
                for _, signal in sell_signals.iterrows():
                    # Find the closest candle to the signal time
                    time_diffs = np.array([abs((t - signal['date']).total_seconds()) for t in price_data.index])
                    closest_idx = np.argmin(time_diffs)
                    price = price_data['Close'].iloc[closest_idx]
                    
                    # Add sell signal marker
                    fig.add_trace(
                        go.Scatter(
                            x=[signal['date']],
                            y=[price],
                            mode='markers+text',
                            marker=dict(
                                symbol='triangle-down',
                                size=20,
                                color='red',
                                line=dict(width=2, color='darkred'),
                                opacity=0.8
                            ),
                            text=[signal['reason']],
                            textposition="bottom center",
                            name='Sell Signal',
                            hoverinfo='text',
                            hovertext=f"Sell Signal<br>Time: {format_time_ist(signal['date'])}<br>Price: ₹{price:.2f}<br>Reason: {signal['reason']}<br>Confidence: {signal['confidence']:.0%}"
                        ),
                        row=1, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=f'{company} Price Chart with Technical Analysis (IST)',
            yaxis_title='Price (₹)',
            yaxis2_title='Volume',
            yaxis3_title='Indicator Value',
            xaxis_rangeslider_visible=False,
            height=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor='grey',
                spikedash='solid',
                spikethickness=1
            ),
            hovermode='x unified',
            hoverdistance=100,
            spikedistance=1000
        )
        
        # Update all x-axes to show IST
        for i in range(1, 4):
            fig.update_xaxes(
                rangeslider_visible=False,
                type="date",
                tickformat="%Y-%m-%d %H:%M:%S IST",
                tickangle=-45,
                title_text="Time (IST)",
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                minor=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    dtick=5*60*1000  # 5 minutes in milliseconds
                ),
                tickformatstops=[
                    dict(dtickrange=[None, None], value="%Y-%m-%d %H:%M:%S IST")
                ],
                row=i,
                col=1
            )
        
        logger.info("Chart created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}", exc_info=True)
        # Return a basic price chart without indicators if technical analysis fails
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price',
                hoverinfo='all',
                xhoverformat='%Y-%m-%d %H:%M:%S IST',
                yhoverformat='.2f'
            )
        )
        fig.update_layout(
            title=f'{company} Price Chart (Basic) (IST)',
            yaxis_title='Price (₹)',
            xaxis_title='Time (IST)',
            xaxis_rangeslider_visible=False,
            height=600
        )
        # Update x-axis for basic chart
        fig.update_xaxes(
            type="date",
            tickformat="%Y-%m-%d %H:%M:%S IST",
            tickangle=-45,
            title_text="Time (IST)",
            tickformatstops=[
                dict(dtickrange=[None, None], value="%Y-%m-%d %H:%M:%S IST")
            ]
        )
        logger.info("Returning basic price chart due to error")
        return fig

def create_dashboard(data=None):
    """Create an interactive dashboard for data visualization."""
    logger.info("Creating visualization dashboard...")
    
    if data is None:
        # Load data from CSV if not provided
        data_file = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
        if os.path.exists(data_file):
            data = pd.read_csv(data_file)
            data['date'] = pd.to_datetime(data['date'])
        else:
            logger.error(f"Data file not found: {data_file}")
            data = pd.DataFrame()
    
    # Log available companies
    available_companies = sorted(data['company_name'].unique())
    logger.info(f"Available companies in data: {available_companies}")
    
    # Initialize the Dash app with a modern theme
    app = Dash(__name__, 
               external_stylesheets=[
                   dbc.themes.BOOTSTRAP,
                   'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
               ],
               meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])
    
    # Create layout with improved styling
    app.layout = dbc.Container([
        # Header with improved styling
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Market Intelligence Dashboard", 
                           className="text-center mb-3",
                           style={'color': '#2c3e50', 'fontWeight': '600'}),
                    html.P("Real-time market sentiment and company analysis", 
                          className="text-center text-muted mb-0",
                          style={'fontSize': '1.1rem'})
                ], style=CUSTOM_CSS['header'])
            ], width=12)
        ]),
        
        # Date Range Filter with improved styling
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Date Range", 
                               className="mb-3",
                               style={'color': '#2c3e50', 'fontWeight': '600'}),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=data['date'].min() if not data.empty else datetime.now() - timedelta(days=7),
                            end_date=data['date'].max() if not data.empty else datetime.now(),
                            className="mb-4",
                            style={'width': '100%'}
                        )
                    ])
                ], style=CUSTOM_CSS['card'])
            ], width=12)
        ]),
        
        # Main Content Tabs with improved styling
        dbc.Tabs([
            # Overview Tab
            dbc.Tab([
                dbc.Row([
                    # Key Metrics Cards with improved styling
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div(id="total-mentions", 
                                            className="metric-value",
                                            style={'color': SENTIMENT_COLORS['NEU']}),
                                    html.Div("Total Mentions", 
                                            className="metric-label")
                                ])
                            ])
                        ], style=CUSTOM_CSS['metric-card']),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div(id="avg-sentiment", 
                                            className="metric-value",
                                            style={'color': SENTIMENT_COLORS['POS']}),
                                    html.Div("Average Sentiment", 
                                            className="metric-label")
                                ])
                            ])
                        ], style=CUSTOM_CSS['metric-card']),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div(id="signal-distribution", 
                                            className="metric-value",
                                            style={'color': SENTIMENT_COLORS['BUY']}),
                                    html.Div("Signal Distribution", 
                                            className="metric-label")
                                ])
                            ])
                        ], style=CUSTOM_CSS['metric-card'])
                    ], width=3),
                    
                    # Main Charts with improved styling
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Sentiment Trend", 
                                       className="card-title mb-4",
                                       style={'color': '#2c3e50', 'fontWeight': '600'}),
                                dcc.Graph(id='sentiment-trend',
                                        style={'height': '400px'})
                            ])
                        ], style=CUSTOM_CSS['card']),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Company Mentions", 
                                       className="card-title mb-4",
                                       style={'color': '#2c3e50', 'fontWeight': '600'}),
                                dcc.Graph(id='company-mentions',
                                        style={'height': '400px'})
                            ])
                        ], style=CUSTOM_CSS['card'])
                    ], width=9)
                ])
            ], label="Overview", tab_style={'backgroundColor': '#f8f9fa'}, 
               active_tab_style={'backgroundColor': '#ffffff', 'color': '#2c3e50', 'fontWeight': '600'}),
            
            # Company Analysis Tab
            dbc.Tab([
                dbc.Row([
                    # Company Selection
                    dbc.Col([
                        html.H5("Select Company", className="mb-3"),
                        dcc.Dropdown(
                            id='company-dropdown',
                            options=[{'label': company, 'value': company} 
                                    for company in sorted(data['company_name'].unique()) 
                                    if company != 'Unknown'],
                            value=None,
                            className="mb-4"
                        ),
                        html.H5("Date Range", className="mb-3"),
                        dcc.DatePickerRange(
                            id='price-date-range',
                            start_date=data['date'].min() if not data.empty else datetime.now() - timedelta(days=7),
                            end_date=data['date'].max() if not data.empty else datetime.now(),
                            className="mb-4"
                        )
                    ], width=12)
                ]),
                dbc.Row([
                    # Company Metrics
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Company Sentiment", className="card-title"),
                                dcc.Graph(id='company-sentiment')
                            ])
                        ], className="mb-4"),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Sentiment Timeline", className="card-title"),
                                dcc.Graph(id='company-timeline')
                            ])
                        ])
                    ], width=6),
                    # Company Details
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Recent Mentions", className="card-title"),
                                html.Div(id='company-mentions-list')
                            ])
                        ], className="mb-4"),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Sentiment Analysis", className="card-title"),
                                html.Div(id='company-analysis')
                            ])
                        ])
                    ], width=6)
                ])
            ], label="Company Analysis"),
            
            # Market Analysis Tab
            dbc.Tab([
                dbc.Row([
                    # Market Overview
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Market Sentiment Heatmap", className="card-title"),
                                dcc.Graph(id='market-heatmap')
                            ])
                        ], className="mb-4"),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Sector Analysis", className="card-title"),
                                dcc.Graph(id='sector-analysis')
                            ])
                        ])
                    ], width=8),
                    # Market Metrics
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Market Metrics", className="card-title"),
                                html.Div(id='market-metrics')
                            ])
                        ], className="mb-4"),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Top Movers", className="card-title"),
                                html.Div(id='top-movers')
                            ])
                        ])
                    ], width=4)
                ])
            ], label="Market Analysis"),
            
            # News & Alerts Tab
            dbc.Tab([
                dbc.Row([
                    # News Feed with Enhanced Sentiment
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H5("Live News Feed", className="d-inline"),
                                    dbc.ButtonGroup([
                                        dbc.Button(
                                            html.I(className="fas fa-sync-alt"),
                                            color="link",
                                            id="refresh-news-feed",
                                            className="me-2"
                                        ),
                                        dbc.DropdownMenu(
                                            [
                                                dbc.DropdownMenuItem("All Sentiments", id="news-filter-all-sentiments"),
                                                dbc.DropdownMenuItem("Positive Only", id="news-filter-positive"),
                                                dbc.DropdownMenuItem("Negative Only", id="news-filter-negative"),
                                                dbc.DropdownMenuItem("Neutral Only", id="news-filter-neutral"),
                                            ],
                                            label="Filter by Sentiment",
                                            color="light",
                                            className="me-2"
                                        ),
                                        dbc.DropdownMenu(
                                            [
                                                dbc.DropdownMenuItem("All Signals", id="news-filter-all-signals"),
                                                dbc.DropdownMenuItem("Buy Signals", id="news-filter-buy"),
                                                dbc.DropdownMenuItem("Sell Signals", id="news-filter-sell"),
                                                dbc.DropdownMenuItem("Hold Signals", id="news-filter-hold"),
                                            ],
                                            label="Filter by Signal",
                                            color="light"
                                        )
                                    ], className="float-end")
                                ], className="mb-3"),
                                html.Div(id='news-feed-content', className="news-feed-container")
                            ])
                        ])
                    ], width=8),
                    # Enhanced Alerts Panel
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H5("Active Alerts", className="d-inline"),
                                    dbc.Badge(
                                        id="active-alerts-count",
                                        color="danger",
                                        className="ms-2"
                                    )
                                ], className="mb-3"),
                                html.Div(id='active-alerts-panel'),
                                html.Hr(),
                                html.H6("Alert Settings", className="mt-3 mb-2"),
                                dbc.Form([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Checkbox(
                                                id="news-alert-sentiment",
                                                label="Sentiment Changes",
                                                value=True,
                                                className="mb-2"
                                            ),
                                            dbc.Checkbox(
                                                id="news-alert-volume",
                                                label="Volume Spikes",
                                                value=True,
                                                className="mb-2"
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Checkbox(
                                                id="news-alert-signals",
                                                label="Signal Changes",
                                                value=True,
                                                className="mb-2"
                                            ),
                                            dbc.Checkbox(
                                                id="news-alert-breaking",
                                                label="Breaking News",
                                                value=True,
                                                className="mb-2"
                                            )
                                        ], width=6)
                                    ]),
                                    html.Hr(),
                                    html.H6("Alert Threshold", className="mt-3 mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Confidence Threshold"),
                                            dcc.Slider(
                                                id="news-confidence-threshold",
                                                min=0,
                                                max=1,
                                                step=0.1,
                                                value=0.7,
                                                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                                                tooltip={"placement": "bottom", "always_visible": True}
                                            )
                                        ], width=12)
                                    ])
                                ])
                            ])
                        ])
                    ], width=4)
                ])
            ], label="News & Alerts"),
            
            # Price Analysis Tab
            dbc.Tab([
                dbc.Row([
                    # Company Selection and Controls
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Price & Sentiment Analysis", className="card-title mb-4"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Select Company", className="mb-2"),
                                        dcc.Dropdown(
                                            id='price-company-dropdown',
                                            options=[{'label': company, 'value': company} 
                                                    for company in sorted(data['company_name'].unique()) 
                                                    if company != 'Unknown'],
                                            value=None,
                                            className="mb-3"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Date Range", className="mb-2"),
                                        dcc.DatePickerRange(
                                            id='price-analysis-date-range',
                                            start_date=datetime.now() - timedelta(days=1),
                                            end_date=datetime.now(),
                                            className="mb-3"
                                        )
                                    ], width=6)
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Time Interval", className="mb-2"),
                                        dcc.RadioItems(
                                            id='price-interval-selector',
                                            options=[
                                                {'label': '1 Minute', 'value': '1m'},
                                                {'label': '5 Minutes', 'value': '5m'},
                                                {'label': '15 Minutes', 'value': '15m'},
                                                {'label': '1 Hour', 'value': '1h'},
                                                {'label': '1 Day', 'value': '1d'}
                                            ],
                                            value='1m',
                                            inline=True,
                                            className="mb-3"
                                        )
                                    ], width=12)
                                ])
                            ])
                        ], style=CUSTOM_CSS['card'])
                    ], width=12)
                ]),
                dbc.Row([
                    # Price & Sentiment Chart
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='price-sentiment-chart', style={'height': '800px'})
                            ])
                        ])
                    ], width=12)
                ])
            ], label="Price Analysis"),
            
            # Alerts & Insights Tab
            dbc.Tab([
                dbc.Row([
                    # Alerts
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Recent Alerts", className="card-title"),
                                html.Div(id='alerts-list')
                            ])
                        ], className="mb-4"),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Alert Preferences", className="card-title"),
                                html.Div([
                                    dbc.Checkbox(
                                        id="insights-alert-sentiment",
                                        label="Sentiment Changes",
                                        value=True,
                                        className="mb-2"
                                    ),
                                    dbc.Checkbox(
                                        id="insights-alert-volume",
                                        label="Volume Spikes",
                                        value=True,
                                        className="mb-2"
                                    ),
                                    dbc.Checkbox(
                                        id="insights-alert-signals",
                                        label="Signal Changes",
                                        value=True,
                                        className="mb-2"
                                    )
                                ])
                            ])
                        ])
                    ], width=6),
                    # Insights
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Market Insights", className="card-title"),
                                html.Div(id='market-insights')
                            ])
                        ], className="mb-4"),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Trend Analysis", className="card-title"),
                                html.Div(id='trend-analysis')
                            ])
                        ])
                    ], width=6)
                ])
            ], label="Alerts & Insights")
        ])
    ], fluid=True, style={'marginTop': '2rem'})
    
    # Callbacks for interactive features
    @app.callback(
        [Output('total-mentions', 'children'),
         Output('avg-sentiment', 'children'),
         Output('signal-distribution', 'children')],
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_metrics(start_date, end_date):
        if data.empty:
            return "N/A", "N/A", "N/A"
            
        filtered_data = data[
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ]
        
        total_mentions = len(filtered_data)
        
        # Calculate average sentiment
        sentiment_map = {'POS': 1, 'NEU': 0, 'NEG': -1}
        avg_sentiment = filtered_data['sentiment'].map(sentiment_map).mean()
        avg_sentiment_text = f"{avg_sentiment:.2f}"
        
        # Calculate signal distribution
        signal_dist = filtered_data['signal'].value_counts()
        signal_dist_text = html.Div([
            html.P(f"BUY: {signal_dist.get('BUY', 0)}"),
            html.P(f"SELL: {signal_dist.get('SELL', 0)}"),
            html.P(f"HOLD: {signal_dist.get('HOLD', 0)}")
        ])
        
        return total_mentions, avg_sentiment_text, signal_dist_text
    
    @app.callback(
        Output('sentiment-trend', 'figure'),
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_sentiment_trend(start_date, end_date):
        if data.empty:
            return px.line(title="No data available")
            
        filtered_data = data[
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ]
        
        # Group by date and sentiment
        sentiment_trend = filtered_data.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        return px.line(
            sentiment_trend,
            x='date',
            y='count',
            color='sentiment',
            title='Sentiment Trend Over Time',
            labels={'date': 'Date', 'count': 'Number of Mentions'},
            color_discrete_map={'POS': 'green', 'NEG': 'red', 'NEU': 'blue'}
        )
    
    @app.callback(
        Output('company-mentions', 'figure'),
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_company_mentions(start_date, end_date):
        if data.empty:
            return px.bar(title="No data available")
            
        filtered_data = data[
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ]
        
        # Filter out Unknown companies
        filtered_data = filtered_data[filtered_data['company_name'] != 'Unknown']
        if filtered_data.empty:
            return px.bar(title="No company data available")
            
        company_counts = filtered_data['company_name'].value_counts().head(10)
        
        return px.bar(
            x=company_counts.index,
            y=company_counts.values,
            title="Top 10 Companies Mentioned",
            labels={'x': 'Company', 'y': 'Count'},
            color=company_counts.values,
            color_continuous_scale='Viridis'
        )
    
    @app.callback(
        [Output('company-sentiment', 'figure'),
         Output('company-timeline', 'figure'),
         Output('company-mentions-list', 'children'),
         Output('company-analysis', 'children')],
        [Input('company-dropdown', 'value'),
         Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_company_analysis(company, start_date, end_date):
        if not company or data.empty:
            return px.pie(), px.line(), "No company selected", "No company selected"
            
        filtered_data = data[
            (data['company_name'] == company) &
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ]
        
        if filtered_data.empty:
            return px.pie(), px.line(), "No data available", "No data available"
        
        # Company sentiment pie chart
        sentiment_counts = filtered_data['sentiment'].value_counts()
        sentiment_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title=f'Sentiment Distribution for {company}',
            color=sentiment_counts.index,
            color_discrete_map={'POS': 'green', 'NEG': 'red', 'NEU': 'blue'}
        )
        
        # Company timeline
        timeline_data = filtered_data.sort_values('date')
        sentiment_timeline = px.line(
            timeline_data,
            x='date',
            y='confidence',
            color='sentiment',
            title=f'Sentiment Timeline for {company}',
            labels={'date': 'Date', 'confidence': 'Confidence'},
            color_discrete_map={'POS': 'green', 'NEG': 'red', 'NEU': 'blue'}
        )
        
        # Recent mentions list
        recent_mentions = html.Div([
            html.Div([
                html.P([
                    html.Strong(format_datetime(row['date'])),
                    html.Br(),
                    f"Reason: {row['reason']}",
                    html.Br(),
                    html.Span(f"Sentiment: {row['sentiment']}", 
                             style={'color': 'green' if row['sentiment'] == 'POS' 
                                   else 'red' if row['sentiment'] == 'NEG' 
                                   else 'blue'}),
                    " | ",
                    html.Span(f"Signal: {row['signal']}", 
                             style={'color': 'green' if row['signal'] == 'BUY' 
                                   else 'red' if row['signal'] == 'SELL' 
                                   else 'blue'})
                ]),
                html.Hr()
            ]) for _, row in filtered_data.sort_values('date', ascending=False).head(5).iterrows()
        ])
        
        # Company analysis
        analysis = html.Div([
            html.H5("Sentiment Analysis"),
            html.P(f"Total Mentions: {len(filtered_data)}"),
            html.P(f"Positive Mentions: {len(filtered_data[filtered_data['sentiment'] == 'POS'])}"),
            html.P(f"Negative Mentions: {len(filtered_data[filtered_data['sentiment'] == 'NEG'])}"),
            html.P(f"Neutral Mentions: {len(filtered_data[filtered_data['sentiment'] == 'NEU'])}"),
            html.H5("Signal Analysis"),
            html.P(f"Buy Signals: {len(filtered_data[filtered_data['signal'] == 'BUY'])}"),
            html.P(f"Sell Signals: {len(filtered_data[filtered_data['signal'] == 'SELL'])}"),
            html.P(f"Hold Signals: {len(filtered_data[filtered_data['signal'] == 'HOLD'])}")
        ])
        
        return sentiment_pie, sentiment_timeline, recent_mentions, analysis
    
    @app.callback(
        [Output('market-heatmap', 'figure'),
         Output('sector-analysis', 'figure'),
         Output('market-metrics', 'children'),
         Output('top-movers', 'children')],
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_market_analysis(start_date, end_date):
        if data.empty:
            return px.imshow(), px.bar(), "No data available", "No data available"
            
        filtered_data = data[
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ]
        
        # Market heatmap
        sentiment_map = {'POS': 1, 'NEU': 0, 'NEG': -1}
        company_sentiment = filtered_data.groupby('company_name')['sentiment'].apply(
            lambda x: x.map(sentiment_map).mean()
        ).reset_index()
        
        heatmap = px.imshow(
            company_sentiment.pivot_table(
                index='company_name',
                values='sentiment',
                aggfunc='mean'
            ),
            title='Market Sentiment Heatmap',
            color_continuous_scale='RdYlGn'
        )
        
        # Sector analysis (placeholder - would need sector data)
        sector_analysis = px.bar(
            title='Sector Analysis (Placeholder)'
        )
        
        # Market metrics
        metrics = html.Div([
            html.H5("Market Overview"),
            html.P(f"Total Companies Mentioned: {len(filtered_data['company_name'].unique())}"),
            html.P(f"Total Mentions: {len(filtered_data)}"),
            html.P(f"Average Sentiment: {filtered_data['sentiment'].map(sentiment_map).mean():.2f}"),
            html.P(f"Signal Distribution:"),
            html.P(f"BUY: {len(filtered_data[filtered_data['signal'] == 'BUY'])}"),
            html.P(f"SELL: {len(filtered_data[filtered_data['signal'] == 'SELL'])}"),
            html.P(f"HOLD: {len(filtered_data[filtered_data['signal'] == 'HOLD'])}")
        ])
        
        # Top movers
        top_movers = html.Div([
            html.H5("Top Positive Movers"),
            html.Div([
                html.P(f"{company}: {sentiment:.2f}")
                for company, sentiment in company_sentiment.nlargest(5, 'sentiment').values
            ]),
            html.H5("Top Negative Movers"),
            html.Div([
                html.P(f"{company}: {sentiment:.2f}")
                for company, sentiment in company_sentiment.nsmallest(5, 'sentiment').values
            ])
        ])
        
        return heatmap, sector_analysis, metrics, top_movers

    @app.callback(
        [Output('alerts-list', 'children'),
         Output('market-insights', 'children'),
         Output('trend-analysis', 'children')],
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date'),
         Input('insights-alert-sentiment', 'value'),
         Input('insights-alert-volume', 'value'),
         Input('insights-alert-signals', 'value')]
    )
    def update_alerts_and_insights(start_date, end_date, 
                                 alert_sentiment, alert_volume, alert_signals):
        if data.empty:
            return "No alerts", "No insights", "No trend analysis"
            
        filtered_data = data[
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ]
        
        # Fix the alert filtering logic
        alerts_mask = pd.Series(False, index=filtered_data.index)
        
        if alert_signals:
            alerts_mask |= filtered_data['signal'].isin(['BUY', 'SELL'])
        
        if alert_sentiment:
            alerts_mask |= filtered_data['confidence'] > 0.7
        
        if alert_volume:
            alerts_mask |= filtered_data['text'].str.contains('volume|spike', case=False, na=False)
        
        alerts_data = filtered_data[alerts_mask].sort_values('date', ascending=False)
        
        # Alerts list with comprehensive information
        alerts = html.Div([
            html.H5("Recent Updates & Alerts", className="mb-4"),
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            # Header with timestamp and company
                            html.Div([
                                html.H6(format_datetime(row['date']), 
                                      className="text-muted mb-2"),
                                html.H5(f"{row['company_name']}", className="mb-3"),
                            ], className="d-flex justify-content-between align-items-center"),
                            
                            # News content with better formatting
                            html.Div([
                                html.Div([
                                    html.Strong("News: ", className="me-2"),
                                    html.Span(
                                        row.get('text', 'No text available'),
                                        style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word'}
                                    )
                                ], className="mb-3 p-2 bg-light rounded"),
                                
                                # Sentiment and signal information with improved layout
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.Strong("Sentiment: ", className="me-2"),
                                                html.Span(
                                                    row['sentiment'],
                                                    style={
                                                        'backgroundColor': get_sentiment_color(row['sentiment'], row['confidence']),
                                                        'color': 'white',
                                                        **CUSTOM_CSS['sentiment-indicator']
                                                    }
                                                ),
                                                html.Small(
                                                    f" ({row['confidence']:.0%})",
                                                    className="ms-2 text-muted"
                                                )
                                            ])
                                        ], width=4),
                                        dbc.Col([
                                            html.Div([
                                                html.Strong("Signal: ", className="me-2"),
                                                html.Span(
                                                    row['signal'],
                                                    style={
                                                        'backgroundColor': '#28a745' if row['signal'] == 'BUY' 
                                                        else '#dc3545' if row['signal'] == 'SELL' 
                                                        else '#17a2b8',
                                                        'color': 'white',
                                                        **CUSTOM_CSS['sentiment-indicator']
                                                    }
                                                )
                                            ])
                                        ], width=4),
                                        dbc.Col([
                                            html.Div([
                                                html.Strong("Impact: ", className="me-2"),
                                                html.Span(
                                                    "High" if row['confidence'] > 0.8 else "Medium" if row['confidence'] > 0.6 else "Low",
                                                    style={
                                                        'backgroundColor': '#dc3545' if row['confidence'] > 0.8 
                                                        else '#ffc107' if row['confidence'] > 0.6 
                                                        else '#28a745',
                                                        'color': 'white',
                                                        **CUSTOM_CSS['sentiment-indicator']
                                                    }
                                                )
                                            ])
                                        ], width=4)
                                    ], className="mb-3")
                                ])
                            ])
                        ], className="p-2")
                    ])
                ], style={
                    **CUSTOM_CSS['alert-card'],
                    'borderLeftColor': '#dc3545' if get_alert_severity(row['signal'], row['confidence']) == 'high'
                    else '#ffc107' if get_alert_severity(row['signal'], row['confidence']) == 'medium'
                    else '#28a745'
                })
                for _, row in alerts_data.head(10).iterrows()
            ])
        ])
        
        # Market insights with more detailed analysis
        insights = html.Div([
            html.H5("Market Insights", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("Overall Market Sentiment", className="mb-2"),
                        html.P([
                            html.Strong("Dominant Sentiment: "),
                            html.Span(
                                filtered_data['sentiment'].value_counts().index[0],
                                style={
                                    'color': 'green' if filtered_data['sentiment'].value_counts().index[0] == 'POS'
                                    else 'red' if filtered_data['sentiment'].value_counts().index[0] == 'NEG'
                                    else 'blue',
                                    'fontWeight': 'bold'
                                }
                            )
                        ]),
                        html.P([
                            html.Strong("Sentiment Distribution: "),
                            html.Br(),
                            f"Positive: {len(filtered_data[filtered_data['sentiment'] == 'POS'])} mentions",
                            html.Br(),
                            f"Negative: {len(filtered_data[filtered_data['sentiment'] == 'NEG'])} mentions",
                            html.Br(),
                            f"Neutral: {len(filtered_data[filtered_data['sentiment'] == 'NEU'])} mentions"
                        ])
                    ], className="mb-3"),
                    
                    html.Div([
                        html.H6("Most Active Companies", className="mb-2"),
                        html.P([
                            html.Strong("Top 3 Companies: "),
                            html.Div([
                                html.Div([
                                    f"{company} ({count} mentions)"
                                ]) for company, count in filtered_data['company_name'].value_counts().head(3).items()
                            ])
                        ])
                    ])
                ])
            ])
        ])
        
        # Trend analysis with more detailed metrics
        trends = html.Div([
            html.H5("Trend Analysis", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("Signal Distribution", className="mb-2"),
                        html.P([
                            html.Strong("Current Signal Distribution: "),
                            html.Br(),
                            f"BUY Signals: {len(filtered_data[filtered_data['signal'] == 'BUY'])}",
                            html.Br(),
                            f"SELL Signals: {len(filtered_data[filtered_data['signal'] == 'SELL'])}",
                            html.Br(),
                            f"HOLD Signals: {len(filtered_data[filtered_data['signal'] == 'HOLD'])}"
                        ])
                    ], className="mb-3"),
                    
                    html.Div([
                        html.H6("Recent Trend Changes", className="mb-2"),
                        html.P([
                            html.Strong("Last 24 Hours: "),
                            html.Br(),
                            f"New Signals: {len(filtered_data[filtered_data['date'] >= (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')])}",
                            html.Br(),
                            f"Sentiment Changes: {len(filtered_data[filtered_data['date'] >= (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')].groupby('company_name')['sentiment'].nunique())}"
                        ])
                    ])
                ])
            ])
        ])
        
        return alerts, insights, trends
    
    @app.callback(
        [Output('news-feed-content', 'children'),
         Output('active-alerts-panel', 'children'),
         Output('active-alerts-count', 'children')],
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date'),
         Input('refresh-news-feed', 'n_clicks'),
         Input('news-confidence-threshold', 'value'),
         Input('news-alert-sentiment', 'value'),
         Input('news-alert-signals', 'value'),
         Input('news-alert-volume', 'value'),
         Input('news-alert-breaking', 'value')]
    )
    def update_news_and_alerts(start_date, end_date, n_clicks, 
                              confidence_threshold,
                              alert_sentiment, alert_signals, alert_volume, alert_breaking):
        if data.empty:
            return "No news available", "No alerts available", "0"
            
        filtered_data = data[
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ].sort_values('date', ascending=False)
        
        # Update the alerts filtering logic to use only confidence threshold
        alerts_mask = pd.Series(False, index=filtered_data.index)
        
        if alert_signals:
            alerts_mask |= (filtered_data['signal'].isin(['BUY', 'SELL']) & 
                           (filtered_data['confidence'] > confidence_threshold))
        
        if alert_sentiment:
            alerts_mask |= filtered_data['confidence'] > confidence_threshold
        
        if alert_volume:
            alerts_mask |= filtered_data['text'].str.contains('volume|spike', case=False, na=False)
        
        if alert_breaking:
            alerts_mask |= filtered_data['text'].str.contains('breaking|urgent|important', case=False, na=False)
        
        alerts_data = filtered_data[alerts_mask].head(10)
        
        # News Feed with enhanced sentiment visualization
        news_feed = html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        # Header with timestamp and company
                        html.Div([
                            html.Div([
                                html.H6(format_datetime(row['date']), 
                                      className="text-muted mb-0"),
                                html.Small(f"Source: {row.get('channel', 'Telegram')}", 
                                         className="text-muted d-block")
                            ]),
                            html.H5(f"{row['company_name']}", className="mb-2")
                        ], className="d-flex justify-content-between align-items-start mb-3"),
                        
                        # News content with sentiment indicator
                        html.Div([
                            html.Div([
                                html.Strong("News: ", className="me-2"),
                                html.Span(
                                    row.get('text', 'No text available'),
                                    style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word'}
                                )
                            ], className="mb-3 p-3 bg-light rounded"),
                            
                            # Enhanced sentiment and signal information
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.Strong("Sentiment: ", className="me-2"),
                                            html.Span(
                                                row['sentiment'],
                                                style={
                                                    'backgroundColor': get_sentiment_color(row['sentiment'], row['confidence']),
                                                    'color': 'white',
                                                    **CUSTOM_CSS['sentiment-indicator']
                                                }
                                            ),
                                            html.Small(
                                                f" ({row['confidence']:.0%})",
                                                className="ms-2 text-muted"
                                            )
                                        ])
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Strong("Signal: ", className="me-2"),
                                            html.Span(
                                                row['signal'],
                                                style={
                                                    'backgroundColor': '#28a745' if row['signal'] == 'BUY' 
                                                    else '#dc3545' if row['signal'] == 'SELL' 
                                                    else '#17a2b8',
                                                    'color': 'white',
                                                    **CUSTOM_CSS['sentiment-indicator']
                                                }
                                            )
                                        ])
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Strong("Impact: ", className="me-2"),
                                            html.Span(
                                                "High" if row['confidence'] > 0.8 else "Medium" if row['confidence'] > 0.6 else "Low",
                                                style={
                                                    'backgroundColor': '#dc3545' if row['confidence'] > 0.8 
                                                    else '#ffc107' if row['confidence'] > 0.6 
                                                    else '#28a745',
                                                    'color': 'white',
                                                    **CUSTOM_CSS['sentiment-indicator']
                                                }
                                            )
                                        ])
                                    ], width=4)
                                ], className="mb-3")
                            ])
                        ])
                    ], className="p-2")
                ])
            ], style=CUSTOM_CSS['news-card'])
            for _, row in alerts_data.iterrows()
        ])
        
        # Enhanced Alerts Panel with severity levels
        alerts = html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6([
                            html.Span(
                                f"{row['company_name']}",
                                className="me-2"
                            ),
                            dbc.Badge(
                                row['signal'],
                                color="success" if row['signal'] == 'BUY' else "danger",
                                className="me-2"
                            ),
                            html.Small(
                                format_datetime(row['date']),
                                className="text-muted float-end"
                            )
                        ], className="alert-heading mb-2"),
                        html.P(
                            row.get('text', '')[:150] + '...' if len(row.get('text', '')) > 150 else row.get('text', ''),
                            className="mb-2"
                        ),
                        html.Div([
                    html.Small([
                                html.Strong("Sentiment: ", className="me-2"),
                                html.Span(
                                    row['sentiment'],
                                    style={
                                        'color': get_sentiment_color(row['sentiment'], row['confidence']),
                                        'fontWeight': 'bold'
                                    }
                                ),
                                html.Span(" | ", className="mx-2"),
                                html.Strong("Confidence: ", className="me-2"),
                                html.Span(
                                    f"{row['confidence']:.0%}",
                                    style={'fontWeight': 'bold'}
                                )
                            ], className="text-muted")
                        ])
                    ])
                ])
            ], style={
                **CUSTOM_CSS['alert-card'],
                'borderLeftColor': '#dc3545' if get_alert_severity(row['signal'], row['confidence']) == 'high'
                else '#ffc107' if get_alert_severity(row['signal'], row['confidence']) == 'medium'
                else '#28a745'
            })
            for _, row in alerts_data.iterrows()
        ])
        
        alert_count = len(alerts_data)
        
        return news_feed, alerts, str(alert_count)
    
    @app.callback(
        Output('price-sentiment-chart', 'figure'),
        [Input('price-company-dropdown', 'value'),
         Input('price-analysis-date-range', 'start_date'),
         Input('price-analysis-date-range', 'end_date'),
         Input('price-interval-selector', 'value')]
    )
    def update_price_sentiment_chart(company, start_date, end_date, interval):
        if not company or data.empty:
            logger.warning("No company selected or data is empty")
            return go.Figure()
        
        try:
            # Log input parameters
            logger.info(f"Updating chart for company: {company}")
            logger.info(f"Date range: {start_date} to {end_date}")
            logger.info(f"Interval: {interval}")
            
            # Convert input dates to UTC timezone-aware datetime objects
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
            
            logger.info(f"Converted dates to UTC: {start_dt} to {end_dt}")
            
            # Ensure data['date'] is timezone-aware UTC
            if data['date'].dt.tz is None:
                data['date'] = data['date'].dt.tz_localize('UTC')
                logger.info("Localized sentiment data dates to UTC")
            
            # Filter sentiment data with exact company name match
            filtered_sentiment = data[
                (data['company_name'] == company) &
                (data['date'] >= start_dt) & 
                (data['date'] <= end_dt)
            ]
            
            logger.info(f"Found {len(filtered_sentiment)} sentiment records for {company}")
            
            if filtered_sentiment.empty:
                logger.warning(f"No sentiment data found for {company} in the selected date range")
                return go.Figure()
            
            # Get stock price data
            try:
                nse = NSEMasterData()
                logger.info("Downloading NSE symbol master data...")
                nse.download_symbol_master()
                
                # Get company symbol with exact match
                logger.info(f"Looking up symbol for company: '{company}'")
                company_symbol = get_company_symbol(company)
                logger.info(f"Found symbol: '{company_symbol}'")
                
                if company_symbol == "Unknown":
                    logger.error(f"Could not find NSE symbol for company: {company}")
                    return go.Figure()
                
                # Get price data with selected interval
                logger.info(f"Fetching price data for {company_symbol} from {start_dt} to {end_dt} with interval {interval}")
                price_data = nse.get_history(
                    symbol=company_symbol,
                    exchange='NSE',
                    start=start_dt,
                    end=end_dt,
                    interval=interval
                )
                
                if price_data.empty:
                    logger.error(f"No price data received for {company_symbol}")
                    return go.Figure()
                
                logger.info(f"Retrieved {len(price_data)} price data points")
                logger.info(f"Price data columns: {price_data.columns.tolist()}")
                logger.info(f"Price data index type: {type(price_data.index)}")
                logger.info(f"First few price data points:\n{price_data.head()}")
                
                # Ensure price data index is timezone-aware UTC
                if price_data.index.tz is None:
                    price_data.index = price_data.index.tz_localize('UTC')
                    logger.info("Localized price data index to UTC")
                
                # Create the chart
                fig = create_price_sentiment_chart(price_data, filtered_sentiment, company)
                
                # Add signal summary
                if not filtered_sentiment.empty:
                    signals = filtered_sentiment[filtered_sentiment['signal'].isin(['BUY', 'SELL'])]
                    if not signals.empty:
                        logger.info(f"Found {len(signals)} signals to display")
                        # Create a string representation of the signal summary
                        signal_summary_text = "Signal Summary:\n"
                        for _, signal in signals.iterrows():
                            signal_summary_text += f"\n{signal['signal']} Signal at {signal['date'].strftime('%Y-%m-%d %H:%M:%S UTC')}: {signal['reason']} (Confidence: {signal['confidence']:.0%})"
                        
                        fig.add_annotation(
                            text=signal_summary_text,
                            xref="paper", yref="paper",
                            x=0.02, y=0.98,
                            showarrow=False,
                            font=dict(size=12),
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4,
                            align="left"
                        )
                else:
                    logger.info("No signals found to display")
                
                return fig
                
            except Exception as e:
                logger.error(f"Error creating price-sentiment chart: {str(e)}", exc_info=True)
                return go.Figure()
            
        except Exception as e:
            logger.error(f"Error in update_price_sentiment_chart: {str(e)}", exc_info=True)
            return go.Figure()
    
    # Run the app
    app.run_server(debug=False, port=8054)
    
    # Save individual charts
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Save sentiment distribution
    sentiment_pie = px.pie(
        values=data['sentiment'].value_counts().values,
        names=data['sentiment'].value_counts().index,
        title='Sentiment Distribution'
    )
    sentiment_pie.write_html(f"{VISUALIZATIONS_DIR}/sentiment_pie.html")
    
    # Save sentiment trend
    sentiment_trend = px.line(
        data.groupby(['date', 'sentiment']).size().reset_index(name='count'),
        x='date',
        y='count',
        color='sentiment',
        title='Sentiment Trend Over Time'
    )
    sentiment_trend.write_html(f"{VISUALIZATIONS_DIR}/sentiment_trend.html")
    
    # Save company mentions
    company_mentions = px.bar(
        x=data['company_name'].value_counts().head(10).index,
        y=data['company_name'].value_counts().head(10).values,
        title='Top 10 Companies Mentioned'
    )
    company_mentions.write_html(f"{VISUALIZATIONS_DIR}/company_mentions.html")

if __name__ == '__main__':
    create_dashboard()