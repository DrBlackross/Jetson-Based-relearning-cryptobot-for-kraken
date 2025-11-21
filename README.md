# Jetson-Based-relearning-cryptobot-for-kraken (improved for jetson)
Transformer-based neural network (inspired by the Temporal Fusion Transformer - TFT) for Kraken ON jetson nano orin


# 🤖 Advanced AI Crypto Trading Bot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated AI-powered cryptocurrency trading bot featuring **double optimization**, **fee-aware trading**, and **real-time web dashboard**. Built with Transformer architecture and optimized for aggressive trading strategies.

## 🚀 Features

### 🤖 AI-Powered Trading
- **Transformer Neural Network** for sequence prediction
- **Double Optimization** of ATR multipliers & learning rates
- **Real-time model retraining** every 6 hours
- **Fee-aware profit calculations** during training and execution

### ⚡ Aggressive Trading Strategy
- **50% position sizing** per trade
- **Fast technical indicators** (3-period RSI, 8/21 MACD)
- **RSI-based overrides** for extreme market conditions
- **Adaptive volatility thresholds**

### 📊 Live Monitoring
- **Real-time web dashboard** (Flask)
- **TensorBoard integration** for performance tracking
- **Trade history with fee tracking**
- **Portfolio performance metrics**

### 🔒 Risk Management
- **Paper trading mode** (default)
- **Live trading support** (Kraken API)
- **Fee-aware position sizing**
- **Minimum trade amount enforcement**

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- Kraken API keys (for live trading)
- TA-Lib library

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/advanced-crypto-trading-bot.git
cd advanced-crypto-trading-bot

2. Install Dependencies
bash

pip install -r requirements.txt

3. Install TA-Lib

Ubuntu/Linux:
bash

sudo apt-get update
sudo apt-get install ta-lib
pip install TA-Lib

macOS:
bash

brew install ta-lib
pip install TA-Lib

Windows:
Download pre-built wheel from here and install:
bash

pip install TA_Lib-0.4.28-cp38-cp38-win_amd64.whl

4. Environment Configuration

Create a .env file:
env

KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here

⚙️ Configuration
Trading Pairs

The bot supports two trading pairs:

    DOGEUSDT (Default) - High volatility, smaller positions

    XBTUSDT (Bitcoin) - Lower volatility, precision trading

Key Settings in Script
python

# Core Configuration
LIVE_TRADING = False  # ⚠️ Set to True for real trading
TRADING_PAIR = 'DOGEUSDT'  # or 'XBTUSDT'
TRADE_PERCENTAGE = 0.5  # 50% position size

# AI Optimization
ATR_MULTIPLIERS_TO_TEST = [0.05, 0.1, 0.15, 0.2, 0.25]
LEARNING_RATES_TO_TEST = [1e-6, 1e-5, 1e-4, 5e-4]

# Trading Fees
EXCHANGE_FEES = {
    'maker': 0.001,   # 0.1%
    'taker': 0.002,   # 0.2%
    'default': 0.002  # Aggressive strategy uses taker fees
}

🎯 Usage
1. Start the Bot
bash

python kraken-trans-bot-jetson.py

2. Access Web Dashboard

Open your browser to: http://localhost:5000
3. Monitor Performance

    Real-time trading decisions

    Portfolio balance and P&L

    AI training progress

    Trade history with fees

Dashboard Features

    📊 Trading Status - Live/paper mode, strategy, balances

    🧠 AI Training - Optimization progress, loss metrics

    📈 Recent Trades - Complete history with fee breakdown

    ⚡ Strategy Info - Aggressive settings and parameters

🔧 Technical Architecture
AI Model
python

class CryptoTransformer(torch.nn.Module):
    # Transformer encoder with 4 layers
    # 128-dimensional embeddings
    # Sequence length: 24 time periods
    # 3-class output: BUY/SELL/HOLD

Feature Engineering

    Price Data: OHLC + Volume

    Technical Indicators: RSI, MACD, Bollinger Bands, ATR, OBV, ADX, CCI, Stochastic, EMA

    Sequence Modeling: 24-period lookback window

Optimization Process

    ATR Multiplier Search - Find optimal volatility thresholds

    Learning Rate Search - Optimize model convergence

    Fee-aware Target Generation - Realistic profit calculations

    Model Selection - Best performing combination

📈 Trading Strategy
Aggressive Configuration

    RSI: 3-period with 75/25 bands

    MACD: 8/21/5 (fast settings)

    Bollinger Bands: 5-period, 1.5 std

    Position Sizing: 50% of available balance

Fee-Aware Logic
python

# All calculations include trading fees
def apply_trading_fees(amount, price, side):
    fee_rate = 0.002  # 0.2%
    trade_value = amount * price
    fee_amount = trade_value * fee_rate
    # Fees affect both entry and exit calculations

🚨 Risk Warning

⚠️ Important Disclaimer:

    This is experimental software

    Paper trading is enabled by default

    Live trading can result in significant financial losses

    Test thoroughly before using real money

    Use at your own risk

📊 Performance Metrics

The bot tracks:

    Portfolio Value (USDT)

    Profit/Loss (Absolute and Percentage)

    Trade Success Rate

    Fee Impact on profitability

    Model Accuracy and Loss

🔄 Retraining Schedule

    Automatic retraining every 6 hours

    Double optimization on each retrain

    Model persistence between sessions

    Performance-based model selection

🐛 Troubleshooting
Common Issues

    TA-Lib installation fails

        Use pre-compiled wheels for your platform

        Ensure system dependencies are installed

    Kraken API errors

        Verify API keys in .env file

        Check API permissions (trading enabled)

    CUDA out of memory (Jetson)

        Reduce batch size in training parameters

        Use smaller sequence length

Logs and Debugging

    Check console output for decision logic

    TensorBoard logs in ./trading_logs/

    Trade history saved to trade_history.csv

🤝 Contributing

    Fork the repository

    Create a feature branch (git checkout -b feature/amazing-feature)

    Commit your changes (git commit -m 'Add amazing feature')

    Push to the branch (git push origin feature/amazing-feature)

    Open a Pull Request

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    Kraken for API access

    Hugging Face for Transformers library

    TA-Lib for technical indicators

    PyTorch for deep learning framework

Happy Trading! 📈✨

Remember: Past performance is not indicative of future results. Always test with paper trading first.
