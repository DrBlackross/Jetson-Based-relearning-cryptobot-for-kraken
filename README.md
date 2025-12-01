# 🤖 Advanced AI Crypto Trading Bot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

(I reworked this from my other script for a raspberrypi, jetson is MUCH faster once you get torch properly installed)

## BTW, this is the kraken test address DBeziNpJSnRqvhJYQxgnVv5Lfc62BTd5WW 
##### /\ (in case you want to track it, 11/28/25 is my latest top up with a new updated version of the script i'll post here if it works) /\



[transformer-base-Ai-kraken](http://https://github.com/DrBlackross/transformer-base-Ai-kraken "transformer-base-Ai-kraken")

## This Script is a AI-powered cryptocurrency trading bot in python featuring **double optimization**, **fee-aware trading**, and **real-time web dashboard**. Built with Transformer architecture and optimized for aggressive trading strategies on a Jetson Nano Orin.

## Key things:
Jetson Nano: You might need to build some packages from source
CUDA Support: PyTorch should (keyword) automatically detect your Jetson's GPU
Memory: The bot is optimized for Jetson's limited RAM

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
- Kraken API keys (for live trading, paper should work without the api keys)
- TA-Lib library (always a hard part on installing the RIGHT one)

### 1. Clone Repository
in bash

    git clone https://github.com/DrBlackross/Jetson-Based-relearning-cryptobot-for-kraken.git
	cd Jetson-Based-relearning-cryptobot-for-kraken

### 2. Create a Virtual Environment / Install Dependencies
in bash

    python -m venv .venv; source ./.venv/bin/activate
	pip install -r requirements.txt

### 3. Install TA-Lib

Ubuntu/Linux:
in bash

	sudo apt-get update
	sudo apt-get install ta-lib
	pip install TA-Lib

### 4. Environment Configuration

Create a .env file:

	nano .env
and enter your 

    KRAKEN_API_KEY=your_api_key_here
    KRAKEN_API_SECRET=your_api_secret_here

(then save file)

### ⚙️ Configuration

#### Trading Pairs

The bot supports two trading pairs:

    DOGEUSDT (Default) - High volatility, smaller positions

    XBTUSDT (Bitcoin) - Lower volatility, precision trading

(these are the only two I care about, but will work with others on kraken)

#### Key Settings in Script

##### # Core Configuration
    LIVE_TRADING = False  # ⚠️ Set to True for real trading
    TRADING_PAIR = 'DOGEUSDT'  # or 'XBTUSDT'
    TRADE_PERCENTAGE = 0.5  # 50% position size
    
##### (NOTE: I use my kraken account as a 'control group' for experimenting in crypto trading, and it is usually seeded from my Coinbase with 100 DOGE when I start these bots, its a CYA, AND IS ISOLATED FROM MY MAIN ACCOUNTS!)
###### (ALWAYS Cover Your $ss, CYA, do not risk what you can't lose)


##### # AI Optimization
    ATR_MULTIPLIERS_TO_TEST = [0.05, 0.1, 0.15, 0.2, 0.25]
    LEARNING_RATES_TO_TEST = [1e-6, 1e-5, 1e-4, 5e-4]

##### # Trading Fees
    EXCHANGE_FEES = {
        'maker': 0.001,   # 0.1%
        'taker': 0.002,   # 0.2%
        'default': 0.002  # Aggressive strategy uses taker fees
    }

### 🎯 Usage
##### 1. Start the Bot

	python kraken-transformerbot-jetson.py

##### 2. Access Web Dashboard

	Open your browser to: http://localhost:5000

##### 3. Monitor Performance

    Real-time trading decisions
    Portfolio balance and P&L
    AI training progress
    Trade history with fees

### Dashboard Features

<img width="1120" height="868" alt="image" src="https://github.com/user-attachments/assets/4d90f859-fd5c-4696-bc72-0de75ed296cc" />

    📊 Trading Status - Live/paper mode, strategy, balances
    🧠 AI Training - Optimization progress, loss metrics
    📈 Recent Trades - Complete history with fee breakdown
    ⚡ Strategy Info - Aggressive settings and parameters

Overnight, this is what I woke up to...

<img width="976" height="932" alt="image" src="https://github.com/user-attachments/assets/0d6e3765-ef8e-4e2d-8cb2-fe20244358bc" />
(pretty cool, works a lot faster than my RPI version of this script, but i did make it more robust)

### 🔧 Technical Architecture
### AI Model
### python based
### CUDA

#### class CryptoTransformer(torch.nn.Module):
    Transformer encoder with 4 layers
    128-dimensional embeddings
    Sequence length: 24 time periods
    3-class output: BUY/SELL/HOLD

#### Feature Engineering

    Price Data: OHLC + Volume
    Technical Indicators: RSI, MACD, Bollinger Bands, ATR, OBV, ADX, CCI, Stochastic, EMA
    Sequence Modeling: 24-period lookback window

#### Optimization Process

    ATR Multiplier Search - Find optimal volatility thresholds
    Learning Rate Search - Optimize model convergence
    Fee-aware Target Generation - Realistic profit calculations
    Model Selection - Best performing combination

### 📈 Trading Strategy
##### Aggressive Configuration

    RSI: 3-period with 75/25 bands
    MACD: 8/21/5 (fast settings)
    Bollinger Bands: 5-period, 1.5 std
    Position Sizing: 50% of available balance
	Fee-Aware Logic


##### All calculations include trading fees
    def apply_trading_fees(amount, price, side):
    fee_rate = 0.002  # 0.2%
    trade_value = amount * price
    fee_amount = trade_value * fee_rate
    # Fees affect both entry and exit calculations

## 🚨 Risk Warning (always)

⚠️ Important Disclaimer:
    This is experimental software
    Paper trading is enabled by default
    Live trading can result in significant financial losses
    Test thoroughly before using real money
    Use at your own risk
### Stressing this again here
(NOTE: I use my kraken account as a 'control group' for experimenting in crypto trading, and it is usually seeded from my Coinbase with 100 DOGE when I start these bots, its a CYA, AND IS ISOLATED FROM MY MAIN ACCOUNTS!)
##### (ALWAYS Cover Your A$$, CYA, do not risk what you can't lose)

### 📊 Performance Metrics

#### The bot tracks:
    Portfolio Value (USDT)
    Profit/Loss (Absolute and Percentage)
    Trade Success Rate
    Fee Impact on profitability
    Model Accuracy and Loss

### 🔄 Retraining Schedule

    Automatic retraining every 6 hours
    Double optimization on each retrain
    Model persistence between sessions
    Performance-based model selection

### 🐛 Troubleshooting
Common Issues (for me TALIB, and getting Torch in a venv with CUDA)

#### TA-Lib installation fails (always a headache)
        Use pre-compiled wheels for your platform
        Ensure system dependencies are installed

#### Kraken API errors
        Verify API keys in .env file
        Check API permissions (trading enabled)

#### CUDA out of memory (Jetson)
        Reduce batch size in training parameters
        Use smaller sequence length

#### Logs and Debugging

    Check console output for decision logic
    TensorBoard logs in ./trading_logs/
    Trade history saved to trade_history.csv

#### Also to do a full reset with restart of the script (just in case you wish to purge the data and start over)

	rm -f ./doge_transformer_model_best.pth ./standard_scaler_best.pkl ./doge_transformer_model_best_multiplier.txt ./doge_transformer_model_best_learning_rate.txt && rm -rf ./results/ ./logs/ ./trading_logs/ && python kraken-transformerbot-jetson.py

### 🤝 Contributing
   Fork the repository
   Create a feature branch (git checkout -b feature/amazing-feature)
   Commit your changes (git commit -m 'Add amazing feature')
   Push to the branch (git push origin feature/amazing-feature)
   Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
### 🙏 Acknowledgments

   Kraken for API access
   Hugging Face for Transformers library
   TA-Lib for technical indicators
   PyTorch for deep learning framework

## Happy Trading! 📈✨

#### Take your time - good trading bots need room to breathe and learn! The double optimization + fee awareness should give much more reliable results.

See you in (hopefully) the profits! 💰😎

*P.S. The bot will automatically retrain every 6 hours with fresh, fee-aware data - it just gets smarter over time!*

Remember: Past performance is not indicative of future results. Always test with paper trading first.


*Side Note... I'm trying to have a reasoning model double check the rsi based transformer model. Not having the best of luck yet (see below), the script posted here without the added reasoning just works better IMHO, but I was working on this for a V2.0 version of the script.

<img width="840" height="914" alt="image" src="https://github.com/user-attachments/assets/3fb64781-a4e5-42ab-a691-fc4773b8c36e" />
