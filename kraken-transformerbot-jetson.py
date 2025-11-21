import pandas as pd
import talib
import krakenex
from pykrakenapi import KrakenAPI
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime, timedelta, timezone
import os
import joblib
import shutil
import warnings
import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
from flask import Flask, render_template_string, jsonify
import threading
import json

load_dotenv()
warnings.filterwarnings('ignore', category=FutureWarning)

# ======================
# CONFIGURATION SETTINGS
# ======================

LIVE_TRADING = False
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')

TRADING_PAIR = 'DOGEUSDT'

if TRADING_PAIR == 'DOGEUSDT':
    MODEL_PATH_BASE = './doge_transformer_model'
    INITIAL_USDT_BALANCE = 50.0
    INITIAL_CRYPTO_BALANCE = 100.0
    MIN_TRADE_AMOUNT = 15.0
    CRYPTO_NAME = 'DOGE'
    CRYPTO_DECIMALS = 4
    DEFAULT_ATR_MULTIPLIER = 0.15
elif TRADING_PAIR == 'XBTUSDT':
    MODEL_PATH_BASE = './btc_transformer_model'
    INITIAL_USDT_BALANCE = 0.0
    INITIAL_CRYPTO_BALANCE = 0.00149596
    MIN_TRADE_AMOUNT = 10.0
    CRYPTO_NAME = 'BTC'
    CRYPTO_DECIMALS = 6
    DEFAULT_ATR_MULTIPLIER = 0.1
else:
    raise ValueError("Invalid trading pair. Use 'DOGEUSDT' or 'XBTUSDT'")

SCALER_PATH_BASE = './standard_scaler'
BEST_MODEL_PATH = f'{MODEL_PATH_BASE}_best.pth'
BEST_SCALER_PATH = f'{SCALER_PATH_BASE}_best.pkl'
BEST_MULTIPLIER_PATH = f'{MODEL_PATH_BASE}_best_multiplier.txt'
BEST_LEARNING_RATE_PATH = f'{MODEL_PATH_BASE}_best_learning_rate.txt'

PAIR = TRADING_PAIR
INTERVAL = 1
LOOKBACK_DAYS_TRAINING = 30
LOOKBACK_DAYS_WINDOW = 2
SEQUENCE_LENGTH = 24
SLEEP_TIME_SECONDS = 60
DECISION_INTERVAL_SECONDS = 60

EXCHANGE_FEES = {
    'maker': 0.001,
    'taker': 0.002,
    'default': 0.002
}

TRADE_PERCENTAGE = 0.5

NUM_TRAINING_RUNS = 2
PER_DEVICE_BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 50
EVAL_STEPS = 20
LOGGING_STEPS = 10
SAVE_STEPS = 20
SAVE_TOTAL_LIMIT = 2
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.001

ATR_MULTIPLIERS_TO_TEST = [0.05, 0.1, 0.15, 0.2, 0.25]
LEARNING_RATES_TO_TEST = [1e-6, 1e-5, 1e-4, 5e-4]
RETRAIN_INTERVAL_HOURS = 6

INDICATOR_CONFIG = {
    'BASE_FEATURES': ['open', 'high', 'low', 'close', 'volume'],
    'RSI': {'enabled': True, 'length': 3, 'overbought': 75, 'oversold': 25},
    'MACD': {'enabled': True, 'fast': 8, 'slow': 21, 'signal': 5},
    'BBANDS': {'enabled': True, 'length': 5, 'std': 1.5},
    'OBV': {'enabled': True},
    'ADX': {'enabled': True, 'length': 10},
    'CCI': {'enabled': True, 'length': 10, 'c': 0.015},
    'ATR': {'enabled': True, 'length': 10},
    'VWAP': {'enabled': False},
    'NATR': {'enabled': True, 'length': 10},
    'TRIX': {'enabled': True, 'length': 3, 'signal': 5},
    'STOCH': {'enabled': True, 'k': 2, 'd': 3, 'smooth_k': 2},
    'EMA': {'enabled': True, 'lengths': [5, 10]}
}

FEATURES_LIST = INDICATOR_CONFIG['BASE_FEATURES'].copy()

if INDICATOR_CONFIG['RSI']['enabled']:
    FEATURES_LIST.append(f"RSI_{INDICATOR_CONFIG['RSI']['length']}")

if INDICATOR_CONFIG['MACD']['enabled']:
    macd = INDICATOR_CONFIG['MACD']
    FEATURES_LIST.extend([
        f"MACD_{macd['fast']}_{macd['slow']}_{macd['signal']}",
        f"MACDh_{macd['fast']}_{macd['slow']}_{macd['signal']}",
        f"MACDs_{macd['fast']}_{macd['slow']}_{macd['signal']}"
    ])

if INDICATOR_CONFIG['BBANDS']['enabled']:
    bb = INDICATOR_CONFIG['BBANDS']
    FEATURES_LIST.extend([
        f"BBL_{bb['length']}_{bb['std']}",
        f"BBM_{bb['length']}_{bb['std']}",
        f"BBU_{bb['length']}_{bb['std']}"
    ])

if INDICATOR_CONFIG['OBV']['enabled']:
    FEATURES_LIST.append('OBV')

if INDICATOR_CONFIG['ADX']['enabled']:
    FEATURES_LIST.append(f"ADX_{INDICATOR_CONFIG['ADX']['length']}")

if INDICATOR_CONFIG['CCI']['enabled']:
    cci = INDICATOR_CONFIG['CCI']
    FEATURES_LIST.append(f"CCI_{cci['length']}_{cci['c']}")

if INDICATOR_CONFIG['ATR']['enabled']:
    FEATURES_LIST.append(f"ATRr_{INDICATOR_CONFIG['ATR']['length']}")

if INDICATOR_CONFIG['VWAP']['enabled']:
    FEATURES_LIST.append('VWAP_D')

if INDICATOR_CONFIG['NATR']['enabled']:
    FEATURES_LIST.append(f"NATR_{INDICATOR_CONFIG['ATR']['length']}")

if INDICATOR_CONFIG['TRIX']['enabled']:
    trix = INDICATOR_CONFIG['TRIX']
    FEATURES_LIST.extend([
        f"TRIX_{trix['length']}_{trix['signal']}",
        f"TRIXs_{trix['length']}_{trix['signal']}"
    ])

if INDICATOR_CONFIG['STOCH']['enabled']:
    stoch = INDICATOR_CONFIG['STOCH']
    FEATURES_LIST.extend([
        f"STOCHk_{stoch['k']}_{stoch['d']}_{stoch['smooth_k']}",
        f"STOCHd_{stoch['k']}_{stoch['d']}_{stoch['smooth_k']}"
    ])

if INDICATOR_CONFIG['EMA']['enabled']:
    for length in INDICATOR_CONFIG['EMA']['lengths']:
        FEATURES_LIST.append(f"EMA_{length}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app_state = {
    'training_progress': 0,
    'training_status': 'Not Training',
    'current_multiplier': 'N/A',
    'current_learning_rate': 'N/A',
    'current_run': 0,
    'total_runs': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0.0,
    'trader': None,
    'last_training_time': None,
    'next_training_time': None,
    'is_training': False,
    'best_multiplier': 'N/A',
    'best_learning_rate': 'N/A',
    'optimization_phase': 'Not Started'
}

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Advanced Crypto Trading Bot</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #0f0f23; color: #00ff00; }
        .container { max-width: 1400px; margin: 0 auto; }
        .card { background: #1a1a2e; padding: 20px; margin: 10px 0; border-radius: 8px; border: 1px solid #00ff00; }
        .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        .status-badge { padding: 5px 10px; border-radius: 4px; color: white; font-weight: bold; }
        .status-training { background: #ffa500; }
        .status-ready { background: #28a745; }
        .status-live { background: #dc3545; }
        .status-paper { background: #17a2b8; }
        .progress-bar { width: 100%; background: #2d2d2d; border-radius: 4px; margin: 10px 0; }
        .progress-fill { height: 20px; background: linear-gradient(90deg, #00ff00, #00cc00); border-radius: 4px; text-align: center; color: white; line-height: 20px; }
        .trade-history { max-height: 300px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #00ff00; }
        th { background: #2a2a3c; }
        .buy { color: #00ff00; font-weight: bold; }
        .sell { color: #ff4444; font-weight: bold; }
        .hold { color: #ffff00; }
        .warning { background: #ff4444; color: white; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .optimization-info { background: #2a2a3c; padding: 15px; border-radius: 4px; margin: 10px 0; }
        .fee-info { background: #2a3c2a; padding: 10px; border-radius: 4px; margin: 5px 0; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Advanced AI Crypto Trading Bot</h1>

        {% if LIVE_TRADING %}
        <div class="warning">
            ⚠️ <strong>LIVE TRADING MODE</strong> - Real money is at risk!
        </div>
        {% endif %}

        <div class="grid">
            <div class="card">
                <h2>📊 Trading Status</h2>
                <p><strong>Mode:</strong> <span class="status-badge {% if LIVE_TRADING %}status-live{% else %}status-paper{% endif %}">
                    {{ "LIVE TRADING" if LIVE_TRADING else "PAPER TRADING" }}
                </span></p>
                <p><strong>Pair:</strong> {{ trading_pair }}</p>
                <p><strong>Device:</strong> {{ device }}</p>
                <p><strong>Strategy:</strong> <span style="color: #ff4444;">🔥 AGGRESSIVE</span></p>
                <p><strong>Trade Size:</strong> {{ (TRADE_PERCENTAGE * 100)|int }}% of balance</p>
                <p><strong>Fees:</strong> {{ (EXCHANGE_FEES.default * 100)|float }}% per trade</p>
                <p><strong>Last Decision:</strong> {{ last_decision_time }}</p>
            </div>

            <div class="card">
                <h2>💰 Balances & Performance</h2>
                <p><strong>{{ crypto_name }} Balance:</strong> {{ "%.4f"|format(crypto_balance) }}</p>
                <p><strong>USDT Balance:</strong> {{ "%.2f"|format(usdt_balance) }}</p>
                <p><strong>Portfolio Value:</strong> {{ "%.2f"|format(portfolio_value) }} USDT</p>
                <p><strong>Profit/Loss:</strong> <span style="color: {{ 'green' if pnl >= 0 else 'red' }}; font-weight: bold;">
                    {{ "%.2f"|format(pnl) }} USDT ({{ "%.2f"|format(pnl_percent) }}%)
                </span></p>
                <div class="fee-info">
                    <strong>Fee-Aware Trading:</strong> All calculations include {{ (EXCHANGE_FEES.default * 100)|float }}% trading fees
                </div>
            </div>

            <div class="card">
                <h2>🎯 Optimization Results</h2>
                <p><strong>Best ATR Multiplier:</strong> {{ best_multiplier }}</p>
                <p><strong>Best Learning Rate:</strong> {{ best_learning_rate }}</p>
                <p><strong>Last Training:</strong> {{ last_training_time }}</p>
                <p><strong>Next Training:</strong> {{ next_training_time }}</p>
                <p><strong>Optimization Phase:</strong> {{ optimization_phase }}</p>
            </div>
        </div>

        <div class="card">
            <h2>🧠 AI Training Progress</h2>
            <p><strong>Status:</strong> <span class="status-badge {% if is_training %}status-training{% else %}status-ready{% endif %}">
                {{ training_status }}
            </span></p>
            {% if is_training %}
            <div class="optimization-info">
                <p><strong>ATR Multiplier:</strong> {{ current_multiplier }}</p>
                <p><strong>Learning Rate:</strong> {{ current_learning_rate }}</p>
                <p><strong>Run:</strong> {{ current_run }}/{{ total_runs }}</p>
                <p><strong>Epoch:</strong> {{ current_epoch }}/{{ total_epochs }}</p>
                <p><strong>Loss:</strong> {{ "%.6f"|format(current_loss) }}</p>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ training_progress }}%;">{{ training_progress }}%</div>
            </div>
            {% endif %}
        </div>

        <div class="card">
            <h2>📈 Recent Trades (Aggressive Mode)</h2>
            <div class="trade-history">
                {% if trade_history %}
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Type</th>
                            <th>Amount</th>
                            <th>Price</th>
                            <th>Total</th>
                            <th>Fees</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for trade in trade_history[-15:] %}
                        <tr>
                            <td>{{ trade.time }}</td>
                            <td class="{{ trade.type.lower() }}">{{ trade.type }}</td>
                            <td>{{ "%.4f"|format(trade.amount) }}</td>
                            <td>{{ "%.5f"|format(trade.price) }}</td>
                            <td>{{ "%.2f"|format(trade.cost if trade.type == 'BUY' else trade.proceeds) }}</td>
                            <td>{{ "%.4f"|format(trade.fees) }} USDT</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                <p>No trades yet</p>
                {% endif %}
            </div>
        </div>

        <div class="card">
            <h2>⚡ Trading Strategy</h2>
            <p><strong>Aggressive Settings:</strong></p>
            <ul>
                <li>Trade Size: {{ (TRADE_PERCENTAGE * 100)|int }}% per trade</li>
                <li>Trading Fees: {{ (EXCHANGE_FEES.default * 100)|float }}% per trade</li>
                <li>RSI Overbought: {{ INDICATOR_CONFIG['RSI']['overbought'] }} (Tight)</li>
                <li>RSI Oversold: {{ INDICATOR_CONFIG['RSI']['oversold'] }} (Tight)</li>
                <li>Fast MACD: {{ INDICATOR_CONFIG['MACD']['fast'] }}/{{ INDICATOR_CONFIG['MACD']['slow'] }}/{{ INDICATOR_CONFIG['MACD']['signal'] }}</li>
                <li>ATR Multipliers Tested: {{ ATR_MULTIPLIERS_TO_TEST|length }}</li>
                <li>Learning Rates Tested: {{ LEARNING_RATES_TO_TEST|length }}</li>
            </ul>
        </div>
    </div>

    <script>
        {% if is_training %}
        setTimeout(() => location.reload(), 3000);
        {% else %}
        setTimeout(() => location.reload(), 10000);
        {% endif %}
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    trader = app_state['trader']
    if trader:
        current_value = trader.crypto_balance * trader._get_current_price() + trader.usdt_balance
        profit_loss = current_value - trader.initial_portfolio_value
        profit_loss_pct = (profit_loss / trader.initial_portfolio_value) * 100 if trader.initial_portfolio_value != 0 else 0
        trade_history = trader.trade_history
    else:
        current_value = 0
        profit_loss = 0
        profit_loss_pct = 0
        trade_history = []

    return render_template_string(HTML_TEMPLATE,
                                  LIVE_TRADING=LIVE_TRADING,
                                  trading_pair=PAIR,
                                  device=str(device),
                                  crypto_name=CRYPTO_NAME,
                                  crypto_balance=trader.crypto_balance if trader else INITIAL_CRYPTO_BALANCE,
                                  usdt_balance=trader.usdt_balance if trader else INITIAL_USDT_BALANCE,
                                  portfolio_value=current_value,
                                  pnl=profit_loss,
                                  pnl_percent=profit_loss_pct,
                                  last_decision_time=trader.last_trade_time.strftime('%Y-%m-%d %H:%M:%S') if trader and trader.last_trade_time else 'Never',
                                  training_progress=app_state['training_progress'],
                                  training_status=app_state['training_status'],
                                  current_multiplier=app_state['current_multiplier'],
                                  current_learning_rate=app_state['current_learning_rate'],
                                  current_run=app_state['current_run'],
                                  total_runs=app_state['total_runs'],
                                  current_epoch=app_state['current_epoch'],
                                  total_epochs=app_state['total_epochs'],
                                  current_loss=app_state['current_loss'],
                                  is_training=app_state['is_training'],
                                  last_training_time=app_state['last_training_time'].strftime('%Y-%m-%d %H:%M:%S') if app_state['last_training_time'] else 'Never',
                                  next_training_time=app_state['next_training_time'].strftime('%Y-%m-%d %H:%M:%S') if app_state['next_training_time'] else 'Calculating...',
                                  best_multiplier=app_state['best_multiplier'],
                                  best_learning_rate=app_state['best_learning_rate'],
                                  optimization_phase=app_state['optimization_phase'],
                                  trade_history=trade_history,
                                  TRADE_PERCENTAGE=TRADE_PERCENTAGE,
                                  INDICATOR_CONFIG=INDICATOR_CONFIG,
                                  ATR_MULTIPLIERS_TO_TEST=ATR_MULTIPLIERS_TO_TEST,
                                  LEARNING_RATES_TO_TEST=LEARNING_RATES_TO_TEST,
                                  EXCHANGE_FEES=EXCHANGE_FEES
                                  )

@app.route('/api/status')
def api_status():
    return jsonify(app_state)

@app.route('/api/trade_history')
def api_trade_history():
    trader = app_state['trader']
    if trader:
        return jsonify(trader.trade_history[-20:])
    return jsonify([])

class CryptoTransformer(torch.nn.Module):
    def __init__(self, num_features, dropout_rate=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(num_features, 128)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = torch.nn.Linear(128 * SEQUENCE_LENGTH, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]).to(device))

    def forward(self, x, labels=None):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        logits = self.output_layer(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        if loss is not None:
            return (loss, logits)
        return logits

class TradingDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        if idx + self.seq_length >= len(self.features):
            raise IndexError("Index out of bounds for sequence length")
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return {"x": torch.FloatTensor(x), "labels": torch.tensor(y, dtype=torch.long)}

def calculate_all_indicators(df):
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values

        if INDICATOR_CONFIG['RSI']['enabled']:
            length = INDICATOR_CONFIG['RSI']['length']
            df[f"RSI_{length}"] = talib.RSI(close, timeperiod=length)

        if INDICATOR_CONFIG['MACD']['enabled']:
            macd_config = INDICATOR_CONFIG['MACD']
            macd, macd_signal, macd_hist = talib.MACD(close,
                                                      fastperiod=macd_config['fast'],
                                                      slowperiod=macd_config['slow'],
                                                      signalperiod=macd_config['signal'])
            df[f"MACD_{macd_config['fast']}_{macd_config['slow']}_{macd_config['signal']}"] = macd
            df[f"MACDs_{macd_config['fast']}_{macd_config['slow']}_{macd_config['signal']}"] = macd_signal
            df[f"MACDh_{macd_config['fast']}_{macd_config['slow']}_{macd_config['signal']}"] = macd_hist

        if INDICATOR_CONFIG['BBANDS']['enabled']:
            bb_config = INDICATOR_CONFIG['BBANDS']
            upper, middle, lower = talib.BBANDS(close,
                                                timeperiod=bb_config['length'],
                                                nbdevup=bb_config['std'],
                                                nbdevdn=bb_config['std'])
            df[f"BBU_{bb_config['length']}_{bb_config['std']}"] = upper
            df[f"BBM_{bb_config['length']}_{bb_config['std']}"] = middle
            df[f"BBL_{bb_config['length']}_{bb_config['std']}"] = lower

        if INDICATOR_CONFIG['OBV']['enabled']:
            df['OBV'] = talib.OBV(close, volume)

        if INDICATOR_CONFIG['ADX']['enabled']:
            length = INDICATOR_CONFIG['ADX']['length']
            df[f"ADX_{length}"] = talib.ADX(high, low, close, timeperiod=length)

        if INDICATOR_CONFIG['CCI']['enabled']:
            cci_config = INDICATOR_CONFIG['CCI']
            df[f"CCI_{cci_config['length']}_{cci_config['c']}"] = talib.CCI(high, low, close, timeperiod=cci_config['length'])

        if INDICATOR_CONFIG['ATR']['enabled']:
            length = INDICATOR_CONFIG['ATR']['length']
            df[f"ATRr_{length}"] = talib.ATR(high, low, close, timeperiod=length)

        if INDICATOR_CONFIG['NATR']['enabled']:
            length = INDICATOR_CONFIG['NATR']['length']
            df[f"NATR_{length}"] = talib.NATR(high, low, close, timeperiod=length)

        if INDICATOR_CONFIG['TRIX']['enabled']:
            trix_config = INDICATOR_CONFIG['TRIX']
            df[f"TRIX_{trix_config['length']}_{trix_config['signal']}"] = talib.TRIX(close, timeperiod=trix_config['length'])
            trix_values = df[f"TRIX_{trix_config['length']}_{trix_config['signal']}"].values
            df[f"TRIXs_{trix_config['length']}_{trix_config['signal']}"] = talib.EMA(trix_values, timeperiod=trix_config['signal'])

        if INDICATOR_CONFIG['STOCH']['enabled']:
            stoch_config = INDICATOR_CONFIG['STOCH']
            slowk, slowd = talib.STOCH(high, low, close,
                                       fastk_period=stoch_config['k'],
                                       slowk_period=stoch_config['smooth_k'],
                                       slowk_matype=0,
                                       slowd_period=stoch_config['d'],
                                       slowd_matype=0)
            df[f"STOCHk_{stoch_config['k']}_{stoch_config['d']}_{stoch_config['smooth_k']}"] = slowk
            df[f"STOCHd_{stoch_config['k']}_{stoch_config['d']}_{stoch_config['smooth_k']}"] = slowd

        if INDICATOR_CONFIG['EMA']['enabled']:
            for length in INDICATOR_CONFIG['EMA']['lengths']:
                df[f"EMA_{length}"] = talib.EMA(close, timeperiod=length)

        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return pd.DataFrame()

def apply_trading_fees(amount, price, side, fee_rate=None):
    if fee_rate is None:
        fee_rate = EXCHANGE_FEES['default']

    trade_value = amount * price
    fee_amount = trade_value * fee_rate

    if side == 'BUY':
        return trade_value + fee_amount
    else:
        return trade_value - fee_amount

def calculate_profit_with_fees(entry_price, exit_price, amount, fee_rate=None):
    if fee_rate is None:
        fee_rate = EXCHANGE_FEES['default']

    entry_cost = apply_trading_fees(amount, entry_price, 'BUY', fee_rate)
    exit_value = apply_trading_fees(amount, exit_price, 'SELL', fee_rate)
    return exit_value - entry_cost

class CryptoTrader:
    def __init__(self, model, public_api, private_api, scaler, device, live_trading, crypto_balance, usdt_balance, trading_writer=None):
        self.model = model
        self.public_api = public_api
        self.private_api = private_api
        self.scaler = scaler
        self.device = device
        self.live_trading = live_trading

        self.crypto_balance = crypto_balance
        self.usdt_balance = usdt_balance

        self.last_trade_time = datetime.now(timezone.utc) - timedelta(seconds=DECISION_INTERVAL_SECONDS)
        self.trade_percentage = TRADE_PERCENTAGE
        self.min_trade_amount = MIN_TRADE_AMOUNT
        self.trade_history = []
        self.trading_writer = trading_writer
        self.trade_step = 0

        self.initial_portfolio_value = self.usdt_balance + (self.crypto_balance * self._get_current_price())

        self.status_colors = {
            'BUY': '\033[92m',
            'SELL': '\033[91m',
            'HOLD': '\033[93m',
            'RESET': '\033[0m'
        }

    def _get_current_price(self):
        try:
            df = self._fetch_latest_data()
            if df is not None and not df.empty:
                return df['close'].iloc[-1]
            return 0
        except Exception as e:
            print(f"Error getting current price: {e}")
            return 0

    def _print_status(self, action, price, confidence=None):
        color = self.status_colors.get(action, '')
        action_str = f"{color}{action}{self.status_colors['RESET']}"
        price_str = f"{price:.5f}"
        confidence_str = f" (Confidence: {confidence:.2f}%)" if confidence is not None else ""
        print(f"\n{'-' * 50}")
        print(f"|  Action: {action_str:<10} | Price: {price_str:<10}{confidence_str}")
        print(f"{'-' * 50}")
        print(f"|  {CRYPTO_NAME} Balance: {self.crypto_balance:.{CRYPTO_DECIMALS}f}")
        print(f"|  USDT Balance: {self.usdt_balance:.2f}")
        current_value = self.crypto_balance * price + self.usdt_balance
        print(f"|  Portfolio Value: {current_value:.2f} USDT")
        print(f"|  Trading Fees: {EXCHANGE_FEES['default'] * 100}% per trade")
        print(f"{'-' * 50}\n")

    def _fetch_latest_data(self):
        try:
            since_time = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS_WINDOW)).timestamp())
            df = self.public_api.get_ohlc_data(PAIR, interval=INTERVAL, since=since_time)[0]
            df.index = pd.to_datetime(df.index, unit='s')
            df = df[~df.index.duplicated(keep='first')]
            new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{INTERVAL}min')
            df = df.reindex(new_index)
            df = df.ffill().infer_objects()
            return df
        except Exception as e:
            print(f"Error fetching latest data: {e}")
            return None

    def _prepare_features(self, df):
        if len(df) < SEQUENCE_LENGTH:
            print(f"Not enough data for a sequence. Need {SEQUENCE_LENGTH}, got {len(df)}.")
            return None

        try:
            latest_data = df[FEATURES_LIST].tail(SEQUENCE_LENGTH).values
            scaled_features = self.scaler.transform(latest_data)
            features_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(self.device)
            return features_tensor
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None

    def _execute_trade(self, trade_type, amount, price=None):
        if price is None or price <= 0:
            print("Invalid price for trade execution")
            return False

        trade_time = datetime.now(timezone.utc)

        try:
            if trade_type == 'buy':
                cost_usdt = amount * price
                fee_amount = cost_usdt * EXCHANGE_FEES['default']
                total_cost = cost_usdt + fee_amount

                if cost_usdt < self.min_trade_amount:
                    print(f"Trade amount too small: {cost_usdt:.2f} USDT (minimum: {self.min_trade_amount} USDT)")
                    return False

                if self.usdt_balance >= total_cost:
                    self.usdt_balance -= total_cost
                    self.crypto_balance += amount
                    trade_record = {
                        'time': trade_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'BUY',
                        'amount': amount,
                        'price': price,
                        'cost': cost_usdt,
                        'fees': fee_amount,
                        'total_cost': total_cost
                    }
                    self.trade_history.append(trade_record)

                    if self.trading_writer:
                        self.trading_writer.add_scalar('Trades/Buy_Price', price, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Buy_Amount_Crypto', amount, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Buy_Cost_USDT', cost_usdt, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Buy_Fees_USDT', fee_amount, self.trade_step)

                    print(f"BUY executed: {amount:.4f} {CRYPTO_NAME} at ${price:.5f}")
                    print(f"   Fees: {fee_amount:.4f} USDT ({EXCHANGE_FEES['default'] * 100}%)")
                    return True
                else:
                    print(f"Insufficient USDT for BUY. Needed {total_cost:.2f} (incl. fees), have {self.usdt_balance:.2f}")
                    return False

            elif trade_type == 'sell':
                if amount * price < self.min_trade_amount:
                    print(f"Trade amount too small: {amount * price:.2f} USDT (minimum: {self.min_trade_amount} USDT)")
                    return False

                if self.crypto_balance >= amount:
                    proceeds = amount * price
                    fee_amount = proceeds * EXCHANGE_FEES['default']
                    net_proceeds = proceeds - fee_amount

                    self.crypto_balance -= amount
                    self.usdt_balance += net_proceeds
                    trade_record = {
                        'time': trade_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'SELL',
                        'amount': amount,
                        'price': price,
                        'proceeds': proceeds,
                        'fees': fee_amount,
                        'net_proceeds': net_proceeds
                    }
                    self.trade_history.append(trade_record)

                    if self.trading_writer:
                        self.trading_writer.add_scalar('Trades/Sell_Price', price, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Sell_Amount_Crypto', amount, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Sell_Proceeds_USDT', proceeds, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Sell_Fees_USDT', fee_amount, self.trade_step)

                    print(f"SELL executed: {amount:.4f} {CRYPTO_NAME} at ${price:.5f}")
                    print(f"   Fees: {fee_amount:.4f} USDT ({EXCHANGE_FEES['default'] * 100}%)")
                    return True
                else:
                    print(f"Insufficient {CRYPTO_NAME} for SELL. Needed {amount:.{CRYPTO_DECIMALS}f}, have {self.crypto_balance:.{CRYPTO_DECIMALS}f}")
                    return False
            else:
                print(f"Unknown trade type: {trade_type}")
                return False
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False

    def make_decision(self):
        current_time = datetime.now(timezone.utc)
        if (current_time - self.last_trade_time).total_seconds() < DECISION_INTERVAL_SECONDS:
            return

        print(f"\nMaking decision at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.last_trade_time = current_time

        try:
            df = self._fetch_latest_data()
            if df is None or df.empty:
                print("Could not fetch data or data is empty. Skipping decision.")
                return

            df = calculate_all_indicators(df.copy())
            if df.empty or len(df) < SEQUENCE_LENGTH:
                print("Insufficient data after indicator calculation. Skipping decision.")
                return

            latest_close = df['close'].iloc[-1]
            rsi_value = df[f"RSI_{INDICATOR_CONFIG['RSI']['length']}"].iloc[-1]

            if self.trading_writer:
                self.trading_writer.add_scalar('Market/Price', latest_close, self.trade_step)
                self.trading_writer.add_scalar('Market/Volume', df['volume'].iloc[-1], self.trade_step)
                self.trading_writer.add_scalar('Indicators/RSI', rsi_value, self.trade_step)

            features_tensor = self._prepare_features(df)
            if features_tensor is None:
                print("Could not prepare features. Skipping decision.")
                return

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[prediction] * 100

            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            predicted_action = action_map[prediction]

            if rsi_value >= INDICATOR_CONFIG['RSI']['overbought']:
                predicted_action = "SELL"
                print(f"RSI {rsi_value:.2f} >= {INDICATOR_CONFIG['RSI']['overbought']} (Overbought) - FORCING SELL")
            elif rsi_value <= INDICATOR_CONFIG['RSI']['oversold']:
                predicted_action = "BUY"
                print(f"RSI {rsi_value:.2f} <= {INDICATOR_CONFIG['RSI']['oversold']} (Oversold) - FORCING BUY")

            self._print_status(predicted_action, latest_close, confidence)

            trade_executed = False
            if predicted_action == "BUY":
                if latest_close > 0:
                    max_usdt_to_spend = self.usdt_balance * self.trade_percentage
                    adjusted_max_spend = max_usdt_to_spend / (1 + EXCHANGE_FEES['default'])
                    amount_to_buy = adjusted_max_spend / latest_close
                    trade_executed = self._execute_trade('buy', amount_to_buy, latest_close)
            elif predicted_action == "SELL":
                if latest_close > 0:
                    amount_to_sell = self.crypto_balance * self.trade_percentage
                    if amount_to_sell > 0:
                        trade_executed = self._execute_trade('sell', amount_to_sell, latest_close)

            current_value = self.crypto_balance * latest_close + self.usdt_balance
            profit_loss = current_value - self.initial_portfolio_value
            profit_loss_pct = (profit_loss / self.initial_portfolio_value) * 100 if self.initial_portfolio_value != 0 else 0

            if self.trading_writer:
                self.trading_writer.add_scalar('Portfolio/Total_Value_USDT', current_value, self.trade_step)
                self.trading_writer.add_scalar('Portfolio/Profit_Loss_USDT', profit_loss, self.trade_step)
                self.trading_writer.add_scalar('Portfolio/Profit_Loss_Pct', profit_loss_pct, self.trade_step)

            if trade_executed:
                self.trade_step += 1
            else:
                self.trade_step += 1

        except Exception as e:
            print(f"Error in decision making: {e}")

def fetch_historical_data(public_api, pair, interval, lookback_days):
    print(f"Fetching {lookback_days} days of historical data for {pair}...")
    since_time = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())

    try:
        ohlc_data, last = public_api.get_ohlc_data(pair, interval=interval, since=since_time)
        print(f"Retrieved {len(ohlc_data)} candles of historical data")
        return ohlc_data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def train_model(train_data, val_data, scaler, current_device, learning_rate, run_idx=0):
    num_features = train_data.features.shape[-1]
    model = CryptoTransformer(num_features=num_features, dropout_rate=DROPOUT_RATE)
    model.to(current_device)

    run_output_dir = f'./results/run_{run_idx}'
    run_logging_dir = f'./logs/run_{run_idx}'
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_logging_dir, exist_ok=True)

    try:
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            logging_dir=run_logging_dir,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            weight_decay=WEIGHT_DECAY
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            evaluation_strategy="steps",
            eval_steps=EVAL_STEPS,
            logging_dir=run_logging_dir,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            weight_decay=WEIGHT_DECAY
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data
    )

    train_result = trainer.train()
    eval_results = trainer.evaluate()

    model_path = os.path.join(run_output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    scaler_path = os.path.join(run_output_dir, f'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    val_results = trainer.evaluate()
    print(f"Final validation loss: {val_results['eval_loss']:.6f}")

    return model_path, scaler_path, val_results['eval_loss']

def train_with_double_optimization(df, device, public_api):
    best_eval_loss = float('inf')
    best_model_info = None

    total_combinations = len(ATR_MULTIPLIERS_TO_TEST) * len(LEARNING_RATES_TO_TEST)
    current_combination = 0

    for multiplier in ATR_MULTIPLIERS_TO_TEST:
        print(f"\n=== Testing ATR Multiplier: {multiplier} ===")

        df_temp = df.copy()
        df_temp['future_price'] = df_temp['close'].shift(-SEQUENCE_LENGTH)

        df_temp['threshold'] = df_temp[f"ATRr_{INDICATOR_CONFIG['ATR']['length']}"] * multiplier
        df_temp['buy_threshold_with_fees'] = df_temp['close'] + df_temp['threshold'] + (df_temp['close'] * EXCHANGE_FEES['default'])
        df_temp['sell_threshold_with_fees'] = df_temp['close'] - df_temp['threshold'] - (df_temp['close'] * EXCHANGE_FEES['default'])

        df_temp['target'] = 1
        df_temp.loc[df_temp['future_price'] > df_temp['buy_threshold_with_fees'], 'target'] = 2
        df_temp.loc[df_temp['future_price'] < df_temp['sell_threshold_with_fees'], 'target'] = 0
        df_temp.dropna(inplace=True)

        features = df_temp[FEATURES_LIST].values
        targets = df_temp['target'].values

        split_idx = int(len(features) * 0.8)
        scaler = StandardScaler().fit(features[:split_idx])
        features = scaler.transform(features)

        for learning_rate in LEARNING_RATES_TO_TEST:
            current_combination += 1
            progress = int((current_combination / total_combinations) * 100)

            app_state['current_multiplier'] = str(multiplier)
            app_state['current_learning_rate'] = f"{learning_rate:.2e}"
            app_state['training_progress'] = progress
            app_state['optimization_phase'] = f"Testing ATR {multiplier}, LR {learning_rate:.2e}"

            print(f"\n--- Testing Learning Rate: {learning_rate:.2e} (Progress: {progress}%) ---")
            print(f"   Fee-aware training: {EXCHANGE_FEES['default'] * 100}% fees included in targets")

            model_infos = []
            for run in range(NUM_TRAINING_RUNS):
                app_state['current_run'] = run + 1
                app_state['total_runs'] = NUM_TRAINING_RUNS

                print(f"Training Run {run + 1}/{NUM_TRAINING_RUNS}")
                train_data = TradingDataset(features[:split_idx], targets[:split_idx], SEQUENCE_LENGTH)
                val_data = TradingDataset(features[split_idx:], targets[split_idx:], SEQUENCE_LENGTH)

                model_path, scaler_path, eval_loss = train_model(
                    train_data, val_data, scaler, device, learning_rate,
                    run_idx=f"ATR{multiplier}_LR{learning_rate:.2e}_run{run}"
                )

                app_state['current_loss'] = eval_loss
                model_infos.append({
                    'model_path': model_path,
                    'scaler_path': scaler_path,
                    'eval_loss': eval_loss,
                    'multiplier': multiplier,
                    'learning_rate': learning_rate
                })

            current_best = min(model_infos, key=lambda x: x['eval_loss'])
            print(f"Best for ATR {multiplier}, LR {learning_rate:.2e}: Loss {current_best['eval_loss']:.6f}")

            if current_best['eval_loss'] < best_eval_loss:
                best_eval_loss = current_best['eval_loss']
                best_model_info = current_best
                app_state['best_multiplier'] = str(best_model_info['multiplier'])
                app_state['best_learning_rate'] = f"{best_model_info['learning_rate']:.2e}"

    if best_model_info:
        shutil.copy(best_model_info['model_path'], BEST_MODEL_PATH)
        shutil.copy(best_model_info['scaler_path'], BEST_SCALER_PATH)
        with open(BEST_MULTIPLIER_PATH, 'w') as f:
            f.write(str(best_model_info['multiplier']))
        with open(BEST_LEARNING_RATE_PATH, 'w') as f:
            f.write(str(best_model_info['learning_rate']))

        print(f"\nBEST COMBINATION FOUND!")
        print(f"ATR Multiplier: {best_model_info['multiplier']}")
        print(f"Learning Rate: {best_model_info['learning_rate']:.2e}")
        print(f"Best validation loss: {best_eval_loss:.6f}")
        print(f"Fee-aware training: {EXCHANGE_FEES['default'] * 100}% fees included")

        return best_model_info['multiplier'], best_model_info['learning_rate']
    return None, None

def _get_kraken_balance_code(symbol):
    kraken_codes = {
        'DOGE': 'XXDG',
        'BTC': 'XBT',
        'ETH': 'ETH',
        'USDT': 'USDT'
    }
    return kraken_codes.get(symbol.upper(), symbol.upper())

def _get_live_balances(krakenex_api):
    print("Fetching live balances from Kraken...")
    crypto_balance, usdt_balance = 0.0, 0.0
    try:
        balance_data = krakenex_api.query_private('Balance')
        if 'error' in balance_data and balance_data['error']:
            raise Exception(f"Kraken API error: {balance_data['error']}")

        balances = balance_data['result']
        print(f"Raw balance data from API: {balances}")

        crypto_code = _get_kraken_balance_code(CRYPTO_NAME)

        crypto_balance = float(balances.get(crypto_code, balances.get(CRYPTO_NAME, 0.0)))
        usdt_balance = float(balances.get('USDT', 0.0))

        print("Successfully fetched balances:")
        print(f"  {CRYPTO_NAME} Balance: {crypto_balance:.{CRYPTO_DECIMALS}f}")
        print(f"  USDT Balance: {usdt_balance:.2f}")

    except Exception as e:
        print(f"Failed to fetch live balances: {e}")
        print("WARNING: Falling back to paper trading balances. Trading will NOT be live.")
        crypto_balance = INITIAL_CRYPTO_BALANCE
        usdt_balance = INITIAL_USDT_BALANCE

    return crypto_balance, usdt_balance

def initial_checks():
    print(f"Using device: {device}")

    mode = "LIVE TRADING" if LIVE_TRADING else "PAPER TRADING"
    print(f"Trading mode set to: {mode}")

    if LIVE_TRADING:
        print("WARNING: LIVE TRADING MODE - REAL MONEY AT RISK!")
        if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
            raise ValueError("API keys must be set in the .env file for live trading.")

    krakenex_api = krakenex.API(key=KRAKEN_API_KEY, secret=KRAKEN_API_SECRET)
    pykrakenapi_api = KrakenAPI(krakenex_api)

    crypto_balance = INITIAL_CRYPTO_BALANCE
    usdt_balance = INITIAL_USDT_BALANCE

    if LIVE_TRADING:
        print("Verifying API keys...")
        try:
            pykrakenapi_api.get_server_time()
            print("Kraken API keys are valid.")
            crypto_balance, usdt_balance = _get_live_balances(krakenex_api)
        except Exception as e:
            raise ValueError(f"Failed to connect to Kraken API. Please check your API keys and permissions: {e}")
    else:
        print("Kraken API initialized for PAPER TRADING.")
        print(f"Paper trading balances initialized: {CRYPTO_NAME}={crypto_balance:.{CRYPTO_DECIMALS}f}, USDT={usdt_balance:.2f}")

    return pykrakenapi_api, krakenex_api, crypto_balance, usdt_balance

def save_trade_history(trader, filename='trade_history.csv'):
    try:
        if not trader.trade_history:
            print("No trade history to save")
            return

        df = pd.DataFrame(trader.trade_history)
        df.to_csv(filename, index=False)
        print(f"Trade history saved to {filename}")
    except Exception as e:
        print(f"Error saving trade history: {e}")

def run_flask_app():
    print(f"Starting Flask web server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def trading_loop():
    global app_state

    public_api, private_api, crypto_balance_init, usdt_balance_init = initial_checks()

    current_time = datetime.now(timezone.utc)
    print(f"\nStarting script at {current_time}")
    print(f"Trading pair: {PAIR}")
    print(f"AGGRESSIVE TRADING MODE: {TRADE_PERCENTAGE * 100}% position size")
    print(f"FEE-AWARE TRADING: {EXCHANGE_FEES['default'] * 100}% fees included in all calculations")

    TRADE_LOG_DIR = './trading_logs'
    trading_writer = SummaryWriter(log_dir=TRADE_LOG_DIR)
    print(f"TensorBoard trading logs will be saved to: {TRADE_LOG_DIR}")

    app_state['last_training_time'] = datetime.now(timezone.utc)
    app_state['next_training_time'] = app_state['last_training_time'] + timedelta(hours=RETRAIN_INTERVAL_HOURS)

    try:
        trained_model = None
        final_scaler = None
        TARGET_ATR_MULTIPLIER = DEFAULT_ATR_MULTIPLIER
        TARGET_LEARNING_RATE = 1e-5

        if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_SCALER_PATH):
            try:
                num_features_for_model = len(FEATURES_LIST)
                trained_model = CryptoTransformer(num_features=num_features_for_model, dropout_rate=DROPOUT_RATE)
                trained_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
                trained_model.to(device)
                trained_model.eval()
                final_scaler = joblib.load(BEST_SCALER_PATH)

                if os.path.exists(BEST_MULTIPLIER_PATH):
                    with open(BEST_MULTIPLIER_PATH, 'r') as f:
                        TARGET_ATR_MULTIPLIER = float(f.read())
                        app_state['best_multiplier'] = str(TARGET_ATR_MULTIPLIER)

                if os.path.exists(BEST_LEARNING_RATE_PATH):
                    with open(BEST_LEARNING_RATE_PATH, 'r') as f:
                        TARGET_LEARNING_RATE = float(f.read())
                        app_state['best_learning_rate'] = f"{TARGET_LEARNING_RATE:.2e}"

                print(f"Loaded existing best model (ATR Multiplier: {TARGET_ATR_MULTIPLIER}, LR: {TARGET_LEARNING_RATE:.2e})")
            except Exception as e:
                print(f"Error loading existing model/scaler: {e}")
                trained_model = None
                final_scaler = None

        current_time = datetime.now(timezone.utc)
        needs_retraining = (trained_model is None or
                            current_time - app_state['last_training_time'] >= timedelta(hours=RETRAIN_INTERVAL_HOURS))

        if needs_retraining:
            print("Starting DOUBLE OPTIMIZATION (ATR Multipliers + Learning Rates)...")
            print(f"   Fee-aware training: {EXCHANGE_FEES['default'] * 100}% fees included in optimization")
            app_state['is_training'] = True
            app_state['training_status'] = 'Double Optimization Running'
            app_state['training_progress'] = 0
            app_state['optimization_phase'] = 'Starting optimization...'

            try:
                historical_data = fetch_historical_data(public_api, PAIR, INTERVAL, LOOKBACK_DAYS_TRAINING)

                if historical_data is None or historical_data.empty:
                    raise RuntimeError("Failed to fetch historical data for training")

                df = historical_data.copy()
                df.index = pd.to_datetime(df.index, unit='s')
                df = df[~df.index.duplicated(keep='first')]
                new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{INTERVAL}min')
                df = df.reindex(new_index)
                df = df.ffill().infer_objects()

                df = calculate_all_indicators(df)
                df.dropna(inplace=True)

                temp_multiplier, temp_learning_rate = train_with_double_optimization(df, device, public_api)
                if temp_multiplier is not None and temp_learning_rate is not None:
                    TARGET_ATR_MULTIPLIER = temp_multiplier
                    TARGET_LEARNING_RATE = temp_learning_rate
                else:
                    TARGET_ATR_MULTIPLIER = DEFAULT_ATR_MULTIPLIER
                    TARGET_LEARNING_RATE = 1e-5

                if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_SCALER_PATH):
                    trained_model = CryptoTransformer(num_features=len(FEATURES_LIST), dropout_rate=DROPOUT_RATE)
                    trained_model.load_state_dict(torch.load(BEST_MODEL_PATH))
                    trained_model.to(device)
                    final_scaler = joblib.load(BEST_SCALER_PATH)
                else:
                    raise RuntimeError("Training search failed and did not save a model.")

                app_state['last_training_time'] = datetime.now(timezone.utc)
                app_state['next_training_time'] = app_state['last_training_time'] + timedelta(hours=RETRAIN_INTERVAL_HOURS)
                app_state['is_training'] = False
                app_state['training_status'] = 'Double Optimization Complete'
                app_state['training_progress'] = 100
                app_state['optimization_phase'] = 'Optimization Complete'

            except Exception as e:
                app_state['training_status'] = f'Training Failed: {e}'
                app_state['is_training'] = False
                raise RuntimeError(f"Failed during training: {e}")

        trader = CryptoTrader(trained_model, public_api, private_api, final_scaler, device, live_trading=LIVE_TRADING,
                              crypto_balance=crypto_balance_init, usdt_balance=usdt_balance_init,
                              trading_writer=trading_writer)

        app_state['trader'] = trader

        print(f"\nStarting TRADING with optimized parameters:")
        print(f"   ATR Multiplier: {TARGET_ATR_MULTIPLIER}")
        print(f"   Learning Rate: {TARGET_LEARNING_RATE:.2e}")
        print(f"   Trade Size: {TRADE_PERCENTAGE * 100}% of balance")
        print(f"   Trading Fees: {EXCHANGE_FEES['default'] * 100}% per trade")
        print("Starting trading loop... (Ctrl+C to stop)")

        try:
            while True:
                trader.make_decision()
                time.sleep(SLEEP_TIME_SECONDS)
        except KeyboardInterrupt:
            print("\nStopping trading bot...")
            save_trade_history(trader)
    finally:
        trading_writer.close()
        print("TensorBoard trading writer closed.")

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    trading_loop()