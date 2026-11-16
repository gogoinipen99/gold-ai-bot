import os
import json
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

DATA_FILE = "gold_ai_memory.csv"
MODEL_FILE = "gold_ai_brain.pkl"
STATE_FILE = "active_trade.json"
IST = pytz.timezone("Asia/Kolkata")

def calculate_indicators(df):
    df['ema9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['ema600'] = df['Close'].ewm(span=600, adjust=False).mean()
    
    # ATR (14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ma'] = df['atr'].rolling(20).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ADX (14)
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(14).mean()
    
    df['vol_ma'] = df['Volume'].rolling(20).mean()
    df['high20'] = df['High'].rolling(20).max()
    df['low20'] = df['Low'].rolling(20).min()
    return df

def train_ai():
    if not os.path.exists(DATA_FILE):
        return 0
    df = pd.read_csv(DATA_FILE)
    if len(df) < 15:
        return len(df)
    df = df.tail(150).copy()
    df['dir_num'] = df['dir'].map({'BUY': 1, 'SELL': 0})
    df['type_num'] = df['type'].map({'BREAKOUT': 1, 'PULLBACK': 0})
    features = ['dir_num', 'type_num', 'rsi', 'adx', 'atr', 'vol_ratio', 'hour']
    X = df[features].fillna(0)
    y = df['win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return "ACTIVE"

def get_ai_probability(trade_data):
    if not os.path.exists(MODEL_FILE):
        return None
    model = joblib.load(MODEL_FILE)
    dir_num = 1 if trade_data['dir'] == 'BUY' else 0
    type_num = 1 if trade_data['type'] == 'BREAKOUT' else 0
    feats = [[dir_num, type_num, trade_data['rsi'], trade_data['adx'], trade_data['atr'], trade_data['vol_ratio'], trade_data['hour']]]
    prob = model.predict_proba(feats)[0][1] * 100
    return round(prob, 1)

@app.route('/scan', methods=['GET'])
def scan_market():
    now_ist = datetime.now(IST)
    # Gold Futures (GC=F) 5m data for live scanning
    df = yf.download("GC=F", period="5d", interval="5m", progress=False)
    if df.empty or len(df) < 605:
        return jsonify({"status": "waiting for market data"}), 200
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = calculate_indicators(df)
    curr = df.iloc[-2] # Last confirmed candle
    prev = df.iloc[-3]
    
    # 1. Check Active Virtual Trade Outcome
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            trade = json.load(f)
        
        result = None
        if trade['dir'] == 'BUY':
            if curr['High'] >= trade['tp']:
                result = 1
            elif curr['Low'] <= trade['sl']:
                result = 0
        elif trade['dir'] == 'SELL':
            if curr['Low'] <= trade['tp']:
                result = 1
            elif curr['High'] >= trade['sl']:
                result = 0
                
        if result is not None:
            trade['win'] = result
            trade['closed_at'] = now_ist.strftime("%Y-%m-%d %H:%M")
            row = pd.DataFrame([trade])
            row.to_csv(DATA_FILE, mode='a', header=not os.path.exists(DATA_FILE), index=False)
            os.remove(STATE_FILE)
            train_ai()
            return jsonify({"status": "Trade Closed & AI Updated", "win": result}), 200
        return jsonify({"status": "Trade Active", "trade": trade}), 200

    # 2. Check Golden Hours & Skip Monday
    if now_ist.weekday() == 0: # Monday
        return jsonify({"status": "Monday Skipped"}), 200
    
    t_str = now_ist.strftime("%H%M")
    in_golden = ("1500" <= t_str <= "1530") or ("1800" <= t_str <= "2130")
    if not in_golden:
        return jsonify({"status": "Off Golden Hours", "ist_time": now_ist.strftime("%H:%M")}), 200

    # 3. Check Strategy Signals
    strong_trend = (curr['adx'] >= 25) and (curr['atr'] > curr['atr_ma'] * 0.9)
    ema_up = (curr['ema21'] > prev['ema21']) and (curr['ema200'] > prev['ema200'])
    ema_dn = (curr['ema21'] < prev['ema21']) and (curr['ema200'] < prev['ema200'])
    
    body = abs(curr['Close'] - curr['Open'])
    rng = curr['High'] - curr['Low']
    vol_ratio = (curr['Volume'] / curr['vol_ma']) if curr['vol_ma'] > 0 else 1.0
    
    s1_buy = (curr['ema9'] > curr['ema21'] > curr['ema200']) and (curr['Close'] > curr['ema600']) and strong_trend and ema_up and (curr['Low'] <= curr['ema21']*1.0005) and (curr['Close'] > curr['ema21']) and (curr['Close'] > curr['Open']) and (body > rng*0.5) and (55 < curr['rsi'] < 70)
    s1_sell = (curr['ema9'] < curr['ema21'] < curr['ema200']) and (curr['Close'] < curr['ema600']) and strong_trend and ema_dn and (curr['High'] >= curr['ema9']) and (curr['Close'] < curr['ema21']) and (curr['Close'] < curr['Open']) and (body > rng*0.45) and (28 < curr['rsi'] < 46)
    
    big_body = (body > curr['atr'] * 1.1) and (vol_ratio > 1.2)
    s2_buy = (curr['High'] >= curr['high20']) and (curr['Close'] > curr['ema200']) and big_body and (60 < curr['rsi'] < 78)
    s2_sell = (curr['Low'] <= curr['low20']) and (curr['Close'] < curr['ema200']) and big_body and (22 < curr['rsi'] < 40)
    
    direction = None
    strat_type = None
    if s1_buy or s2_buy:
        direction = "BUY"
        strat_type = "PULLBACK" if s1_buy else "BREAKOUT"
        risk = curr['atr'] * 2.0
        sl = curr['Close'] - risk
        tp = curr['Close'] + (risk * 1.4)
    elif s1_sell or s2_sell:
        direction = "SELL"
        strat_type = "PULLBACK" if s1_sell else "BREAKOUT"
        risk = curr['atr'] * 1.7
        sl = curr['Close'] + risk
        tp = curr['Close'] - (risk * 1.4)
        
    if direction:
        new_trade = {
            "dir": direction,
            "type": strat_type,
            "entry": round(float(curr['Close']), 2),
            "sl": round(float(sl), 2),
            "tp": round(float(tp), 2),
            "rsi": round(float(curr['rsi']), 2),
            "adx": round(float(curr['adx']), 2),
            "atr": round(float(curr['atr']), 2),
            "vol_ratio": round(float(vol_ratio), 2),
            "hour": now_ist.hour,
            "opened_at": now_ist.strftime("%Y-%m-%d %H:%M")
        }
        new_trade["ai_win_prob"] = get_ai_probability(new_trade)
        with open(STATE_FILE, "w") as f:
            json.dump(new_trade, f)
        return jsonify({"status": "New Virtual Trade Opened", "trade": new_trade}), 200

    return jsonify({"status": "Scanning - No Setup Yet", "ist_time": now_ist.strftime("%H:%M")}), 200

@app.route('/', methods=['GET'])
@app.route('/status', methods=['GET'])
def status():
    total_trades = 0
    win_rate = 0
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        total_trades = len(df)
        if total_trades > 0:
            win_rate = round((df['win'].sum() / total_trades) * 100, 1)
            
    active = None
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            active = json.load(f)
            
    return jsonify({
        "AI_Status": "BRAIN ACTIVE" if os.path.exists(MODEL_FILE) else f"LEARNING ({total_trades}/15 trades)",
        "Total_Learned_Trades": total_trades,
        "Historical_Win_Rate": f"{win_rate}%",
        "Current_Active_Setup": active
    }), 200
