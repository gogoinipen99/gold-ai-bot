import os
import requests
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

TOPIC = "gold_ai_alerts_9876"
s = chr(47)
NTFY_URL = "https:" + s + s + "ntfy.sh" + s + TOPIC

DATA_FILE = "gold_trades_memory.csv"
MODEL_FILE = "gold_ai_brain.pkl"

def send_alert(msg):
    try:
        requests.post(NTFY_URL, data=msg.encode('utf-8'), timeout=5)
    except Exception as e:
        print("Alert Error:", e)

def train_ai_model():
    if not os.path.exists(DATA_FILE):
        return 0
    df = pd.read_csv(DATA_FILE)
    if len(df) < 15:
        return len(df)
    df = df.tail(150).copy()
    df['dir_num'] = df['dir'].map({'BUY': 1, 'SELL': 0})
    df['type_num'] = df['type'].map({'BREAKOUT': 1, 'PULLBACK': 0})
    features = ['dir_num', 'type_num', 'rsi', 'adx', 'atr', 'vol_ratio', 'mtf_score']
    X = df[features]
    y = df['win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return "UPDATED"

@app.route('/', methods=['GET'])
def home():
    return "Gold AI Server is Alive!", 200

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if not data:
        return jsonify({"status": "no data"}), 400
    event = data.get('event')
    if event == "ENTRY":
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
            dir_num = 1 if data['dir'] == 'BUY' else 0
            type_num = 1 if data['type'] == 'BREAKOUT' else 0
            features = [[dir_num, type_num, data['rsi'], data['adx'], data['atr'], data['vol_ratio'], data['mtf_score']]]
            win_prob = model.predict_proba(features) * 100
            if win_prob >= 60.0:
                msg = f"AI APPROVED ({win_prob:.1f}%)\nGold {data['dir']} ({data['type']})\nEntry: {data['entry']} | SL: {data['sl']} | TP: {data['tp']}"
                send_alert(msg)
            else:
                msg = f"AI SKIPPED ({win_prob:.1f}%)\nGold {data['dir']} ({data['type']}) @ {data['entry']}"
                send_alert(msg)
        else:
            msg = f"AI LEARNING MODE\nGold {data['dir']} ({data['type']}) @ {data['entry']}\nSL: {data['sl']} | TP: {data['tp']}"
            send_alert(msg)
    elif event == "RESULT":
        new_row = pd.DataFrame([data])
        if os.path.exists(DATA_FILE):
            new_row.to_csv(DATA_FILE, mode='a', header=False, index=False)
        else:
            new_row.to_csv(DATA_FILE, mode='w', header=True, index=False)
        status = train_ai_model()
        res_icon = "WIN" if data['win'] == 1 else "LOSS"
        if status == "UPDATED":
            send_alert(f"{res_icon} (PnL: {data['pnl']})\nAI Brain updated with new trade!")
        else:
            send_alert(f"{res_icon} (PnL: {data['pnl']})\nAI Memory: {status}/15 trades.")
    return jsonify({"status": "ok"}), 200
