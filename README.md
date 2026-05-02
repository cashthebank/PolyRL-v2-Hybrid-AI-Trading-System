🤖 PolyRL v2 — Hybrid AI Trading System for Prediction Markets
Author: Aarya Jindal | BITS Pilani (F20221100)
Project: Windfall Research Internship — RL Agents for Prediction Market Trading

Python PyTorch Stable Baselines3 Streamlit License: MIT

A 3-layer hybrid AI system that combines Anomaly Detection, Supervised ML, and Reinforcement Learning to trade on Polymarket prediction markets.

📌 What is This?
Polymarket is a prediction market where you can trade on real-world events (elections, crypto prices, geopolitics). Each market has a YES token priced between $0 and $1 — if the event happens, it pays $1.

PolyRL v2 is an AI system that watches these markets and decides when to Buy, Sell, or Hold using a 3-layer AI pipeline:

  Live Market Data (Polymarket + News)
            ↓
  Layer 1: Autoencoder — "Is something weird happening?"
            ↓
  Layer 2: Random Forest + SVM — "Will price go up or down?"
            ↓
  Layer 3: RL Agent (DQN / PPO / A2C) — "Should I BUY, SELL, or HOLD?"
            ↓
  Streamlit Dashboard — Live monitoring & results
🖼️ Dashboard Screenshots
🌐 Live Markets — Real-time Polymarket Data
Live Markets

Fetches the top active prediction markets by 24h volume directly from Polymarket's API. The biggest market shown: "Will Bitcoin hit $150k by June 30, 2026?" with $5.8M in 24h volume.

🔍 Anomaly Monitor — Autoencoder Detection
Anomaly Monitor

The Autoencoder detects unusual orderbook patterns (whale orders, liquidity crises, arbitrage windows). The purple line shows the anomaly score over time — spikes above the red threshold line trigger alerts (marked with ✕).

🌲 RF/SVM Signals — Supervised ML Predictions
RF SVM Signals

The Random Forest outputs a probability (0–1) of price going up in the next 5 steps. The blue line is the actual YES price, the green dotted line is the RF bullish confidence. RF mean confidence: 0.509 | Signal agreement between RF & SVM: 50%.

🤖 RL Architecture Comparison — DQN vs PPO vs A2C
RL Comparison

Simulated equity curves comparing the three RL agents. PPO (purple) showed the most stable growth. A2C (green) was more volatile. DQN (blue) underperformed on limited training.

Architecture Notes

📋 Backtest Results — Real Trained Model Performance
Backtest Results

Model	Total Return	Ann. Sharpe	Max Drawdown	Win Rate	Final Value
🥇 RF Signal	+7.47%	4.35	-6.50%	18.99%	$1,074.73
🥈 PPO	+2.12%	2.09	-4.68%	17.88%	$1,021.17
A2C	0.00%	0.00	0.00%	0.00%	$1,000.00
Buy & Hold	-13.63%	-2.39	-28.54%	51.76%	$863.73
DQN	-5.85%	-4.94	-10.43%	15.64%	$941.49
Random	-12.95%	-11.40	-15.65%	25.14%	$870.49
Charts

Both DQN and PPO heavily favoured SELL actions (~95-100% of trades), indicating the models need more training steps to develop a balanced strategy.

🏗️ System Architecture
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│   Polymarket CLOB REST + WebSocket Ticks + GDELT Sentiment  │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 1 — ANOMALY DETECTION                    │
│   Autoencoder on 40-dim orderbook depth vectors             │
│   High reconstruction error = unusual market condition      │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│           LAYER 2 — SUPERVISED SIGNAL GENERATION            │
│   Random Forest (300 trees) + SVM (RBF kernel)              │
│   Predicts: will price be higher in 5 steps?                │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 3 — RL EXECUTION AGENT                   │
│   DQN vs PPO vs A2C — trained for 25,000 steps              │
│   Actions: BUY YES / SELL / HOLD                            │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│            STREAMLIT DASHBOARD (via ngrok)                  │
│   5 tabs: Markets · Anomaly · Signals · RL · Backtest       │
└─────────────────────────────────────────────────────────────┘
📡 APIs Used
API	URL	Auth Required	Purpose
Gamma API	gamma-api.polymarket.com	❌ Free	Market discovery & metadata
CLOB REST	clob.polymarket.com	❌ Read / ✅ Trade	Prices, orderbook, history
CLOB WebSocket	ws-subscriptions-clob.polymarket.com	❌	Real-time tick stream
GDELT DOC 2.0	api.gdeltproject.org	❌ Free	News sentiment scoring
🧠 Models Explained
Layer 1 — Autoencoder
Input: 40-dimensional orderbook vector (10 bid prices + 10 bid sizes + 10 ask prices + 10 ask sizes)
Architecture: Linear encoder (40→32→16→8) + decoder (8→16→32→40)
Output: Anomaly score 0–1 per market snapshot
Training: 60 epochs on real or synthetic orderbook snapshots
Layer 2 — Random Forest + SVM
Label: 1 if price[t+5] > price[t], else 0
Features: 8 technical indicators + anomaly score = 9 features
RF: 300 estimators, max depth 8, balanced class weights
SVM: RBF kernel, StandardScaler pipeline
Split: Walk-forward (75% train / 25% test, no shuffling)
Layer 3 — RL Agents
Agent	Key Strength	Hyperparams
DQN	Experience replay, Q-value learning	LR=1e-4, buffer=20k, γ=0.99
PPO	Stable policy updates, clip=0.2	LR=3e-4, n_steps=256, epochs=10
A2C	Actor-critic, low variance	LR=7e-4, n_steps=128, γ=0.99
All agents use a [256, 256, 128] MLP policy network and train for 25,000 timesteps.

🚀 Quick Start
Run in Google Colab (Recommended)
Open In Colab

Open the notebook in Colab
Go to Runtime → Change runtime type → T4 GPU for faster training
Fill in your ngrok token in Section 2 (get it free at dashboard.ngrok.com)
Run all cells top to bottom
Open the ngrok URL printed at the end of Section 10
Keys Required
Key	Where to Get	Required?
NGROK_AUTH_TOKEN	dashboard.ngrok.com — free	✅ For dashboard
POLYMARKET_PRIVATE_KEY	MetaMask → Export Private Key	❌ Section 11 only
📁 Project Structure
PolyRL-v2/
│
├── PolyRL_v2_Hybrid_AI.ipynb   # Main notebook (run this)
├── README.md                    # This file
├── screenshots/                 # Dashboard screenshots
│   ├── live_markets.png
│   ├── anomaly_monitor.png
│   ├── rf_svm_signals.png
│   ├── rl_comparison.png
│   ├── architecture_notes.png
│   ├── backtest_results.png
│   └── charts.png
│
└── results/                     # Generated after running notebook
    ├── all_metrics.csv          # Backtest comparison table
    ├── full_comparison.png      # Equity curves chart
    ├── ae_loss.png              # Autoencoder training loss
    └── rf_importance.png        # Feature importance chart
📊 Key Results
✅ RF Signal was the top performer with +7.47% return and Sharpe ratio of 4.35
✅ PPO was the best RL agent with +2.12% return — beat Buy & Hold by ~15%
✅ Both RL agents beat the Random baseline significantly
⚠️ DQN and A2C need more training steps (try TOTAL_STEPS = 100_000) to reach full potential
📌 The Autoencoder successfully detected 3 anomaly events in the demo data
🛠️ Tech Stack
Category	Libraries
RL	stable-baselines3, gymnasium
Deep Learning	PyTorch
Supervised ML	scikit-learn (Random Forest, SVM)
Technical Analysis	ta (RSI, EMA, Bollinger Bands)
Data	pandas, numpy, pyarrow
News Sentiment	gdeltdoc
Dashboard	streamlit, plotly, pyngrok
Market Data	requests, websockets, py-clob-client
⚠️ Disclaimer
This project is for educational and research purposes only. It is not financial advice. Prediction market trading involves real financial risk. Never trade with money you cannot afford to lose. The live trading section (Section 11) is commented out by default for safety.

📚 References
Polymarket Docs
py-clob-client
GDELT Project
Stable Baselines3 Docs
ngrok Dashboard
