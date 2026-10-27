# Crowding Terminal｜加密永續合約擁擠度反轉交易系統

ML4T 期中專題「Crypto Perps Crowding Reversal」的互動式前端：行情總覽與篩選、K 線（1H/4H/1D/1W）、
每日選股排行、回測實驗室、即時訊號、模型診斷、風險監控、2025 封存期驗證、策略規格。

- 資料：Binance USDT-M 公開資料（data.binance.vision），19 檔永續合約，2020-01 ～ 2025-12。
- 前端只讀已存好的資料與研究結果（`midterm_crowding_reversal/outputs/`），不訓練模型。
- 本機執行：`pip install -r requirements.txt`，再 `streamlit run midterm_crowding_reversal/app/terminal.py`。
