import streamlit as st
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Sniper AI Monitor - Sincronizado", layout="centered")

# =========================================================
# 1. CONFIGURAÇÕES IDENTICAS AO MT5 (WIN.txt)
# =========================================================
BETAS_WIN = {'^GSPC': 1.2, 'USDBRL=X': -1.0, 'USDMXN=X': -0.5, '^TNX': -0.4, 'EWZ': 1.0}
TICKERS_MACRO = list(BETAS_WIN.keys())

# Parâmetros de Entrada (Sincronizados com Inputs do MT5)
INP_WAIT_CANDLES = 5       # 
INP_BAND_BUFFER  = 10      # [cite: 13]
INP_BREAKOUT     = 20      # [cite: 14]
INP_RSI_UPPER    = 70      # [cite: 16]
INP_RSI_LOWER    = 30      # [cite: 17]
INP_TAKE_POINTS  = 300     # [cite: 21]
INP_STOP_POINTS  = 1500    # [cite: 20]
INP_PARTIAL_PTS  = 50      # [cite: 24]
INP_PARTIAL_VOL  = 1.0     # [cite: 25]

# =========================================================
# 2. MOTOR DE DADOS QUANT
# =========================================================
class DataEngine:
    def get_market_data(self, symbol="BOVA11.SA", interval="1m", n_bars=100):
        try:
            data = yf.download(symbol, period='7d', interval=interval, progress=False)
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            return data.tail(n_bars)
        except: return pd.DataFrame()

    def get_macro_and_vol(self):
        """Calcula Score, Shift e Volatilidade igual ao Sniper_Data_Feed.py"""
        macro_results = {}
        vol = 0.0035
        try:
            # Pega dados de 1m para bater com o Data_Feed.py
            data = yf.download(TICKERS_MACRO, period="2d", interval="1m", progress=False)
            if not data.empty:
                df_close = data['Close'].ffill()
                # Cálculo de Shift e Score
                changes = (df_close.iloc[-1] / df_close.iloc[-2]) - 1
                shift_total = sum(changes[t] * BETAS_WIN[t] for t in TICKERS_MACRO if t in changes)
                score = sum(1 if (changes[t]*BETAS_WIN[t]) > 0.001 else -1 if (changes[t]*BETAS_WIN[t]) < -0.001 else 0 for t in TICKERS_MACRO if t in changes)
                # Cálculo de Volatilidade
                vol = df_close.iloc[:,0].pct_change().std()
                return int(max(min(score, 5), -5)), shift_total, vol
        except: pass
        return 0, 0.0, 0.0035

    def get_prev_day_close(self, symbol="BOVA11.SA"):
        """Busca o fechamento de ontem para o Preço Justo """
        df = yf.download(symbol, period="2d", interval="1d", progress=False)
        return df['Close'].iloc[-2] if len(df) >= 2 else df['Close'].iloc[-1]

# =========================================================
# 3. GESTÃO DE ESTADO (MEMÓRIA DO ROBÔ)
# =========================================================
if 'sim_active' not in st.session_state:
    st.session_state.update({
        'sim_active': False, 'trades_history': [], 'total_points': 0.0,
        'wins': 0, 'losses': 0, 'pending_side': 0, 'wait_counter': 0,
        'trigger_price': 0.0, 'last_profit_time': None, 'peak_price': 0.0,
        'partial_done': False, 'current_lots': 2.0, 'profit_closed': 0.0
    })

# =========================================================
# 4. LÓGICA DE EXECUÇÃO (CÓPIA DO OnTick DO MT5)
# =========================================================
def main():
    engine = DataEngine()
    df_m1 = engine.get_market_data()
    prev_close = engine.get_prev_day_close()
    score, shift, vol = engine.get_macro_and_vol()

    if df_m1.empty: return

    # --- CÁLCULO DAS LINHAS QUANT (IDENTICO AO MT5) ---
    fair_value = prev_close * (1.0 + shift) # [cite: 557]
    daily_vol_pts = fair_value * vol        # [cite: 558]
    scalp_vol_pts = daily_vol_pts / 12.0    # [cite: 558]
    
    # Linhas de Gatilho (Sólidas no MT5)
    q_up = fair_value + (scalp_vol_pts * 2.5) # [cite: 566]
    q_dn = fair_value - (scalp_vol_pts * 2.5) # 

    current_price = df_m1['close'].iloc[-1]
    
    # --- MONITORAMENTO DE ENTRADA ---
    if not st.session_state.sim_active:
        buffer = INP_BAND_BUFFER
        # Se tocar na linha Quant Verde (Compra) [cite: 187]
        if current_price <= (q_dn + buffer) and st.session_state.pending_side == 0:
            st.session_state.pending_side = 1
            st.session_state.wait_counter = INP_WAIT_CANDLES
        # Se tocar na linha Quant Vermelha (Venda) [cite: 189]
        elif current_price >= (q_up - buffer) and st.session_state.pending_side == 0:
            st.session_state.pending_side = -1
            st.session_state.wait_counter = INP_WAIT_CANDLES
            
        # Lógica de Gatilho (Wait Counter) [cite: 204, 209]
        if st.session_state.wait_counter > 0:
            st.session_state.wait_counter -= 1
        elif st.session_state.pending_side != 0:
            # Define gatilho na máxima/mínima anterior [cite: 209]
            h1, l1 = df_m1['high'].iloc[-2], df_m1['low'].iloc[-2]
            if st.session_state.pending_side == 1:
                st.session_state.trigger_price = h1 + INP_BREAKOUT
                if current_price >= st.session_state.trigger_price:
                    open_trade(1, current_price, score)
            else:
                st.session_state.trigger_price = l1 - INP_BREAKOUT
                if current_price <= st.session_state.trigger_price:
                    open_trade(-1, current_price, score)

    # --- GESTÃO DE TRADE ATIVO (TRAILING E PARCIAL) ---
    else:
        manage_active_trade(current_price)

    render_hud(current_price, score, shift, fair_value, q_up, q_dn)
    time.sleep(2)
    st.rerun()

def open_trade(side, price, score):
    # Verifica filtros macro antes de abrir [cite: 240, 241]
    if (side == 1 and score < 2) or (side == -1 and score > -2):
        st.session_state.pending_side = 0
        return

    st.session_state.update({
        'sim_active': True, 'sim_side': side, 'open_price': price,
        'peak_price': price, 'partial_done': False, 'current_lots': 2.0,
        'sl_price': price - 1500 if side == 1 else price + 1500,
        'tp_price': price + 300 if side == 1 else price - 300
    })

def manage_active_trade(price):
    s = st.session_state
    side = s.sim_side
    
    # 1. Atualiza Pico (Peak) para Trailing [cite: 311]
    if side == 1: s.peak_price = max(s.peak_price, price)
    else: s.peak_price = min(s.peak_price, price)
    
    # 2. Realização Parcial [cite: 337, 343]
    points = (price - s.open_price) if side == 1 else (s.open_price - price)
    if not s.partial_done and points >= INP_PARTIAL_PTS:
        s.profit_closed += (INP_PARTIAL_PTS * 0.20 * INP_PARTIAL_VOL)
        s.current_lots -= INP_PARTIAL_VOL
        s.partial_done = True
        # Move para Break-Even [cite: 345]
        s.sl_price = s.open_price + 10 if side == 1 else s.open_price - 10

    # 3. Verificação de Saída Final [cite: 352]
    if (side == 1 and (price >= s.tp_price or price <= s.sl_price)) or \
       (side == -1 and (price <= s.tp_price or price >= s.sl_price)):
        exit_price = s.tp_price if points > 0 else s.sl_price
        final_points = (exit_price - s.open_price) if side == 1 else (s.open_price - exit_price)
        st.session_state.total_points += final_points
        st.session_state.sim_active = False
        st.session_state.pending_side = 0

def render_hud(price, score, shift, fair, q_up, q_dn):
    st.markdown(f"### 🎯 SNIPER HUD (Sincronizado MT5)")
    c1, c2, c3 = st.columns(3)
    c1.metric("PREÇO ATUAL", f"{price:.2f}")
    c2.metric("SCORE MACRO", f"{score:+}")
    c3.metric("FAIR VALUE", f"{fair:.2f}")
    
    st.sidebar.markdown(f"**LINHAS QUANT (MT5)**")
    st.sidebar.error(f"Venda: {q_up:.2f}")
    st.sidebar.warning(f"Justo: {fair:.2f}")
    st.sidebar.success(f"Compra: {q_dn:.2f}")

if __name__ == "__main__":
    main()
