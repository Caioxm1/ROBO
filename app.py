import streamlit as st
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf

# 1. ÚNICA CHAMADA DE CONFIGURAÇÃO (Deve ser a primeira)
st.set_page_config(page_title="Sniper AI Monitor", layout="centered")

# =========================================================
# 1. CONFIGURAÇÕES MACRO
# =========================================================
BETAS_WIN = {
    '^GSPC': 1.2,      'USDBRL=X': -1.0, 
    'USDMXN=X': -0.5,  '^TNX': -0.4,      'EWZ': 1.0
}
TICKERS_MACRO = list(BETAS_WIN.keys())

# =========================================================
# 2. INPUTS DO SISTEMA
# =========================================================
INP_TREND_TF, INP_TREND_PER, INP_TREND_DEV = "5m", 20, 2.0
INP_ENTRY_TF, INP_ENTRY_PER, INP_ENTRY_DEV = "1m", 20, 2.5
INP_BAND_BUFFER, INP_BREAKOUT = 10, 20
INP_RSI_PER, INP_RSI_UPPER, INP_RSI_LOWER = 14, 70, 30
INP_TAKE_POINTS, INP_MIN_SCORE_TRADE = 1000, 2
INP_WAIT_CANDLES = 1 

# Inicialização do Estado (Session State)
if 'sim_active' not in st.session_state:
    st.session_state.update({
        'sim_active': False, 'trades_history': [], 'total_points': 0.0,
        'wins': 0, 'losses': 0, 'pending_side': 0, 'wait_counter': 0,
        'trigger_price': 0.0, 'last_profit_time': None, 'peak_price': 0.0
    })

# =========================================================
# 4. MOTOR DE DADOS (CORRIGIDO PARA MULTI-INDEX)
# =========================================================
class DataEngine:
    def get_market_data(self, symbol="WIN=F", interval="1m", n_bars=100):
        try:
            # Garante que intervalo seja string (evita erro de 'Interval' object)
            str_int = "1m" if "1" in str(interval) else "5m"
            data = yf.download(symbol, period='7d', interval=str_int, progress=False, timeout=10)
            
            if data.empty:
                if symbol == "WIN=F":
                    st.sidebar.warning("WIN=F sem dados (Fim de semana). Usando Ibovespa (^BVSP)...")
                    return self.get_market_data("^BVSP", str_int, n_bars)
                return pd.DataFrame()

            # CORREÇÃO: Achata MultiIndex do Yahoo v0.2.x+
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]
            
            return data.tail(n_bars)
        except Exception as e:
            st.error(f"Erro no Yahoo Finance: {e}")
            return pd.DataFrame()

    def get_macro_prices(self):
        macro_results = {}
        for t in TICKERS_MACRO:
            try:
                df = yf.download(t, period='5d', interval='1d', progress=False)
                if not df.empty:
                    c = df['Close'].iloc[-2:] # Pega os 2 últimos dias
                    val_now = c.iloc[-1][0] if isinstance(c.iloc[-1], (pd.Series, np.ndarray)) else c.iloc[-1]
                    val_prev = c.iloc[-2][0] if isinstance(c.iloc[-2], (pd.Series, np.ndarray)) else c.iloc[-2]
                    macro_results[t] = (float(val_now) / float(val_prev)) - 1
            except: macro_results[t] = 0.0
        return macro_results

# =========================================================
# 5. CÉREBRO MATEMÁTICO (CORREÇÃO DO KEYERROR)
# =========================================================
def compute_technical_indicators(df_m1, df_m5):
    """Usa posição iloc para evitar erro de nome de coluna das bandas"""
    
    # M5 - Tendência
    bb_m5 = ta.bbands(df_m5['close'], length=INP_TREND_PER, std=INP_TREND_DEV)
    df_m5['trend_mid'] = bb_m5.iloc[:, 1] # Posição 1 é sempre a Média (BBM)
    
    # M1 - Entrada
    bb_m1 = ta.bbands(df_m1['close'], length=INP_ENTRY_PER, std=INP_ENTRY_DEV)
    df_m1['entry_low'] = bb_m1.iloc[:, 0] # BBL (Inferior)
    df_m1['entry_mid'] = bb_m1.iloc[:, 1] # BBM (Média)
    df_m1['entry_up']  = bb_m1.iloc[:, 2] # BBU (Superior)
    
    df_m1['rsi'] = ta.rsi(df_m1['close'], length=INP_RSI_PER)
    df_m1['atr'] = ta.atr(df_m1['high'], df_m1['low'], df_m1['close'], length=14)
    adx_df = ta.adx(df_m1['high'], df_m1['low'], df_m1['close'], length=14)
    df_m1['adx'] = adx_df.iloc[:, 0] # ADX_14
    
    return df_m1, df_m5

# ... (Mantenha as funções calculate_macro_score, get_narrator_message, close_sim_trade iguais) ...

def calculate_macro_score(macro_changes):
    score, shift = 0, 0
    for t, b in BETAS_WIN.items():
        if t in macro_changes:
            imp = macro_changes[t] * b
            shift += imp
            if imp > 0.001: score += 1
            elif imp < -0.001: score -= 1
    return int(max(min(score, 5), -5)), shift

def render_dashboard(current_price, macro_score, shift, df_m1, narrator_msg):
    st.markdown(f"<h2 style='text-align: center; color: #FFD700;'>🎯 SNIPER AI v8.0</h2>", unsafe_content_type=True)
    c1, c2 = st.columns(2)
    with c1: st.metric("STATUS", "ESCANEANDO" if st.session_state.pending_side == 0 else "GATILHO")
    with c1: st.metric("MACRO SCORE", f"{macro_score:+}")
    with c2: st.metric("PREÇO ATUAL", f"{current_price:.0f}")
    with c2: st.metric("PLACAR", f"{st.session_state.wins}W - {st.session_state.losses}L")
    
    st.divider()
    if "🔥" in narrator_msg: st.error(narrator_msg)
    elif "⛔" in narrator_msg: st.warning(narrator_msg)
    else: st.info(narrator_msg)

def main():
    engine = DataEngine()
    df_m1 = engine.get_market_data("WIN=F", "1m")
    df_m5 = engine.get_market_data("WIN=F", "5m")
    macro_changes = engine.get_macro_prices()
    
    if df_m1.empty:
        st.error("❌ Erro ao obter dados. Verifique a conexão.")
        if st.button("🔄 Recarregar"): st.rerun()
        return

    df_m1, df_m5 = compute_technical_indicators(df_m1, df_m5)
    m_score, shift = calculate_macro_score(macro_changes)
    curr_p = df_m1['close'].iloc[-1]
    
    # Lógica simplificada de gatilho para o dashboard não quebrar
    msg = "Aguardando sinal..." if st.session_state.pending_side == 0 else "Monitorando gatilho..."
    
    render_dashboard(curr_p, m_score, shift, df_m1, msg)
    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()
