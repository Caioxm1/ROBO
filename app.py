import streamlit as st
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="Sniper AI Monitor - v8.0 Perfeito", layout="wide")

# =========================================================
# 1. CONFIGURAÇÕES E INPUTS (Sincronizados com WIN.txt)
# =========================================================
BETAS_WIN = {'^GSPC': 1.2, 'USDBRL=X': -1.0, 'USDMXN=X': -0.5, '^TNX': -0.4, 'EWZ': 1.0}
TICKERS_MACRO = list(BETAS_WIN.keys())

INP_WAIT_CANDLES = 1       # [cite: 18, 193]
INP_BAND_BUFFER  = 10      # [cite: 14, 187]
INP_BREAKOUT     = 20      # [cite: 15, 186]
INP_RSI_UPPER    = 70      # [cite: 17, 152]
INP_RSI_LOWER    = 30      # [cite: 18, 168]
INP_ADX_LEVEL    = 35      # [cite: 35, 154]
INP_TAKE_POINTS  = 300     # [cite: 22, 255]
INP_STOP_POINTS  = 1500    # [cite: 21, 274]

# =========================================================
# 2. MOTOR DE DADOS E CÁLCULOS TÉCNICOS
# =========================================================
class DataEngine:
    def get_market_data(self, symbol="^BVSP", interval="1m", n_bars=100):
        try:
            data = yf.download(symbol, period='7d', interval=interval, progress=False)
            if data.empty: return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            return data.tail(n_bars)
        except: return pd.DataFrame()

    # --- AJUSTE IDENTICO AO Sniper_Data_Feed.py ---
def get_macro_data(self):
    try:
        # Baixa dados diários para calcular a Volatilidade Real do dia anterior
        data = yf.download(TICKERS_MACRO, period="5d", interval="1d", progress=False)
        if data.empty: return 0, 0.0, 0.0035
        
        df_close = data['Close'].ffill()
        # Variação e Score
        changes = (df_close.iloc[-1] / df_close.iloc[-2]) - 1
        shift, score = 0.0, 0
        for t in TICKERS_MACRO:
            val = float(changes[t])
            impacto = val * BETAS_WIN[t]
            shift += impacto
            if impacto > 0.001: score += 1
            elif impacto < -0.001: score -= 1
            
        # Volatilidade Exata do S&P500
        vol = df_close['^GSPC'].pct_change().std() 
        return int(max(min(score, 5), -5)), shift, float(vol)
    except: return 0, 0.0, 0.0035

    def get_ref_price(self):
        df = yf.download("^BVSP", period="5d", interval="1d", progress=False)
        val = df['Close'].iloc[-2] if len(df) >= 2 else df['Close'].iloc[-1]
        return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)

def get_zscore(df):
    """Calcula a tensão do preço igual ao GetCurrentZScore do MT5"""
    last = df.iloc[-1]
    std_puro = (last['bb_up'] - last['bb_mid']) / 2.5
    if std_puro == 0: return 0
    return (last['close'] - last['bb_mid']) / std_puro

# =========================================================
# 3. GESTÃO DE TRADE (TRAILING E PLACAR)
# =========================================================
def manage_active_trade(price, df):
    """Trailing Stop Elástico de 3 Fases"""
    s = st.session_state
    side = s.sim_side  
    entry = s.open_price
    
    if side == 1: s.peak_price = max(s.peak_price, price)
    else: s.peak_price = min(s.peak_price, price)
    
    dist_pts = abs(s.peak_price - entry)
    z = get_zscore(df)
    
    # FASE A: Proteção Zero (60 pts) [cite: 319, 482]
    if 60 <= dist_pts < 100:
        be = entry + 10 if side == 1 else entry - 10
        if (side == 1 and s.sl_price < be) or (side == -1 and s.sl_price > be): s.sl_price = be

    # FASE B: Elástico (100 pts) [cite: 322, 484]
    elif 100 <= dist_pts < 150:
        if side == 1: s.sl_price = max(s.peak_price - 100, entry + 50)
        else: s.sl_price = min(s.peak_price + 100, entry - 50)

    # FASE C: Tendência (150 pts) [cite: 327, 491]
    elif dist_pts >= 150:
        gap = 130 if abs(z) > 3.0 else 60
        if side == 1: s.sl_price = max(s.sl_price, s.peak_price - gap)
        else: s.sl_price = min(s.sl_price, s.peak_price + gap)

    # Saída Final [cite: 356]
    points = (price - entry) if side == 1 else (entry - price)
    if (side == 1 and (price >= s.tp_price or price <= s.sl_price)) or \
       (side == -1 and (price <= s.tp_price or price >= s.sl_price)):
        s.total_points += points
        if points > 0: s.wins += 1
        else: s.losses += 1
        s.sim_active = False

# =========================================================
# 4. GESTÃO DE ESTADO E NARRADOR
# =========================================================
if 'sim_active' not in st.session_state:
    st.session_state.update({
        'sim_active': False, 'trades_history': [], 'total_points': 0.0,
        'wins': 0, 'losses': 0, 'pending_side': 0, 'wait_counter': 0,
        'peak_price': 0.0, 'sl_price': 0.0, 'tp_price': 0.0, 'open_price': 0.0, 'sim_side': 0
    })

def get_narrator_message(price, df, score, fair_value):
    if st.session_state.sim_active:
        return "🚀 POSIÇÃO ABERTA: Monitorando Trailing Stop Elástico."
    
    if st.session_state.pending_side == 0:
        return "💤 MEIO DE CAMPO. Aguardando toque nas extremidades Quant."

    last = df.iloc[-1]
    rsi = last['rsi']
    dist_fair = abs(price - fair_value)

    if st.session_state.wait_counter > 0:
        return f"✋ FILTRO TEMPO: Faltam {st.session_state.wait_counter} velas para autorizar." 
    
    if st.session_state.pending_side == 1: 
        if rsi <= INP_RSI_LOWER: return f"⛔ BLOQUEIO RSI: {rsi:.1f} (Abaixo de 30)" 
        if score < 2: return f"⛔ BLOQUEIO MACRO: Score {score} insuficiente." 
        return "🔥 DISPARANDO COMPRA AGORA!!!"
    
    if st.session_state.pending_side == -1: 
        if rsi >= INP_RSI_UPPER: return f"⛔ BLOQUEIO RSI: {rsi:.1f} (Acima de 70)" 
        if score > -2: return f"⛔ BLOQUEIO MACRO: Score {score} insuficiente." 
        return "🔥 DISPARANDO VENDA AGORA!!!"

# =========================================================
# 5. EXECUÇÃO PRINCIPAL (DASHBOARD)
# =========================================================
def main():
    engine = DataEngine()
    df = engine.get_market_data()
    if df.empty: return

    # Indicadores M1
    df['rsi'] = ta.rsi(df['close'], length=14).fillna(50)
    # --- AJUSTE POSICIONAL: EVITA ERRO DE NOME DE COLUNA ---
    bb = ta.bbands(df['close'], length=20, std=2.5)
    
    # Pegamos pelas posições fixas que a biblioteca sempre entrega:
    # 0 = Low (BBL), 1 = Mid (BBM), 2 = Up (BBU)
    df['bb_low'] = bb.iloc[:, 0]
    df['bb_mid'] = bb.iloc[:, 1]
    df['bb_up']  = bb.iloc[:, 2]
    # -------------------------------------------------------

    score, shift, vol = engine.get_macro_data()
    ref_price = engine.get_ref_price()
    fair_value = ref_price * (1.0 + shift)
    # --- CÁLCULO DE DESVIOS IDENTICO AO MT5 ---
    fair_value = ref_price * (1.0 + shift)
    scalp_vol_pts = (fair_value * vol) / 12.0
    q_up = fair_value + (scalp_vol_pts * 2.5) # Venda Scalper (Linha Sólida)
    q_dn = fair_value - (scalp_vol_pts * 2.5) # Compra Scalper (Linha Sólida)
    
    daily_vol_pts = fair_value * vol
    m_up = fair_value + (daily_vol_pts * 2.0) # Máxima Macro (Linha Pontilhada)
    m_dn = fair_value - (daily_vol_pts * 2.0) # Mínima Macro (Linha Pontilhada)
    current_price = float(df['close'].iloc[-1])

    # Lógica de Execução
    if st.session_state.sim_active:
        manage_active_trade(current_price, df)
    else:
        if current_price <= (q_dn + INP_BAND_BUFFER): 
            st.session_state.pending_side = 1; st.session_state.wait_counter = INP_WAIT_CANDLES
        elif current_price >= (q_up - INP_BAND_BUFFER): 
            st.session_state.pending_side = -1; st.session_state.wait_counter = INP_WAIT_CANDLES
        
        if st.session_state.wait_counter > 0: 
            st.session_state.wait_counter -= 1
        elif st.session_state.pending_side != 0:
            msg = get_narrator_message(current_price, df, score, fair_value)
            if "DISPARANDO" in msg:
                st.session_state.update({
                    'sim_active': True, 'sim_side': st.session_state.pending_side,
                    'open_price': current_price, 'peak_price': current_price,
                    'sl_price': current_price - 1500 if st.session_state.pending_side == 1 else current_price + 1500,
                    'tp_price': current_price + 300 if st.session_state.pending_side == 1 else current_price - 300
                })
                st.session_state.trades_history.append({"Hora": datetime.now().strftime("%H:%M"), "Lado": "Compra" if st.session_state.pending_side == 1 else "Venda", "Preço": current_price})

    # Interface Visual # [cite:: 389]
    st.markdown(f"<h1 style='text-align: center; color: #FFD700;'>🎯 SNIPER AI - MONITOR v8.0</h1>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("STATUS", "CAÇANDO" if st.session_state.pending_side != 0 else "ESCANEANDO")
    c2.metric("SCORE MACRO", f"{score:+}", delta=f"{shift:.4f}")
    c3.metric("PLACAR (W/L)", f"{st.session_state.wins} x {st.session_state.losses}")
    c4.metric("PONTOS HOJE", f"{int(st.session_state.total_points)} pts")

    st.divider()
    st.subheader("O QUE FALTA?")
    msg = get_narrator_message(current_price, df, score, fair_value)
    if "⛔" in msg or "✋" in msg: st.warning(msg)
    elif "🔥" in msg: st.error(msg)
    else: st.info(msg)

    st.sidebar.title("📊 Linhas Quant (MT5)")
    st.sidebar.error(f"Máxima Dia: {m_up:,.0f}")
    st.sidebar.write(f"Venda Scalper: {q_up:,.0f}")
    st.sidebar.warning(f"Preço Justo: {fair_value:,.0f}")
    st.sidebar.write(f"Compra Scalper: {q_dn:,.0f}")
    st.sidebar.success(f"Mínima Dia: {m_dn:,.0f}")

    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()





