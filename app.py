import streamlit as st
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="Sniper AI Monitor - HUD Completo", layout="wide")

# =========================================================
# 1. CONFIGURAÇÕES E INPUTS (Sincronizados com WIN.txt)
# =========================================================
BETAS_WIN = {'^GSPC': 1.2, 'USDBRL=X': -1.0, 'USDMXN=X': -0.5, '^TNX': -0.4, 'EWZ': 1.0}
TICKERS_MACRO = list(BETAS_WIN.keys())

INP_WAIT_CANDLES = 1       # Velas de espera após o toque [cite: 18, 193]
INP_BAND_BUFFER  = 10      # Buffer para antecipar toque [cite: 14, 187]
INP_BREAKOUT     = 20      # Gordura de rompimento [cite: 15, 186]
INP_RSI_UPPER    = 70      # Nível de Venda [cite: 17, 152]
INP_RSI_LOWER    = 30      # Nível de Compra [cite: 18, 168]
INP_ADX_LEVEL    = 35      # Filtro de tendência forte [cite: 35, 154]
INP_TAKE_POINTS  = 300     # Take Profit [cite: 22, 255]
INP_STOP_POINTS  = 1500    # Stop Loss [cite: 21, 274]

# =========================================================
# 2. MOTOR DE DADOS (Blindado contra erros de Series)
# =========================================================
class DataEngine:
    def get_market_data(self, symbol="BOVA11.SA", interval="1m", n_bars=100):
        try:
            data = yf.download(symbol, period='7d', interval=interval, progress=False)
            if data.empty: return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            return data.tail(n_bars)
        except: return pd.DataFrame()

def get_zscore(df):
    """Calcula a tensão do preço igual ao GetCurrentZScore do MT5"""
    last = df.iloc[-1]
    # No MT5 usamos Desvio 2.5 para achar o desvio padrão puro
    std_puro = (last['bb_up'] - last['bb_mid']) / 2.5
    if std_puro == 0: return 0
    return (last['close'] - last['bb_mid']) / std_puro
    
    def get_macro_data(self):
        try:
            data = yf.download(TICKERS_MACRO, period="2d", interval="1m", progress=False)
            if data.empty: return 0, 0.0, 0.0035
            df_close = data['Close'].ffill()
            changes = (df_close.iloc[-1] / df_close.iloc[-2]) - 1
            shift, score = 0.0, 0
            for t in TICKERS_MACRO:
                if t in changes.index:
                    val = float(changes[t].iloc[0]) if hasattr(changes[t], 'iloc') else float(changes[t])
                    impacto = val * BETAS_WIN[t]
                    shift += impacto
                    if impacto > 0.001: score += 1
                    elif impacto < -0.001: score -= 1
            vol = df_close.iloc[:,0].pct_change().std()
            return int(max(min(score, 5), -5)), shift, float(vol)
        except: return 0, 0.0, 0.0035

    def get_ref_price(self):
        df = yf.download("BOVA11.SA", period="5d", interval="1d", progress=False)
        val = df['Close'].iloc[-2] if len(df) >= 2 else df['Close'].iloc[-1]
        return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)

# =========================================================
# 3. GESTÃO DE ESTADO (PLACAR E HISTÓRICO)
# =========================================================
if 'sim_active' not in st.session_state:
    st.session_state.update({
        'sim_active': False, 'trades_history': [], 'total_points': 0.0,
        'wins': 0, 'losses': 0, 'pending_side': 0, 'wait_counter': 0,
        'trigger_price': 0.0, 'partial_done': False, 'current_lots': 2.0,
        'peak_price': 0.0, 'sl_price': 0.0, 'tp_price': 0.0  # Adicionadas [cite: 80]
    })

# =========================================================
# 4. LÓGICA DO NARRADOR (O QUE FALTA?)
# =========================================================
def get_narrator_message(price, df, score, fair_value):
    if st.session_state.sim_active:
    manage_active_trade(current_price, df)
        return "🚀 POSIÇÃO ABERTA: Monitorando Alvo/Stop."
    
    if st.session_state.pending_side == 0:
        return "💤 MEIO DE CAMPO. Aguardando toque nas extremidades Quant."

    last = df.iloc[-1]
    rsi, adx = last['rsi'], last['adx']
    dist_fair = abs(price - fair_value)

    # Checklist de Bloqueios (Idêntico ao WIN.txt) 
    if st.session_state.wait_counter > 0:
        return f"✋ FILTRO TEMPO: Faltam {st.session_state.wait_counter} velas para autorizar." 
    
    if st.session_state.pending_side == 1: # Compra
        if rsi <= INP_RSI_LOWER: return f"⛔ BLOQUEIO RSI: {rsi:.1f} (Muito Frio)" 
        if score < 2: return f"⛔ BLOQUEIO MACRO: Score {score} insuficiente para Compra." 
        if dist_fair < 180: return f"⛔ BLOQUEIO MÉDIA: Muito perto do Preço Justo." 
        return "🔥 DISPARANDO COMPRA AGORA!!!"
    
    if st.session_state.pending_side == -1: # Venda
        if rsi >= INP_RSI_UPPER: return f"⛔ BLOQUEIO RSI: {rsi:.1f} (Muito Quente)" 
        if score > -2: return f"⛔ BLOQUEIO MACRO: Score {score} insuficiente para Venda." 
        if dist_fair < 180: return f"⛔ BLOQUEIO MÉDIA: Muito perto do Preço Justo." # [cite: 161, 242]
        return "🔥 DISPARANDO VENDA AGORA!!!" # [cite: 162]

def manage_active_trade(price, df):
    s = st.session_state
    side = s.sim_side  # 1 p/ Compra, -1 p/ Venda
    entry = s.open_price
    
    # 1. Atualiza o Pico (Peak Price) [cite: 311, 467]
    if side == 1: s.peak_price = max(s.peak_price, price)
    else: s.peak_price = min(s.peak_price, price)
    
    dist_pts = abs(s.peak_price - entry)
    z = get_zscore(df)
    
    # 2. FASES DO TRAILING (Gatilhos do MT5: 60, 100, 150) [cite: 313, 458]
    if 60 <= dist_pts < 100: # FASE A: Proteção Zero (BE + 10 pts) [cite: 319, 482]
        be = entry + 10 if side == 1 else entry - 10
        if (side == 1 and s.sl_price < be) or (side == -1 and s.sl_price > be): s.sl_price = be

    elif 100 <= dist_pts < 150: # FASE B: Elástico (Gap 100/Min 50) [cite: 322, 484]
        if side == 1: s.sl_price = max(s.peak_price - 100, entry + 50)
        else: s.sl_price = min(s.peak_price + 100, entry - 50)

    elif dist_pts >= 150: # FASE C: Tendência (Trailing 60 ou 130 pts) [cite: 327, 491]
        gap = 130 if abs(z) > 3.0 else 60
        if side == 1: s.sl_price = max(s.sl_price, s.peak_price - gap)
        else: s.sl_price = min(s.sl_price, s.peak_price + gap)

    # 3. VERIFICA SAÍDA FINAL [cite: 352, 356]
    points = (price - entry) if side == 1 else (entry - price)
    if (side == 1 and (price >= s.tp_price or price <= s.sl_price)) or \
       (side == -1 and (price <= s.tp_price or price >= s.sl_price)):
        s.total_points += points
        if points > 0: s.wins += 1; s.last_profit_time = time.time()
        else: s.losses += 1
        s.sim_active = False

# =========================================================
# 5. EXECUÇÃO PRINCIPAL
# =========================================================
def main():
    engine = DataEngine()
    df = engine.get_market_data()
    if df.empty: return

    # Indicadores
    df['rsi'] = ta.rsi(df['close'], length=14).fillna(50)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14'].fillna(20)
    bb = ta.bands(df['close'], length=20, std=2.5) # [cite: 13, 83]
    df['bb_mid'] = bb['BBM_20_2.5']
    df['bb_up']  = bb['BBU_20_2.5']

    score, shift, vol = engine.get_macro_data()
    ref_price = engine.get_ref_price()
    fair_value = ref_price * (1.0 + shift)
    q_up = fair_value + (fair_value * vol / 12.0 * 2.5)
    q_dn = fair_value - (fair_value * vol / 12.0 * 2.5)
    current_price = float(df['close'].iloc[-1])

    # Lógica de Gatilho
    if not st.session_state.sim_active:
        if current_price <= (q_dn + INP_BAND_BUFFER): st.session_state.pending_side = 1; st.session_state.wait_counter = INP_WAIT_CANDLES
        elif current_price >= (q_up - INP_BAND_BUFFER): st.session_state.pending_side = -1; st.session_state.wait_counter = INP_WAIT_CANDLES
        
        if st.session_state.wait_counter > 0: st.session_state.wait_counter -= 1
        elif st.session_state.pending_side != 0:
            msg = get_narrator_message(current_price, df, score, fair_value)
            if "DISPARANDO" in msg:
                # Abre trade simulado
                st.session_state.sim_active = True
                st.session_state.open_price = current_price
                st.session_state.trades_history.append({"Hora": datetime.now().strftime("%H:%M"), "Lado": "Compra" if st.session_state.pending_side == 1 else "Venda", "Preço": current_price})

    # --- RENDERIZAÇÃO DO DASHBOARD ---
    st.markdown(f"<h1 style='text-align: center; color: #FFD700;'>🎯 SNIPER AI - MONITOR v8.0</h1>", unsafe_allow_html=True) # [cite: 389]
    
    # HUD de Métricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("STATUS", "CAÇANDO" if st.session_state.pending_side != 0 else "ESCANEANDO") # [cite: 391]
    c2.metric("SCORE MACRO", f"{score:+}", delta=f"{shift:.4f}") # [cite: 409, 414]
    c3.metric("PLACAR (W/L)", f"{st.session_state.wins} x {st.session_state.losses}") # [cite: 398, 427]
    c4.metric("PONTOS HOJE", f"{int(st.session_state.total_points)} pts") # [cite: 399, 430]

    st.divider()

    # Narrador (O que falta?)
    st.subheader("O QUE FALTA?")
    msg = get_narrator_message(current_price, df, score, fair_value)
    if "⛔" in msg or "✋" in msg: st.warning(msg)
    elif "🔥" in msg: st.error(msg)
    else: st.info(msg)

    # Resultado Financeiro
    res_val = st.session_state.total_points * 0.20 
    st.markdown(f"### RESULTADO: <span style='color: #00FF00;'>R$ {res_val:.2f}</span>", unsafe_allow_html=True) # [cite: 401, 432]

    # Histórico e Sidebar
    with st.expander("Ver Histórico de Trades"):
        if st.session_state.trades_history: st.table(pd.DataFrame(st.session_state.trades_history))
    
    st.sidebar.title("📊 Linhas Quant (MT5)") 
    st.sidebar.error(f"Venda Scalper: {q_up:,.0f}") 
    st.sidebar.warning(f"Preço Justo: {fair_value:,.0f}") 
    st.sidebar.success(f"Compra Scalper: {q_dn:,.0f}") 

    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()

