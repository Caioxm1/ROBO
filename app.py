import streamlit as st
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="Sniper AI Monitor - Sincronizado", layout="centered")

# =========================================================
# 1. CONFIGURAÇÕES IDENTICAS AO MT5 (WIN.txt)
# =========================================================
BETAS_WIN = {'^GSPC': 1.2, 'USDBRL=X': -1.0, 'USDMXN=X': -0.5, '^TNX': -0.4, 'EWZ': 1.0}
TICKERS_MACRO = list(BETAS_WIN.keys())

# [cite_start]Parâmetros Sincronizados [cite: 18, 21, 25]
INP_WAIT_CANDLES = 5
INP_BAND_BUFFER  = 10
INP_BREAKOUT     = 20
INP_TAKE_POINTS  = 300
INP_STOP_POINTS  = 1500
INP_PARTIAL_PTS  = 50
INP_PARTIAL_VOL  = 1.0

# =========================================================
# 2. MOTOR DE DADOS QUANT (COM CORREÇÃO DE MULTIINDEX)
# =========================================================
class DataEngine:
    def get_market_data(self, symbol="BOVA11.SA", interval="1m", n_bars=100):
        try:
            data = yf.download(symbol, period='7d', interval=interval, progress=False)
            if data.empty: return pd.DataFrame()
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data.columns = [str(col).lower() for col in data.columns]
            return data.tail(n_bars)
        except: return pd.DataFrame()

    def get_macro_and_vol(self):
        """Calcula Score e Volatilidade com extração escalar (.item())"""
        try:
            data = yf.download(TICKERS_MACRO, period="2d", interval="1m", progress=False)
            if not data.empty:
                df_close = data['Close'].ffill()
                # .iloc[-1] e .iloc[-2] podem retornar Series, usamos .item() para extrair o valor único
                changes = (df_close.iloc[-1] / df_close.iloc[-2]) - 1
                
                shift_total = 0.0
                score = 0
                for t in TICKERS_MACRO:
                    if t in changes.index:
                        # O segredo para não ter erro é o .item() ou .iloc[0]
                        val_change = float(changes[t].item()) if hasattr(changes[t], 'item') else float(changes[t])
                        impacto = val_change * BETAS_WIN[t]
                        shift_total += impacto
                        if impacto > 0.001: score += 1
                        elif impacto < -0.001: score -= 1
                
                # Cálculo da Volatilidade Diária (Base 252 dias para as bandas não colapsarem)
                vol = df_close.iloc[:,0].pct_change().std() * (252**0.5) 
                # Se vol for nan ou 0, usamos o padrão do MT5
                final_vol = float(vol.item()) if (not pd.isna(vol) and vol > 0) else 0.0035
                
                return int(max(min(score, 5), -5)), shift_total, final_vol
        except: pass
        return 0, 0.0, 0.0035

    def get_prev_day_close(self, symbol="BOVA11.SA"):
        """Extrai o preço de fechamento como um escalar puro"""
        df = yf.download(symbol, period="5d", interval="1d", progress=False)
        if len(df) >= 2:
            val = df['Close'].iloc[-2]
            return float(val.item()) if hasattr(val, 'item') else float(val)
        val = df['Close'].iloc[-1]
        return float(val.item()) if hasattr(val, 'item') else float(val)

# =========================================================
# 3. GESTÃO DE ESTADO
# =========================================================
if 'sim_active' not in st.session_state:
    st.session_state.update({
        'sim_active': False, 'total_points': 0.0, 'wins': 0, 'losses': 0,
        'pending_side': 0, 'wait_counter': 0, 'trigger_price': 0.0,
        'partial_done': False, 'current_lots': 2.0, 'profit_closed': 0.0
    })

# =========================================================
# 4. LÓGICA DE EXECUÇÃO (ONTICK)
# =========================================================
def main():
    engine = DataEngine()
    df_m1 = engine.get_market_data()
    
    if df_m1.empty:
        st.warning("Aguardando dados do mercado...")
        return

    prev_close = engine.get_prev_day_close()
    score, shift, vol = engine.get_macro_and_vol()

    # [cite_start]--- CÁLCULO DAS LINHAS QUANT (IDENTICO AO MT5) [cite: 557, 558] ---
    fair_value = float(prev_close * (1.0 + shift))
    daily_vol_pts = fair_value * vol
    scalp_vol_pts = daily_vol_pts / 12.0
    
    q_up = float(fair_value + (scalp_vol_pts * 2.5))
    q_dn = float(fair_value - (scalp_vol_pts * 2.5))

    # Forçamos o preço atual para float para evitar o erro relatado
    current_price = float(df_m1['close'].iloc[-1])
    
    # --- MONITORAMENTO DE ENTRADA ---
    if not st.session_state.sim_active:
        # [cite_start]Se tocar nas bandas (com buffer de 10 pts) [cite: 187]
        if st.session_state.pending_side == 0:
            if current_price <= (q_dn + INP_BAND_BUFFER):
                st.session_state.pending_side = 1
                st.session_state.wait_counter = INP_WAIT_CANDLES
            elif current_price >= (q_up - INP_BAND_BUFFER):
                st.session_state.pending_side = -1
                st.session_state.wait_counter = INP_WAIT_CANDLES
            
        # Lógica de Gatilho (Wait Counter)
        if st.session_state.wait_counter > 0:
            st.session_state.wait_counter -= 1
        elif st.session_state.pending_side != 0:
            # [cite_start]Pega High/Low da vela anterior para o gatilho [cite: 209]
            h1 = float(df_m1['high'].iloc[-2])
            l1 = float(df_m1['low'].iloc[-2])
            
            if st.session_state.pending_side == 1:
                st.session_state.trigger_price = h1 + INP_BREAKOUT
                if current_price >= st.session_state.trigger_price:
                    open_trade(1, current_price, score, q_up, q_dn)
            else:
                st.session_state.trigger_price = l1 - INP_BREAKOUT
                if current_price <= st.session_state.trigger_price:
                    open_trade(-1, current_price, score, q_up, q_dn)

    else:
        manage_active_trade(current_price)

    # --- DASHBOARD HUD ---
    st.markdown(f"## 🎯 SNIPER AI - MONITOR")
    col1, col2, col3 = st.columns(3)
    col1.metric("PREÇO WIN", f"{current_price:,.0f}")
    col2.metric("SCORE MACRO", f"{score:+}")
    col3.metric("FAIR VALUE", f"{fair_value:,.0f}")

    # Visualização das Zonas
    st.divider()
    st.write(f"**Status:** {'🟢 COMPRA' if st.session_state.pending_side == 1 else '🔴 VENDA' if st.session_state.pending_side == -1 else '⚪ BUSCANDO TOQUE'}")
    if st.session_state.wait_counter > 0:
        st.info(f"✋ FILTRO TEMPO: Faltam {st.session_state.wait_counter} barras.")
    
    st.sidebar.markdown("### 📊 Linhas Quant (MT5)")
    st.sidebar.error(f"Venda Scalper: {q_up:,.0f}")
    st.sidebar.warning(f"Preço Justo: {fair_value:,.0f}")
    st.sidebar.success(f"Compra Scalper: {q_dn:,.0f}")
    
    time.sleep(2)
    st.rerun()

def open_trade(side, price, score, q_up, q_dn):
    # [cite_start]Filtro Macro [cite: 241, 242]
    if (side == 1 and score < 2) or (side == -1 and score > -2):
        return

    st.session_state.update({
        'sim_active': True, 'sim_side': side, 'open_price': price,
        'peak_price': price, 'partial_done': False,
        'sl_price': price - 1500 if side == 1 else price + 1500,
        'tp_price': price + 300 if side == 1 else price - 300
    })

def manage_active_trade(price):
    s = st.session_state
    side = s.sim_side
    points = (price - s.open_price) if side == 1 else (s.open_price - price)
    
    # [cite_start]1. Parcial (50 pts) [cite: 25, 292]
    if not s.partial_done and points >= INP_PARTIAL_PTS:
        s.partial_done = True
        s.sl_price = s.open_price + 10 if side == 1 else s.open_price - 10
        st.toast("✅ Parcial Executada! Stop no Break-Even.")

    # 2. Saída Final
    if (side == 1 and (price >= s.tp_price or price <= s.sl_price)) or \
       (side == -1 and (price <= s.tp_price or price >= s.sl_price)):
        st.session_state.total_points += points
        st.session_state.sim_active = False
        st.session_state.pending_side = 0

if __name__ == "__main__":
    main()


