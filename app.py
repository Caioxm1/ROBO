import streamlit as st
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval
import numpy as np
import yfinance as yf # Importante para a redundância

# =========================================================
# 1. CONFIGURAÇÕES E ESTADO
# =========================================================
BETAS_WIN = {'^GSPC': 1.2, 'USDBRL=X': -1.0, 'USDMXN=X': -0.5, '^TNX': -0.4, 'EWZ': 1.0}
TICKERS_MACRO = list(BETAS_WIN.keys())

INP_TREND_PER, INP_TREND_DEV = 20, 2.0
INP_ENTRY_PER, INP_ENTRY_DEV = 20, 2.5
INP_BAND_BUFFER, INP_BREAKOUT = 10, 20
INP_RSI_PER, INP_RSI_UPPER, INP_RSI_LOWER = 14, 70, 30
INP_TAKE_POINTS = 1000
INP_WAIT_CANDLES = 1
INP_MIN_SCORE_TRADE = 2

if 'sim_active' not in st.session_state:
    st.session_state.update({
        'sim_active': False, 'trades_history': [], 'total_points': 0.0,
        'wins': 0, 'losses': 0, 'pending_side': 0, 'wait_counter': 0,
        'trigger_price': 0.0, 'last_profit_time': None, 'peak_price': 0.0
    })

# =========================================================
# 2. SISTEMA DE DADOS (COM REDUNDÂNCIA)
# =========================================================

class DataEngine:
    def __init__(self):
        try:
            self.tv = TvDatafeed()
        except:
            self.tv = None

    def get_market_data(self, symbol="WIN1!", exchange="BMF", interval=Interval.in_1_minute, n_bars=100):
        if not self.tv: return pd.DataFrame()
        try:
            data = self.tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
            return data if data is not None else pd.DataFrame()
        except:
            return pd.DataFrame()

    def get_macro_prices(self):
        macro_results = {}
        for ticker in TICKERS_MACRO:
            try:
                # Se TradingView falhar, usa Yahoo Finance para os macros
                df = yf.download(ticker, period="2d", interval="1d", progress=False)
                if not df.empty and len(df) >= 2:
                    close_now = df['Close'].iloc[-1]
                    close_prev = df['Close'].iloc[-2]
                    macro_results[ticker] = (float(close_now) / float(close_prev)) - 1
                else: macro_results[ticker] = 0.0
            except: macro_results[ticker] = 0.0
        return macro_results

def get_fallback_data():
    """Redundância extrema via Yahoo Finance"""
    try:
        data = yf.download("WIN=F", period="1d", interval="1m", progress=False)
        if not data.empty:
            data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
            return data
    except: return pd.DataFrame()
    return pd.DataFrame()

# =========================================================
# 3. LÓGICA TÉCNICA
# =========================================================

def calculate_macro_score(macro_changes):
    score, shift_total = 0, 0
    for ticker, beta in BETAS_WIN.items():
        if ticker in macro_changes:
            impacto = macro_changes[ticker] * beta
            shift_total += impacto
            if impacto > 0.001: score += 1
            elif impacto < -0.001: score -= 1
    return int(max(min(score, 5), -5)), shift_total

def compute_technical_indicators(df_m1, df_m5):
    bb_m5 = ta.bbands(df_m5['close'], length=INP_TREND_PER, std=INP_TREND_DEV)
    df_m5['trend_mid'] = bb_m5.iloc[:, 1]
    bb_m1 = ta.bbands(df_m1['close'], length=INP_ENTRY_PER, std=INP_ENTRY_DEV)
    df_m1['entry_up'], df_m1['entry_low'], df_m1['entry_mid'] = bb_m1.iloc[:, 2], bb_m1.iloc[:, 0], bb_m1.iloc[:, 1]
    df_m1['rsi'] = ta.rsi(df_m1['close'], length=INP_RSI_PER)
    df_m1['atr'] = ta.atr(df_m1['high'], df_m1['low'], df_m1['close'], length=14)
    df_m1['adx'] = ta.adx(df_m1['high'], df_m1['low'], df_m1['close'], length=14).iloc[:, 0]
    return df_m1, df_m5

def get_narrator_message(current_price, df_m1, macro_score, fair_value):
    last_row = df_m1.iloc[-1]
    dist_fair = abs(current_price - fair_value)
    if st.session_state.pending_side == 0:
        dist_up = df_m1['entry_up'].iloc[-1] - current_price
        dist_low = current_price - df_m1['entry_low'].iloc[-1]
        if dist_up < 500 and dist_up > 150: return f"⏳ SUBA + {int(dist_up)} pts p/ Vender."
        return f"⏳ DESÇA + {int(dist_low)} pts p/ Comprar." if dist_low < 500 and dist_low > 150 else "💤 MEIO DE CAMPO."
    
    if st.session_state.wait_counter > 0: return f"✋ FILTRO TEMPO: {st.session_state.wait_counter} velas."
    if (st.session_state.pending_side == -1 and macro_score <= -INP_MIN_SCORE_TRADE) or \
       (st.session_state.pending_side == 1 and macro_score >= INP_MIN_SCORE_TRADE):
        return "🔥 DISPARANDO AGORA!!!"
    return "⛔ BLOQUEIO: Aguardando Checklist..."

# =========================================================
# 4. GESTÃO DE TRADES
# =========================================================

def open_sim_trade(side, price, sl, tp, is_macro=False):
    st.session_state.update({'sim_active': True, 'sim_side': side, 'open_price': price, 
                             'peak_price': price, 'sl_price': sl, 'tp_price': tp, 'is_macro_trade': is_macro})
    st.toast(f"🚀 {'COMPRA' if side==1 else 'VENDA'} EM {price}")

def close_sim_trade(exit_price, reason="TP/SL"):
    side, open_p = st.session_state.sim_side, st.session_state.open_price
    points = (exit_price - open_p) if side == 1 else (open_p - exit_price)
    st.session_state.total_points += points
    if points > 0: st.session_state.wins += 1
    else: st.session_state.losses += 1
    st.session_state.trades_history.append({"Data": datetime.now().strftime("%H:%M:%S"), "Lado": "C" if side==1 else "V", "Pontos": points, "Motivo": reason})
    st.session_state.update({'sim_active': False, 'pending_side': 0, 'trigger_price': 0.0})

def manage_smart_trailing(bid, ask):
    if not st.session_state.sim_active: return
    side, entry, peak = st.session_state.sim_side, st.session_state.open_price, st.session_state.peak_price
    if side == 1:
        if bid > peak: st.session_state.peak_price = bid
        if (bid - entry) > 150: st.session_state.sl_price = max(st.session_state.sl_price, entry + 10)
    else:
        if ask < peak: st.session_state.peak_price = ask
        if (entry - ask) > 150: st.session_state.sl_price = min(st.session_state.sl_price, entry - 10)

# =========================================================
# 5. DASHBOARD E MAIN
# =========================================================

def render_dashboard(current_price, macro_score, shift, df_m1, narrator_msg):
    st.set_page_config(page_title="Sniper AI", layout="centered")
    st.markdown("<h2 style='text-align: center; color: #FFD700;'>🎯 SNIPER AI v8.1</h2>", unsafe_content_type=True)
    c1, c2 = st.columns(2)
    c1.metric("STATUS", "CAÇANDO" if st.session_state.pending_side != 0 else "ESCANEANDO")
    c1.metric("MACRO SCORE", f"{macro_score:+}", delta=f"{shift:.4f}")
    c2.metric("GATILHO", f"{st.session_state.trigger_price:.0f}" if st.session_state.trigger_price > 0 else "OFF")
    c2.metric("PLACAR", f"{st.session_state.wins}W - {st.session_state.losses}L")
    st.divider()
    if "🔥" in narrator_msg: st.error(narrator_msg)
    elif "⛔" in narrator_msg: st.warning(narrator_msg)
    else: st.info(narrator_msg)
    st.sidebar.button("Resetar Placar", on_click=lambda: st.session_state.update({'total_points': 0, 'wins': 0, 'losses': 0, 'trades_history': []}))

def main():
    engine = DataEngine()
    df_m1 = engine.get_market_data()
    df_m5 = engine.get_market_data(interval=Interval.in_5_minute, n_bars=50)
    
    # --- REDUNDÂNCIA ---
    if df_m1.empty:
        df_m1 = get_fallback_data()
        df_m5 = df_m1.resample('5min').last().ffill() if not df_m1.empty else pd.DataFrame()

    if df_m1.empty or len(df_m1) < 20:
        st.error("❌ Erro de Conexão com a B3.")
        st.info("O TradingView bloqueou o IP do servidor. Tente novamente em alguns minutos.")
        if st.button("🔄 Tentar Reconectar Agora"): st.rerun()
        return

    macro_changes = engine.get_macro_prices()
    df_m1, df_m5 = compute_technical_indicators(df_m1, df_m5)
    macro_score, shift = calculate_macro_score(macro_changes)
    
    cur_bid = float(df_m1['close'].iloc[-1])
    cur_ask = cur_bid + 5
    fair_value = float(df_m5['close'].iloc[-1]) * (1.0 + shift)

    if st.session_state.sim_active:
        manage_smart_trailing(cur_bid, cur_ask)
        if st.session_state.sim_side == 1:
            if cur_bid >= st.session_state.tp_price: close_sim_trade(st.session_state.tp_price, "TP")
            elif cur_bid <= st.session_state.sl_price: close_sim_trade(st.session_state.sl_price, "SL")
        else:
            if cur_ask <= st.session_state.tp_price: close_sim_trade(st.session_state.tp_price, "TP")
            elif cur_ask >= st.session_state.sl_price: close_sim_trade(st.session_state.sl_price, "SL")
    else:
        last = df_m1.iloc[-1]
        if st.session_state.pending_side == 0:
            if cur_ask <= (last['entry_low'] + INP_BAND_BUFFER): st.session_state.update({'pending_side': 1, 'wait_counter': INP_WAIT_CANDLES})
            elif cur_bid >= (last['entry_up'] - INP_BAND_BUFFER): st.session_state.update({'pending_side': -1, 'wait_counter': INP_WAIT_CANDLES})
        else:
            msg = get_narrator_message(cur_bid, df_m1, macro_score, fair_value)
            if "🔥" in msg:
                sl_dist = max(last['atr'] * 2.5, 150)
                if st.session_state.pending_side == 1: open_sim_trade(1, cur_ask, cur_bid - sl_dist, cur_bid + INP_TAKE_POINTS)
                else: open_sim_trade(-1, cur_bid, cur_ask + sl_dist, cur_ask - INP_TAKE_POINTS)

    render_dashboard(cur_bid, macro_score, shift, df_m1, get_narrator_message(cur_bid, df_m1, macro_score, fair_value))
    time.sleep(5) # Aumentei para 5s para evitar bloqueio de IP
    st.rerun()

if __name__ == "__main__":
    main()
