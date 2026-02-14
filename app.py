import streamlit as st
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Sniper AI Monitor", layout="centered")

# =========================================================
# 1. CONFIGURAÇÕES MACRO (Vindo do Sniper_Data_Feed.py)
# =========================================================
# Betas de correlação para cálculo do Score Global
BETAS_WIN = {
    '^GSPC': 1.2,      # S&P 500
    'USDBRL=X': -1.0,  # Dólar Real
    'USDMXN=X': -0.5,  # Dólar Peso Mexicano
    '^TNX': -0.4,      # Treasury 10Y
    'EWZ': 1.0         # ETF Brasil
}

TICKERS_MACRO = list(BETAS_WIN.keys())

# =========================================================
# 2. INPUTS DO SISTEMA (Espelhados do WIN.txt)
# =========================================================
INP_TREND_TF    = "5m"      # Timeframe de Tendência (M5)
INP_TREND_PER   = 20        # Período Bollinger M5
INP_TREND_DEV   = 2.0       # Desvio Bollinger M5

INP_ENTRY_TF    = "1m"      # Timeframe de Entrada (M1)
INP_ENTRY_PER   = 20        # Período Bollinger M1
INP_ENTRY_DEV   = 2.5       # Desvio Bollinger M1
INP_BAND_BUFFER = 10        # Buffer para antecipar toque
INP_BREAKOUT    = 20        # Gordura de rompimento

INP_RSI_PER     = 14        # Período RSI
INP_RSI_UPPER   = 70        # Nível de Venda
INP_RSI_LOWER   = 30        # Nível de Compra

INP_STOP_POINTS = 1500      # Stop Loss em pontos
INP_TAKE_POINTS = 1000      # Take Profit em pontos
INP_WAIT_CANDLES = 1        # Velas de espera (Sniper Logic) 

# Configurações de Integração Macro
INP_USE_MACRO_FILTER = True # Filtro S&P500/Dólar
INP_MIN_SCORE_TRADE  = 2    # Score mínimo para operar

# =========================================================
# 3. GERENCIAMENTO DE ESTADO (SIMULADOR)
# =========================================================
# Inicializa as variáveis de memória da página (Streamlit Session State)
if 'sim_active' not in st.session_state:
    st.session_state.sim_active = False
    st.session_state.trades_history = []
    st.session_state.current_pnl = 0.0
    st.session_state.total_points = 0.0
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.pending_side = 0      # 0=Neutro, 1=Compra, -1=Venda
    st.session_state.wait_counter = 0      # Contador de exaustão
    st.session_state.trigger_price = 0.0   # Preço de gatilho travado
    st.session_state.last_profit_time = None
    st.session_state.peak_price = 0.0      # Para Trailing Stop

# =========================================================
# 4. DATA FEED - BUSCA DE PREÇOS EM TEMPO REAL
# =========================================================
# 1. MOVA ISSO PARA O TOPO (Antes de qualquer lógica)
st.set_page_config(page_title="Sniper AI Monitor", layout="centered")

class DataEngine:
    def get_market_data(self, symbol="^BVSP", interval="1m", n_bars=100):
        """Busca dados do Yahoo Finance com tratamento de MultiIndex"""
        try:
            search_period = '7d' if interval in ['1m', '5m'] else '1mo'
            
            # Download dos dados
            data = yf.download(
                tickers=symbol, 
                period=search_period, 
                interval=interval, 
                progress=False,
                timeout=15
            )
            
            if data.empty:
                # Se falhar com ^BVSP, tenta automaticamente o Índice Bovespa
                if symbol != "^BVSP":
                    return self.get_market_data(symbol="^BVSP", interval=interval, n_bars=n_bars)
                return pd.DataFrame()

            # --- CORREÇÃO DO ERRO DE TUPLE ---
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            # ---------------------------------
            
            return data.tail(n_bars)
        except Exception as e:
            st.error(f"Erro no Yahoo Finance: {e}")
            return pd.DataFrame()

    def get_macro_prices(self):
        """Busca preços macro tratando as colunas corretamente"""
        macro_results = {}
        for ticker in TICKERS_MACRO:
            try:
                df = yf.download(ticker, period='2d', interval='1d', progress=False)
                if not df.empty and len(df) >= 2:
                    # Garante acesso correto à coluna mesmo com MultiIndex
                    close_col = ('Close', ticker) if ticker in df.columns else 'Close'
                    if isinstance(df.columns, pd.MultiIndex):
                        close_now = df['Close'].iloc[-1].values[0]
                        close_prev = df['Close'].iloc[-2].values[0]
                    else:
                        close_now = df['Close'].iloc[-1]
                        close_prev = df['Close'].iloc[-2]
                    
                    macro_results[ticker] = (close_now / close_prev) - 1
            except:
                macro_results[ticker] = 0.0
        return macro_results

# --- FUNÇÃO QUE ESTAVA FALTANDO ---
def get_fallback_data(symbol):
    """Função de redundância usando Yahoo Finance diretamente"""
    engine = DataEngine()
    return engine.get_market_data(symbol=symbol, interval="1m")

# =========================================================
# 5. CÉREBRO MATEMÁTICO - INDICADORES E MACRO SCORE
# =========================================================

def calculate_macro_score(macro_changes):
    """Calcula o Score Global (1 a 5) baseado nos Betas"""
    score = 0
    shift_total = 0
    
    for ticker, beta in BETAS_WIN.items():
        if ticker in macro_changes:
            pct_change = macro_changes[ticker]
            impacto = pct_change * beta
            shift_total += impacto
            
            # Lógica de pontuação baseada no impacto individual
            if impacto > 0.001: 
                score += 1
            elif impacto < -0.001: 
                score -= 1
                
    # Limita o score entre -5 e 5
    final_score = int(max(min(score, 5), -5))
    return final_score, shift_total

def compute_technical_indicators(df_m1, df_m5):
    """Processa todos os indicadores do Sniper Ultimate"""
    
    # --- INDICADORES M5 (TENDÊNCIA) ---
    bb_m5 = ta.bbands(df_m5['close'], length=INP_TREND_PER, std=INP_TREND_DEV)
    df_m5['trend_mid'] = bb_m5[f'BBM_{INP_TREND_PER}_{INP_TREND_DEV}']
    
    # --- INDICADORES M1 (ENTRADA) ---
    # Bollinger Bands M1
    bb_m1 = ta.bbands(df_m1['close'], length=INP_ENTRY_PER, std=INP_ENTRY_DEV)
    df_m1['entry_up'] = bb_m1[f'BBU_{INP_ENTRY_PER}_{INP_ENTRY_DEV}']
    df_m1['entry_low'] = bb_m1[f'BBL_{INP_ENTRY_PER}_{INP_ENTRY_DEV}']
    df_m1['entry_mid'] = bb_m1[f'BBM_{INP_ENTRY_PER}_{INP_ENTRY_DEV}']
    
    # RSI M1
    df_m1['rsi'] = ta.rsi(df_m1['close'], length=INP_RSI_PER)
    
    # ATR M1 (Volatilidade para Exaustão e Stop)
    df_m1['atr'] = ta.atr(df_m1['high'], df_m1['low'], df_m1['close'], length=14)
    
    # ADX M1 (Filtro de Tendência Forte)
    adx_df = ta.adx(df_m1['high'], df_m1['low'], df_m1['close'], length=14)
    df_m1['adx'] = adx_df['ADX_14']
    
    return df_m1, df_m5

# =========================================================
# 6. LÓGICA DO NARRADOR (DIAGNÓSTICO EM TEMPO REAL)
# =========================================================

def get_narrator_message(current_price, df_m1, macro_score, fair_value):
    """Gera a mensagem do Narrador baseada no checklist Sniper"""
    
    last_row = df_m1.iloc[-1]
    rsi = last_row['rsi']
    adx = last_row['adx']
    atr = last_row['atr']
    
    # Verifica distância do Preço Justo (Quant)
    dist_fair_value = abs(current_price - fair_value)
    is_safe_dist = dist_fair_value > (180 * 1) # 1 é o _Point simplificado
    
    # Verifica se o Score autoriza
    macro_buy_ok = macro_score >= INP_MIN_SCORE_TRADE
    macro_sell_ok = macro_score <= -INP_MIN_SCORE_TRADE

    # --- CENÁRIO: ESPERANDO TOQUE (Neutro) ---
    if st.session_state.pending_side == 0:
        dist_to_up = df_m1['entry_up'].iloc[-1] - current_price
        dist_to_low = current_price - df_m1['entry_low'].iloc[-1]
        
        if dist_to_up < 500 and dist_to_up > 150:
            return f"⏳ SUBA + {int(dist_to_up)} pts p/ Vender."
        elif dist_to_low < 500 and dist_to_low > 150:
            return f"⏳ DESÇA + {int(dist_to_low)} pts p/ Comprar."
        else:
            return "💤 MEIO DE CAMPO. Aguardando extremos."

    # --- CENÁRIO: AGUARDANDO GATILHO DE VENDA ---
    elif st.session_state.pending_side == -1:
        if st.session_state.wait_counter > 0:
            return f"✋ FILTRO TEMPO: Faltam {st.session_state.wait_counter} velas."
        
        # Checklist de Bloqueios
        if rsi >= INP_RSI_UPPER:
            return f"⛔ BLOQUEIO RSI: {rsi:.1f} (Limite {INP_RSI_UPPER})"
        elif adx > 35 and rsi < 75:
            return f"⛔ BLOQUEIO ADX: {adx:.1f} (Tendência Forte)"
        elif not macro_sell_ok:
            return f"⛔ BLOQUEIO MACRO: Score {macro_score} não permite Venda."
        elif not is_safe_dist:
            return f"⛔ BLOQUEIO MÉDIA: Muito perto ({int(dist_fair_value)} pts)"
        
        return "🔥 DISPARANDO VENDA AGORA!!!"

    # --- CENÁRIO: AGUARDANDO GATILHO DE COMPRA ---
    elif st.session_state.pending_side == 1:
        if st.session_state.wait_counter > 0:
            return f"✋ FILTRO TEMPO: Faltam {st.session_state.wait_counter} velas."
        
        if rsi <= INP_RSI_LOWER:
            return f"⛔ BLOQUEIO RSI: {rsi:.1f} (Limite {INP_RSI_LOWER})"
        elif adx > 35 and rsi > 25:
            return f"⛔ BLOQUEIO ADX: {adx:.1f} (Tendência Forte)"
        elif not macro_buy_ok:
            return f"⛔ BLOQUEIO MACRO: Score {macro_score} não permite Compra."
        elif not is_safe_dist:
            return f"⛔ BLOQUEIO MÉDIA: Muito perto ({int(dist_fair_value)} pts)"
            
        return "🔥 DISPARANDO COMPRA AGORA!!!"
    
    return "Analisando mercado..."

# =========================================================
# 7. GESTÃO DE ORDENS SIMULADAS (PAPER TRADING)
# =========================================================

def open_sim_trade(side, price, sl, tp, is_macro=False):
    """Abre uma posição simulada no estado da sessão"""
    st.session_state.sim_active = True
    st.session_state.sim_side = side       # 1 para Compra, -1 para Venda
    st.session_state.open_price = price
    st.session_state.peak_price = price
    st.session_state.sl_price = sl
    st.session_state.tp_price = tp
    st.session_state.is_macro_trade = is_macro
    st.session_state.partial_done = False
    
    # Registra no log do narrador
    tipo = "COMPRA" if side == 1 else "VENDA"
    st.toast(f"🚀 {tipo} SIMULADA: {price} | SL: {sl} | TP: {tp}")

def close_sim_trade(exit_price, reason="TP/SL"):
    """Fecha a posição e calcula o resultado"""
    side = st.session_state.sim_side
    open_price = st.session_state.open_price
    
    # Cálculo de pontos (simplificado para WIN)
    if side == 1:
        points = exit_price - open_price
    else:
        points = open_price - exit_price
        
    st.session_state.total_points += points
    
    if points > 0:
        st.session_state.wins += 1
        st.session_state.last_profit_time = datetime.now()
    else:
        st.session_state.losses += 1
        
    # Salva no histórico para o Dashboard
    st.session_state.trades_history.append({
        "Data": datetime.now().strftime("%H:%M:%S"),
        "Lado": "Compra" if side == 1 else "Venda",
        "Entrada": open_price,
        "Saída": exit_price,
        "Pontos": points,
        "Motivo": reason
    })
    
    # Reseta o estado
    st.session_state.sim_active = False
    st.session_state.pending_side = 0
    st.session_state.trigger_price = 0.0

# =========================================================
# 8. TRAILING STOP ELÁSTICO V2
# =========================================================

def manage_smart_trailing(current_bid, current_ask, df_m1):
    """Gerencia o Stop Loss dinamicamente baseado no pico de preço"""
    if not st.session_state.sim_active:
        return

    side = st.session_state.sim_side
    entry = st.session_state.open_price
    current_sl = st.session_state.sl_price
    is_macro = st.session_state.is_macro_trade
    
    # Atualiza o pico de preço (Peak) 
    if side == 1: # Compra
        if current_bid > st.session_state.peak_price:
            st.session_state.peak_price = current_bid
        dist_points = st.session_state.peak_price - entry
    else: # Venda
        if current_ask < st.session_state.peak_price:
            st.session_state.peak_price = current_ask
        dist_points = entry - st.session_state.peak_price

    # Ajuste dinâmico de gatilhos (Modo Sniper vs Macro)
    gatilho_be = 150 if is_macro else 60
    gatilho_elastico = 250 if is_macro else 100
    gatilho_tendencia = 400 if is_macro else 150
    gap_financeiro = 180 if is_macro else 70
    min_lucro = 80 if is_macro else 50

    new_sl = current_sl
    should_modify = False

    # FASE 1: Proteção Inicial (Break-Even)
    if dist_points >= gatilho_be and dist_points < gatilho_elastico:
        be_price = entry + 10 if side == 1 else entry - 10
        if (side == 1 and current_sl < be_price) or (side == -1 and current_sl > be_price):
            new_sl = be_price
            should_modify = True

    # FASE 2: Elástico (Garante Lucro)
    elif dist_points >= gatilho_elastico and dist_points < gatilho_tendencia:
        if side == 1:
            target = max(st.session_state.peak_price - gap_financeiro, entry + min_lucro)
            if target > current_sl:
                new_sl = target
                should_modify = True
        else:
            target = min(st.session_state.peak_price + gap_financeiro, entry - min_lucro)
            if target < current_sl:
                new_sl = target
                should_modify = True

    # FASE 3: Tendência Longa
    elif dist_points >= gatilho_tendencia:
        trailing_gap = 60 # Simplificado sem Z-Score para nuvem
        if side == 1:
            target = st.session_state.peak_price - trailing_gap
            if target > current_sl:
                new_sl = target
                should_modify = True
        else:
            target = st.session_state.peak_price + trailing_gap
            if target < current_sl:
                new_sl = target
                should_modify = True

    if should_modify:
        st.session_state.sl_price = new_sl

# =========================================================
# 9. FILTRO DE EXAUSTÃO E GATILHOS
# =========================================================

def check_exhaustion_filter(current_price, df_m1):
    """Cancela o trade se o preço fugir demais do toque original"""
    if st.session_state.wait_counter > 0:
        atr = df_m1['atr'].iloc[-1]
        max_run = atr * 2.0
        
        # Simula o Anchor_Price_Touch
        anchor = df_m1['entry_up'].iloc[-5] if st.session_state.pending_side == -1 else df_m1['entry_low'].iloc[-5]
        dist_run = abs(current_price - anchor)
        
        if dist_run > max_run:
            st.session_state.pending_side = 0
            st.session_state.wait_counter = 0
            return True # Exausto
    return False

# =========================================================
# 10. INTERFACE VISUAL - DASHBOARD STREAMLIT
# =========================================================

def render_dashboard(current_price, macro_score, shift, df_m1, narrator_msg):
    """Renderiza o painel estilo HUD para visualização mobile"""
    
    # Configuração de Página para Celular
    st.set_page_config(page_title="Sniper AI Monitor", layout="centered")
    
    # Título Principal
    st.markdown(f"<h2 style='text-align: center; color: #FFD700;'>🎯 SNIPER AI - NARRADOR v8.0</h2>", unsafe_content_type=True)
    
    # --- LINHA 1: MÉTRICAS PRINCIPAIS (HUD) ---
    col1, col2 = st.columns(2)
    
    with col1:
        # Status Atual com cor dinâmica
        state = "CAÇANDO GATILHO" if st.session_state.pending_side != 0 else "ESCANEANDO"
        st.metric("STATUS ATUAL", state)
        
        # Leitura de Fluxo/Macro
        flow_txt = f"SCORE: {macro_score:+}"
        st.metric("LEITURA FLUXO", flow_txt, delta=f"{shift:.4f}")

    with col2:
        # Preço de Gatilho
        trig_val = f"{st.session_state.trigger_price:.0f}" if st.session_state.trigger_price > 0 else "OFF"
        st.metric("PREÇO GATILHO", trig_val)
        
        # Placar (W/L)
        score_txt = f"{st.session_state.wins} x {st.session_state.losses}"
        st.metric("PLACAR (W/L)", score_txt)

    st.divider()

    # --- LINHA 2: NARRADOR (O QUE FALTA?) ---
    st.subheader("O QUE FALTA?")
    if "🔥" in narrator_msg:
        st.error(narrator_msg) # Destaque para disparo
    elif "⛔" in narrator_msg or "✋" in narrator_msg:
        st.warning(narrator_msg) # Destaque para bloqueio
    else:
        st.info(narrator_msg) # Destaque neutro

    # --- LINHA 3: RESULTADOS FINANCEIROS E PONTOS ---
    c_pnl, c_pts = st.columns(2)
    
    with c_pnl:
        pnl_color = "green" if st.session_state.total_points >= 0 else "red"
        st.markdown(f"**RESULTADO:** <span style='color:{pnl_color}; font-size:20px;'>R$ {st.session_state.total_points * 0.20:.2f}</span>", unsafe_content_type=True)
    
    with c_pts:
        st.markdown(f"**TOTAL PONTOS:** <span style='color:cyan; font-size:20px;'>{int(st.session_state.total_points)} pts</span>", unsafe_content_type=True)

    # --- LINHA 4: GRÁFICO E HISTÓRICO --- 
    if st.session_state.trades_history:
        with st.expander("Ver Histórico de Trades Simulados"):
            hist_df = pd.DataFrame(st.session_state.trades_history)
            st.table(hist_df.tail(10)) # Mostra os últimos 10 trades

    # --- SIDEBAR: CONFIGURAÇÕES AO VIVO ---
    st.sidebar.header("Configurações Sniper")
    mode = st.sidebar.toggle("Modo Simulação Ativo", value=True)
    if not mode:
        st.sidebar.error("Atenção: Apenas Simulação nesta versão Cloud.")
    
    # Inputs que você pode mudar pelo celular
    st.session_state.lots = st.sidebar.number_input("Lotes (Contratos)", value=2.0, step=1.0)
    st.sidebar.markdown(f"**Preço Atual WIN:** {current_price}")
    
    # Botão de Reset
    if st.sidebar.button("Resetar Placar do Dia"):
        st.session_state.trades_history = []
        st.session_state.total_points = 0.0
        st.session_state.wins = 0
        st.session_state.losses = 0
        st.rerun()

# =========================================================
# 11. MOTOR DE EXECUÇÃO - LÓGICA DE DECISÃO (ONTICK)
# =========================================================

@st.cache_resource
def get_engine():
    return DataEngine()

def main():
    engine = DataEngine()
    
    # 2. Correção de tipos: Passando strings diretas para evitar o erro de 'Interval'
    df_m1 = engine.get_market_data(symbol="^BVSP", interval="1m", n_bars=100)
    df_m5 = engine.get_market_data(symbol="^BVSP", interval="5m", n_bars=50)
    macro_changes = engine.get_macro_prices()
    
    # --- AJUSTE DE SEGURANÇA: FALLBACK ---
    if df_m1.empty or df_m5.empty:
        st.warning("📡 Feed Principal indisponível. Tentando redundância...")
        # Agora a função existe!
        df_m1_alt = get_fallback_data("^BVSP") 
        
        if not df_m1_alt.empty:
            df_m1 = df_m1_alt
            df_m5 = df_m1.resample('5min').last().ffill()
            st.success("✅ Conectado via Redundância (Yahoo Finance).")
        else:
            st.error("❌ ERRO: Não foi possível obter dados.")
            if st.button("🔄 Tentar Reconectar Agora"):
                st.rerun()
            return

    # 2. PROCESSAMENTO TÉCNICO
    df_m1, df_m5 = compute_technical_indicators(df_m1, df_m5)
    macro_score, shift = calculate_macro_score(macro_changes)
    
    # Preços atuais para o Simulador 
    current_bid = df_m1['close'].iloc[-1]
    current_ask = current_bid + 5 # Simulação de Spread fixo (5 pts)
    
    # Cálculo do Preço Justo (Fair Value / Quant)
    ref_price = df_m5['close'].iloc[-1] # Simplificado para o fechamento anterior
    fair_value = ref_price * (1.0 + shift)
    
    # 3. GESTÃO DE TRADES ATIVOS (Verifica SL/TP e Trailing)
    if st.session_state.sim_active:
        side = st.session_state.sim_side
        sl = st.session_state.sl_price
        tp = st.session_state.tp_price
        
        # Gerencia Trailing Stop
        manage_smart_trailing(current_bid, current_ask, df_m1)
        
        # Verifica saída por Stop ou Take
        if side == 1: # Compra
            if current_bid >= tp: close_sim_trade(tp, "Take Profit")
            elif current_bid <= sl: close_sim_trade(sl, "Stop Loss")
        else: # Venda
            if current_ask <= tp: close_sim_trade(tp, "Take Profit")
            elif current_ask >= sl: close_sim_trade(sl, "Stop Loss")

    # 4. MONITORAMENTO DE NOVAS ENTRADAS
    else:
        last_row = df_m1.iloc[-1]
        
        # FASE 1: Busca de Toque nas Bandas Quant/Bollinger 
        if st.session_state.pending_side == 0:
            buffer = INP_BAND_BUFFER
            if current_ask <= (last_row['entry_low'] + buffer):
                st.session_state.pending_side = 1
                st.session_state.wait_counter = INP_WAIT_CANDLES
                # Reentrada Inteligente (Reduz espera se houve lucro recente) 
                if st.session_state.last_profit_time and (datetime.now() - st.session_state.last_profit_time).seconds < 600:
                    st.session_state.wait_counter = 2
            
            elif current_bid >= (last_row['entry_up'] - buffer):
                st.session_state.pending_side = -1
                st.session_state.wait_counter = INP_WAIT_CANDLES
                if st.session_state.last_profit_time and (datetime.now() - st.session_state.last_profit_time).seconds < 600:
                    st.session_state.wait_counter = 2

        # FASE 2: Gestão do Gatilho e Exaustão
        else:
            # Verifica se o preço já atingiu o alvo antes da entrada (Proteção Quant)
            if (st.session_state.pending_side == 1 and current_bid >= fair_value) or \
               (st.session_state.pending_side == -1 and current_ask <= fair_value):
                st.session_state.pending_side = 0
                st.session_state.trigger_price = 0
            
            # Filtro de Exaustão 
            is_exhausted = check_exhaustion_filter(current_bid, df_m1)
            
            if not is_exhausted:
                # Define Gatilho no fechamento da última vela
                h1, l1 = df_m1['high'].iloc[-2], df_m1['low'].iloc[-2]
                pad = INP_BREAKOUT
                
                if st.session_state.pending_side == 1: # Compra
                    st.session_state.trigger_price = h1 + pad
                else: # Venda
                    st.session_state.trigger_price = l1 - pad
                
                # FASE 3: Disparo da Ordem (Verifica checklist do Narrador) 
                msg = get_narrator_message(current_bid, df_m1, macro_score, fair_value)
                if "DISPARANDO" in msg:
                    # Define Stop e Take baseados na volatilidade atual
                    atr_pts = last_row['atr']
                    sl_dist = max(atr_pts * 2.5, 150)
                    
                    if st.session_state.pending_side == 1:
                        sl = current_bid - sl_dist
                        tp = current_bid + INP_TAKE_POINTS
                        # Se for Modo Macro, alvo é no Preço Justo
                        if abs(macro_score) >= 3: tp = fair_value
                        open_sim_trade(1, current_ask, sl, tp, is_macro=(abs(macro_score) >= 3))
                    else:
                        sl = current_ask + sl_dist
                        tp = current_ask - INP_TAKE_POINTS
                        if abs(macro_score) >= 3: tp = fair_value
                        open_sim_trade(-1, current_bid, sl, tp, is_macro=(abs(macro_score) >= 3))

    # 5. ATUALIZAÇÃO DO DASHBOARD
    narrator_msg = get_narrator_message(current_bid, df_m1, macro_score, fair_value)
    render_dashboard(current_bid, macro_score, shift, df_m1, narrator_msg)

    # 6. LOOP INFINITO (Auto-Refresh a cada 2 segundos)
    time.sleep(2)
    st.rerun()

# --- INICIALIZAÇÃO DO SCRIPT ---
if __name__ == "__main__":

    main()






