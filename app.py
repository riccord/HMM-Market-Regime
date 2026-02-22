import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from hmmlearn.hmm import GaussianHMM 

class HMM():
    def __init__(self, df):
        self.df = df.ffill().bfill()

    def modellazione(self):
        data = self.df.copy()
        returns = np.log(data / data.shift(1))
        returns = returns.dropna()
        return returns

    def regimi(self, returns):
        vol_30 = returns.rolling(window=30).std().dropna()
        soglia_bassa = vol_30.quantile(0.33)
        soglia_alta = vol_30.quantile(0.66)

        def assegna_stato(v):
            if v <= soglia_bassa: return 0
            elif v <= soglia_alta: return 1
            else: return 2
            
        stati = vol_30.apply(assegna_stato)
        return stati, soglia_alta, soglia_bassa

    def plot_streamlit(self, stati, returns):
        cum = returns.cumsum().apply(np.exp)
        cum = cum.loc[stati.index]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(cum.index, cum.values, color='black', linewidth=1.5, label='Price Equity', zorder=2)
        
        colori = {0: '#dcedc8', 1: '#fff9c4', 2: '#ffccbc'}
        
        for i in range(len(stati) - 1):
            ax.axvspan(stati.index[i], stati.index[i+1], color=colori[stati.iloc[i]], alpha=0.6, zorder=1)

        ax.set_title("Analisi dei Regimi di Volatilità", fontsize=14)
        ax.set_ylabel("Equity / Prezzo")
        ax.legend()
        ax.grid(alpha=0.3)
        return fig

class Returns():
    def __init__(self, stati, returns):
        self.returns_allineati = returns.loc[stati.index]
        self.stati = stati
        self.ret_0 = self.returns_allineati.where(stati == 0, 0)
        self.ret_1 = self.returns_allineati.where(stati == 1, 0)
        self.ret_2 = self.returns_allineati.where(stati == 2, 0)
        
    def plot_rit_streamlit(self):
        eq_0 = self.ret_0.cumsum().apply(np.exp)
        eq_1 = self.ret_1.cumsum().apply(np.exp)
        eq_2 = self.ret_2.cumsum().apply(np.exp)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(eq_0, label="Bassa Volatilità (0)", color='#2ecc71')
        ax.plot(eq_1, label="Media Volatilità (1)", color='#f1c40f')
        ax.plot(eq_2, label="Alta Volatilità (2)", color='#e74c3c')
        ax.set_title("Performance Cumulata per Regime")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    def statistiche(self):
        r0 = (np.exp(self.ret_0.sum()) - 1) * 100
        r1 = (np.exp(self.ret_1.sum()) - 1) * 100
        r2 = (np.exp(self.ret_2.sum()) - 1) * 100
        
        v0 = self.ret_0[self.stati == 0].std() * np.sqrt(252) * 100
        v1 = self.ret_1[self.stati == 1].std() * np.sqrt(252) * 100
        v2 = self.ret_2[self.stati == 2].std() * np.sqrt(252) * 100
        
        s0 = (self.ret_0[self.stati == 0].mean() * 252) / (v0 / 100) if v0 != 0 else 0
        s1 = (self.ret_1[self.stati == 1].mean() * 252) / (v1 / 100) if v1 != 0 else 0
        s2 = (self.ret_2[self.stati == 2].mean() * 252) / (v2 / 100) if v2 != 0 else 0

        return pd.DataFrame({
            'Stato': ['Bassa Vol (0)', 'Media Vol (1)', 'Alta Vol (2)'],
            'Rendimento Tot %': [r0, r1, r2],
            'Volatilità Annua %': [v0, v1, v2],
            'Sharpe Ratio': [s0, s1, s2]
        }).set_index('Stato')

# === FUNZIONE PER CALCOLO HMM DINAMICO ===
def calcola_matrice_hmm(data_subset):
    X = data_subset.values.reshape(-1, 1)
    
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)
    
    trans_mat = model.transmat_
    
    varianze = np.array([model.covars_[i][0][0] for i in range(3)])
    ordine = np.argsort(varianze) # Indici dal più piccolo al più grande
    
    mat_ordinata = trans_mat[ordine, :][:, ordine]
    
    nomi = ['Bassa Vol (0)', 'Media Vol (1)', 'Alta Vol (2)']
    return pd.DataFrame(mat_ordinata, index=nomi, columns=nomi)

# === STREAMLIT APP ===
st.set_page_config(page_title="Analisi Regime", layout="wide")

st.sidebar.title("👤 About Me")
st.sidebar.info("""
**Cordeschi Riccardo**
*Quantitative Finance Enthusiast*
- [LinkedIn](https://linkedin.com/in/tuoprofilo)
- [GitHub](https://github.com/tuoprofilo)
""")

st.title("Quantitative Regime Analysis")
uploaded_file = st.sidebar.file_uploader("Carica il tuo database CSV", type=["csv"])

@st.cache_data 
def load_data():
    return pd.read_csv("database_asset_management.csv", index_col=0, parse_dates=True)

try:
    full_df = load_data()
except FileNotFoundError:
    st.error("Errore: Il file 'database_asset_management.csv' non è stato trovato nel repository.")
    st.stop()

asset_name = st.sidebar.selectbox("Seleziona l'asset da analizzare", full_df.columns)
df_asset = full_df[asset_name]

    hmm_class = HMM(df_asset)
    ret = hmm_class.modellazione()
    stati, s_alta, s_bassa = hmm_class.regimi(ret)
    analisi_rit = Returns(stati, ret)

    tab1, tab2, tab3 = st.tabs(["Grafico Regimi", "Performance", "Probabilità (HMM)"])

    with tab1:
        st.subheader(f"Equity Line e Regimi: {asset_name}")
        st.pyplot(hmm_class.plot_streamlit(stati, ret))

    with tab2:
        st.subheader("Statistiche di Performance")
        stats = analisi_rit.statistiche()
        st.table(stats.style.format("{:.2f}"))
        st.pyplot(analisi_rit.plot_rit_streamlit())

    with tab3:
        st.header("Teoria dei Modelli di Markov Nascosti (HMM)")
        
        col_teoria_1, col_teoria_2 = st.columns(2)
        with col_teoria_1:
            st.markdown("""
            Un **Hidden Markov Model** è un modello statistico in cui si assume che il sistema sia un processo di Markov con stati non osservabili (nascosti). 
            Nel nostro caso, lo stato "nascosto" è il **Regime di Volatilità**.
            
            La probabilità di transizione $P$ definisce la dinamica del passaggio tra stati:
            """)
            st.latex(r"P(S_{t+1} = j \mid S_t = i) = p_{ij}")
        
        with col_teoria_2:
            st.markdown("""
            La matrice di transizione deve soddisfare la proprietà per cui la somma di ogni riga è pari a 1:
            """)
            st.latex(r"\sum_{j=1}^{N} p_{ij} = 1")
            st.markdown("Dove $N$ è il numero di stati (nel nostro caso 3).")

        st.divider()
        
        st.subheader("Analisi Dinamica della Matrice di Transizione")
        st.write("Usa la barra sottostante per selezionare la finestra temporale di analisi. Il modello HMM ricalcolerà le probabilità basandosi esclusivamente sui dati selezionati.")

        # Slider per la selezione della data
        date_min = ret.index.min().to_pydatetime()
        date_max = ret.index.max().to_pydatetime()
        
        selected_range = st.slider(
            "Seleziona l'intervallo temporale:",
            min_value=date_min,
            max_value=date_max,
            value=(date_min, date_max),
            format="DD/MM/YYYY"
        )
        
        # Filtraggio dati in base allo slider
        mask = (ret.index >= selected_range[0]) & (ret.index <= selected_range[1])
        ret_filtrati = ret.loc[mask]

        if len(ret_filtrati) > 50: # Evitiamo errori con troppi pochi dati
            try:
                mat_hmm = calcola_matrice_hmm(ret_filtrati)
                
                col_mat_1, col_mat_2 = st.columns([2, 1])
                
                with col_mat_1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(mat_hmm, annot=True, cmap='Blues', fmt='.2%', ax=ax, cbar=False)
                    ax.set_title(f"Matrice di Transizione HMM ({selected_range[0].year} - {selected_range[1].year})")
                    st.pyplot(fig)
                
                with col_mat_2:
                    st.write("**Insights dei Regimi:**")
                    prob_persistenza = np.diag(mat_hmm).mean()
                    st.metric("Persistenza Media", f"{prob_persistenza:.2%}")
                    st.write("""
                    I valori sulla diagonale principale indicano la probabilità che il mercato rimanga nel regime attuale. 
                    Valori elevati (>90%) indicano regimi molto stabili e persistenti.
                    """)
            except:
                st.error("Errore nel calcolo del modello HMM per le date selezionate. Prova ad ampliare l'intervallo.")
        else:
            st.warning("Seleziona un intervallo più ampio per permettere l'allenamento del modello HMM.")

else:

    st.info("Carica un file CSV per iniziare.")


