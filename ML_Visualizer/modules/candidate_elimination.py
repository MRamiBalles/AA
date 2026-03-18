import streamlit as st
import pandas as pd
import numpy as np
from .dataset_factory import get_enjoysport, get_playtennis, get_synthetic_symbolic

class CandidateElimination:
    def __init__(self, n_features):
        self.n_features = n_features
        self.S = [['ø'] * n_features]
        self.G = [['?'] * n_features]

    def is_consistent(self, h, x, target):
        return self.h_predict(h, x) == target

    def h_predict(self, h, x):
        for i in range(len(h)):
            if h[i] != '?' and h[i] != x[i]:
                return False
        return True

    def train_step(self, x, target):
        if target: # Positive Example
            # 1. Remove inconsistent hypothesis from G
            self.G = [g for g in self.G if self.h_predict(g, x)]
            
            # 2. Generalize S to be consistent with x
            new_S = []
            for s in self.S:
                if not self.h_predict(s, x):
                    # Generalize s
                    for i in range(self.n_features):
                        if s[i] == 'ø':
                            s[i] = x[i]
                        elif s[i] != x[i]:
                            s[i] = '?'
                new_S.append(s)
            self.S = new_S

        else: # Negative Example
            # 1. Remove inconsistent hypothesis from S
            self.S = [s for s in self.S if not self.h_predict(s, x)]
            
            # 2. Specialize G to exclude x
            new_G = []
            for g in self.G:
                if self.h_predict(g, x):
                    # Specialize g
                    for i in range(self.n_features):
                        if g[i] == '?':
                            for val in self.get_possible_values(i): # This needs domain knowledge
                                if val != x[i]:
                                    g_spec = list(g)
                                    g_spec[i] = val
                                    # Check consistency with S
                                    if any(self.is_more_specific(s, g_spec) for s in self.S):
                                        new_G.append(g_spec)
                else:
                    new_G.append(g)
            self.G = new_G

    def is_more_specific(self, h1, h2):
        for i in range(len(h1)):
            if h2[i] != '?' and (h1[i] == '?' or h1[i] != h2[i]):
                return False
        return True

def render():
    st.header("Candidate Elimination: El Espacio de Versiones")
    st.markdown("""
    Este algoritmo mantiene **dos fronteras**:
    *   **S (Específica):** La hipótesis más restrictiva que cubre lo visto.
    *   **G (General):** Las hipótesis más amplias que excluyen lo negativo.
    """)

    # Selection of Dataset
    talla = st.radio("Talla del problema", ["S (EnjoySport)", "M (PlayTennis)", "L (Synthetic)"], horizontal=True)
    if talla.startswith("S"):
        df = get_enjoysport()
    elif talla.startswith("M"):
        df = get_playtennis()
    else:
        df = get_synthetic_symbolic(n_samples=20)

    st.subheader("Dataset Seleccionado")
    st.dataframe(df)

    # Simplified implementation for the UI step-by-step
    n_features = len(df.columns) - 1
    S = [['ø'] * n_features]
    G = [['?'] * n_features]
    
    # Track history for visualization
    history = []

    # Domain values for G specialization
    domain = []
    for col in df.columns[:-1]:
        domain.append(list(df[col].unique()))

    def is_consistent(h, x):
        for i in range(len(h)):
            if h[i] != '?' and h[i] != x[i]:
                return False
        return True

    def is_more_general(h1, h2):
        # h1 more general than h2
        for i in range(len(h1)):
            if h1[i] != '?' and (h1[i] != h2[i] or h2[i] == '?'):
                return False
        return True

    # Simulation loop
    for i, row in df.iterrows():
        x = row[:-1].values
        target = row.iloc[-1] == 'Yes'
        
        if target: # POSITIVE
            G = [g for g in G if is_consistent(g, x)]
            for s in S:
                if not is_consistent(s, x):
                    for j in range(n_features):
                        if s[j] == 'ø': s[j] = x[j]
                        elif s[j] != x[j]: s[j] = '?'
        else: # NEGATIVE
            S = [s for s in S if not is_consistent(s, x)]
            new_G = []
            for g in G:
                if is_consistent(g, x):
                    for j in range(n_features):
                        if g[j] == '?':
                            for val in domain[j]:
                                if val != x[j]:
                                    g_spec = list(g)
                                    g_spec[j] = val
                                    if any(is_more_general(g_spec, s_val) for s_val in S):
                                        if g_spec not in new_G: new_G.append(g_spec)
                else:
                    new_G.append(g)
            G = [g for g in new_G if any(is_more_general(g, s_val) for s_val in S)]

        history.append({'S': [list(s) for s in S], 'G': [list(g) for g in G]})

    # Visualizer
    idx = st.slider("Ver evolución tras ejemplo:", 0, len(df)-1, len(df)-1)
    
    col_s, col_g = st.columns(2)
    with col_s:
        st.info("🎯 Frontera Específica (S)")
        st.write(history[idx]['S'])
    with col_g:
        st.warning("🌐 Frontera General (G)")
        st.write(history[idx]['G'])

    st.success(f"Ejemplo {idx+1}: {df.iloc[idx].values}")
