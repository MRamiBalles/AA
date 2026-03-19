import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .dataset_factory import get_enjoysport, get_playtennis, get_synthetic_symbolic

class VersionSpaceEngine:
    """
    Motor lógico para el algoritmo de Eliminación de Candidatos.
    Mantiene las fronteras S (específica) y G (general).
    """
    def __init__(self, n_features: int, domain: List[List[str]]):
        self.n_features = n_features
        self.domain = domain
        self.S = [['ø'] * n_features]
        self.G = [['?'] * n_features]

    def _is_consistent(self, h: List[str], x: np.ndarray) -> bool:
        """Verifica si la hipótesis h cubre el ejemplo x."""
        for i in range(len(h)):
            if h[i] != '?' and h[i] != x[i]:
                return False
        return True

    def _is_more_general(self, h_gen: List[str], h_spec: List[str]) -> bool:
        """Determina si h_gen es igual o más general que h_spec."""
        for i in range(len(h_gen)):
            if h_gen[i] != '?' and (h_gen[i] != h_spec[i] or h_spec[i] == '?'):
                return False
        return True

    def update(self, x: np.ndarray, is_positive: bool):
        """Aplica un paso de aprendizaje basado en un nuevo ejemplo."""
        if is_positive:
            # Ejemplo Positivo: Podar G inconsistente y Generalizar S
            self.G = [g for g in self.G if self._is_consistent(g, x)]
            for s in self.S:
                if not self._is_consistent(s, x):
                    for j in range(self.n_features):
                        if s[j] == 'ø':
                            s[j] = x[j]
                        elif s[j] != x[j]:
                            s[j] = '?'
        else:
            # Ejemplo Negativo: Podar S inconsistente y Especializar G
            self.S = [s for s in self.S if not self._is_consistent(s, x)]
            new_G = []
            for g in self.G:
                if self._is_consistent(g, x):
                    for j in range(self.n_features):
                        if g[j] == '?':
                            for val in self.domain[j]:
                                if val != x[j]:
                                    g_cand = list(g)
                                    g_cand[j] = val
                                    # Solo mantener si es consistente con al menos un s en S
                                    if any(self._is_more_general(g_cand, s) for s in self.S):
                                        if g_cand not in new_G:
                                            new_G.append(g_cand)
                else:
                    new_G.append(g)
            
            # Limpieza: eliminar g que no sean más generales que algún s
            self.G = [g for g in new_G if any(self._is_more_general(g, s) for s in self.S)]

def render():
    st.header("Candidate Elimination: Gestión del Version Space")
    st.markdown("""
    Este módulo visualiza la convergencia de las fronteras lógica **S** (Específica) y **G** (General). 
    A medida que procesamos evidencias, el espacio de hipótesis consistentes se contrae.
    """)

    talla = st.radio("Escala del Dataset", ["S (EnjoySport)", "M (PlayTennis)", "L (Synthetic)"], horizontal=True)
    if talla.startswith("S"):
        df = get_enjoysport()
    elif talla.startswith("M"):
        df = get_playtennis()
    else:
        df = get_synthetic_symbolic(n_samples=20)

    st.subheader("Datos de Entrenamiento")
    st.dataframe(df)

    # Inicialización del motor
    n_features = len(df.columns) - 1
    domain = [list(df[col].unique()) for col in df.columns[:-1]]
    engine = VersionSpaceEngine(n_features, domain)
    
    history: List[Dict[str, Any]] = []

    # Ejecución de la traza
    for i, row in df.iterrows():
        x = row[:-1].values
        label = row.iloc[-1]
        # Heurística para detectar si el ejemplo es positivo (Yes, Normal, o primer valor del dominio target)
        is_pos = (label.lower() in ['yes', 'target_yes', '1', 'true', 'normal'])
        if talla.startswith("L"): is_pos = (label == 'Yes') # Específico para el sintético
        
        engine.update(x, is_pos)
        history.append({'S': [list(s) for s in engine.S], 'G': [list(g) for g in engine.G]})

    # Navegación Interactiva
    step = st.select_slider("Evolución temporal del aprendizaje", options=list(range(len(df))), value=len(df)-1)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("🎯 **Frontera S** (Más Específica)")
        for s in history[step]['S']:
            st.code(f"S: {s}")
    with c2:
        st.warning("🌐 **Frontera G** (Más General)")
        for g in history[step]['G']:
            st.code(f"G: {g}")

    st.markdown(f"**Estado tras ejemplo {step+1}:** `{df.iloc[step].values}`")
