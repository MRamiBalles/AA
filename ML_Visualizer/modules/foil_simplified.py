import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .dataset_factory import get_playtennis

class FOILRuleLearner:
    """
    Motor de inducción de reglas basado en FOIL (First Order Inductive Learner).
    En esta versión, se enfoca en lógica proposicional y ganancia de información.
    """
    def __init__(self, target_name: str):
        self.target_name = target_name

    def _foil_gain(self, p: int, n: int, P: int, N: int) -> float:
        """Calcula la ganancia de FOIL para un nuevo literal."""
        if p == 0:
            return 0.0
        return p * (np.log2(p / (p + n)) - np.log2(P / (P + N)))

    def get_candidate_literals(self, df: pd.DataFrame, pos_examples: pd.DataFrame, neg_examples: pd.DataFrame) -> List[Dict[str, Any]]:
        """Busca el mejor literal para añadir a la regla actual."""
        literals = []
        P, N = len(pos_examples), len(neg_examples)
        
        for col in df.columns[:-1]:
            for val in df[col].unique():
                subset_pos = pos_examples[pos_examples[col] == val]
                subset_neg = neg_examples[neg_examples[col] == val]
                p, n = len(subset_pos), len(subset_neg)
                
                gain = self._foil_gain(p, n, P, N)
                literals.append({
                    'Atributo': col,
                    'Valor': val,
                    'p': p,
                    'n': n,
                    'Gain': gain
                })
        return literals

def render():
    st.header("FOIL: Inducción de Reglas Secuenciales")
    st.markdown("""
    FOIL construye reglas mediante una búsqueda **greedy** de cláusulas que maximizan la ganancia 
    de información lógica, reduciendo la cobertura de ejemplos negativos sin perder los positivos.
    """)

    df = get_playtennis()
    st.subheader("Dataset de Entrenamiento")
    st.dataframe(df)

    target_name = df.columns[-1]
    pos_examples = df[df[target_name] == 'Yes']
    neg_examples = df[df[target_name] == 'No']

    # Motor de inducción
    learner = FOILRuleLearner(target_name)
    candidates = learner.get_candidate_literals(df, pos_examples, neg_examples)
    
    cand_df = pd.DataFrame(candidates).sort_values(by='Gain', ascending=False)
    
    st.subheader("Cálculo de Ganancia para el Primer Literal")
    st.write("Literales candidatos ordenados por ganancia:")
    st.dataframe(cand_df)

    best = cand_df.iloc[0]
    st.success(f"**Mejor Literal seleccionado:** `SI {best['Atributo']} ES {best['Valor']} ENTONCES ...` (Ganancia: {best['Gain']:.3f})")

    st.markdown("---")
    st.subheader("Borrador de Reglas (Simulación de FOIL)")
    st.code("""
    1. SI Outlook = Overcast ENTONCES Play = Yes
    2. SI Humidity = Normal Y Wind = Weak ENTONCES Play = Yes
    3. SI Outlook = Rain Y Wind = Weak ENTONCES Play = Yes
    """)
