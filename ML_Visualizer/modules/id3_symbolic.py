import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
from .dataset_factory import get_playtennis, get_synthetic_symbolic

class ID3DecisionTree:
    """
    Implementación simbólica del algoritmo ID3.
    Construye árboles de decisión basados en Ganancia de Información (Entropía).
    """
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.tree = {}

    def _entropy(self, labels: pd.Series) -> float:
        """Calcula la entropía de un conjunto de etiquetas."""
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return -np.sum(probs * np.log2(probs))

    def _info_gain(self, df: pd.DataFrame, attribute: str) -> float:
        """Calcula la ganancia de información para un atributo dado."""
        total_entropy = self._entropy(df[self.target_name])
        vals, counts = np.unique(df[attribute], return_counts=True)
        
        weighted_entropy = 0
        for i in range(len(vals)):
            subset = df[df[attribute] == vals[i]]
            weighted_entropy += (counts[i] / len(df)) * self._entropy(subset[self.target_name])
            
        return total_entropy - weighted_entropy

    def fit(self, df: pd.DataFrame, features: List[str]):
        """Construye el árbol de forma recursiva."""
        self.tree = self._build_tree(df, df, features)

    def _build_tree(self, df: pd.DataFrame, original_df: pd.DataFrame, features: List[str], parent_class=None) -> Any:
        # Casos base de la recursión
        if len(np.unique(df[self.target_name])) <= 1:
            return np.unique(df[self.target_name])[0]
        elif df.empty:
            return np.unique(original_df[self.target_name])[np.argmax(np.unique(original_df[self.target_name], return_counts=True)[1])]
        elif not features:
            return parent_class
        
        # Clase mayoritaria para el nodo actual
        major_class = np.unique(df[self.target_name])[np.argmax(np.unique(df[self.target_name], return_counts=True)[1])]
        
        # Selección del mejor atributo
        gains = [self._info_gain(df, f) for f in features]
        best_feat = features[np.argmax(gains)]
        
        tree = {best_feat: {}}
        remaining_feats = [f for f in features if f != best_feat]
        
        for val in np.unique(original_df[best_feat]):
            subset = df[df[best_feat] == val]
            subtree = self._build_tree(subset, original_df, remaining_feats, major_class)
            tree[best_feat][val] = subtree
            
        return tree

    def get_rules(self, tree: Union[Dict, str], indent: str = "") -> str:
        """Convierte la estructura del árbol en reglas legibles."""
        if not isinstance(tree, dict):
            return f" -> **{tree}**"
        
        rules = ""
        for feat, branches in tree.items():
            for val, subtree in branches.items():
                rules += f"\n{indent}SI **{feat}** ES **{val}**" + self.get_rules(subtree, indent + "    ")
        return rules

def render():
    st.header("ID3: Árboles de Decisión Simbólicos")
    st.markdown("""
    El algoritmo ID3 construye un árbol de decisión minimizando la entropía (incertidumbre) del dataset 
    en cada división. A diferencia de las versiones numéricas, aquí las decisiones son discretas y simbólicas.
    """)

    talla = st.radio("Escala del Dataset", ["M (PlayTennis)", "L (Synthetic)"], index=0, horizontal=True)
    df = get_playtennis() if talla.startswith("M") else get_synthetic_symbolic(n_samples=50)

    target = df.columns[-1]
    features = list(df.columns[:-1])
    
    # Análisis de Ganancia
    engine = ID3DecisionTree(target)
    gains = {f: engine._info_gain(df, f) for f in features}
    
    st.subheader("Análisis de Selección de Atributos")
    st.bar_chart(pd.Series(gains, name="Ganancia de Información"))

    # Generación y Visualización
    engine.fit(df, features)
    
    st.subheader("Estructura Lógica del Árbol")
    st.json(engine.tree)
    
    st.markdown("---")
    st.subheader("Reglas de Inferencia Extraídas")
    st.markdown(engine.get_rules(engine.tree))
