import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
import streamlit.components.v1 as components
from .dataset_factory import get_playtennis, get_synthetic_symbolic

def render_entropy_intuition():
    """
    Visualización interactiva para explicar la entropía como medida de desorden.
    """
    st.subheader("Intuición: Entropía y Ganancia")
    
    st.markdown("""
    La **Entropía ($H$)** cuantifica la incertidumbre o el "desorden" de un conjunto de datos. 
    En clasificación, un conjunto es **homogéneo** (Entropía 0) si todas las muestras pertenecen 
    a la misma clase. Si las clases están equitativamente repartidas, la incertidumbre es máxima ($H=1$ en binario).
    """)

    # --- Animación de Bolas ---
    html_code = """
    <div id="container" style="border: 2px solid #555; width: 500px; height: 300px; position: relative; border-radius: 10px; background: #111; overflow: hidden; margin: auto;">
        <div id="divider" style="position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; border-left: 1px dashed #444; display: none;"></div>
    </div>
    <div style="text-align: center; margin-top: 15px;">
        <button onclick="randomize()" style="padding: 10px 20px; border-radius: 5px; cursor: pointer; background: #444; color: white; border: none; margin-right: 10px;">Mezclar (Alta Entropía)</button>
        <button onclick="sortBalls()" style="padding: 10px 20px; border-radius: 5px; cursor: pointer; background: #007bff; color: white; border: none;">Particionar (Baja Entropía / Ganancia)</button>
    </div>
    <p id="stats" style="text-align: center; font-family: sans-serif; color: #aaa; margin-top: 10px;">Estado: Inicial</p>

    <script>
        const container = document.getElementById('container');
        const stats = document.getElementById('stats');
        const divider = document.getElementById('divider');
        const balls = [];
        const n = 20;

        for (let i = 0; i < n; i++) {
            const ball = document.createElement('div');
            ball.style.width = '15px';
            ball.style.height = '15px';
            ball.style.borderRadius = '50%';
            ball.style.position = 'absolute';
            ball.className = i < n/2 ? 'blue' : 'orange';
            ball.style.backgroundColor = i < n/2 ? '#3498db' : '#e67e22';
            ball.style.transition = 'all 1s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
            container.appendChild(ball);
            balls.push(ball);
        }

        function randomize() {
            divider.style.display = 'none';
            balls.forEach(b => {
                b.style.left = Math.random() * 470 + 'px';
                b.style.top = Math.random() * 270 + 'px';
            });
            stats.innerText = "Entropía Máxima: Las clases están repartidas uniformemente.";
        }

        function sortBalls() {
            divider.style.display = 'block';
            balls.forEach(b => {
                if(b.className === 'blue') {
                    b.style.left = Math.random() * 210 + 10 + 'px';
                } else {
                    b.style.left = Math.random() * 210 + 260 + 'px';
                }
                b.style.top = Math.random() * 250 + 20 + 'px';
            });
            stats.innerText = "Entropía Nula tras Partición: El sistema está ordenado por clases.";
        }

        randomize();
    </script>
    """
    components.html(html_code, height=420)

    st.markdown("""
    ### Análisis Técnico
    Cuando particionamos los datos (el "split" de una rama del árbol), buscamos que cada subconjunto resultante sea lo más puro posible. 
    
    1.  **Estado Desordenado:** Si tenemos 10 bolas azules y 10 naranjas mezcladas, la probabilidad de elegir una al azar y acertar su color es baja ($1/2$). La entropía es **alta** ($1.0$).
    2.  **Estado Ordenado:** Si una división (por ejemplo, "Color de Ojos = Azul") separa perfectamente las azules a la izquierda, la incertidumbre en ese subconjunto desaparece. La entropía cae a **0.0**.
    3.  **Ganancia de Información:** Es simplemente la **diferencia** entre la entropía inicial y la suma ponderada de las entropías tras la división. El algoritmo ID3 selecciona siempre el atributo que provoque la mayor caída de entropía.
    """)

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
        if len(probs) <= 1: return 0.0
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
    
    render_entropy_intuition()
    
    st.divider()
    
    st.markdown("""
    ### Aplicación del Algoritmo
    Una vez comprendida la métrica, el algoritmo ID3 construye el árbol de forma recursiva 
    maximizando esta ganancia en cada nodo.
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
