import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree

def render():
    st.header("Tema Final: Árboles de Decisión (Aprendizaje Simbólico)")
    st.markdown("""
    Visualiza cómo un Árbol divide el espacio usando cortes **ortogonales** (rectángulos).
    Diferencia clave con SVM/Redes: ¡No hay curvas suaves!
    """)

    # --- 1. Configuración de Datos ---
    st.sidebar.subheader("1. Datos")
    dataset_type = st.sidebar.selectbox("Datasets", ["Lunas (Moons)", "Clasificación Simple"])
    noise = st.sidebar.slider("Ruido", 0.0, 0.5, 0.2)
    
    if dataset_type == "Lunas (Moons)":
        X, y = make_moons(n_samples=200, noise=noise, random_state=42)
    else:
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=1, n_clusters_per_class=1)
        # Shift to be positive for nice plotting
        X += 2 

    # --- 2. Hiperparámetros del Árbol ---
    st.sidebar.subheader("2. Parámetros del Árbol")
    max_depth = st.sidebar.slider("Profundidad Máxima (Max Depth)", 1, 10, 3)
    criterion = st.sidebar.selectbox("Criterio", ["gini", "entropy"])
    
    # Entrenar
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    clf.fit(X, y)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Frontera de Decisión")
        fig, ax = plt.subplots()
        
        # Plot Scatter
        ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='coolwarm', edgecolors='k')
        
        # Decision Boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
        ax.set_title(f"Árbol (Profundidad={max_depth})")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Estructura del Árbol")
        st.write(f"Nodos Totales: **{clf.tree_.node_count}**")
        st.write(f"Profundidad Actual: **{clf.get_depth()}**")
        
        if st.checkbox("Ver Reglas (Texto)"):
            from sklearn.tree import export_text
            st.code(export_text(clf, feature_names=["X1", "X2"]))

    # --- 3. Plot del Árbol Completo ---
    st.markdown("---")
    st.subheader("Visualización del Grafo")
    if st.checkbox("Mostrar Grafo del Árbol (puede ser grande)"):
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        plot_tree(clf, filled=True, feature_names=["X1", "X2"], class_names=["Clase 0", "Clase 1"], ax=ax2)
        st.pyplot(fig2)
