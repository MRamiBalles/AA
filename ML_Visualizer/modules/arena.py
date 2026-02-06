import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def render():
    st.header("⚔️ La Arena: Comparativa de Algoritmos")
    st.markdown("""
    **Idea Innovadora:** No te cases con un solo modelo. 
    Enfréntalos en el mismo dataset y usa **PCA (Análisis de Componentes Principales)** para visualizar en 2D datos complejos.
    """)

    # --- 1. CONFIGURACIÓN (Sidebar) ---
    st.sidebar.subheader("1. Configuración de Arena")
    
    # Dataset Selection
    ds_name = st.sidebar.selectbox("Elige el Campo de Batalla", 
                                   ["Iris (Multiclase)", "Wine (Complejo)", "Breast Cancer (Binario)", "Moons (No Lineal)", "Circles (Difícil)"])
    
    # Model Selection
    st.sidebar.write("Elige los Gladiadores:")
    use_svm = st.sidebar.checkbox("SVM (RBF)", value=True)
    use_tree = st.sidebar.checkbox("Árbol de Decisión", value=True)
    use_knn = st.sidebar.checkbox("KNN (Vecinos)", value=False)
    use_mlp = st.sidebar.checkbox("Red Neuronal (MLP)", value=False)
    
    # --- 2. DATA LOADING & PREP ---
    X, y, target_names = None, None, None
    
    if ds_name == "Iris (Multiclase)":
        data = load_iris()
        X, y, target_names = data.data, data.target, data.target_names
    elif ds_name == "Wine (Complejo)":
        data = load_wine()
        X, y, target_names = data.data, data.target, data.target_names
    elif ds_name == "Breast Cancer (Binario)":
        data = load_breast_cancer()
        X, y, target_names = data.data, data.target, data.target_names
    elif ds_name == "Moons (No Lineal)":
        X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
        target_names = ["Clase 0", "Clase 1"]
        # Moons only has 2 features, so PCA will just be identity (or close)
    else: # Circles
        X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
        target_names = ["Clase 0", "Clase 1"]

    # Scaling (Crucial for SVM/KNN/MLP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # --- 3. DIMENSIONALITY REDUCTION (INNOVATION: PCA) ---
    pca = PCA(n_components=2)
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test = pca.transform(X_test)
    
    # Visualization of the Dataset (2D Projection)
    col_data, col_res = st.columns([1, 2])
    
    with col_data:
        st.subheader("Visualización (PCA 2D)")
        st.caption(f"Dimensión original: {X.shape[1]} features → Proyectado a 2D")
        
        fig_pca, ax_pca = plt.subplots()
        scatter = ax_pca.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c=y_train, cmap='viridis', edgecolors='k', s=50)
        ax_pca.set_xlabel("Componente Principal 1")
        ax_pca.set_ylabel("Componente Principal 2")
        # Add legend logic if needed, but color is enough for quick view
        st.pyplot(fig_pca)
        
        st.info(f"Varianza explicada por PCA: {np.sum(pca.explained_variance_ratio_):.1%}")

    # --- 4. BATTLE ROYALE (TRAINING & EVAL) ---
    results = []
    
    models = []
    if use_svm: models.append(("SVM (RBF)", SVC(kernel='rbf', C=1.0, gamma='scale')))
    if use_tree: models.append(("Árbol (Depth=3)", DecisionTreeClassifier(max_depth=3)))
    if use_knn: models.append(("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)))
    if use_mlp: models.append(("MLP (20,20)", MLPClassifier(hidden_layer_sizes=(20,20), max_iter=500)))

    with col_res:
        st.subheader("Resultados de la Batalla")
        
        if not models:
            st.warning("Selecciona al menos un modelo en la barra lateral.")
        else:
            # We will plot Decision Boundaries on the PCA space (Approximate visualization)
            # NOTE: Training on Full Scaled Data for metrics, but training on PCA data for visualization?
            # To be fair: We train on Full Data to get real Accuracy. 
            # Generating Decision Boundary on PCA 2D is tricky because models expect N features.
            # INNOVATION: We will train a separate "Visual Model" on just the 2 PCA components to show boundaries.
            
            tabs = st.tabs([name for name, _ in models])
            
            for i, (name, model) in enumerate(models):
                with tabs[i]:
                    # 1. Train Real Model (High Dim)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    
                    col_met, col_viz = st.columns([1, 2])
                    
                    with col_met:
                        st.metric("Precisión (Accuracy)", f"{acc:.1%}")
                        st.write("Matriz de Confusión:")
                        cm = confusion_matrix(y_test, y_pred)
                        st.text(cm)
                    
                    with col_viz:
                        st.write("👀 **Frontera de Decisión ( Aproximada en 2D )**")
                        # Train a 'Shadow Model' just on 2D PCA data for viz purposes
                        # Clone model manually or just re-instantiate similar one
                        from sklearn.base import clone
                        viz_model = clone(model)
                        viz_model.fit(X_pca_train, y_train)
                        
                        # Plot
                        fig_bd, ax_bd = plt.subplots()
                        
                        # Grid
                        x_min, x_max = X_pca_train[:, 0].min() - 1, X_pca_train[:, 0].max() + 1
                        y_min, y_max = X_pca_train[:, 1].min() - 1, X_pca_train[:, 1].max() + 1
                        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                        
                        Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
                        Z = Z.reshape(xx.shape)
                        
                        ax_bd.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                        ax_bd.scatter(X_pca_test[:, 0], X_pca_test[:, 1], c=y_test, cmap='viridis', edgecolors='k', s=20, label='Test Data')
                        
                        ax_bd.set_title(f"Frontera 2D ({name})")
                        st.pyplot(fig_bd)
                        st.caption("*Nota: La frontera 2D es ilustrativa (entrenada en datos reducidos). La precisión se calculó con todas las variables.*")

