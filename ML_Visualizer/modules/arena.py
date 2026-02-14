import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def render():
    st.header("⚔️ LA ARENA: Battle Royale de Algoritmos")
    st.markdown("""
    Compara el rendimiento de distintos gladiadores (algoritmos) en datasets complejos.
    **Innovación:** Utilizamos **PCA (Principal Component Analysis)** para proyectar datos multidimensionales 
    en 2D y visualizar las fronteras de decisión aproximadas.
    """)

    # --- 1. Selección de Dataset y Gladiadores ---
    col_config, col_main = st.columns([1, 2])

    with col_config:
        st.subheader("1. Configuración")
        dataset_name = st.selectbox("Elige el Campo de Batalla", ["Iris (4D)", "Wine (13D)", "Breast Cancer (30D)"])
        
        # Carga de datos
        test_size = st.slider("Tamaño de Test (%)", 10, 50, 20) / 100.0

        # Selección de Modelos
        selected_models = st.multiselect(
            "Selecciona Gladiadores",
            ["SVM", "Decision Tree", "Random Forest", "KNN", "MLP (Neural Net)"],
            default=["SVM", "Decision Tree"]
        )

    # --- 2. Procesamiento (Pipeline) ---
    @st.cache_data
    def load_and_process_data(ds_name):
        if ds_name == "Iris (4D)":
            data = datasets.load_iris()
        elif ds_name == "Wine (13D)":
            data = datasets.load_wine()
        else:
            data = datasets.load_breast_cancer()
        return data.data, data.target, data.target_names

    # Use cached function
    X, y, class_names = load_and_process_data(dataset_name)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    # PCA para Visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_explicada = np.sum(pca.explained_variance_ratio_) * 100
    
    with col_main:
        st.info(f"ℹ️ **PCA Info:** Proyección 2D conserva el **{var_explicada:.2f}%** de la varianza.")
        if var_explicada < 50.0:
            st.warning("⚠️ **Alerta Pedagógica:** La varianza explicada es baja (<50%). La visualización 2D puede no reflejar la complejidad real de los datos.")
        
        if st.checkbox("Mostrar Vectores Propios (Bonus)"):
             st.caption("Los ejes (PC1, PC2) se alinean con la mayor varianza de los datos.")

    # --- 3. La Batalla ---
    if st.button("🔥 ¡LUCHAR!", use_container_width=True) and selected_models:
        
        st.divider()
        cols = st.columns(len(selected_models))

        for i, model_name in enumerate(selected_models):
            with cols[i]:
                st.subheader(f"🛡️ {model_name}")
                
                # Instanciar modelo
                if model_name == "SVM":
                    clf = SVC(kernel="rbf", C=1.0, probability=True)
                elif model_name == "Decision Tree":
                    clf = DecisionTreeClassifier(max_depth=5)
                elif model_name == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
                elif model_name == "KNN":
                    clf = KNeighborsClassifier(n_neighbors=5)
                elif model_name == "MLP (Neural Net)":
                    clf = MLPClassifier(hidden_layer_sizes=(50, ), max_iter=500)

                # A. Entrenamiento Real
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                st.metric(label="Accuracy (Real)", value=f"{acc:.2%}")
                
                # Matriz de Confusión
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                            xticklabels=class_names if len(class_names) < 5 else "auto", 
                            yticklabels=class_names if len(class_names) < 5 else "auto")
                plt.title("Matriz de Confusión")
                st.pyplot(fig_cm)

                # B. Visualización PCA (Modelado visual)
                # Entrenar modelo gemelo en espacio reducido
                clf_viz = type(clf)(**clf.get_params())
                clf_viz.fit(X_pca, y)

                # Malla
                h = .02
                x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

                Z = clf_viz.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig_viz, ax_viz = plt.subplots(figsize=(4, 3))
                ax_viz.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                ax_viz.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=20, edgecolor='k', cmap='viridis')
                ax_viz.set_title(f"Frontera 2D ({model_name})")
                st.pyplot(fig_viz)

    elif not selected_models:
        st.warning("Selecciona al menos un gladiador.")
