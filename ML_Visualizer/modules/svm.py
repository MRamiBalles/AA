import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

def render():
    st.header("Tema 3: Máquinas de Vectores de Soporte (SVM)")
    st.markdown("""
    Explora cómo el **Kernel** y los parámetros **C** y **Gamma** afectan a la frontera de decisión.
    *   **Margen Duro (C alto):** Menor tolerancia al error.
    *   **Margen Blando (C bajo):** Mayor tolerancia, frontera más suave.
    *   **Kernel Trick:** Proyectar a dimensiones superiores para separar datos no lineales.
    """)
    
    # --- 1. Configuración de Datos ---
    st.sidebar.subheader("1. Datos")
    dataset_type = st.sidebar.selectbox("Tipo de Dataset", ["Lineal (Blobs)", "No Lineal (Círculos)", "No Lineal (Lunas)"])
    noise = st.sidebar.slider("Nivel de Ruido", 0.0, 0.5, 0.1)
    
    if dataset_type == "Lineal (Blobs)":
        X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.5 + noise)
    elif dataset_type == "No Lineal (Círculos)":
        X, y = datasets.make_circles(n_samples=100, factor=0.5, noise=noise)
    else:
        X, y = datasets.make_moons(n_samples=100, noise=noise)

    # --- 2. Configuración del Modelo ---
    st.sidebar.subheader("2. Hiperparámetros SVM")
    kernel = st.sidebar.radio("Kernel", ["linear", "rbf", "poly"])
    C = st.sidebar.slider("Regularización (C)", 0.01, 10.0, 1.0)
    gamma = "scale"
    if kernel != "linear":
        gamma = st.sidebar.slider("Gamma (solo RBF/Poly)", 0.1, 10.0, 1.0)

    # --- 3. Entrenamiento ---
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X, y)

    # --- 4. Visualización ---
    st.subheader(f"Frontera de Decisión con Kernel: {kernel.upper()}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Puntos
    # c=y uses the class labels for color. cmap handles the mapping.
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn', edgecolors='k')

    # Grid para evaluar el modelo
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Fix: linspace takes (start, stop, num). xlim is (min, max).
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # decision_function returns distance to hyperplane
    Z = model.decision_function(xy).reshape(XX.shape)

    # Fronteras y Márgenes
    # Level 0 is the hyperplane. Levels -1 and 1 are the margins.
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # Vectores de Soporte (Resaltarlos)
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=150,
               linewidth=1.5, facecolors='none', edgecolors='k', label='Vectores Soporte')
    
    ax.legend()
    st.pyplot(fig)
    
    # Stats
    st.info(f"Número de Vectores de Soporte: {len(model.support_vectors_)}")
    if st.checkbox("Ver explicación de Vectores de Soporte"):
        st.write("""
        **Vectores de Soporte:** Son los puntos de datos más "difíciles" de clasificar (los más cercanos a la línea divisoria).
        En SVM, **solo estos puntos importan**. Si mueves los otros puntos (los lejanos), la línea no cambiará.
        Observa cómo los círculos negros encierran estos puntos críticos.
        """)
