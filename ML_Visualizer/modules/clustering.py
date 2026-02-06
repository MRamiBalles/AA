import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def render():
    st.header("Tema 4: Clustering (K-Means Interactivo)")
    
    # Tabs for different activities
    tab1, tab2 = st.tabs(["🔬 Laboratorio K-Means", "📈 Método del Codo (Elbow)"])
    
    # --- GLOBAL DATA SETUP ---
    # We use session state to keep data persistent across reruns
    if 'X_blobs' not in st.session_state:
        X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
        st.session_state['X_blobs'] = X

    X = st.session_state['X_blobs']

    # --- TAB 1: Step-by-Step K-Means ---
    with tab1:
        st.markdown("""
        **Algoritmo Paso a Paso:**
        1.  **Asignación:** Colorear cada punto según su centroide más cercano.
        2.  **Actualización:** Mover los centroides al promedio de sus puntos.
        """)
        
        col_conf, col_plot = st.columns([1, 2])
        
        with col_conf:
            K = st.slider("Número de Clusters (K)", 2, 6, 4)
            
            # Reset / Init Button
            if st.button("🔄 Reiniciar / Generar Nuevos Centros"):
                # Initialize random centroids
                indices = np.random.choice(X.shape[0], K, replace=False)
                centroids = X[indices]
                st.session_state['centroids'] = centroids
                st.session_state['iteration'] = 0
                st.session_state['labels'] = np.zeros(X.shape[0])
            
            # Initialization check
            if 'centroids' not in st.session_state or len(st.session_state['centroids']) != K:
                indices = np.random.choice(X.shape[0], K, replace=False)
                st.session_state['centroids'] = X[indices]
                st.session_state['iteration'] = 0
            
            # Step Buttons
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("1. Asignar Puntos"):
                    # Calculate distances
                    centroids = st.session_state['centroids']
                    # Broadcasting distance calculation
                    # distances: (N_samples, K)
                    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                    labels = np.argmin(distances, axis=1)
                    st.session_state['labels'] = labels
            
            with col_b2:
                if st.button("2. Mover Centroides"):
                    labels = st.session_state.get('labels', np.zeros(X.shape[0]))
                    new_centroids = []
                    for k in range(K):
                        points = X[labels == k]
                        if len(points) > 0:
                            new_centroids.append(points.mean(axis=0))
                        else:
                            # Handle empty cluster (re-init random point purely for visualization stability)
                            new_centroids.append(X[np.random.choice(X.shape[0])])
                    st.session_state['centroids'] = np.array(new_centroids)
                    st.session_state['iteration'] += 1

            st.metric("Iteración Actual", st.session_state['iteration'])

        with col_plot:
            fig, ax = plt.subplots()
            # Plot Data Points
            labels = st.session_state.get('labels', np.zeros(X.shape[0]))
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', alpha=0.6, label='Datos')
            
            # Plot Centroids
            centroids = st.session_state['centroids']
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', edgecolors='black', label='Centroides')
            
            ax.set_title(f"K-Means (K={K})")
            ax.legend()
            st.pyplot(fig)

    # --- TAB 2: Elbow Method ---
    with tab2:
        st.markdown("Calcula la **Inercia** (Suma de distancias al cuadrado) para diferentes K.")
        
        if st.button("🚀 Calcular Curva de Codo"):
            inertias = []
            k_range = range(1, 10)
            
            progress = st.progress(0)
            for i, k in enumerate(k_range):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
                progress.progress((i+1)/len(k_range))
            
            fig2, ax2 = plt.subplots()
            ax2.plot(k_range, inertias, 'bo-', markerfacecolor='red')
            ax2.set_xlabel('Número de Clusters (K)')
            ax2.set_ylabel('Inercia (Costo)')
            ax2.set_title('Método del Codo')
            ax2.grid(True)
            
            # Highlight K=4 (True center count)
            ax2.axvline(x=4, color='green', linestyle='--', label='K=4 (Real)')
            ax2.legend()
            
            st.pyplot(fig2)
            st.info("Nota el 'codo' en K=4, donde la ganancia marginal disminuye.")
