import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def render():
    st.header("🏆 Modo Reto: La Arena de Datos")
    st.markdown("Pon a prueba tu intuición matemática. ¿Puedes ganar a la heurística?")

    tab1, tab2, tab3, tab4 = st.tabs(["1. Ajuste Manual", "2. Maestro del Kernel", "3. Susurrador de Clusters", "4. El Arquitecto"])

    # --- CHALLENGE 1: Manual Regression (With R2) ---
    with tab1:
        st.subheader("Objetivo: Maximizar $R^2$ (y minimizar Costo)")
        st.write("No basta con pasar cerca de los puntos. Debes explicar la varianza.")
        
        # Data
        np.random.seed(42)
        X = np.linspace(0, 10, 50)
        true_w, true_b = 2.5, 5.0
        # Add a bit more noise/outliers to make R2 tricky
        y = true_w * X + true_b + np.random.normal(0, 2.5, 50)
        
        col_ctrl, col_plot = st.columns([1, 2])
        with col_ctrl:
            w = st.slider("Pendiente (w)", 0.0, 5.0, 1.0, key="chal1_w")
            b = st.slider("Intercepto (b)", 0.0, 15.0, 0.0, key="chal1_b")
            
            # Real-time Metrics
            y_pred = w * X + b
            mse = np.mean((y_pred - y) ** 2)
            r2 = r2_score(y, y_pred)
            
            st.metric("Costo (MSE)", f"{mse:.2f}")
            st.metric("R² Score", f"{r2:.4f}", help="1.0 es perfecto. < 0 es terrible.")
            
            if r2 > 0.90:
                st.balloons()
                st.success("¡Perfecto! Has capturado la tendencia subyacente.")
            elif r2 > 0.70:
                st.info("Vas bien. Ajusta un poco más.")
            elif r2 < 0:
                st.error("¡Cuidado! Tu modelo es peor que una línea horizontal.")

        with col_plot:
            fig, ax = plt.subplots()
            ax.scatter(X, y, label='Datos Reales', alpha=0.6)
            ax.plot(X, y_pred, 'r-', linewidth=2, label='Tu Hipótesis')
            ax.set_ylim(-5, 40)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # --- CHALLENGE 2: SVM Tuning (Generalization) ---
    with tab2:
        st.subheader("Objetivo: Generalización (>95% en Test)")
        st.write("Ajusta el modelo viendo SOLO los datos de entrenamiento (Negros). Los puntos de Test (Rojos/Azules pálido) determinarán tu victoria.")
        
        # Data Setup (Cached logic could be here, but lightweight enough)
        X_m, y_m = make_moons(n_samples=300, noise=0.25, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_m, y_m, test_size=0.3, random_state=42)
        
        col_c2, col_p2 = st.columns([1, 2])
        
        with col_c2:
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], key="chal2_k")
            C = st.slider("C (Regularización)", 0.1, 50.0, 1.0, key="chal2_c", help="C alto = Menos error en train (riesgo overfitting).")
            gamma = st.slider("Gamma", 0.1, 10.0, 1.0, key="chal2_g", help="Gamma alto = Curvas más complejas ajustadas a cada punto.")
            
            # Train model
            clf = SVC(kernel=kernel, C=C, gamma=gamma)
            clf.fit(X_train, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            
            st.markdown("---")
            st.metric("Train Accuracy", f"{train_acc:.1%}")
            st.metric("Test Accuracy (Tu Puntuación)", f"{test_acc:.1%}")
            
            if test_acc > 0.96:
                st.balloons()
                st.success("¡Maestro! Modelo generalizado perfectamente.")
            elif train_acc > 0.99 and test_acc < 0.90:
                st.warning("⚠️ OVERFITTING DETECTADO. Memorizaste los datos de train.")
            elif test_acc < 0.85:
                st.error("Modelo pobre (Underfitting). Prueba otro Kernel.")

        with col_p2:
            fig2, ax2 = plt.subplots()
            
            # Plot Decision Boundary
            x_min, x_max = X_m[:, 0].min() - .5, X_m[:, 0].max() + .5
            y_min, y_max = X_m[:, 1].min() - .5, X_m[:, 1].max() + .5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax2.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
            
            # Plot Train (Solid)
            ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=50, label='Entrenamiento')
            
            # Plot Test (Faint/Smaller)
            ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=20, alpha=0.6, label='Test (Validación)')
            
            ax2.legend()
            st.pyplot(fig2)

    # --- CHALLENGE 3: Clustering Whisperer ---
    with tab3:
        st.subheader("Objetivo: Encontrar la K oculta")
        st.markdown("Los datos no tienen etiquetas. Usa la inercia y silueta para adivinar $K_{real}$.")

        # 1. State Management for 'Hidden' data
        if 'chal3_k' not in st.session_state:
            st.session_state['chal3_k'] = np.random.randint(3, 7)
            st.session_state['chal3_X'], _ = make_blobs(n_samples=300, 
                                                        centers=st.session_state['chal3_k'], 
                                                        cluster_std=0.70, random_state=None)

        X_blob = st.session_state['chal3_X']
        true_k = st.session_state['chal3_k']

        col_c3, col_p3 = st.columns([1, 2])
        
        with col_c3:
            k_user = st.slider("¿Cuántos Clusters ves?", 2, 8, 2, key="chal3_slider")
            
            if st.button("Generar Nuevo Problema"):
                st.session_state.pop('chal3_k')
                st.rerun()

            # Model
            kmeans = KMeans(n_clusters=k_user, n_init=10)
            labels = kmeans.fit_predict(X_blob)
            
            # Metrics
            inertia = kmeans.inertia_
            sil = silhouette_score(X_blob, labels)
            
            st.markdown("#### Pistas:")
            st.metric("Inercia (Codo)", f"{inertia:.0f}")
            st.metric("Silueta (Calidad)", f"{sil:.2f}")

            # Guess Button
            if st.button("¡Es mi respuesta final!"):
                if k_user == true_k:
                    st.balloons()
                    st.success(f"¡CORRECTO! Había {true_k} grupos naturales.")
                else:
                    st.error(f"Fallaste. La respuesta real era {true_k}.")
        
        with col_p3:
            fig3, ax3 = plt.subplots()
            # Always color by predicted label to show what the user sees
            ax3.scatter(X_blob[:, 0], X_blob[:, 1], c=labels, cmap='viridis', s=40, alpha=0.8)
            
            # Centroids
            centers = kmeans.cluster_centers_
            ax3.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='*', label='Centroides Propuestos')
            ax3.legend()
            st.pyplot(fig3)

    # --- CHALLENGE 4: The Architect (Neural Nets) ---
    with tab4:
        st.subheader("Objetivo: Domar el Gradiente")
        st.markdown("""
        Estás entrenando una Red Neuronal profunda. La superficie de error es **No Convexa** (llena de valles falsos).
        Configura el **Learning Rate** y el **Momentum** para que la bola (tu modelo) llegue al **Mínimo Global (Centro)** sin quedarse atrapada ni salir volando.
        """)

        col_c4, col_p4 = st.columns([1, 2])

        with col_c4:
            lr = st.slider("Learning Rate (Tasa de Aprendizaje)", 0.01, 1.0, 0.1, step=0.01, key="chal4_lr")
            momentum = st.slider("Momentum (Inercia)", 0.0, 0.99, 0.0, step=0.01, key="chal4_mom")
            iterations = st.slider("Iteraciones (Épocas)", 20, 200, 50, key="chal4_iter")
            
            if st.button("▶️ Lanzar Entrenamiento", key="chal4_btn"):
                # Simulation Logic
                # Surface: Z = x^2 + y^2 - 4*cos(2*x) - 4*cos(2*y)  (Rastrigin-like)
                # Global Min at (0,0)
                
                path_x, path_y, path_z = [], [], []
                
                # Start at a tricky point (corner)
                x, y = 3.0, 3.0
                vx, vy = 0.0, 0.0 # Velocity for momentum
                
                success = False
                diverged = False
                
                for _ in range(iterations):
                    path_x.append(x)
                    path_y.append(y)
                    
                    # Calculate Loss Z
                    z = x**2 + y**2 - 4*np.cos(2*x) - 4*np.cos(2*y)
                    path_z.append(z)
                    
                    # Gradients
                    # dZ/dx = 2x + 8*sin(2x)
                    dz_dx = 2*x + 8*np.sin(2*x)
                    dz_dy = 2*y + 8*np.sin(2*y)
                    
                    # Update with Momentum
                    # v_t = gamma * v_{t-1} + lr * grad
                    vx = momentum * vx + lr * dz_dx
                    vy = momentum * vy + lr * dz_dy
                    
                    # x_t = x_{t-1} - v_t
                    x = x - vx
                    y = y - vy
                    
                    # Check Divergence
                    if abs(x) > 6 or abs(y) > 6:
                        diverged = True
                        break
                        
                # End state analysis
                final_dist = np.sqrt(x**2 + y**2)
                
                if diverged:
                    st.error("💥 ¡DIVERGENCIA! La red explotó (NaN). Baja el Learning Rate.")
                elif final_dist < 0.5:
                    st.balloons()
                    st.success("🏆 ¡CONVERGENCIA! Has llegado al Mínimo Global.")
                elif final_dist < 3.0:
                    st.warning("Te has atascado en un Mínimo Local. Intenta usar más Momentum para 'saltar' la colina.")
                else:
                    st.info("Sigues muy lejos. Necesitas más velocidad (LR o Momentum).")

                # --- 3D Visualization (Plotly) ---
                import plotly.graph_objects as go
                
                # Create Grid
                grid_x = np.linspace(-4, 4, 50)
                grid_y = np.linspace(-4, 4, 50)
                X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
                Z_grid = X_grid**2 + Y_grid**2 - 4*np.cos(2*X_grid) - 4*np.cos(2*Y_grid)
                
                fig4 = go.Figure(data=[go.Surface(z=Z_grid, x=grid_x, y=grid_y, colorscale='Viridis', opacity=0.7)])
                
                # Add Trajectory
                fig4.add_trace(go.Scatter3d(
                    x=path_x, y=path_y, z=path_z,
                    mode='lines+markers',
                    marker=dict(size=4, color='red'),
                    line=dict(color='yellow', width=5),
                    name='Descenso'
                ))
                
                # Mark Global Min
                fig4.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[-8],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='diamond'),
                    name='Meta (Mínimo Global)'
                ))

                fig4.update_layout(title="Superficie de Error (Loss Landscape)", 
                                   width=600, height=500,
                                   scene=dict(zaxis=dict(range=[-10, 20])))
                st.plotly_chart(fig4)

        with col_p4:
            if 'fig4' not in locals():
               # Placeholder logic for first render
               st.info("Configura los parámetros y pulsa 'Lanzar' para ver la simulación.")
               import plotly.graph_objects as go
               grid_x = np.linspace(-4, 4, 50)
               grid_y = np.linspace(-4, 4, 50)
               X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
               Z_grid = X_grid**2 + Y_grid**2 - 4*np.cos(2*X_grid) - 4*np.cos(2*Y_grid)
               fig_preview = go.Figure(data=[go.Surface(z=Z_grid, x=grid_x, y=grid_y, colorscale='Viridis', opacity=0.8)])
               fig_preview.update_layout(title="Tu campo de batalla (Vista previa)", width=600, height=500)
               st.plotly_chart(fig_preview)
