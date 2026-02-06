import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def render():
    st.header("Tema 2: Regresión Lineal con Descenso por Gradiente")
    st.markdown(r"""
    El objetivo es minimizar la función de costo:
    $$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 $$
    donde la hipótesis es $h_\theta(x) = \theta_0 + \theta_1 x$.
    """)

    # --- 1. Carga de Datos ---
    # Usamos datos sintéticos si no existe el csv, para que el código sea ejecutable ya mismo.
    try:
        df = pd.read_csv('d:/AA/ML_Visualizer/data/regresion 1.csv', header=None, names=['Poblacion', 'Beneficio'])
    except FileNotFoundError:
        st.warning("⚠️ 'regresion 1.csv' no encontrado. Usando datos sintéticos para demostración.")
        df = pd.DataFrame({
            'Poblacion': np.linspace(5, 22, 50),
            'Beneficio': 2 * np.linspace(5, 22, 50) - 5 + np.random.normal(0, 2, 50)
        })

    X = df['Poblacion'].values
    y = df['Beneficio'].values
    m = len(y)

    # --- 2. Configuración Interactiva (Sidebar) ---
    st.sidebar.subheader("Hiperparámetros")
    # Learning Rate (alpha)
    alpha = st.sidebar.slider("Tasa de aprendizaje (alpha)", 0.001, 0.05, 0.01, format="%.3f")
    # Iteraciones
    iterations = st.sidebar.slider("Iteraciones", 1, 100, 10)
    
    # Inicialización de Thetas
    theta_0 = st.sidebar.number_input("Theta 0 inicial", value=0.0)
    theta_1 = st.sidebar.number_input("Theta 1 inicial", value=0.0)

    # --- 3. Lógica del Gradiente (Manual) ---
    # Historial para visualizar la trayectoria en la superficie de costo
    cost_history = []
    theta_history = []
    
    t0, t1 = theta_0, theta_1
    
    # Bucle de entrenamiento
    for _ in range(iterations):
        theta_history.append((t0, t1)) # Save state before update for proper visualization
        
        # Hipótesis h(x)
        h = t0 + t1 * X
        
        # Cálculo de Costo J(theta)
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Actualización de gradientes (Derivadas parciales)
        d_t0 = (1/m) * np.sum(h - y)
        d_t1 = (1/m) * np.sum((h - y) * X)
        
        t0 = t0 - alpha * d_t0
        t1 = t1 - alpha * d_t1

    # --- 4. Visualización ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ajuste de la Hipótesis")
        fig, ax = plt.subplots()
        ax.scatter(X, y, marker='x', color='red', label='Datos de entrenamiento') #
        
        # Línea de regresión actual
        ax.plot(X, t0 + t1*X, label=f'Hipótesis: {t0:.2f} + {t1:.2f}x', color='blue')
        ax.set_xlabel("Población (10k)")
        ax.set_ylabel("Beneficio ($10k)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.success(f"Costo Final J(θ): {cost_history[-1]:.4f}")

    with col2:
        st.subheader("Superficie de Costo J(θ)")
        
        # Crear malla para la superficie. Centrarla en el valor final aproximado
        # Ajustamos el rango de visualizacion para que sea relevante
        t0_range = np.linspace(t0 - 10, t0 + 10, 50)
        t1_range = np.linspace(t1 - 5, t1 + 5, 50)
        
        # Si estamos al inicio (0,0), usamos rango estandar para ver el descenso
        if iterations < 5 and theta_0 == 0 and theta_1 == 0:
             t0_range = np.linspace(-10, 15, 50)
             t1_range = np.linspace(-5, 5, 50)

        t0_mesh, t1_mesh = np.meshgrid(t0_range, t1_range)
        J_vals = np.zeros(t0_mesh.shape)
        
        for i in range(t0_mesh.shape[0]):
            for j in range(t0_mesh.shape[1]):
                # t calcule for grid
                t_grid_0 = t0_mesh[i, j]
                t_grid_1 = t1_mesh[i, j]
                h_val = t_grid_0 + t_grid_1 * X
                J_vals[i, j] = (1/(2*m)) * np.sum((h_val - y)**2)
        
        # Gráfico de Contorno con trayectoria
        fig2 = go.Figure(data=[go.Surface(z=J_vals, x=t0_range, y=t1_range, opacity=0.8, colorscale='Viridis')])
        
        # Trayectoria del descenso
        th_0 = [t[0] for t in theta_history]
        th_1 = [t[1] for t in theta_history]
        
        # Add trajectory on top of the surface (lifting z slightly to be visible)
        fig2.add_trace(go.Scatter3d(x=th_0, y=th_1, z=[c + 0.1 for c in cost_history], 
                                    mode='markers+lines', marker=dict(size=4, color='red'),
                                    name='Trayectoria GD'))
        
        fig2.update_layout(scene = dict(
                    xaxis_title='Theta 0',
                    yaxis_title='Theta 1',
                    zaxis_title='Costo J'),
                    width=500, height=500,
                    margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig2)

    # --- 5. Comparativa con Scikit-Learn ---
    if st.checkbox("Comparar con Solución Exacta (Scikit-Learn/Ecuación Normal)"):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        st.info(f"Solución Scikit-Learn: Intercepto (Theta 0) ={model.intercept_:.4f}, Pendiente (Theta 1) ={model.coef_[0]:.4f}")
        st.write("Nota: La solución exacta es el mínimo global. Tu descenso por gradiente debería acercarse a estos valores.")
