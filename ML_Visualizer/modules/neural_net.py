import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def render():
    st.header("Tema 5: Redes Neuronales (Entrenamiento en Vivo)")
    st.markdown("""
    Visualiza cómo una Red Neuronal "aprende" época tras época.
    Observa cómo la **función de pérdida (Loss)** disminuye y la **frontera de decisión** se ajusta a los datos.
    """)

    # --- 1. Configuración de Datos ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuración")
        dataset_name = st.selectbox("Dataset", ["Lunas (Moons)", "Círculos (Circles)"])
        noise = st.slider("Ruido", 0.0, 0.3, 0.1)
        n_samples = 300
        
        # Generar datos
        if dataset_name == "Lunas (Moons)":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        else:
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
            
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.markdown("---")
        st.write("**Hiperparámetros de la Red:**")
        hidden_layers = st.text_input("Capas Ocultas (ej. 10,10)", "10,10")
        try:
            hidden_layer_sizes = tuple(map(int, hidden_layers.split(',')))
        except:
            st.error("Formato inválido. Usando (10,10)")
            hidden_layer_sizes = (10, 10)
            
        lr = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
        max_epochs = st.slider("Total Épocas", 10, 500, 100)
        
        start_btn = st.button("▶️ Iniciar Entrenamiento")

    # --- 2. Preparación visualización ---
    with col2:
        plot_placeholder = st.empty()
        metrics_placeholder = st.empty()

    # --- 3. Lógica de Entrenamiento ---
    if start_btn:
        # Inicializar modelo
        # warm_start=True permite entrenar incrementalmente (partial_fit)
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                            learning_rate_init=lr,
                            max_iter=1,  # Solo 1 paso por llamada
                            warm_start=True,
                            random_state=42)
        
        # Listas para métricas
        loss_history = []
        accuracy_history = []
        classes = np.unique(y)

        # Crear meshgrid estático para la frontera
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))

        # Bucle de Entrenamiento
        progress_bar = st.progress(0)
        
        for epoch in range(max_epochs):
            # Paso de entrenamiento
            mlp.partial_fit(X_train, y_train, classes=classes)
            
            # Métricas
            loss_history.append(mlp.loss_)
            accuracy = mlp.score(X_test, y_test)
            accuracy_history.append(accuracy)
            
            # --- Visualización ---
            # Solo actualizar cada N épocas para rendimiento
            if epoch % 5 == 0 or epoch == max_epochs - 1:
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Gráfico 1: Frontera de Decisión
                if hasattr(mlp, "coefs_"):
                     Z = mlp.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                     Z = Z.reshape(xx.shape)
                     ax1.contourf(xx, yy, Z, cmap="RdBu", alpha=0.8)
                
                ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap="RdBu_r")
                ax1.set_title(f"Frontera (Época {epoch})")
                ax1.axis('off')

                # Gráfico 2: Curva de Pérdida
                ax2.plot(loss_history, label='Loss')
                ax2.set_title("Evolución del Error")
                ax2.set_xlabel("Época")
                ax2.set_ylabel("Loss")
                ax2.grid(True)
                
                plot_placeholder.pyplot(fig)
                plt.close(fig)
                
                metrics_placeholder.markdown(f"""
                **Estado Actual:**
                *   Pérdida (Loss): `{mlp.loss_:.4f}`
                *   Precisión Test: `{accuracy:.2%}`
                """)
            
            progress_bar.progress((epoch + 1) / max_epochs)
            time.sleep(0.01) # Pequeña pausa para ver la animación

        st.success("Entrenamiento Completado!")
    
    elif not start_btn:
        with col2:
            st.info("Configura los parámetros a la izquierda y pulsa 'Iniciar' para ver la magia de las Redes Neuronales.")

