import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def render():
    st.header("🏆 Modo Reto: Demuestra lo que sabes")
    st.markdown("Aquí no hay 'piloto automático'. Tú eres el algoritmo.")

    tab1, tab2 = st.tabs(["Reto 1: El Ojo Humano", "Reto 2: Maestro del Kernel"])

    # --- CHALLENGE 1: Manual Regression ---
    with tab1:
        st.subheader("Objetivo: Minimizar el Costo J(θ) manualmente")
        st.write("Ajusta los parámetros w (pendiente) y b (intercepto) para que la línea roja cubra los puntos azules.")
        
        # Data
        np.random.seed(42)
        X = np.linspace(0, 10, 50)
        true_w, true_b = 2.5, 5.0
        y = true_w * X + true_b + np.random.normal(0, 2, 50)
        
        col_ctrl, col_plot = st.columns([1, 2])
        with col_ctrl:
            w = st.slider("Pendiente (w)", 0.0, 5.0, 0.0, key="chal1_w")
            b = st.slider("Intercepto (b)", 0.0, 10.0, 0.0, key="chal1_b")
            
            # Real-time Cost Calc
            y_pred = w * X + b
            cost = np.mean((y_pred - y) ** 2)
            
            st.metric("Tu Costo (MSE)", f"{cost:.2f}")
            
            if cost < 5.0:
                st.balloons()
                st.success("¡Excelente! Has encontrado el mínimo global (aprox).")
            elif cost < 20.0:
                st.warning("Cerca... afina un poco más.")
            else:
                st.error("Estás lejos. ¡Sigue probando!")

        with col_plot:
            fig, ax = plt.subplots()
            ax.scatter(X, y, label='Datos')
            ax.plot(X, y_pred, 'r-', linewidth=2, label='Tu Hipótesis')
            ax.set_ylim(0, 35)
            ax.legend()
            st.pyplot(fig)

    # --- CHALLENGE 2: SVM Tuning ---
    with tab2:
        st.subheader("Objetivo: Separar las Lunas (Accuracy > 95%)")
        st.write("El modelo lineal falla. Encuentra la configuración de SVM correcta.")
        
        # Hard Data
        X_m, y_m = make_moons(n_samples=200, noise=0.25, random_state=42)
        
        col_c2, col_p2 = st.columns([1, 2])
        
        with col_c2:
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], key="chal2_k")
            C = st.slider("C (Regularización)", 0.1, 10.0, 1.0, key="chal2_c")
            gamma = st.slider("Gamma", 0.1, 10.0, 1.0, key="chal2_g")
            
            clf = SVC(kernel=kernel, C=C, gamma=gamma)
            clf.fit(X_m, y_m)
            acc = accuracy_score(y_m, clf.predict(X_m))
            
            st.metric("Precisión", f"{acc:.1%}")
            
            if acc > 0.95:
                st.balloons()
                st.success("¡Logro Desbloqueado: Maestro de SVM!")
            else:
                st.info("Intenta cambiar el Kernel o ajustar Gamma.")

        with col_p2:
            fig2, ax2 = plt.subplots()
            ax2.scatter(X_m[:, 0], X_m[:, 1], c=y_m, cmap='coolwarm', edgecolors='k')
            
            # Decision boundary
            xlim = ax2.get_xlim()
            ylim = ax2.get_ylim()
            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = clf.decision_function(xy).reshape(XX.shape)
            
            ax2.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
            st.pyplot(fig2)
