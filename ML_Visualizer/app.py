import streamlit as st
from modules import regression, svm, neural_net, clustering, trees, challenges, arena, logic

# Configuración de la página
st.set_page_config(page_title="ML-Visualizer: Aprendizaje Automático", layout="wide")

st.title("🎓 AA: Visualizador de Algoritmos")
st.markdown("Herramienta interactiva para explorar los conceptos de la asignatura.")

# Menú Lateral basado en el Temario
st.sidebar.title("Temario")
opcion = st.sidebar.radio(
    "Selecciona un tema:",
    ("1. Regresión Lineal", "2. SVM (Vectores de Soporte)", "3. Lógica (Concept Learning)", "4. Árboles (Lógica Simbólica)", "5. Redes Neuronales (En Vivo)", "6. Clustering (K-Means)", "🏆 MODO RETO", "⚔️ LA ARENA")
)

# Enrutamiento de Módulos
if opcion == "1. Regresión Lineal":
    regression.render()
elif opcion == "2. SVM (Vectores de Soporte)":
    svm.render()
elif opcion == "3. Lógica (Concept Learning)":
    logic.render()
elif opcion == "4. Árboles (Lógica Simbólica)":
    trees.render()
elif opcion == "5. Redes Neuronales (En Vivo)":
    neural_net.render()
elif opcion == "6. Clustering (K-Means)":
    clustering.render()
elif opcion == "🏆 MODO RETO":
    challenges.render()
elif opcion == "⚔️ LA ARENA":
    arena.render()
else:
    st.info("Módulo en desarrollo.")
