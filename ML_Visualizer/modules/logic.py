import streamlit as st
import pandas as pd
import numpy as np

def render():
    st.header("Tema 3: Aprendizaje Simbólico (Concept Learning)")
    st.markdown("""
    A diferencia del aprendizaje numérico (ajustar pesos $w$), aquí buscamos **Reglas Lógicas**.
    
    **Algoritmo Find-S**: Comenzamos con la hipótesis más específica posible y la "relajamos" (generalizamos) 
    solo cuando vemos un ejemplo positivo que la contradice.
    """)

    # --- 1. Dataset EnjoySport ---
    st.subheader("1. El Dataset: EnjoySport")
    st.markdown("¿Bajo qué condiciones Aldo disfruta de su deporte acuático?")
    
    data = [
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
    ]
    columns = ['Cielo', 'Temp', 'Humedad', 'Viento', 'Agua', 'Pronostico', 'EnjoySport']
    df = pd.DataFrame(data, columns=columns)
    
    st.dataframe(df)

    # --- 2. Algoritmo Find-S Paso a Paso ---
    st.subheader("2. Traza del Algoritmo Find-S")
    
    # Init Hypothesis
    hypothesis = ['ø', 'ø', 'ø', 'ø', 'ø', 'ø']
    st.write(f"**Hipótesis Inicial ($h_0$):** `{hypothesis}` (Nada es posible)")
    
    step = 0
    for i, row in df.iterrows():
        is_positive = row['EnjoySport'] == 'Yes'
        attributes = row[:-1].values
        
        step_col1, step_col2 = st.columns([1, 3])
        
        with step_col1:
            st.markdown(f"**Ejemplo {i+1}:**")
            st.info(f"{attributes} -> **{row['EnjoySport']}**")
        
        with step_col2:
            if not is_positive:
                st.write(f"❌ Ejemplo Negativo. Find-S lo ignora.")
                st.code(f"h_{i+1} = {hypothesis}")
            else:
                st.write("✅ Ejemplo Positivo. Generalizamos la hipótesis.")
                
                # Update Logic
                new_h = []
                changes = []
                
                if hypothesis[0] == 'ø': # First positive example
                    hypothesis = list(attributes)
                    st.success("Primera inicialización con el primer ejemplo positivo.")
                else:
                    for j in range(len(hypothesis)):
                        if hypothesis[j] == attributes[j]:
                            new_h.append(hypothesis[j])
                        else:
                            new_h.append('?') # Generalize
                            if hypothesis[j] != '?':
                                changes.append(f"{columns[j]}: {hypothesis[j]} -> ?")
                    
                    hypothesis = new_h
                    if changes:
                        st.warning(f"Generalización forzada por contradicción: {', '.join(changes)}")
                    else:
                        st.write("El ejemplo ya era consistente. No hay cambios.")
                
                st.code(f"h_{i+1} = {hypothesis}")
        
        st.divider()

    # --- 3. Resultado Final ---
    st.subheader("3. Hipótesis Final Aprendida")
    st.success(f"**H_final:** {hypothesis}")
    st.markdown("""
    **Interpretación:**
    Aldo hace deporte si:
    *   Cielo es **Sunny**
    *   Temperatura es **Warm**
    *   Viento es **Strong**
    *   *Lo demás (Humedad, Agua, Pronóstico) no importa (?)*.
    """)
    
    # --- 4. Playground ---
    st.subheader("4. Prueba la Regla")
    col_p1, col_p2, col_p3 = st.columns(3)
    cielo = col_p1.selectbox("Cielo", ["Sunny", "Rainy", "Cloudy"])
    temp = col_p2.selectbox("Temp", ["Warm", "Cold"])
    viento = col_p3.selectbox("Viento", ["Strong", "Weak"])
    
    # Check logic
    matches = True
    if hypothesis[0] != '?' and hypothesis[0] != cielo: matches = False
    if hypothesis[1] != '?' and hypothesis[1] != temp: matches = False
    if hypothesis[3] != '?' and hypothesis[3] != viento: matches = False
    
    if matches:
        st.balloons()
        st.success("✅ ¡Aldo hará deporte!")
    else:
        st.error("⛔ Aldo se queda en casa.")
