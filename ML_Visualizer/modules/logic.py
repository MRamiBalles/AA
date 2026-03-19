import streamlit as st
import pandas as pd
import numpy as np
from . import candidate_elimination, id3_symbolic, foil_simplified

def render_find_s():
    st.subheader("Algoritmo Find-S")
    st.markdown("""
    **Find-S**: Comenzamos con la hipótesis más específica posible y la "relajamos" (generalizamos) 
    solo cuando vemos un ejemplo positivo que la contradice.
    """)

    # --- 1. Dataset EnjoySport ---
    st.markdown("**Dataset: EnjoySport**")
    
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
    hypothesis = ['ø', 'ø', 'ø', 'ø', 'ø', 'ø']
    
    for i, row in df.iterrows():
        is_positive = row['EnjoySport'] == 'Yes'
        attributes = row[:-1].values
        
        if is_positive:
            if hypothesis[0] == 'ø':
                hypothesis = list(attributes)
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != attributes[j]:
                        hypothesis[j] = '?'
    
    st.success(f"**Hipótesis Final Aprendida:** {hypothesis}")

def render():
    st.header("Simbólico y Lógica: El Enfoque de Caja Blanca")
    st.markdown("""
    En esta sección exploramos algoritmos que aprenden **reglas legibles por humanos** 
    en lugar de pesos numéricos opacos.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Find-S", 
        "🤺 Candidate Elimination", 
        "🌳 ID3 (Simbólico)", 
        "📜 FOIL (Reglas)"
    ])

    with tab1:
        render_find_s()
    
    with tab2:
        candidate_elimination.render()
        
    with tab3:
        id3_symbolic.render()
        
    with tab4:
        foil_simplified.render()
