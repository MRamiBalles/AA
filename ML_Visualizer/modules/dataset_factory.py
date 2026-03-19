import pandas as pd
import numpy as np
from typing import Dict, List

def get_enjoysport() -> pd.DataFrame:
    """
    Retorna el dataset clásico EnjoySport de 4 instancias (Talla S).
    Utilizado para validación inicial de algoritmos de Espacio de Versiones.
    """
    data: List[List[str]] = [
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
    ]
    columns: List[str] = ['Cielo', 'Temp', 'Humedad', 'Viento', 'Agua', 'Pronostico', 'Disfruta']
    return pd.DataFrame(data, columns=columns)

def get_playtennis() -> pd.DataFrame:
    """
    Retorna el dataset estándar PlayTennis de 14 instancias (Talla M).
    Ideal para testing de ID3 y cálculo de Ganancia de Información.
    """
    data: List[List[str]] = [
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']
    ]
    columns: List[str] = ['Outlook', 'Temp', 'Humidity', 'Wind', 'Play']
    return pd.DataFrame(data, columns=columns)

def get_synthetic_symbolic(n_samples: int = 50, n_features: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Generador de datasets categóricos sintéticos (Talla L).
    Permite evaluar el escalado del Version Space y la profundidad de árboles simbólicos.
    """
    np.random.seed(seed)
    feature_vals: Dict[str, List[str]] = {
        f'Attr_{i}': [f'V{i}_{j}' for j in range(3)] 
        for i in range(n_features)
    }
    
    rows: List[List[str]] = []
    for _ in range(n_samples):
        row = [np.random.choice(feature_vals[f'Attr_{i}']) for i in range(n_features)]
        # Lógica de clasificación arbitraria pero consistente
        target = 'Yes' if (row[0] == 'V0_0' and row[1] != 'V1_2') else 'No'
        row.append(target)
        rows.append(row)
        
    columns: List[str] = [f'Attr_{i}' for i in range(n_features)] + ['Target']
    return pd.DataFrame(rows, columns=columns)
