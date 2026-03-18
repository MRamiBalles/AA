import pandas as pd
import numpy as np

def get_enjoysport():
    """Dataset clásico de 4 ejemplos (Small)"""
    data = [
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
    ]
    columns = ['Cielo', 'Temp', 'Humedad', 'Viento', 'Agua', 'Pronostico', 'Disfruta']
    return pd.DataFrame(data, columns=columns)

def get_playtennis():
    """Dataset estándar de 14 ejemplos (Medium)"""
    data = [
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
    columns = ['Outlook', 'Temp', 'Humidity', 'Wind', 'Play']
    return pd.DataFrame(data, columns=columns)

def get_synthetic_symbolic(n_samples=50, n_features=4, seed=42):
    """Generador de datos categóricos aleatorios (Large)"""
    np.random.seed(seed)
    feature_vals = {
        f'Attr_{i}': [f'V{i}_{j}' for j in range(3)] 
        for i in range(n_features)
    }
    
    rows = []
    for _ in range(n_samples):
        row = [np.random.choice(feature_vals[f'Attr_{i}']) for i in range(n_features)]
        # Regla simple para determinar el target: si Attr_0 es V0_0 y Attr_1 no es V1_2
        target = 'Yes' if (row[0] == 'V0_0' and row[1] != 'V1_2') else 'No'
        row.append(target)
        rows.append(row)
        
    columns = [f'Attr_{i}' for i in range(n_features)] + ['Target']
    return pd.DataFrame(rows, columns=columns)
