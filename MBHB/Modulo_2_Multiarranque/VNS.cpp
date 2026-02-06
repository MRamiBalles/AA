#include <vector>
#include <iostream>
// Mutation.cpp se incluye en ILS o en main, aseguramos no redefinición con #pragma o header guards en prod.
// Aquí asumimos unidad de compilación o que main incluye todo.

using namespace std;

// --- VNS: Variable Neighborhood Search ---
// Estrategia: Entornos [k=1..kmax]
// k determina la intensidad de la perturbación (tamaño sublista).
// Si BL mejora -> k=1. Si no -> k++.

struct ResultadoVNS {
    vector<int> solucion;
    long long coste;
};

ResultadoVNS variableNeighborhoodSearch(const vector<vector<int>>& flujo, const vector<vector<int>>& distancia, int n) {
    // 1. Solución Inicial
    vector<int> solActual = generarSolucionAleatoria(n);
    // VNS a veces comienza con BL, a veces no. La guía dice: "Generar S_act ... optimizar S_vec". 
    // Asumiremos que S_act es un óptimo local inicial para ser justos.
    solActual = busquedaLocalBestImprovement(solActual, flujo, distancia);
    long long costeActual = evaluarSolucion(solActual, flujo, distancia);
    
    int k = 1;
    int kMax = 5;
    
    // Criterio de Parada: Para ser comparables con ILS (10 BLs), podríamos limitar por iteraciones o evaluciones.
    // O un límite fijo como "50 iteraciones VNS". Pondremos un límite razonable para el 'while'.
    int maxFallos = 50; // Safety break
    int iter = 0;
    
    while (iter < maxFallos) {
        // Definir tamaño de vecindario k
        // k=1 -> n/8, k=2 -> n/7, ... k=5 -> n/4
        // Fórmula: s = n / (9 - k)
        // Ejemplo n=25:
        // k=1 (8) -> s=3
        // k=5 (4) -> s=6
        int tamSublista = n / (9 - k);
        if (tamSublista < 2) tamSublista = 2; // Mínimo 2 swap
        
        // 1. Shaking (Perturbación)
        vector<int> solVecino = solActual;
        mutarSublista(solVecino, tamSublista);
        
        // 2. Búsqueda Local
        vector<int> solOptima = busquedaLocalBestImprovement(solVecino, flujo, distancia);
        long long costeOptimo = evaluarSolucion(solOptima, flujo, distancia);
        
        iter++;
        
        // 3. Cambio de Entorno
        if (costeOptimo < costeActual) {
            // Move (Mejora)
            solActual = solOptima;
            costeActual = costeOptimo;
            k = 1; // Reiniciar entorno (Intensificar)
        } else {
            // Next Neighborhood (Diversificar)
            k++;
            if (k > kMax) k = 1; // Volver a empezar ciclo
        }
    }
    
    return {solActual, costeActual};
}
