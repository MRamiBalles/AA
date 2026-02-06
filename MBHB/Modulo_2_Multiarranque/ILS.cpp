#include <vector>
#include <iostream>
#include "Mutation.cpp"

using namespace std;

// --- ILS: Iterated Local Search ---
// Estrategia: Búsqueda Local -> Perturbación -> Búsqueda Local -> Criterio Aceptación
// Ejecución: 10 Iteraciones (1 inicial + 9 reinicios)
// Perturbación: Mutación Sublista n/4

struct ResultadoILS {
    vector<int> solucion;
    long long coste;
};

ResultadoILS iteratedLocalSearch(const vector<vector<int>>& flujo, const vector<vector<int>>& distancia, int n) {
    // 1. Solución Inicial
    vector<int> solActual = generarSolucionAleatoria(n);
    solActual = busquedaLocalBestImprovement(solActual, flujo, distancia);
    long long costeActual = evaluarSolucion(solActual, flujo, distancia);
    
    // Mejor Global
    vector<int> mejorGlobal = solActual;
    long long costeMejorGlobal = costeActual;
    
    // Bucle de Reinicios (9 iteraciones adicionales, total 10 óptimos locales explorados)
    int iteraciones = 9; 
    
    for(int i = 0; i < iteraciones; i++) {
        // Copia para perturbar
        vector<int> solCandidata = mejorGlobal; // ILS suele perturbar desde el MEJOR encontrado (Trayectoria)
                                                // O desde el actual? La guía dice "si Coste(S') < Coste(Best)... actualizamos". 
                                                // Implica que siempre intentamos mejorar el Best.
        
        // 2. Perturbación (Mutación)
        int tamSublista = n / 4;
        mutarSublista(solCandidata, tamSublista);
        
        // 3. Búsqueda Local
        solCandidata = busquedaLocalBestImprovement(solCandidata, flujo, distancia);
        long long costeCandidata = evaluarSolucion(solCandidata, flujo, distancia);
        
        // 4. Criterio de Aceptación (El Mejor)
        // Aceptamos solo si mejora estrictamente el histórico
        if (costeCandidata < costeMejorGlobal) {
            mejorGlobal = solCandidata;
            costeMejorGlobal = costeCandidata;
            // solActual = solCandidata; // En esta variante "Best", la base de mutación siempre es el mejor.
        }
        // Si no mejora, el bucle siguiente volverá a mutar 'mejorGlobal'.
    }
    
    return {mejorGlobal, costeMejorGlobal};
}
