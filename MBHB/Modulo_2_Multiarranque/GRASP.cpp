#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

// --- GRASP: Greedy Randomized Adaptive Search Procedure ---

// 1. Construcción Greedy Aleatorizada (LRC - Lista Restringida de Candidatos)
// alpha: Factor de relajación (0.0 = Greedy Puro, 1.0 = Aleatorio Puro)
// En QAP constructivo (Potenciales), aleatorizamos la elección de la asignación.
// Estrategia: Calcular potenciales, tomar los 'k' mejores candidatos y elegir uno al azar.

struct ElementoCandidato {
    int id;
    long long potencial;
};

// Reutilizamos estructuras de comparacion de Greedy puro si es posible, o redefinimos
bool comparePotMayor(const ElementoCandidato& a, const ElementoCandidato& b) { return a.potencial > b.potencial; }
bool comparePotMenor(const ElementoCandidato& a, const ElementoCandidato& b) { return a.potencial < b.potencial; }

vector<int> construccionGreedyAleatorizada(const vector<vector<int>>& flujo, const vector<vector<int>>& distancia, float alpha) {
    int n = flujo.size();
    vector<int> solucion(n, -1); // -1 = no asignado
    
    // Potenciales Flujo (Unidades)
    vector<ElementoCandidato> potFlujo(n);
    for(int i=0; i<n; i++) {
        potFlujo[i].id = i;
        potFlujo[i].potencial = 0;
        for(int j=0; j<n; j++) potFlujo[i].potencial += flujo[i][j] + flujo[j][i];
    }
    sort(potFlujo.begin(), potFlujo.end(), comparePotMayor); // Mayor flujo primero
    
    // Potenciales Distancia (Locaciones)
    vector<ElementoCandidato> potDist(n);
    for(int k=0; k<n; k++) {
        potDist[k].id = k;
        potDist[k].potencial = 0;
        for(int l=0; l<n; l++) potDist[k].potencial += distancia[k][l] + distancia[l][k];
    }
    sort(potDist.begin(), potDist.end(), comparePotMenor); // Menor distancia (centro) primero
    
    // Asignación:
    // Para la unidad i (en orden de flujo), elegimos una locación de la LRC.
    // Ojo: El Greedy de QAP suele ser directo (sort vs sort). 
    // Para GRASP, relajamos el emparejamiento.
    
    // Estrategia simplificada para QAP-GRASP:
    // 1. Tomamos la Unidad con mayor flujo NO asignada.
    // 2. Definimos LRC de Locaciones disponibles (las mejores 'k' o por umbral alpha).
    // 3. Asignamos Unidad -> Locación Random(LRC).
    
    vector<bool> locacionOcupada(n, false);
    
    for(int i=0; i<n; i++) {
        int unidad = potFlujo[i].id;
        
        // Construir lista de locaciones disponibles
        vector<ElementoCandidato> locacionesDisponibles;
        for(int k=0; k<n; k++) {
            if(!locacionOcupada[potDist[k].id]) {
                locacionesDisponibles.push_back(potDist[k]);
            }
        }
        
        // Definir LRC
        // Tamaño LRC = max(1, alpha * disponibles)
        int nDisp = locacionesDisponibles.size();
        int lrcSize = max(1, (int)(alpha * nDisp));
        
        // Elegir aleatoriamente de los mejores lrcSize
        int indexAleatorio = aleatorio(0, lrcSize - 1);
        int locacionElegida = locacionesDisponibles[indexAleatorio].id;
        
        solucion[unidad] = locacionElegida;
        locacionOcupada[locacionElegida] = true;
    }
    
    return solucion;
}

// 2. GRASP Principal
struct ResultadoGRASP {
    vector<int> solucion;
    long long coste;
};

ResultadoGRASP grasp(const vector<vector<int>>& flujo, const vector<vector<int>>& distancia, int maxIter, float alpha) {
    vector<int> mejorSolucion;
    long long mejorCoste = -1;
    
    for(int i = 0; i < maxIter; i++) {
        // Fase Constructiva
        vector<int> sol = construccionGreedyAleatorizada(flujo, distancia, alpha);
        
        // Fase de Mejora (Búsqueda Local)
        // Usamos Best Improvement (importado de Modulo 1)
        sol = busquedaLocalBestImprovement(sol, flujo, distancia);
        
        long long coste = evaluarSolucion(sol, flujo, distancia);
        
        if (mejorCoste == -1 || coste < mejorCoste) {
            mejorCoste = coste;
            mejorSolucion = sol;
        }
    }
    
    return {mejorSolucion, mejorCoste};
}
