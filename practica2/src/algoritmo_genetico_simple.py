import math
import random
import numpy as np

from algoritmo_genetico import AlgoritmoGenetico

class AlgoritmoGeneticoSimple(AlgoritmoGenetico):
    def __init__(self, tamanio_problema, matriz_pesos, matriz_distancias, prob_mut, prob_cruce, num_individuos_inicial, num_padres_torneo, num_mutaciones, porcentaje_torneo):
        super().__init__(tamanio_problema, matriz_pesos, matriz_distancias, prob_mut, prob_cruce, num_individuos_inicial, num_padres_torneo, num_mutaciones, porcentaje_torneo)

    def executeAlgorithm(self, num_iter_max=350):
        # Se podría mandar al constructor y tenerlos como atributos, pero así permitimos diferentes poblaciones con un mismo objeto del algoritmo.
        population = self.inicializarPoblacion()
        scores = self.fitnessTodaPoblacion(population)
        
        num_iter = 0
        mejor_global = (np.zeros(self.tamanio_problema), math.inf)    # La mejor solución global es nula de inicio
        while num_iter < num_iter_max:
            print("Generación: ", num_iter+1)
            parent = [self.seleccionPorTorneo(scores) for _ in range(self.num_padres_torneo)]
            for i in range(len(parent)-1):
                s1, s2 = self.cruce(parent[i], parent[i+1])
                parent.append(s1)   # Se añaden los hijos al final de la lista de los padres.
                parent.append(s2)   # Por reusar un poco el nombre de la variable, pero hace más referencia a la población que se mantiene.
            mejor_hijo = list(min(parent, key=lambda x: x[1]))
            if mejor_hijo[1] < mejor_global[1]:
                mejor_global = mejor_hijo

            new_population = self.inicializarPoblacion(self.num_individuos_inicial - len(parent))   
            scores = self.fitnessTodaPoblacion(new_population) + parent 
            random.shuffle(scores)

            num_iter += 1
            print("Mejor fitness por ahora:", mejor_global[1], "\n")
            # print("Mejor solución por ahora: ", mejor_global[0])


        print("Mejor combinación al final:", mejor_global)