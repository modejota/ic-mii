import math
import random
import numpy as np

from algoritmo_genetico import AlgoritmoGenetico


class AlgoritmoGeneticoLamarckiano(AlgoritmoGenetico):
    def __init__(self, tamanio_problema, matriz_pesos, matriz_distancias, prob_mut, prob_cruce, num_individuos_inicial, num_padres_torneo, num_mutaciones, porcentaje_torneo):
        super().__init__(tamanio_problema, matriz_pesos, matriz_distancias, prob_mut, prob_cruce, num_individuos_inicial, num_padres_torneo, num_mutaciones, porcentaje_torneo)

    def opt_greedy_lamarckiana(self, scores):
        print("Ejecución greedy. This may take a while...")
        for x in range(len(scores)):
            # print("I am doing calculation ", x+1, " of ", len(scores))
            ind = scores[x][0]
            best = ind.copy()
            for i in range(len(ind)):
                for j in range(i + 1, len(ind)):
                    if i != j:
                        T = best.copy()
                        T[i], T[j] = T[j], T[i]
                        if self.fitnessIndividuo(T) < self.fitnessIndividuo(best):
                            best = T
            scores[x] = (best, self.fitnessIndividuo(best))
        # print("Finalizado Greedy Lamarck")
        return scores

    def executeAlgorithm(self, num_iter_max=10):
        population = self.inicializarPoblacion()
        scores = self.fitnessTodaPoblacion(population)
        scores_lamarck = self.opt_greedy_lamarckiana(scores.copy())

        num_iter = 0
        mejor_global = (np.zeros(self.tamanio_problema), math.inf)
        while num_iter < num_iter_max:
            print("Generación: ", num_iter+1)
            next_gen = [self.seleccionPorTorneo(scores_lamarck) for _ in range(self.num_padres_torneo)]
            for i in range(len(next_gen) - 1):
                s1, s2 = self.cruce(next_gen[i], next_gen[i + 1])
                next_gen.append(s1)
                next_gen.append(s2)
            mejor_hijo = list(min(next_gen, key=lambda x: x[1]))
            if mejor_hijo[1] < mejor_global[1]:
                mejor_global = mejor_hijo
                print("Mejor fitness por ahora: ", mejor_global[1])
                print("Mejor individuo por ahora: ", mejor_global[0])   

            population = self.inicializarPoblacion(self.num_individuos_inicial - len(next_gen))
            scores = self.fitnessTodaPoblacion(population) + next_gen
            scores_lamarck = self.opt_greedy_lamarckiana(scores)
            random.shuffle(scores_lamarck)

            num_iter += 1
        
        print("Mejor combinación al final:", mejor_global)            