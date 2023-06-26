import math
import random
import numpy as np

# Esta clase es una base para las diferentes implementaciones de algoritmos genéticos
# Contiene los métodos básicos y los parámetros que se pueden modificar
class AlgoritmoGenetico:

    # Constructor. 
    # El tamaño y matrices debería venir del lector. El resto pueden ser especificados por usuario.
    def __init__(self, tamanio_problema, matriz_pesos, matriz_distancias,
                 prob_mut, prob_cruce, num_individuos_inicial, 
                 num_padres_torneo, num_mutaciones, porcentaje_torneo):
       
        self.tamanio_problema = tamanio_problema
        self.matriz_pesos = matriz_pesos
        self.matriz_distancias = matriz_distancias
        self.prob_mut = prob_mut
        self.prob_cruce = prob_cruce
        self.num_individuos_inicial = num_individuos_inicial
        self.num_padres_torneo = num_padres_torneo    # Lo usarán las clases hijas
        self.num_mutaciones = num_mutaciones
        self.porcentaje_torneo = porcentaje_torneo

    def inicializarPoblacion(self, sizepop=None):
        if sizepop is None:
            sizepop = self.num_individuos_inicial
        taman = self.tamanio_problema  # Longitud del vector igual al número de localizaciones
        poblacion = [np.random.permutation(taman) for _ in range(sizepop)]
        return poblacion

    def fitnessIndividuo(self, indv):
        return sum(np.sum(self.matriz_pesos * self.matriz_distancias[indv[:, None], indv], 1))

    def fitnessTodaPoblacion(self, poblacion):
        return [(indv, self.fitnessIndividuo(indv)) for indv in poblacion]

    def mutacion(self, indv):
        if self.prob_mut >= random.random():    # Rango [0,1]. Tener en cuenta al definir probabilidad
            for _ in range(self.num_mutaciones):
                # Número de mutaciones parametrizables, para problemas grandes puede interesar sea mayor.
                cromo1 = int(random.choice(indv[0]))
                cromo2 = int(random.choice(indv[0]))
                indv[0][cromo1], indv[0][cromo2] = indv[0][cromo2], indv[0][cromo1]
        return indv

    def gestionarCromosomasRepes(self, indv):
        for n in range(len(indv)):
            ocurrences = np.count_nonzero(indv == indv[n])
            if ocurrences > 1:
                replaces = [r for r in range(len(indv)) if r not in list(indv)]
                indv[n] = random.choice(replaces)
        return indv

    def cruce(self, indv1, indv2):
        if self.prob_cruce >= random.random():
            cut = random.randrange(1, len(indv1[0]))  # Punto de corte aleatorio, puede aportar más diversidad y es más "real"

            hijoA = np.concatenate((indv1[0][:cut], indv2[0][cut:]), axis=None)
            hijoB = np.concatenate((indv2[0][:cut], indv1[0][cut:]), axis=None)
            # Resolver problemas de cromosomas repetidos, es probable que le vengan de los padres algunos repetidos.
            hijoA = (self.gestionarCromosomasRepes(hijoA), self.fitnessIndividuo(hijoA))
            hijoB = (self.gestionarCromosomasRepes(hijoB), self.fitnessIndividuo(hijoB))
        # No se cruzan, se devuelven los padres
        else:
            hijoA = indv1
            hijoB = indv2
        # Despues de cruzar, vemos si mutan
        return self.mutacion(hijoA), self.mutacion(hijoB)

    def seleccionPorTorneo(self, scores):
        min_score = min(scores, key=lambda x: x[1])[1]
        num_participants = math.ceil(self.porcentaje_torneo * len(scores))  # Escogemos participantes de acuerdo a parámetro
        random.shuffle(scores)
        participants = []
        p = 0
        while len(participants) != num_participants:
            prob_selec = min_score / scores[p][1]
            if prob_selec > random.random():
                participants.append(scores[p])
            p += 1

        return list(min(participants, key=lambda x: x[1]))
