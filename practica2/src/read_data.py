import numpy as np

def read_data_from_file(filename):

    with open(filename, 'r' ) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        both_matrix = [list(map(int, line)) for line in lines if line != []]    # Quitar líneas en blanco del fichero

        problem_size = int(lines[0][0])     # Primer elemento de la primera línea es el tamaño del problema
        # Separar ambas matrices se puede hacer con slicing fácilmente.
        flow_matrix = both_matrix[1:problem_size+1]
        distance_matrix = both_matrix[problem_size+1:]
        # Matrices como arrays de numpy, a ver si consguimos optimizar un poco los cálculos
        return problem_size, np.asfarray(flow_matrix), np.asfarray(distance_matrix)

