import sys
from tkinter import Tk, filedialog
from read_data import read_data_from_file
from algoritmo_genetico_simple import AlgoritmoGeneticoSimple
from algoritmo_genetico_vbaldwiniana import AlgoritmoGeneticoBaldwiniano
from algoritmo_genetico_vlamarckiana import AlgoritmoGeneticoLamarckiano

Tk().withdraw()
try:
    data_file = filedialog.askopenfilename(initialdir="./data", title="Seleccione fichero",
                                           filetypes=[("Fichero de datos", ".dat")])
except(OSError, FileNotFoundError):
    print(f'No se ha podido abrir el fichero seleccionado.')
    sys.exit(100)
except Exception as error:
    print(f'Ha ocurrido un error: <{error}>')
    sys.exit(101)
if len(data_file) == 0 or data_file is None:
    print(f'No se ha seleccionado ningún archivo.')
    sys.exit(102)

tamanio, pesos, distancias = read_data_from_file(data_file)

opcion = -1
while opcion < 0 or opcion > 3:
    print("Seleccione el algoritmo a ejecutar:")
    print("1. Algoritmo evolutivo simple")
    print("2. Algoritmo evolutivo baldwiniano")
    print("3. Algoritmo evolutivo lamarckiano")
    opcion = int(input("Opción: "))
    print("")

# Se han dejado los valores de los parámetros para los que se obtuvieron mejores resultados.
if opcion == 1:
    alg = AlgoritmoGeneticoSimple(
        tamanio, pesos, distancias,
        prob_mut=0.1, prob_cruce=0.8, num_individuos_inicial=500,
        num_padres_torneo=30, num_mutaciones=2, porcentaje_torneo=0.2)
    alg.executeAlgorithm(num_iter_max=300)

elif opcion == 2:
    alg = AlgoritmoGeneticoBaldwiniano(
        tamanio, pesos, distancias,
        prob_mut=0.1, prob_cruce=0.75, num_individuos_inicial=30,
        num_padres_torneo=2, num_mutaciones=2, porcentaje_torneo=0.2)
    alg.executeAlgorithm(num_iter_max=5)

elif opcion == 3:
    alg = AlgoritmoGeneticoLamarckiano(
        tamanio, pesos, distancias,
        prob_mut=0.1, prob_cruce=0.7, num_individuos_inicial=12,
        num_padres_torneo=2, num_mutaciones=2, porcentaje_torneo=0.2)
    alg.executeAlgorithm(num_iter_max=5)

else:
    # Este mensaje no debería aparecer nunca, pero por si acaso...
    print("Opción no válida. Saliendo...")
    sys.exit(103)

# Convertir lista de números a string para el formulario
# numbers_string = " ".join(map(str, numbers))
# print(numbers_string)