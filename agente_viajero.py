import pandas as pd
import numpy as np
import itertools
import time
import random

# =====================================================================
# SECCIÓN ESPECIAL: PROCESO DE CARGA DEL ARCHIVO DE PRUEBA
# =====================================================================
def cargar_matriz(nombre_archivo):
    """
    Carga el archivo Excel y devuelve la matriz de distancias.
    Ignora filas completamente vacías y la primera columna (nombres de ciudades).
    """
    try:
        # Detecta la ruta exacta de la carpeta donde está guardado el script
        import os
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_absoluta = os.path.join(directorio_actual, nombre_archivo)
        
        df = pd.read_excel(ruta_absoluta)
        
        # LIMPIEZA ESTRICTA: Elimina la fila si la celda de la 'ciudad 1' no tiene datos
        df = df.dropna(subset=[df.columns[1]]) 
        
        # Extrae la matriz numérica y fuerza cualquier otro error tipográfico a 0
        matriz = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
        return matriz
        
    except Exception as e:
        print(f"Error al cargar el archivo '{nombre_archivo}': {e}")
        return None
# =====================================================================

def formato_tiempo(segundos, metodo):
    """
    Formatea el tiempo requerido al formato estricto: 
    HH:MM:SS X horas, Y minutos y Z segundos para X método
    """
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    return f"{h:02d}:{m:02d}:{s:02d} {h} horas, {m} minutos y {s} segundos para el método de {metodo}"

def fuerza_bruta(matriz, num_ciudades_selec):
    start_time = time.time()
    min_costo = [float('inf')]
    mejor_ruta = []
    
    def explorar(nodo_actual, visitados, costo_acumulado, ruta_actual):
        # PODA: Si el trayecto parcial ya es más largo que la mejor ruta, abortamos
        if costo_acumulado >= min_costo[0]:
            return
            
        # Si ya visitamos todas las ciudades requeridas
        if len(visitados) == num_ciudades_selec:
            costo_final = costo_acumulado + matriz[nodo_actual][0]
            if costo_final < min_costo[0]:
                min_costo[0] = costo_final
                mejor_ruta.clear()
                mejor_ruta.extend(ruta_actual + [0])
            return
            
        # Explorar los siguientes nodos
        for siguiente in range(1, num_ciudades_selec):
            if siguiente not in visitados:
                visitados.add(siguiente)
                ruta_actual.append(siguiente)
                
                explorar(
                    siguiente, 
                    visitados, 
                    costo_acumulado + matriz[nodo_actual][siguiente], 
                    ruta_actual
                )
                
                # Retroceder (Backtracking)
                visitados.remove(siguiente)
                ruta_actual.pop()

    # Iniciar la exploración desde la Ciudad 1 (índice 0)
    explorar(0, {0}, 0.0, [0])
    
    end_time = time.time()
    tiempo_str = formato_tiempo(end_time - start_time, "Fuerza Bruta (con Poda)")
    ruta_str = " -> ".join([str(c + 1) for c in mejor_ruta])
    return min_costo[0], ruta_str, tiempo_str

def algoritmo_voraz(matriz):
    start_time = time.time()
    num_ciudades = len(matriz)
    no_visitadas = set(range(1, num_ciudades))
    
    actual = 0
    ruta = [0]
    costo = 0
    
    while no_visitadas:
        # Vecino más cercano
        siguiente = min(no_visitadas, key=lambda c: matriz[actual][c])
        costo += matriz[actual][siguiente]
        ruta.append(siguiente)
        no_visitadas.remove(siguiente)
        actual = siguiente
        
    # Regresar al origen (Ciudad 1 / índice 0)
    costo += matriz[actual][0]
    ruta.append(0)
    
    end_time = time.time()
    tiempo_str = formato_tiempo(end_time - start_time, "Voraz (Greedy)")
    ruta_str = " -> ".join([str(c + 1) for c in ruta])
    return costo, ruta_str, tiempo_str

def algoritmo_genetico(matriz, tam_poblacion=150, generaciones=800, tasa_mutacion=0.1):
    start_time = time.time()
    num_ciudades = len(matriz)
    
    def calcular_costo(ruta):
        c = 0
        r = [0] + ruta + [0]
        for i in range(len(r)-1):
            c += matriz[r[i]][r[i+1]]
        return c
        
    ruta_base = list(range(1, num_ciudades))
    poblacion = []
    
    for _ in range(tam_poblacion):
        r = ruta_base.copy()
        random.shuffle(r)
        poblacion.append(r)
        
    mejor_ruta_global = None
    min_costo_global = float('inf')
    
    for gen in range(generaciones):
        fitness = [(ind, calcular_costo(ind)) for ind in poblacion]
        
        for ind, c in fitness:
            if c < min_costo_global:
                min_costo_global = c
                mejor_ruta_global = ind
                
        fitness.sort(key=lambda x: x[1])
        siguiente_gen = [x[0] for x in fitness[:int(tam_poblacion*0.2)]] # Elitismo
        
        while len(siguiente_gen) < tam_poblacion:
            p1, p2 = random.choices(fitness[:int(tam_poblacion*0.5)], k=2)
            p1, p2 = p1[0], p2[0]
            
            # Cruzamiento ordenado (OX)
            inicio, fin = sorted(random.sample(range(len(p1)), 2))
            hijo = [-1]*len(p1)
            hijo[inicio:fin] = p1[inicio:fin]
            p2_filtrado = [x for x in p2 if x not in hijo]
            
            idx = 0
            for i in range(len(hijo)):
                if hijo[i] == -1:
                    hijo[i] = p2_filtrado[idx]
                    idx += 1
            
            # Mutación
            if random.random() < tasa_mutacion:
                i, j = random.sample(range(len(hijo)), 2)
                hijo[i], hijo[j] = hijo[j], hijo[i]
                
            siguiente_gen.append(hijo)
        poblacion = siguiente_gen
        
    end_time = time.time()
    ruta_final = [0] + (mejor_ruta_global or []) + [0]
    tiempo_str = formato_tiempo(end_time - start_time, "Genéticos")
    ruta_str = " -> ".join([str(c + 1) for c in ruta_final])
    
    return min_costo_global, ruta_str, tiempo_str

def principal():
    # ARCHIVO A CARGAR (Cámbialo aquí durante la revisión en clase)
    archivo_prueba = 'CIUDADES_gasolina.xlsx'
    
    print("Iniciando sistema de ruteo...")
    matriz = cargar_matriz(archivo_prueba)
    if matriz is None:
        return
        
    total_ciudades = len(matriz)
    print(f"[{archivo_prueba}] cargado con éxito. ({total_ciudades} ciudades detectadas).")

    while True:
        print("\n" + "="*55)
        print("  PROYECTO: PROBLEMA DEL AGENTE VIAJERO (TSP)")
        print("="*55)
        print("1) Método Exhaustivo (Fuerza Bruta)")
        print("2) Algoritmo de tipo Voraz")
        print("3) Algoritmos Genéticos")
        print("4) Salir del Programa")
        opcion = input("Seleccione una opción (1-4): ")
        
        if opcion == '1':
            try:
                num = int(input(f"¿Cuántas ciudades desea evaluar? (Máximo seguro recomendado: 10, Total en matriz: {total_ciudades}): "))
                if num < 2 or num > total_ciudades:
                    print(f"Rango inválido. Ingrese entre 2 y {total_ciudades}.")
                    continue
                print("Calculando, por favor espere...")
                costo, ruta, tiempo = fuerza_bruta(matriz, num)
                print(f"\n1) Costo total de la solución: {costo}")
                print(f"2) Ruta seleccionada: {ruta}")
                print(f"3) Tiempo requerido: {tiempo}")
            except ValueError:
                print("Por favor, ingrese un número válido.")
                
        elif opcion == '2':
            print(f"Calculando para {total_ciudades} ciudades...")
            costo, ruta, tiempo = algoritmo_voraz(matriz)
            print(f"\n1) Costo total de la solución: {costo}")
            print(f"2) Ruta seleccionada: {ruta}")
            print(f"3) Tiempo requerido: {tiempo}")
            
        elif opcion == '3':
            print(f"Calculando para {total_ciudades} ciudades...")
            costo, ruta, tiempo = algoritmo_genetico(matriz)
            print(f"\n1) Costo total de la solución: {costo}")
            print(f"2) Ruta seleccionada: {ruta}")
            print(f"3) Tiempo requerido: {tiempo}")
            
        elif opcion == '4':
            print("Cerrando programa. ¡Éxito en tu entrega!")
            break
        else:
            print("Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    principal()