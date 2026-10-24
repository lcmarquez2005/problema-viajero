# Problema del Agente Viajero (TSP) — Versión JavaScript con WebGPU

Proyecto idéntico al desarrollo en Python (`agente_viajero.py`), migrado a **JavaScript moderno (ES Modules)** con procesamiento acelerado en la **tarjeta gráfica (GPU) mediante WebGPU** y lectura de hojas de cálculo con `xlsx`.

---

## 🚀 Características Principales

1. **Aceleración por Tarjeta Gráfica con WebGPU (WGSL Compute Shaders)**:
   - Evaluación masiva en paralelo de las rutas del **Algoritmo Genético** directamente en los núcleos de cómputo de la GPU.
   - Detecta automáticamente el hardware gráfico del sistema (ej. **NVIDIA GeForce RTX 4050 Laptop GPU**).
   - Selector en tiempo real para alternar y comparar el rendimiento entre **WebGPU (GPU)** y **CPU**.

2. **Lectura idéntica de Excel**:
   - Carga y limpia automáticamente el archivo de prueba `CIUDADES_gasolina.xlsx`.
   - Limpieza estricta: ignora filas vacías basándose en la columna de la 'ciudad 1' y convierte celdas no numéricas a 0.
   - Permite arrastrar o seleccionar cualquier otro archivo Excel `.xlsx` directamente desde la interfaz.

3. **Formatos y Algoritmos 100% Idénticos**:
   - **1) Fuerza Bruta con Poda**: exploración exacta con backtracking y poda de trayectos subóptimos.
   - **2) Algoritmo Voraz (Greedy)**: heurística del vecino más cercano partiendo de la Ciudad 1 y regresando al origen.
   - **3) Algoritmos Genéticos**: cruce OX (Order Crossover), elitismo (top 20%), selección de padres (top 50%) y mutación por intercambio.
   - **Formato estricto de tiempo**:
     ```text
     HH:MM:SS X horas, Y minutos y Z segundos para el método de [Método]
     ```
   - **Formato de salida idéntico**:
     ```text
     1) Costo total de la solución: [costo]
     2) Ruta seleccionada: 1 -> X -> Y -> ... -> 1
     3) Tiempo requerido: [tiempo]
     ```

4. **Visualizador de Ruta en Tiempo Real**:
   - Canvas interactivo que dibuja la posición de las ciudades y anima el recorrido óptimo calculado.

---

## 🛠️ Ejecución con `pnpm`

Para iniciar el servidor de desarrollo en el navegador:

```bash
pnpm dev
```

Luego abre la dirección en tu navegador (por ejemplo, Google Chrome o Microsoft Edge con WebGPU habilitado):

👉 **http://localhost:5173**
