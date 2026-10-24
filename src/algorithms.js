import * as XLSX from 'xlsx';

/**
 * Formatea el tiempo requerido al formato estricto: 
 * HH:MM:SS X horas, Y minutos y Z segundos para el método de {metodo}
 */
export function formatoTiempo(segundos, metodo) {
  const h = Math.floor(segundos / 3600);
  const m = Math.floor((segundos % 3600) / 60);
  const s = Math.floor(segundos % 60);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)} ${h} horas, ${m} minutos y ${s} segundos para el método de ${metodo}`;
}

/**
 * Calcula factorial exacto o en notación científica si es muy grande
 */
export function calcularFactorialStr(n) {
  if (n <= 0) return '1';
  let f = 1n;
  for (let i = 2n; i <= BigInt(n); i++) {
    f *= i;
  }
  const s = f.toString();
  if (s.length > 15) {
    const exponente = s.length - 1;
    const mantisa = (Number(s.slice(0, 5)) / 10000).toFixed(2);
    return `${mantisa} × 10^${exponente}`;
  }
  return Number(f).toLocaleString();
}

/**
 * Formatea el estimado de tiempo restante en formato legible
 */
export function formatoEstimado(segundos) {
  if (!isFinite(segundos) || segundos < 0) return 'Estimando tiempo...';
  if (segundos > 365 * 86400) {
    const years = (segundos / (365 * 86400)).toFixed(1);
    return `~${years} años (espacio factorial inmenso)`;
  }
  if (segundos > 86400) {
    const days = Math.floor(segundos / 86400);
    const h = Math.floor((segundos % 86400) / 3600);
    return `~${days} días y ${h} horas`;
  }
  const h = Math.floor(segundos / 3600);
  const m = Math.floor((segundos % 3600) / 60);
  const s = Math.floor(segundos % 60);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)} (${h}h ${m}m ${s}s restantes)`;
}

/**
 * Carga el archivo Excel y devuelve la matriz de distancias.
 */
export function cargarMatrizDesdeBuffer(arrayBuffer) {
  const wb = XLSX.read(arrayBuffer, { type: 'array' });
  const sheet = wb.Sheets[wb.SheetNames[0]];
  const rawRows = XLSX.utils.sheet_to_json(sheet, { header: 1 });

  if (!rawRows || rawRows.length < 2) {
    throw new Error('El archivo Excel no tiene suficientes datos.');
  }

  const validRows = rawRows.slice(1).filter(r => r && r[1] !== undefined && r[1] !== null && String(r[1]).trim() !== '');
  const nombresCiudades = validRows.map((r, i) => (r[0] && String(r[0]).trim() !== '') ? String(r[0]).trim() : `Ciudad ${i + 1}`);

  const matriz = validRows.map(row => {
    return row.slice(1).map(val => {
      const num = Number(val);
      return (isNaN(num) || num === null || num === undefined) ? 0 : num;
    });
  });

  return { matriz, nombresCiudades };
}

/**
 * Algoritmo 1: Fuerza Bruta con Poda Asíncrona (NO bloqueante)
 * Permite ejecutar de 2 a 20 ciudades sin congelar el navegador,
 * calculando el tiempo estimado restante y permitiendo pausar o detener.
 */
export async function fuerzaBrutaAsync(matriz, numCiudadesSelec, options = {}) {
  const { onProgress, shouldCancel } = options;
  const startTime = performance.now();

  let minCosto = Infinity;
  let mejorRuta = [];

  // 1. Inicialización de límite superior (Greedy) para podar agresivamente desde el inicio
  {
    const unvisited = new Set();
    for (let i = 1; i < numCiudadesSelec; i++) unvisited.add(i);
    let cur = 0;
    let cost = 0;
    const r = [0];
    while (unvisited.size > 0) {
      let next = null;
      let minD = Infinity;
      for (const c of unvisited) {
        if (matriz[cur][c] < minD) {
          minD = matriz[cur][c];
          next = c;
        }
      }
      cost += matriz[cur][next];
      r.push(next);
      unvisited.delete(next);
      cur = next;
    }
    cost += matriz[cur][0];
    r.push(0);
    minCosto = cost;
    mejorRuta = r;
  }

  const visitados = new Uint8Array(numCiudadesSelec);
  visitados[0] = 1;
  let visitadosCount = 1;
  const rutaActual = [0];

  let nodosExplorados = 0;
  let rutasCompletadas = 0;
  let lastYieldTime = performance.now();
  let lastSampleTime = performance.now();
  let cancelled = false;

  // Variables para estimación de progreso del árbol
  const branchIndices = new Array(numCiudadesSelec).fill(0);

  async function explorar(nodoActual, costoAcumulado, depth) {
    if (cancelled) return;

    nodosExplorados++;

    // Solo chequeamos tiempo cada 1024 nodos para no saturar con performance.now()
    if ((nodosExplorados & 1023) === 0) {
      const now = performance.now();

      // Consulta y emite el intento en el que va cada 0.1 segundos (100 ms)
      if (now - lastSampleTime >= 100) {
        lastSampleTime = now;
        if (shouldCancel && shouldCancel()) {
          cancelled = true;
          return;
        }

        // Reconstruir el camino completo del intento en curso
        const intentoEnCurso = [...rutaActual];
        for (let i = 1; i < numCiudadesSelec; i++) {
          if (visitados[i] === 0) {
            intentoEnCurso.push(i);
          }
        }
        intentoEnCurso.push(0);

        // Calcular progreso y tiempo estimado
        let progressFraction = 0;
        let factor = 1;
        const maxTrackDepth = Math.min(depth, 4);
        for (let d = 1; d <= maxTrackDepth; d++) {
          const totalBranches = numCiudadesSelec - d;
          if (totalBranches > 0) {
            factor *= totalBranches;
            progressFraction += branchIndices[d] / factor;
          }
        }

        const elapsedSec = (now - startTime) / 1000;
        let estimatedRemaining = Infinity;
        if (progressFraction > 0.0000001) {
          const totalEst = elapsedSec / progressFraction;
          estimatedRemaining = Math.max(0, totalEst - elapsedSec);
        }

        if (onProgress) {
          onProgress({
            nodosExplorados,
            rutasCompletadas,
            mejorCosto: minCosto,
            mejorRuta,
            intentoActual: intentoEnCurso,
            tiempoTranscurrido: elapsedSec,
            tiempoEstimadoRestante: estimatedRemaining,
            progresoFraccion: progressFraction
          });
        }

        // Ceder control al Event Loop para renderizar
        await new Promise(resolve => setTimeout(resolve, 0));
        lastYieldTime = performance.now();
      } else if (now - lastYieldTime > 50) {
        if (shouldCancel && shouldCancel()) {
          cancelled = true;
          return;
        }
        await new Promise(resolve => setTimeout(resolve, 0));
        lastYieldTime = performance.now();
      }
    }

    // PODA: Si el trayecto parcial ya supera la mejor ruta conocida, abortamos
    if (costoAcumulado >= minCosto) {
      return;
    }

    // Si ya visitamos todas las ciudades requeridas
    if (visitadosCount === numCiudadesSelec) {
      rutasCompletadas++;
      const costoFinal = costoAcumulado + matriz[nodoActual][0];
      if (costoFinal < minCosto) {
        minCosto = costoFinal;
        mejorRuta = [...rutaActual, 0];
      }
      return;
    }

    // Explorar los siguientes nodos
    let branchIdx = 0;
    for (let siguiente = 1; siguiente < numCiudadesSelec; siguiente++) {
      if (cancelled) return;

      if (visitados[siguiente] === 0) {
        branchIndices[depth] = branchIdx;
        branchIdx++;

        visitados[siguiente] = 1;
        visitadosCount++;
        rutaActual.push(siguiente);

        await explorar(
          siguiente,
          costoAcumulado + matriz[nodoActual][siguiente],
          depth + 1
        );

        visitados[siguiente] = 0;
        visitadosCount--;
        rutaActual.pop();
      }
    }
  }

  await explorar(0, 0.0, 1);

  const endTime = performance.now();
  const duracionSegundos = (endTime - startTime) / 1000;
  const metodoLabel = cancelled ? 'Fuerza Bruta (Detenido por usuario)' : 'Fuerza Bruta (con Poda)';
  const tiempoStr = formatoTiempo(duracionSegundos, metodoLabel);
  const rutaStr = mejorRuta.map(c => c + 1).join(' -> ');

  return {
    costo: minCosto,
    rutaStr,
    tiempoStr,
    ruta: mejorRuta,
    duracionSegundos,
    nodosExplorados,
    rutasCompletadas,
    cancelled
  };
}

/**
 * Algoritmo 1 Síncrono (Original para pruebas o benchmarks directos)
 */
export function fuerzaBruta(matriz, numCiudadesSelec) {
  const startTime = performance.now();
  let minCosto = Infinity;
  let mejorRuta = [];
  const visitados = new Set([0]);
  const rutaActual = [0];

  function explorar(nodoActual, costoAcumulado) {
    if (costoAcumulado >= minCosto) return;
    if (visitados.size === numCiudadesSelec) {
      const costoFinal = costoAcumulado + matriz[nodoActual][0];
      if (costoFinal < minCosto) {
        minCosto = costoFinal;
        mejorRuta = [...rutaActual, 0];
      }
      return;
    }
    for (let siguiente = 1; siguiente < numCiudadesSelec; siguiente++) {
      if (!visitados.has(siguiente)) {
        visitados.add(siguiente);
        rutaActual.push(siguiente);
        explorar(siguiente, costoAcumulado + matriz[nodoActual][siguiente]);
        visitados.delete(siguiente);
        rutaActual.pop();
      }
    }
  }

  explorar(0, 0.0);

  const endTime = performance.now();
  const duracionSegundos = (endTime - startTime) / 1000;
  const tiempoStr = formatoTiempo(duracionSegundos, 'Fuerza Bruta (con Poda)');
  const rutaStr = mejorRuta.map(c => c + 1).join(' -> ');

  return { costo: minCosto, rutaStr, tiempoStr, ruta: mejorRuta, duracionSegundos };
}

/**
 * Algoritmo 2: Algoritmo de tipo Voraz (Greedy CPU)
 */
export function algoritmoVoraz(matriz) {
  const startTime = performance.now();
  const numCiudades = matriz.length;
  const noVisitadas = new Set(Array.from({ length: numCiudades - 1 }, (_, i) => i + 1));

  let actual = 0;
  const ruta = [0];
  let costo = 0;

  while (noVisitadas.size > 0) {
    let siguiente = null;
    let menorDistancia = Infinity;

    for (const c of noVisitadas) {
      const d = matriz[actual][c];
      if (d < menorDistancia) {
        menorDistancia = d;
        siguiente = c;
      }
    }

    costo += matriz[actual][siguiente];
    ruta.push(siguiente);
    noVisitadas.delete(siguiente);
    actual = siguiente;
  }

  costo += matriz[actual][0];
  ruta.push(0);

  const endTime = performance.now();
  const duracionSegundos = (endTime - startTime) / 1000;
  const tiempoStr = formatoTiempo(duracionSegundos, 'Voraz (Greedy)');
  const rutaStr = ruta.map(c => c + 1).join(' -> ');

  return { costo, rutaStr, tiempoStr, ruta, duracionSegundos };
}

/**
 * Algoritmo 3: Algoritmo Genético (CPU)
 */
export async function algoritmoGeneticoCPU(matriz, tamPoblacion = 150, generaciones = 800, tasaMutacion = 0.1, onProgress = null) {
  const startTime = performance.now();
  const numCiudades = matriz.length;

  function calcularCosto(ruta) {
    let c = 0;
    const r = [0, ...ruta, 0];
    for (let i = 0; i < r.length - 1; i++) {
      c += matriz[r[i]][r[i + 1]];
    }
    return c;
  }

  const rutaBase = Array.from({ length: numCiudades - 1 }, (_, i) => i + 1);
  let poblacion = [];

  for (let i = 0; i < tamPoblacion; i++) {
    const r = [...rutaBase];
    for (let j = r.length - 1; j > 0; j--) {
      const k = Math.floor(Math.random() * (j + 1));
      [r[j], r[k]] = [r[k], r[j]];
    }
    poblacion.push(r);
  }

  let mejorRutaGlobal = null;
  let minCostoGlobal = Infinity;

  const numElitismo = Math.floor(tamPoblacion * 0.2);
  const topPadres = Math.floor(tamPoblacion * 0.5);

  let lastProgressTime = performance.now();

  for (let gen = 0; gen < generaciones; gen++) {
    const fitness = poblacion.map(ind => ({ ind, costo: calcularCosto(ind) }));

    for (const f of fitness) {
      if (f.costo < minCostoGlobal) {
        minCostoGlobal = f.costo;
        mejorRutaGlobal = f.ind;
      }
    }

    fitness.sort((a, b) => a.costo - b.costo);
    const siguienteGen = fitness.slice(0, numElitismo).map(f => f.ind);

    while (siguienteGen.length < tamPoblacion) {
      const p1 = fitness[Math.floor(Math.random() * topPadres)].ind;
      const p2 = fitness[Math.floor(Math.random() * topPadres)].ind;

      let i1 = Math.floor(Math.random() * p1.length);
      let i2 = Math.floor(Math.random() * p1.length);
      if (i1 > i2) [i1, i2] = [i2, i1];

      const hijo = new Array(p1.length).fill(-1);
      const enHijo = new Set();
      for (let i = i1; i <= i2; i++) {
        hijo[i] = p1[i];
        enHijo.add(p1[i]);
      }

      const p2Filtrado = p2.filter(x => !enHijo.has(x));
      let idx = 0;
      for (let i = 0; i < hijo.length; i++) {
        if (hijo[i] === -1) {
          hijo[i] = p2Filtrado[idx++];
        }
      }

      if (Math.random() < tasaMutacion) {
        const m1 = Math.floor(Math.random() * hijo.length);
        const m2 = Math.floor(Math.random() * hijo.length);
        [hijo[m1], hijo[m2]] = [hijo[m2], hijo[m1]];
      }

      siguienteGen.push(hijo);
    }

    poblacion = siguienteGen;

    const now = performance.now();
    if (now - lastProgressTime >= 100) {
      lastProgressTime = now;
      if (onProgress) {
        onProgress({
          generacion: gen,
          mejorCosto: minCostoGlobal,
          intentoActual: [0, ...(mejorRutaGlobal || poblacion[0]), 0]
        });
      }
      await new Promise(r => setTimeout(r, 0));
    }
  }

  const endTime = performance.now();
  const duracionSegundos = (endTime - startTime) / 1000;
  const rutaFinal = [0, ...(mejorRutaGlobal || []), 0];
  const tiempoStr = formatoTiempo(duracionSegundos, 'Genéticos');
  const rutaStr = rutaFinal.map(c => c + 1).join(' -> ');

  return { costo: minCostoGlobal, rutaStr, tiempoStr, ruta: rutaFinal, duracionSegundos };
}
