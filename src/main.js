import { cargarMatrizDesdeBuffer, fuerzaBrutaAsync, algoritmoVoraz, algoritmoGeneticoCPU, formatoEstimado, calcularFactorialStr } from './algorithms.js';
import { WebGPUTSP } from './webgpu-tsp.js';

// DOM Elements
const statFilename = document.getElementById('stat-filename');
const excelFileInput = document.getElementById('excel-file-input');

// Stepper (- / +) & FB Controls
const citiesCounterInput = document.getElementById('cities-counter-input');
const btnCitiesDec = document.getElementById('btn-cities-dec');
const btnCitiesInc = document.getElementById('btn-cities-inc');
const btnRunFB = document.getElementById('btn-run-fb');
const btnStopFB = document.getElementById('btn-stop-fb');
const fbProgressBox = document.getElementById('fb-progress-box');
const fbProgressBarFill = document.getElementById('fb-progress-bar-fill');
const fbLiveCost = document.getElementById('fb-live-cost');
const fbLiveNodes = document.getElementById('fb-live-nodes');
const fbLiveEta = document.getElementById('fb-live-eta');
const cardFB = document.getElementById('card-fb');

// Voraz & Genético Buttons
const btnRunVoraz = document.getElementById('btn-run-voraz');
const btnRunGen = document.getElementById('btn-run-gen');
const cardVoraz = document.getElementById('card-voraz');
const cardGen = document.getElementById('card-genetico');

// Results elements
const statCostVal = document.getElementById('stat-cost-val');
const statCostDesc = document.getElementById('stat-cost-desc');
const statCombVal = document.getElementById('stat-comb-val');
const statCombDesc = document.getElementById('stat-comb-desc');
const statTimeVal = document.getElementById('stat-time-val');
const statTimeDetail = document.getElementById('stat-time-detail');
const statAlgoVal = document.getElementById('stat-algo-val');
const statAlgoDesc = document.getElementById('stat-algo-desc');
const routeTagsContainer = document.getElementById('route-tags-container');
const btnCopyRoute = document.getElementById('btn-copy-route');

// Canvas Map
const mapCanvas = document.getElementById('map-canvas');
const ctx = mapCanvas.getContext('2d');
const canvasStatusText = document.getElementById('canvas-status-text');

// State
let matriz = null;
let nombresCiudades = [];
let archivoActual = 'CIUDADES_gasolina.xlsx';
let webgpu = new WebGPUTSP();
let ultimasCoordenadas = [];
let animacionFrame = null;
let isExecutionCancelled = false;
let ultimaRutaCalculada = [];

// =====================================================================
// INICIALIZACIÓN DE WEBGPU
// =====================================================================
async function initGPU() {
  await webgpu.init();
}

function actualizarContador(valor) {
  if (!matriz) return;
  const maxCiudades = matriz.length;
  const clamped = Math.max(2, Math.min(valor, maxCiudades));
  citiesCounterInput.value = clamped;
  btnCitiesDec.disabled = (clamped <= 2);
  btnCitiesInc.disabled = (clamped >= maxCiudades);
}

// =====================================================================
// CARGA Y PARSEO DE EXCEL
// =====================================================================
async function cargarArchivoPorDefecto() {
  try {
    const res = await fetch(`/${archivoActual}`);
    if (!res.ok) throw new Error(`No se pudo cargar /${archivoActual}`);
    const buf = await res.arrayBuffer();
    const resultado = cargarMatrizDesdeBuffer(buf);
    matriz = resultado.matriz;
    nombresCiudades = resultado.nombresCiudades;

    const total = matriz.length;
    statFilename.textContent = archivoActual;
    
    citiesCounterInput.max = total;
    actualizarContador(10);

    generarCoordenadasCiudades(total);
    dibujarMapa([]);
  } catch (err) {
    console.error('Error al cargar archivo inicial:', err);
  }
}

async function manejarCargaArchivoLocal(file) {
  if (!file) return;
  try {
    const buf = await file.arrayBuffer();
    const resultado = cargarMatrizDesdeBuffer(buf);
    matriz = resultado.matriz;
    nombresCiudades = resultado.nombresCiudades;
    archivoActual = file.name;

    const total = matriz.length;
    statFilename.textContent = archivoActual;
    
    citiesCounterInput.max = total;
    actualizarContador(Math.min(10, total));

    generarCoordenadasCiudades(total);
    dibujarMapa([]);
  } catch (err) {
    alert(`Error al procesar el archivo Excel: ${err.message}`);
  }
}

// =====================================================================
// RENDERIZADO DE RESULTADOS
// =====================================================================
function mostrarResultados(resultado, nombreMetodo, usoGPU, infoCombinaciones = null) {
  const { costo, rutaStr, tiempoStr, ruta, duracionSegundos } = resultado;
  ultimaRutaCalculada = ruta || [];

  // 1. Costo
  statCostVal.textContent = costo.toLocaleString();
  if (statCostDesc) statCostDesc.textContent = 'Distancia total';

  // 2. Combinaciones (KPI propio al lado del costo)
  if (statCombVal && statCombDesc) {
    if (infoCombinaciones) {
      statCombVal.textContent = infoCombinaciones.valor;
      statCombDesc.textContent = infoCombinaciones.descripcion;
    } else {
      statCombVal.textContent = '--';
      statCombDesc.textContent = 'Espacio evaluado';
    }
  }

  // 2. Tiempo (Formato corto en valor, formato completo estricto en detalle)
  const h = Math.floor(duracionSegundos / 3600);
  const m = Math.floor((duracionSegundos % 3600) / 60);
  const s = Math.floor(duracionSegundos % 60);
  const ms = Math.floor((duracionSegundos % 1) * 1000);
  const pad = (n) => String(n).padStart(2, '0');
  statTimeVal.textContent = `${pad(h)}:${pad(m)}:${pad(s)}.${String(ms).padStart(3, '0')}`;
  statTimeDetail.textContent = tiempoStr;

  // 3. Algoritmo Utilizado
  if (statAlgoVal) statAlgoVal.textContent = nombreMetodo;
  if (statAlgoDesc) {
    statAlgoDesc.textContent = usoGPU 
      ? `Acelerado en Tarjeta Gráfica (${webgpu.getGPUDescription()})`
      : 'Procesamiento en CPU';
  }

  // 4. Cadena de etiquetas de ruta
  renderizarRutaTags(ruta);

  // 6. Mapa Canvas
  canvasStatusText.textContent = `Ruta calculada con ${nombreMetodo} (Costo: ${costo})`;
  animarRuta(ruta);
}

function renderizarRutaTags(ruta) {
  routeTagsContainer.innerHTML = '';
  if (!ruta || ruta.length === 0) {
    routeTagsContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 12px;">Sin ruta generada.</span>';
    return;
  }

  for (let i = 0; i < ruta.length; i++) {
    const cityIdx = ruta[i];
    const isOrigin = (cityIdx === 0);

    const pill = document.createElement('span');
    pill.className = `route-node-pill ${isOrigin ? 'origin' : ''}`;
    pill.title = nombresCiudades[cityIdx] || `Ciudad ${cityIdx + 1}`;
    pill.textContent = String(cityIdx + 1);
    routeTagsContainer.appendChild(pill);

    if (i < ruta.length - 1) {
      const arrow = document.createElement('span');
      arrow.className = 'route-arrow';
      arrow.textContent = '➔';
      routeTagsContainer.appendChild(arrow);
    }
  }
}

// =====================================================================
// EJECUCIÓN DE FUERZA BRUTA
// =====================================================================
btnRunFB.addEventListener('click', async () => {
  if (!matriz) return;
  const num = parseInt(citiesCounterInput.value, 10);
  const useGPU = webgpu.isSupported;

  // Actualizar UI para ejecución
  isExecutionCancelled = false;
  cardFB.classList.add('active-run');
  btnRunFB.style.display = 'none';
  btnStopFB.style.display = 'inline-flex';
  fbProgressBox.style.display = 'flex';
  fbProgressBarFill.style.width = '0%';
  fbLiveCost.textContent = 'Mejor: --';
  fbLiveNodes.textContent = 'Nodos: 0';
  fbLiveEta.textContent = 'Tiempo restante: Estimando...';
  canvasStatusText.textContent = 'Iniciando búsqueda de camino...';

  try {
    const resultado = await fuerzaBrutaAsync(matriz, num, {
      shouldCancel: () => isExecutionCancelled,
      onProgress: (p) => {
        fbLiveCost.textContent = `Mejor: ${p.mejorCosto.toLocaleString()}`;
        fbLiveNodes.textContent = `Nodos: ${p.nodosExplorados.toLocaleString()}`;
        fbLiveEta.textContent = `Restante: ${formatoEstimado(p.tiempoEstimadoRestante)}`;

        const pct = Math.min(100, (p.progresoFraccion * 100)).toFixed(1);
        fbProgressBarFill.style.width = `${Math.max(5, pct)}%`;

        // Renderiza el intento en el que va cada 0.1 segundos
        if (p.intentoActual && p.intentoActual.length > 1) {
          dibujarMapa(p.intentoActual, 1.0);
          canvasStatusText.textContent = `Consultando intento en curso: ${p.intentoActual.slice(0, 5).map(c => c + 1).join('➔')}... (Nodos: ${p.nodosExplorados.toLocaleString()})`;
        }
      }
    });

    fbProgressBarFill.style.width = '100%';
    const totalComb = calcularFactorialStr(num - 1);
    mostrarResultados(
      resultado, 
      `Fuerza Bruta (${num} ciudades)`, 
      useGPU, 
      {
        valor: totalComb,
        descripcion: `Nodos explorados: ${resultado.nodosExplorados.toLocaleString()}`
      }
    );
  } catch (err) {
    alert(`Error en Fuerza Bruta: ${err.message}`);
  } finally {
    cardFB.classList.remove('active-run');
    btnRunFB.style.display = 'inline-flex';
    btnStopFB.style.display = 'none';
    setTimeout(() => {
      fbProgressBox.style.display = 'none';
    }, 1500);
  }
});

btnStopFB.addEventListener('click', () => {
  isExecutionCancelled = true;
  fbLiveEta.textContent = 'Deteniendo búsqueda...';
});

// =====================================================================
// EJECUCIÓN DE ALGORITMO VORAZ (GREEDY)
// =====================================================================
btnRunVoraz.addEventListener('click', async () => {
  if (!matriz) return;
  const useGPU = webgpu.isSupported;
  cardVoraz.classList.add('active-run');
  btnRunVoraz.disabled = true;

  try {
    let resultado;
    if (useGPU) {
      resultado = await webgpu.ejecutarVorazGPU(matriz);
    } else {
      resultado = algoritmoVoraz(matriz);
    }
    const n = matriz.length;
    mostrarResultados(
      resultado, 
      'Algoritmo Voraz (Greedy)', 
      useGPU,
      {
        valor: `${n - 1} pasos`,
        descripcion: `De ${calcularFactorialStr(n - 1)} combinaciones posibles`
      }
    );
  } catch (err) {
    alert(`Error en Voraz: ${err.message}`);
  } finally {
    cardVoraz.classList.remove('active-run');
    btnRunVoraz.disabled = false;
  }
});

// =====================================================================
// EJECUCIÓN DE ALGORITMO GENÉTICO (TRADICIONAL)
// =====================================================================
btnRunGen.addEventListener('click', async () => {
  if (!matriz) return;
  cardGen.classList.add('active-run');
  btnRunGen.disabled = true;
  btnRunGen.textContent = 'Evolucionando generaciones...';
  canvasStatusText.textContent = 'Ejecutando Algoritmo Genético...';

  try {
    const resultado = await algoritmoGeneticoCPU(matriz, 150, 800, 0.1, (p) => {
      if (p.intentoActual && p.intentoActual.length > 1) {
        dibujarMapa(p.intentoActual, 1.0);
        canvasStatusText.textContent = `Generación ${p.generacion} | Mejor costo temporal: ${Math.round(p.mejorCosto).toLocaleString()}`;
      }
    });
    const totalCombEspacio = calcularFactorialStr(matriz.length - 1);
    mostrarResultados(
      resultado, 
      'Algoritmos Genéticos (Tradicional)', 
      false,
      {
        valor: '120,000',
        descripcion: `Evaluadas de ${totalCombEspacio} posibles`
      }
    );
  } catch (err) {
    alert(`Error en Algoritmo Genético: ${err.message}`);
  } finally {
    cardGen.classList.remove('active-run');
    btnRunGen.disabled = false;
    btnRunGen.textContent = 'Ejecutar Algoritmo Genético';
  }
});

// =====================================================================
// STEPPER CONTROLS & LISTENERS
// =====================================================================
btnCitiesDec.addEventListener('click', () => {
  const cur = parseInt(citiesCounterInput.value, 10) || 10;
  actualizarContador(cur - 1);
});

btnCitiesInc.addEventListener('click', () => {
  const cur = parseInt(citiesCounterInput.value, 10) || 10;
  actualizarContador(cur + 1);
});

if (btnCopyRoute) {
  btnCopyRoute.addEventListener('click', () => {
    if (!ultimaRutaCalculada || ultimaRutaCalculada.length === 0) return;
    const txt = ultimaRutaCalculada.map(c => c + 1).join(' -> ');
    navigator.clipboard.writeText(txt);
    btnCopyRoute.textContent = '✓ ¡Copiado!';
    setTimeout(() => {
      btnCopyRoute.textContent = '📋 Copiar Ruta';
    }, 1800);
  });
}

excelFileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    manejarCargaArchivoLocal(e.target.files[0]);
  }
});

function ajustarDimensionesCanvas() {
  if (!mapCanvas) return;
  const w = mapCanvas.clientWidth || 800;
  const h = mapCanvas.clientHeight || 320;
  if (mapCanvas.width !== w || mapCanvas.height !== h) {
    mapCanvas.width = w;
    mapCanvas.height = h;
    if (matriz) {
      generarCoordenadasCiudades(matriz.length);
      dibujarMapa(ultimaRutaCalculada || []);
    }
  }
}

// =====================================================================
// CANVAS MAP & GRAPH VISUALIZATION
// =====================================================================
function generarCoordenadasCiudades(total) {
  ultimasCoordenadas = [];
  const width = mapCanvas.width || 800;
  const height = mapCanvas.height || 320;
  const padding = 55;
  const radioX = Math.max(30, (width - padding * 2) / 2);
  const radioY = Math.max(30, (height - padding * 2) / 2);
  const centroX = width / 2;
  const centroY = height / 2;

  for (let i = 0; i < total; i++) {
    const angulo = (i / total) * Math.PI * 2 - Math.PI / 2;
    const perturb = 0.85 + 0.28 * Math.sin(i * 3.7 + 1.2);
    const x = centroX + Math.cos(angulo) * radioX * perturb;
    const y = centroY + Math.sin(angulo) * radioY * perturb;
    ultimasCoordenadas.push({ x, y, id: i + 1, nombre: nombresCiudades[i] || `Ciudad ${i + 1}` });
  }
}

function dibujarMapa(ruta = [], progresoAnimacion = 1.0) {
  const width = mapCanvas.width;
  const height = mapCanvas.height;
  ctx.clearRect(0, 0, width, height);

  // Cuadrícula sutil
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Conexiones de fondo (red de grafo más brillante y visible)
  if (ultimasCoordenadas.length > 0) {
    ctx.strokeStyle = 'rgba(129, 140, 248, 0.24)';
    ctx.lineWidth = 1.25;
    for (let i = 0; i < ultimasCoordenadas.length; i++) {
      for (let j = i + 1; j < ultimasCoordenadas.length; j++) {
        ctx.beginPath();
        ctx.moveTo(ultimasCoordenadas[i].x, ultimasCoordenadas[i].y);
        ctx.lineTo(ultimasCoordenadas[j].x, ultimasCoordenadas[j].y);
        ctx.stroke();
      }
    }
  }

  // Línea de Ruta Resaltada con Glow y Brillo Neón
  if (ruta && ruta.length > 1) {
    const segmentosTotal = ruta.length - 1;
    const segmentosADibujar = Math.floor(segmentosTotal * progresoAnimacion);
    const fraccionUltimo = (segmentosTotal * progresoAnimacion) - segmentosADibujar;

    ctx.save();
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 18;
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();

    const inicio = ultimasCoordenadas[ruta[0]];
    if (inicio) {
      ctx.moveTo(inicio.x, inicio.y);
      for (let s = 0; s < segmentosADibujar; s++) {
        const sig = ultimasCoordenadas[ruta[s + 1]];
        if (sig) ctx.lineTo(sig.x, sig.y);
      }
      if (fraccionUltimo > 0 && segmentosADibujar < segmentosTotal) {
        const u1 = ultimasCoordenadas[ruta[segmentosADibujar]];
        const u2 = ultimasCoordenadas[ruta[segmentosADibujar + 1]];
        if (u1 && u2) {
          const interX = u1.x + (u2.x - u1.x) * fraccionUltimo;
          const interY = u1.y + (u2.y - u1.y) * fraccionUltimo;
          ctx.lineTo(interX, interY);
        }
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  // Nodos (Ciudades)
  for (let i = 0; i < ultimasCoordenadas.length; i++) {
    const { x, y, id } = ultimasCoordenadas[i];
    const esOrigen = (id === 1);

    ctx.save();
    if (esOrigen) {
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 16;
      ctx.fillStyle = '#10b981';
    } else {
      ctx.shadowColor = '#6366f1';
      ctx.shadowBlur = 8;
      ctx.fillStyle = '#1e293b';
    }

    ctx.beginPath();
    ctx.arc(x, y, esOrigen ? 18 : 13.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = esOrigen ? '#34d399' : '#6366f1';
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.fillStyle = esOrigen ? '#000' : '#f8fafc';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(id), x, y);
    ctx.restore();
  }
}

function animarRuta(ruta) {
  if (animacionFrame) cancelAnimationFrame(animacionFrame);
  let start = performance.now();
  const duracionAnim = 1000;

  function frame(now) {
    const progress = Math.min((now - start) / duracionAnim, 1.0);
    const eased = 1 - Math.pow(1 - progress, 3);
    dibujarMapa(ruta, eased);
    if (progress < 1.0) {
      animacionFrame = requestAnimationFrame(frame);
    }
  }
  animacionFrame = requestAnimationFrame(frame);
}

window.addEventListener('resize', ajustarDimensionesCanvas);

// Inicialización
(async () => {
  await initGPU();
  ajustarDimensionesCanvas();
  await cargarArchivoPorDefecto();
  ajustarDimensionesCanvas();
})();
