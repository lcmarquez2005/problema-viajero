import { formatoTiempo } from './algorithms.js';

/**
 * WebGPU Manager con pipelines acelerados para los 3 métodos:
 * 1) Fuerza Bruta (Evaluación de lotes y ramas en GPU)
 * 2) Voraz / Greedy (ArgMin paralelo por reducción en GPU)
 * 3) Algoritmos Genéticos (Evaluación masiva de población en GPU)
 */
export class WebGPUTSP {
  constructor() {
    this.device = null;
    this.adapter = null;
    this.adapterInfo = null;
    this.isSupported = false;

    // 1. Pipeline Voraz en GPU
    this.vorazPipeline = null;
    this.vorazBindGroupLayout = null;

    // 2. Pipeline Genéticos / Evaluación de Rutas en GPU
    this.routesPipeline = null;
    this.routesBindGroupLayout = null;
  }

  async init() {
    if (!navigator.gpu) {
      console.warn('WebGPU no está disponible en este navegador.');
      return false;
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });

      if (!this.adapter) {
        console.warn('No se pudo obtener un adaptador WebGPU con alto rendimiento.');
        return false;
      }

      this.device = await this.adapter.requestDevice();

      if (this.adapter.info) {
        this.adapterInfo = this.adapter.info;
      } else if (this.adapter.requestAdapterInfo) {
        this.adapterInfo = await this.adapter.requestAdapterInfo();
      } else {
        this.adapterInfo = { description: 'Dispositivo WebGPU Genérico' };
      }

      await this._compileVorazShader();
      await this._compileRoutesShader();
      this.isSupported = true;
      return true;
    } catch (e) {
      console.error('Error al inicializar WebGPU:', e);
      this.isSupported = false;
      return false;
    }
  }

  getGPUDescription() {
    if (!this.adapterInfo) return 'GPU Desconocida';
    const desc = this.adapterInfo.description || this.adapterInfo.device || '';
    const vendor = this.adapterInfo.vendor || '';
    const arch = this.adapterInfo.architecture || '';
    if (desc) return `${desc} (${arch || vendor || 'WebGPU'})`;
    return `${vendor} ${arch}`.trim() || 'Acelerador WebGPU';
  }

  /**
   * Shader 1: Algoritmo Voraz (Búsqueda paralela de vecino más cercano en GPU)
   */
  async _compileVorazShader() {
    const wgsl = /* wgsl */ `
      struct VorazParams {
        num_cities: u32,
        current_city: u32,
      };

      @group(0) @binding(0) var<uniform> params: VorazParams;
      @group(0) @binding(1) var<storage, read> distMatrix: array<f32>;
      @group(0) @binding(2) var<storage, read> visitedMask: array<u32>;
      @group(0) @binding(3) var<storage, read_write> outputBest: array<f32, 2>;

      var<workgroup> wg_dist: array<f32, 64>;
      var<workgroup> wg_city: array<u32, 64>;

      @compute @workgroup_size(64)
      fn main(
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>
      ) {
        let city_idx = global_id.x;
        let local_idx = local_id.x;
        let N = params.num_cities;
        let cur = params.current_city;

        var d: f32 = 1e20;
        var c: u32 = 0u;

        if (city_idx < N) {
          if (city_idx != cur && visitedMask[city_idx] == 0u) {
            d = distMatrix[cur * N + city_idx];
            c = city_idx;
          }
        }

        wg_dist[local_idx] = d;
        wg_city[local_idx] = c;

        workgroupBarrier();

        for (var stride = 32u; stride > 0u; stride = stride / 2u) {
          if (local_idx < stride) {
            if (wg_dist[local_idx + stride] < wg_dist[local_idx]) {
              wg_dist[local_idx] = wg_dist[local_idx + stride];
              wg_city[local_idx] = wg_city[local_idx + stride];
            }
          }
          workgroupBarrier();
        }

        if (local_idx == 0u) {
          outputBest[0] = wg_dist[0];
          outputBest[1] = f32(wg_city[0]);
        }
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: 'TSP Voraz Neighbor Search Shader',
      code: wgsl
    });

    this.vorazBindGroupLayout = this.device.createBindGroupLayout({
      label: 'TSP Voraz Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.vorazBindGroupLayout]
    });

    this.vorazPipeline = this.device.createComputePipeline({
      label: 'TSP Voraz Compute Pipeline',
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
  }

  /**
   * Shader 2: Evaluador masivo de rutas en paralelo para Genéticos y Fuerza Bruta
   */
  async _compileRoutesShader() {
    const wgsl = /* wgsl */ `
      struct Params {
        num_cities: u32,
        route_len: u32,
        pop_size: u32,
        pad: u32,
      };

      @group(0) @binding(0) var<uniform> params: Params;
      @group(0) @binding(1) var<storage, read> distMatrix: array<f32>;
      @group(0) @binding(2) var<storage, read> routes: array<u32>;
      @group(0) @binding(3) var<storage, read_write> costs: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= params.pop_size) {
          return;
        }

        let N = params.num_cities;
        let K = params.route_len;
        let offset = idx * K;

        var total_cost: f32 = 0.0;
        let first_city = routes[offset];
        total_cost += distMatrix[0u * N + first_city];

        for (var i = 0u; i < K - 1u; i = i + 1u) {
          let u = routes[offset + i];
          let v = routes[offset + i + 1u];
          total_cost += distMatrix[u * N + v];
        }

        let last_city = routes[offset + K - 1u];
        total_cost += distMatrix[last_city * N + 0u];

        costs[idx] = total_cost;
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: 'TSP Route Batch Cost Evaluator',
      code: wgsl
    });

    this.routesBindGroupLayout = this.device.createBindGroupLayout({
      label: 'TSP Routes Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.routesBindGroupLayout]
    });

    this.routesPipeline = this.device.createComputePipeline({
      label: 'TSP Routes Compute Pipeline',
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
  }

  /**
   * Ejecuta el Algoritmo Voraz (Greedy) acelerado en la GPU
   */
  async ejecutarVorazGPU(matriz) {
    if (!this.isSupported || !this.device) {
      throw new Error('WebGPU no disponible');
    }

    const startTime = performance.now();
    const numCiudades = matriz.length;

    const distFlat = new Float32Array(numCiudades * numCiudades);
    for (let i = 0; i < numCiudades; i++) {
      for (let j = 0; j < numCiudades; j++) {
        distFlat[i * numCiudades + j] = matriz[i][j];
      }
    }

    const distBuffer = this.device.createBuffer({
      size: Math.max(64, distFlat.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(distBuffer.getMappedRange()).set(distFlat);
    distBuffer.unmap();

    const visitedArray = new Uint32Array(numCiudades);
    const visitedBuffer = this.device.createBuffer({
      size: Math.max(64, visitedArray.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    const outputBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const readbackBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.vorazBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: distBuffer } },
        { binding: 2, resource: { buffer: visitedBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } }
      ]
    });

    visitedArray[0] = 1;
    let actual = 0;
    const ruta = [0];
    let costoTotal = 0;

    for (let paso = 1; paso < numCiudades; paso++) {
      this.device.queue.writeBuffer(visitedBuffer, 0, visitedArray);
      this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numCiudades, actual]));

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.vorazPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();

      encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, 8);
      this.device.queue.submit([encoder.finish()]);

      await readbackBuffer.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(readbackBuffer.getMappedRange(0, 8));
      const minDist = data[0];
      const nextCity = Math.round(data[1]);
      readbackBuffer.unmap();

      costoTotal += minDist;
      ruta.push(nextCity);
      visitedArray[nextCity] = 1;
      actual = nextCity;
    }

    costoTotal += matriz[actual][0];
    ruta.push(0);

    distBuffer.destroy();
    visitedBuffer.destroy();
    outputBuffer.destroy();
    readbackBuffer.destroy();
    paramsBuffer.destroy();

    const endTime = performance.now();
    const duracionSegundos = (endTime - startTime) / 1000;
    const tiempoStr = formatoTiempo(duracionSegundos, 'Voraz (WebGPU)');
    const rutaStr = ruta.map(c => c + 1).join(' -> ');

    return {
      costo: costoTotal,
      rutaStr,
      tiempoStr,
      ruta,
      duracionSegundos
    };
  }

  /**
   * Evalúa un lote de rutas arbitrarias en paralelo sobre la GPU
   */
  async evaluarLoteRutasGPU(matriz, rutasArray, popSize, routeLen) {
    if (!this.isSupported || !this.device) return null;

    const numCiudades = matriz.length;
    const distFlat = new Float32Array(numCiudades * numCiudades);
    for (let i = 0; i < numCiudades; i++) {
      for (let j = 0; j < numCiudades; j++) {
        distFlat[i * numCiudades + j] = matriz[i][j];
      }
    }

    const distBuffer = this.device.createBuffer({
      size: Math.max(64, distFlat.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(distBuffer.getMappedRange()).set(distFlat);
    distBuffer.unmap();

    const paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint32Array(paramsBuffer.getMappedRange()).set(new Uint32Array([numCiudades, routeLen, popSize, 0]));
    paramsBuffer.unmap();

    const routesBuffer = this.device.createBuffer({
      size: Math.max(64, rutasArray.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint32Array(routesBuffer.getMappedRange()).set(rutasArray);
    routesBuffer.unmap();

    const costsByteLength = Math.max(64, popSize * 4);
    const costsBuffer = this.device.createBuffer({
      size: costsByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const readbackBuffer = this.device.createBuffer({
      size: costsByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.routesBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: distBuffer } },
        { binding: 2, resource: { buffer: routesBuffer } },
        { binding: 3, resource: { buffer: costsBuffer } }
      ]
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.routesPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(popSize / 64));
    pass.end();

    encoder.copyBufferToBuffer(costsBuffer, 0, readbackBuffer, 0, costsByteLength);
    this.device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const results = new Float32Array(readbackBuffer.getMappedRange().slice(0, popSize * 4));
    readbackBuffer.unmap();

    distBuffer.destroy();
    paramsBuffer.destroy();
    routesBuffer.destroy();
    costsBuffer.destroy();
    readbackBuffer.destroy();

    return results;
  }

  /**
   * Ejecuta Algoritmos Genéticos utilizando WebGPU para evaluar masivamente el fitness
   */
  async ejecutarGeneticoGPU(matriz, tamPoblacion = 150, generaciones = 800, tasaMutacion = 0.1, onProgress = null) {
    if (!this.isSupported || !this.device) {
      throw new Error('WebGPU no está inicializado o no es compatible.');
    }

    const startTime = performance.now();
    const numCiudades = matriz.length;
    const routeLen = numCiudades - 1;

    const distFlat = new Float32Array(numCiudades * numCiudades);
    for (let i = 0; i < numCiudades; i++) {
      for (let j = 0; j < numCiudades; j++) {
        distFlat[i * numCiudades + j] = matriz[i][j];
      }
    }

    const distBuffer = this.device.createBuffer({
      size: Math.max(64, distFlat.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(distBuffer.getMappedRange()).set(distFlat);
    distBuffer.unmap();

    const paramsArray = new Uint32Array([numCiudades, routeLen, tamPoblacion, 0]);
    const paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint32Array(paramsBuffer.getMappedRange()).set(paramsArray);
    paramsBuffer.unmap();

    const routesArray = new Uint32Array(tamPoblacion * routeLen);
    const routesBuffer = this.device.createBuffer({
      size: Math.max(64, routesArray.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    const costsByteLength = Math.max(64, tamPoblacion * Float32Array.BYTES_PER_ELEMENT);
    const costsBuffer = this.device.createBuffer({
      size: costsByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const readbackBuffer = this.device.createBuffer({
      size: costsByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.routesBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: distBuffer } },
        { binding: 2, resource: { buffer: routesBuffer } },
        { binding: 3, resource: { buffer: costsBuffer } }
      ]
    });

    const rutaBase = Array.from({ length: routeLen }, (_, i) => i + 1);
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

    const workgroupCount = Math.max(1, Math.ceil(tamPoblacion / 64));
    const numElitismo = Math.floor(tamPoblacion * 0.2);
    const topPadres = Math.floor(tamPoblacion * 0.5);

    for (let gen = 0; gen < generaciones; gen++) {
      for (let i = 0; i < tamPoblacion; i++) {
        const r = poblacion[i];
        for (let j = 0; j < routeLen; j++) {
          routesArray[i * routeLen + j] = r[j];
        }
      }

      this.device.queue.writeBuffer(routesBuffer, 0, routesArray);

      const commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(this.routesPipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(workgroupCount);
      passEncoder.end();

      commandEncoder.copyBufferToBuffer(costsBuffer, 0, readbackBuffer, 0, costsByteLength);
      this.device.queue.submit([commandEncoder.finish()]);

      await readbackBuffer.mapAsync(GPUMapMode.READ);
      const gpuCosts = new Float32Array(readbackBuffer.getMappedRange().slice(0));
      readbackBuffer.unmap();

      const fitness = [];
      for (let i = 0; i < tamPoblacion; i++) {
        const c = gpuCosts[i];
        fitness.push({ ind: poblacion[i], costo: c });
        if (c < minCostoGlobal) {
          minCostoGlobal = c;
          mejorRutaGlobal = [...poblacion[i]];
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

      if (onProgress && gen % 100 === 0) {
        onProgress({ generacion: gen, mejorCosto: minCostoGlobal });
      }
    }

    distBuffer.destroy();
    paramsBuffer.destroy();
    routesBuffer.destroy();
    costsBuffer.destroy();
    readbackBuffer.destroy();

    const endTime = performance.now();
    const duracionSegundos = (endTime - startTime) / 1000;
    const rutaFinal = [0, ...(mejorRutaGlobal || []), 0];
    const tiempoStr = formatoTiempo(duracionSegundos, 'Genéticos (WebGPU)');
    const rutaStr = rutaFinal.map(c => c + 1).join(' -> ');

    return {
      costo: minCostoGlobal,
      rutaStr,
      tiempoStr,
      ruta: rutaFinal,
      duracionSegundos
    };
  }
}
