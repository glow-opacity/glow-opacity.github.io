import {
  COLLAPSE_KIND,
  applyCollapsePlan,
  buildCollapsePlan,
  findLogBracket,
  interpolateOpacityLogLog,
  makeLogEdges,
  parsePositiveList,
  parseStrictlyIncreasingEdges,
  requiredFieldsForCollapse,
} from './opacity_math.mjs';
import {
  isDrawablePoint,
  loadPlotPoints,
  makeSpectrumPlotPoints,
  preparePlotDomain,
  processSpectrumChunkBatches,
} from './plot_math.mjs';
import {
  temperatureFromEV,
  temperatureToEV,
  temperatureUnitLabel,
  spectralUnitLabel,
} from './unit_math.mjs';

const TABLE_CATALOG_URL = 'tables.json';

// GLOW spectrum plot repair v9
const GLOW_KELVIN_PER_EV = 11604.518121550082;
const GLOW_HZ_PER_EV = 2.417989242084918e14;

const PREVIEW_ROW_LIMIT = 300;
const MAX_OUTPUT_ROWS = 2_000_000;
const MAX_TEMPERATURE_VALUES = 256;
const MAX_DENSITY_VALUES = 256;
const MAX_SPECTRA = 4096;
const TRANSFER_WARNING_BYTES = 100 * 1024 * 1024;
const MAX_CONCURRENT_SPECTRUM_CHUNKS = 4;

const state = {
  tableCatalog: null,
  activeTable: null,
  dataRoot: null,
  manifest: null,
  axes: null,
  chunkCache: new Map(),
  collapsePlanCache: new Map(),
  plotManifest: null,
  plotCache: new Map(),
  lastPlot: null,
  plotTemperatureUnit: 'eV',
};

const elements = {
  loadingMessage: document.querySelector('#loading-message'),
  form: document.querySelector('#selection-form'),
  tableSelect: document.querySelector('#table-select'),
  tableDescription: document.querySelector('#table-description'),
  plotTableSelect: document.querySelector('#plot-table-select'),
  fieldSelect: document.querySelector('#field-select'),
  temperatureSelect: document.querySelector('#temperature-select'),
  temperatureList: document.querySelector('#temperature-list'),
  densityMinSelect: document.querySelector('#density-min-select'),
  densityMaxSelect: document.querySelector('#density-max-select'),
  densityList: document.querySelector('#density-list'),
  groupMinInput: document.querySelector('#group-min-input'),
  groupMaxInput: document.querySelector('#group-max-input'),
  energyEdgeList: document.querySelector('#energy-edge-list'),
  energyLogMin: document.querySelector('#energy-log-min'),
  energyLogMax: document.querySelector('#energy-log-max'),
  energyLogCount: document.querySelector('#energy-log-count'),
  temperatureNativePanel: document.querySelector('#temperature-native-panel'),
  temperatureCustomPanel: document.querySelector('#temperature-custom-panel'),
  densityNativePanel: document.querySelector('#density-native-panel'),
  densityCustomPanel: document.querySelector('#density-custom-panel'),
  energyNativePanel: document.querySelector('#energy-native-panel'),
  energyCustomPanel: document.querySelector('#energy-custom-panel'),
  energyLogPanel: document.querySelector('#energy-log-panel'),
  temperatureSummary: document.querySelector('#temperature-summary'),
  densitySummary: document.querySelector('#density-summary'),
  groupSummary: document.querySelector('#group-summary'),
  rowSummary: document.querySelector('#row-summary'),
  chunkSummary: document.querySelector('#chunk-summary'),
  transferSummary: document.querySelector('#transfer-summary'),
  formError: document.querySelector('#form-error'),
  previewButton: document.querySelector('#preview-button'),
  downloadButton: document.querySelector('#download-button'),
  previewSection: document.querySelector('#preview-section'),
  previewDescription: document.querySelector('#preview-description'),
  previewLimit: document.querySelector('#preview-limit'),
  previewBody: document.querySelector('#preview-body'),
  previewTemperatureHeading: document.querySelector('#preview-temperature-heading'),
  temperatureOutputUnitSelect: document.querySelector('#temperature-output-unit-select'),
  plotForm: document.querySelector('#plot-form'),
  plotFieldSelect: document.querySelector('#plot-field-select'),
  plotSweepSelect: document.querySelector('#plot-sweep-select'),
  plotTemperatureUnitSelect: document.querySelector('#plot-temperature-unit-select'),
  plotSpectralUnitSelect: document.querySelector('#plot-spectral-unit-select'),
  plotLogX: document.querySelector('#plot-log-x'),
  plotLogY: document.querySelector('#plot-log-y'),
  plotDensityMode: document.querySelector('#plot-density-mode'),
  plotDensitySelect: document.querySelector('#plot-density-select'),
  plotDensityCustom: document.querySelector('#plot-density-custom'),
  plotTemperatureMode: document.querySelector('#plot-temperature-mode'),
  plotTemperatureSelect: document.querySelector('#plot-temperature-select'),
  plotTemperatureCustom: document.querySelector('#plot-temperature-custom'),
  plotFixedDensityPanel: document.querySelector('#plot-fixed-density-panel'),
  plotFixedTemperaturePanel: document.querySelector('#plot-fixed-temperature-panel'),
  plotDensityNativeLabel: document.querySelector('#plot-density-native-label'),
  plotDensityCustomLabel: document.querySelector('#plot-density-custom-label'),
  plotTemperatureNativeLabel: document.querySelector('#plot-temperature-native-label'),
  plotTemperatureCustomLabel: document.querySelector('#plot-temperature-custom-label'),
  plotTemperatureNativeLabelText: document.querySelector('#plot-temperature-native-label-text'),
  plotTemperatureCustomLabelText: document.querySelector('#plot-temperature-custom-label-text'),
  plotSpectralAxisPanel: document.querySelector('#plot-spectral-axis-panel'),
  plotEnergyTreatment: document.querySelector('#plot-energy-treatment'),
  plotGroupPanel: document.querySelector('#plot-group-panel'),
  plotIntegratedHelp: document.querySelector('#plot-integrated-help'),
  plotGroupInput: document.querySelector('#plot-group-input'),
  plotGroupRange: document.querySelector('#plot-group-range'),
  plotPointSummary: document.querySelector('#plot-point-summary'),
  plotTransferSummary: document.querySelector('#plot-transfer-summary'),
  plotEnergySummary: document.querySelector('#plot-energy-summary'),
  plotEnergySummaryLabel: document.querySelector('#plot-energy-summary-label'),
  plotFormError: document.querySelector('#plot-form-error'),
  plotButton: document.querySelector('#plot-button'),
  plotDownloadButton: document.querySelector('#plot-download-button'),
  plotSection: document.querySelector('#plot-section'),
  plotDescription: document.querySelector('#plot-description'),
  plotWarningBadge: document.querySelector('#plot-warning-badge'),
  plotChart: document.querySelector('#plot-chart'),
  plotZeroNote: document.querySelector('#plot-zero-note'),
};

function selectedMode(name) {
  return document.querySelector(`input[name="${name}"]:checked`).value;
}

function formatScientific(value, digits = 6) {
  if (!Number.isFinite(value)) return 'nan';
  if (value === 0) return '0';
  return value.toExponential(digits);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(2)} MiB`;
}

function glowPlotTemperatureUnit() {
  return elements.plotTemperatureUnitSelect.value === 'K' ? 'K' : 'eV';
}

function glowTemperatureFromEv(valueEv, unit) {
  return unit === 'K' ? valueEv * GLOW_KELVIN_PER_EV : valueEv;
}

function glowTemperatureToEv(value, unit) {
  return unit === 'K' ? value / GLOW_KELVIN_PER_EV : value;
}

function glowFormatPlotTemperature(valueEv, unit, digits = 6) {
  return `${formatScientific(glowTemperatureFromEv(valueEv, unit), digits)} ${unit}`;
}

function glowRefreshPlotTemperatureControls() {
  if (!state.axes) return;
  const unit = glowPlotTemperatureUnit();
  elements.plotTemperatureNativeLabelText.textContent = `Fixed temperature [${unit}]`;
  elements.plotTemperatureCustomLabelText.textContent = `Fixed temperature [${unit}]`;
  state.axes.temp_eV.forEach((temperatureEv, index) => {
    const option = elements.plotTemperatureSelect.options[index];
    if (option) {
      option.textContent = `${index}: ${formatScientific(glowTemperatureFromEv(temperatureEv, unit), 5)} ${unit}`;
    }
  });
}

function glowHandlePlotTemperatureUnitChange() {
  const previousUnit = state.plotTemperatureUnit;
  const nextUnit = glowPlotTemperatureUnit();
  const displayed = Number(elements.plotTemperatureCustom.value);
  if (Number.isFinite(displayed) && displayed > 0 && previousUnit !== nextUnit) {
    const valueEv = glowTemperatureToEv(displayed, previousUnit);
    elements.plotTemperatureCustom.value = String(glowTemperatureFromEv(valueEv, nextUnit));
  }
  state.plotTemperatureUnit = nextUnit;
  glowRefreshPlotTemperatureControls();
  updatePlotPanels();
}

function showError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = false;
}

function clearError() {
  elements.formError.textContent = '';
  elements.formError.hidden = true;
}

function setBusy(isBusy, message = '') {
  elements.tableSelect.disabled = isBusy;
  elements.plotTableSelect.disabled = isBusy;
  elements.previewButton.disabled = isBusy;
  elements.downloadButton.disabled = isBusy;
  elements.plotButton.disabled = isBusy;
  elements.plotDownloadButton.disabled = isBusy || !state.lastPlot;
  if (message) {
    elements.loadingMessage.textContent = message;
    elements.loadingMessage.hidden = false;
  } else if (state.manifest) {
    elements.loadingMessage.hidden = true;
  }
}

function datasetUrl(relativePath) {
  if (!state.dataRoot) throw new Error('No opacity table is selected.');
  return `${state.dataRoot}/${relativePath}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
  return response.json();
}

async function decompressGzip(response) {
  if (!('DecompressionStream' in window)) {
    throw new Error('This browser does not support DecompressionStream(gzip).');
  }
  const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

function decodeOpacityArray(buffer, metadata) {
  const dtype = metadata.dtype || state.manifest?.storage?.dtype;
  const ArrayType = dtype === 'float32-le'
    ? Float32Array
    : dtype === 'float64-le'
      ? Float64Array
      : dtype === 'uint8-le'
        ? Uint8Array
        : null;
  if (!ArrayType) throw new Error(`Unsupported opacity data type: ${dtype}.`);

  const expectedValues = metadata.shape.reduce((product, size) => product * size, 1);
  const expectedBytes = expectedValues * ArrayType.BYTES_PER_ELEMENT;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(`${metadata.file} has ${buffer.byteLength} bytes; expected ${expectedBytes}.`);
  }
  return new ArrayType(buffer);
}

async function fetchChunk(chunkInfo) {
  const url = datasetUrl(chunkInfo.file);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
  const buffer = await decompressGzip(response);
  return decodeOpacityArray(buffer, chunkInfo);
}

async function loadChunk(chunkInfo) {
  if (state.chunkCache.has(chunkInfo.file)) return state.chunkCache.get(chunkInfo.file);

  const promise = fetchChunk(chunkInfo);

  state.chunkCache.set(chunkInfo.file, promise);
  try {
    return await promise;
  } catch (error) {
    state.chunkCache.delete(chunkInfo.file);
    throw error;
  }
}


async function loadPlotField(field) {
  if (!state.plotManifest || !(field in state.plotManifest.fields)) {
    throw new Error(`No integrated plot data are available for ${field}.`);
  }
  const info = state.plotManifest.fields[field];
  if (state.plotCache.has(info.file)) return state.plotCache.get(info.file);

  const promise = (async () => {
    const url = datasetUrl(`plot/${info.file}`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
    const buffer = await decompressGzip(response);
    return decodeOpacityArray(buffer, info);
  })();

  state.plotCache.set(info.file, promise);
  try {
    return await promise;
  } catch (error) {
    state.plotCache.delete(info.file);
    throw error;
  }
}

function findTemperaturePart(globalTemperatureIndex) {
  const part = state.manifest.parts.find((candidate) => (
    globalTemperatureIndex >= candidate.temperature_global_start &&
    globalTemperatureIndex < candidate.temperature_global_stop
  ));
  if (!part) throw new Error(`No temperature part contains global index ${globalTemperatureIndex}.`);
  return part;
}

function findChunkInfo(part, field, groupIndex) {
  const chunk = part.chunks.find((candidate) => (
    candidate.field === field &&
    groupIndex >= candidate.group_start &&
    groupIndex < candidate.group_stop
  ));
  if (!chunk) {
    throw new Error(`No ${field} chunk in part ${part.part_index} contains group ${groupIndex}.`);
  }
  return chunk;
}

// Quality flags: per-group bits written by the table generator.  Each field reads one flag
// array (manifest.quality_flags.fields[field]) restricted to the bits that apply to it.
function fieldQualityInfo(field) {
  return state.manifest?.quality_flags?.fields?.[field] ?? null;
}

function findFlagChunkInfo(part, arrayName, groupIndex) {
  return (part.flag_chunks ?? []).find((candidate) => (
    candidate.flags === arrayName &&
    groupIndex >= candidate.group_start &&
    groupIndex < candidate.group_stop
  )) ?? null;
}

function readNativeQuality(loadedChunks, field, groupIndex, globalTemperatureIndex, densityIndex) {
  const info = fieldQualityInfo(field);
  if (!info) return 0;
  const part = findTemperaturePart(globalTemperatureIndex);
  const chunkInfo = findFlagChunkInfo(part, info.array, groupIndex);
  if (!chunkInfo) throw new Error(`No ${info.array} chunk in part ${part.part_index} contains group ${groupIndex}.`);
  const values = loadedChunks.get(chunkInfo.file);
  if (!values) throw new Error(`Required chunk ${chunkInfo.file} was not loaded.`);
  const localGroup = groupIndex - chunkInfo.group_start;
  const localTemperature = globalTemperatureIndex - part.temperature_global_start;
  const flatIndex = (
    (localGroup * chunkInfo.shape[1] + densityIndex) * chunkInfo.shape[2] + localTemperature
  );
  return values[flatIndex] & info.mask;
}

function stencilQuality(readQuality, temperatureBracket, densityBracket) {
  const temperatures = temperatureBracket.lowerIndex === temperatureBracket.upperIndex
    ? [temperatureBracket.lowerIndex]
    : [temperatureBracket.lowerIndex, temperatureBracket.upperIndex];
  const densities = densityBracket.lowerIndex === densityBracket.upperIndex
    ? [densityBracket.lowerIndex]
    : [densityBracket.lowerIndex, densityBracket.upperIndex];
  let bits = 0;
  for (const temperatureIndex of temperatures) {
    for (const densityIndex of densities) bits |= readQuality(temperatureIndex, densityIndex);
  }
  return bits;
}

function describeQuality(bits) {
  if (!bits) return '';
  const names = Object.entries(state.manifest?.quality_flags?.bits ?? {})
    .filter(([, bit]) => bits & bit)
    .map(([name]) => name);
  return names.join('+');
}

function updateModePanels() {
  const temperatureMode = selectedMode('temperature-mode');
  const densityMode = selectedMode('density-mode');
  const energyMode = selectedMode('energy-mode');

  elements.temperatureNativePanel.hidden = temperatureMode !== 'native';
  elements.temperatureCustomPanel.hidden = temperatureMode !== 'custom';
  elements.densityNativePanel.hidden = densityMode !== 'native';
  elements.densityCustomPanel.hidden = densityMode !== 'custom';
  elements.energyNativePanel.hidden = energyMode !== 'native';
  elements.energyCustomPanel.hidden = energyMode !== 'custom';
  elements.energyLogPanel.hidden = energyMode !== 'log';
  updateSummary();
}

function checkInsideAxis(values, axis, label) {
  const minimum = axis[0];
  const maximum = axis[axis.length - 1];
  for (const value of values) {
    if (value < minimum || value > maximum) {
      throw new Error(`${label} ${value} is outside [${minimum}, ${maximum}]. Extrapolation is disabled.`);
    }
  }
}

function getTemperatureValues() {
  if (selectedMode('temperature-mode') === 'native') {
    const index = Number.parseInt(elements.temperatureSelect.value, 10);
    return [state.axes.temp_eV[index]];
  }
  const values = parsePositiveList(elements.temperatureList.value, 'Temperature');
  if (values.length > MAX_TEMPERATURE_VALUES) {
    throw new Error(`At most ${MAX_TEMPERATURE_VALUES} temperatures may be requested.`);
  }
  checkInsideAxis(values, state.axes.temp_eV, 'Temperature');
  return values;
}

function getDensityValues() {
  if (selectedMode('density-mode') === 'native') {
    const minimumIndex = Number.parseInt(elements.densityMinSelect.value, 10);
    const maximumIndex = Number.parseInt(elements.densityMaxSelect.value, 10);
    if (minimumIndex > maximumIndex) {
      throw new Error('Minimum native density must not exceed maximum native density.');
    }
    return state.axes.rho_gcc.slice(minimumIndex, maximumIndex + 1);
  }
  const values = parsePositiveList(elements.densityList.value, 'Density');
  if (values.length > MAX_DENSITY_VALUES) {
    throw new Error(`At most ${MAX_DENSITY_VALUES} densities may be requested.`);
  }
  checkInsideAxis(values, state.axes.rho_gcc, 'Density');
  return values;
}

function energyRangeToNativeGroups(outputEdges) {
  const nativeEdges = state.axes.hnu_ev_edges;
  const minimum = outputEdges[0];
  const maximum = outputEdges[outputEdges.length - 1];
  if (minimum < nativeEdges[0] || maximum > nativeEdges[nativeEdges.length - 1]) {
    throw new Error(
      `Energy edges must remain within [${nativeEdges[0]}, ${nativeEdges[nativeEdges.length - 1]}] eV.`,
    );
  }

  let start = 0;
  while (start < nativeEdges.length - 1 && nativeEdges[start + 1] <= minimum) start += 1;
  let stopExclusive = start;
  while (stopExclusive < nativeEdges.length - 1 && nativeEdges[stopExclusive] < maximum) {
    stopExclusive += 1;
  }
  return { start, stopExclusive };
}

function getEnergyDefinition() {
  const mode = selectedMode('energy-mode');
  const nativeGroupCount = state.manifest.dimensions.groups;

  if (mode === 'native') {
    const start = Number.parseInt(elements.groupMinInput.value, 10);
    const last = Number.parseInt(elements.groupMaxInput.value, 10);
    if (!Number.isInteger(start) || !Number.isInteger(last)) {
      throw new Error('Native group indices must be integers.');
    }
    if (start < 0 || last >= nativeGroupCount || start > last) {
      throw new Error(`Native groups must satisfy 0 <= first <= last <= ${nativeGroupCount - 1}.`);
    }
    return {
      mode,
      outputEdges: state.axes.hnu_ev_edges.slice(start, last + 2),
      nativeGroupStart: start,
      nativeGroupStopExclusive: last + 1,
      outputBinCount: last - start + 1,
    };
  }

  let outputEdges;
  if (mode === 'custom') {
    outputEdges = parseStrictlyIncreasingEdges(elements.energyEdgeList.value);
    if (outputEdges.length - 1 > 1024) {
      throw new Error('At most 1024 custom output groups may be requested.');
    }
  } else {
    const minimum = Number(elements.energyLogMin.value);
    const maximum = Number(elements.energyLogMax.value);
    const count = Number.parseInt(elements.energyLogCount.value, 10);
    outputEdges = makeLogEdges(minimum, maximum, count);
  }

  const range = energyRangeToNativeGroups(outputEdges);
  return {
    mode,
    outputEdges,
    nativeGroupStart: range.start,
    nativeGroupStopExclusive: range.stopExclusive,
    outputBinCount: outputEdges.length - 1,
  };
}

function uniqueRequiredChunks(field, temperatureBrackets, energy) {
  const requiredFields = new Set(requiredFieldsForCollapse(field));
  const temperatureIndices = new Set();
  for (const bracket of temperatureBrackets) {
    temperatureIndices.add(bracket.lowerIndex);
    temperatureIndices.add(bracket.upperIndex);
  }

  const parts = new Map();
  for (const temperatureIndex of temperatureIndices) {
    const part = findTemperaturePart(temperatureIndex);
    parts.set(part.part_index, part);
  }

  const flagArrays = new Set(
    [...requiredFields].map((name) => fieldQualityInfo(name)?.array).filter(Boolean),
  );
  const chunks = new Map();
  for (const part of parts.values()) {
    for (const chunk of [...part.chunks, ...(part.flag_chunks ?? [])]) {
      const wanted = chunk.flags ? flagArrays.has(chunk.flags) : requiredFields.has(chunk.field);
      if (
        wanted &&
        chunk.group_stop > energy.nativeGroupStart &&
        chunk.group_start < energy.nativeGroupStopExclusive
      ) {
        chunks.set(chunk.file, chunk);
      }
    }
  }
  return [...chunks.values()];
}

function getQuery() {
  const field = elements.fieldSelect.value;
  if (!(field in state.manifest.field_metadata)) throw new Error(`Unknown opacity field: ${field}`);

  const temperatures = getTemperatureValues();
  const densities = getDensityValues();
  const energy = getEnergyDefinition();
  const spectra = temperatures.length * densities.length;
  const rowCount = spectra * energy.outputBinCount;

  if (spectra > MAX_SPECTRA) {
    throw new Error(`The query contains ${spectra.toLocaleString()} spectra; the limit is ${MAX_SPECTRA}.`);
  }
  if (rowCount > MAX_OUTPUT_ROWS) {
    throw new Error(
      `The query contains ${rowCount.toLocaleString()} output rows; the limit is ${MAX_OUTPUT_ROWS.toLocaleString()}.`,
    );
  }

  const temperatureBrackets = temperatures.map((value) => (
    findLogBracket(state.axes.temp_eV, value, 'Temperature')
  ));
  const densityBrackets = densities.map((value) => (
    findLogBracket(state.axes.rho_gcc, value, 'Density')
  ));
  const chunks = uniqueRequiredChunks(field, temperatureBrackets, energy);
  const compressedBytes = chunks.reduce((sum, chunk) => sum + chunk.compressed_bytes, 0);

  return {
    field,
    temperatureOutputUnit: elements.temperatureOutputUnitSelect.value,
    temperatures,
    densities,
    temperatureBrackets,
    densityBrackets,
    energy,
    spectra,
    rowCount,
    chunks,
    compressedBytes,
  };
}

function updateSummary() {
  if (!state.manifest) return;
  clearError();
  try {
    const query = getQuery();
    elements.temperatureSummary.textContent = query.temperatures.length.toLocaleString();
    elements.densitySummary.textContent = query.densities.length.toLocaleString();
    elements.groupSummary.textContent = query.energy.outputBinCount.toLocaleString();
    elements.rowSummary.textContent = query.rowCount.toLocaleString();
    elements.chunkSummary.textContent = query.chunks.length.toLocaleString();
    elements.transferSummary.textContent = formatBytes(query.compressedBytes) + (
      query.compressedBytes > TRANSFER_WARNING_BYTES ? ' — large transfer' : ''
    );
  } catch (error) {
    elements.temperatureSummary.textContent = '—';
    elements.densitySummary.textContent = '—';
    elements.groupSummary.textContent = '—';
    elements.rowSummary.textContent = '—';
    elements.chunkSummary.textContent = '—';
    elements.transferSummary.textContent = '—';
    showError(error.message);
  }
}

async function loadQueryChunks(query) {
  const entries = await Promise.all(query.chunks.map(async (chunkInfo) => (
    [chunkInfo.file, await loadChunk(chunkInfo)]
  )));
  return new Map(entries);
}

function readNativeValue(
  query, loadedChunks, field, groupIndex, globalTemperatureIndex, densityIndex,
) {
  const part = findTemperaturePart(globalTemperatureIndex);
  const chunkInfo = findChunkInfo(part, field, groupIndex);
  const values = loadedChunks.get(chunkInfo.file);
  if (!values) throw new Error(`Required chunk ${chunkInfo.file} was not loaded.`);

  const localGroup = groupIndex - chunkInfo.group_start;
  const localTemperature = globalTemperatureIndex - part.temperature_global_start;
  const densityCount = chunkInfo.shape[1];
  const temperatureCount = chunkInfo.shape[2];
  const flatIndex = (
    (localGroup * densityCount + densityIndex) * temperatureCount + localTemperature
  );
  return values[flatIndex];
}

function buildInterpolatedNativeSpectrum(
  query, loadedChunks, temperatureIndex, densityIndex, field = query.field,
) {
  const tBracket = query.temperatureBrackets[temperatureIndex];
  const rhoBracket = query.densityBrackets[densityIndex];
  const start = query.energy.nativeGroupStart;
  const stop = query.energy.nativeGroupStopExclusive;
  const values = new Float64Array(stop - start);
  const statuses = new Array(stop - start);
  const quality = new Uint8Array(stop - start);

  for (let group = start; group < stop; group += 1) {
    quality[group - start] = stencilQuality(
      (globalTIndex, rhoIndex) => readNativeQuality(loadedChunks, field, group, globalTIndex, rhoIndex),
      tBracket,
      rhoBracket,
    );
    const result = interpolateOpacityLogLog(
      (globalTIndex, rhoIndex) => readNativeValue(
        query,
        loadedChunks,
        field,
        group,
        globalTIndex,
        rhoIndex,
      ),
      tBracket,
      rhoBracket,
    );
    values[group - start] = result.value;
    statuses[group - start] = result.status;
  }
  return { values, statuses, quality };
}

function getCollapsePlan(query, temperature) {
  const kind = COLLAPSE_KIND[query.field];
  if (!kind) throw new Error(`No collapse convention is defined for ${query.field}.`);
  const start = query.energy.nativeGroupStart;
  const stop = query.energy.nativeGroupStopExclusive;
  const nativeEdges = state.axes.hnu_ev_edges.slice(start, stop + 1);
  const edgeKey = query.energy.outputEdges.map((value) => value.toPrecision(17)).join(',');
  const key = `${kind}|${temperature.toPrecision(17)}|${start}|${stop}|${edgeKey}`;
  if (!state.collapsePlanCache.has(key)) {
    state.collapsePlanCache.set(
      key,
      buildCollapsePlan(kind, nativeEdges, query.energy.outputEdges, temperature),
    );
  }
  return state.collapsePlanCache.get(key);
}

function outputForSpectrum(query, nativeSpectrum, totalSpectrum, temperature) {
  if (query.energy.mode === 'native') {
    return Array.from(nativeSpectrum.values, (value, index) => ({
      value,
      status: nativeSpectrum.statuses[index],
      quality: nativeSpectrum.quality[index],
    }));
  }
  const options = COLLAPSE_KIND[query.field] === 'rosseland_absorption'
    ? {
      totalOpacity: totalSpectrum.values,
      totalStatus: totalSpectrum.statuses,
    }
    : {};
  const plan = getCollapsePlan(query, temperature);
  const results = applyCollapsePlan(plan, nativeSpectrum.values, nativeSpectrum.statuses, options);
  return results.map((result, index) => ({
    ...result,
    quality: plan.bins[index].terms.reduce((bits, term) => bits | nativeSpectrum.quality[term.nativeIndex], 0),
  }));
}

function forEachResult(query, loadedChunks, callback, limit = Number.POSITIVE_INFINITY) {
  let emitted = 0;
  let warningCount = 0;
  let flaggedCount = 0;

  for (let ti = 0; ti < query.temperatures.length; ti += 1) {
    const temperature = query.temperatures[ti];
    for (let ri = 0; ri < query.densities.length; ri += 1) {
      const density = query.densities[ri];
      const nativeSpectrum = buildInterpolatedNativeSpectrum(
        query, loadedChunks, ti, ri, query.field,
      );
      const totalSpectrum = COLLAPSE_KIND[query.field] === 'rosseland_absorption'
        ? buildInterpolatedNativeSpectrum(query, loadedChunks, ti, ri, 'krosseland')
        : null;
      const output = outputForSpectrum(query, nativeSpectrum, totalSpectrum, temperature);

      for (let group = 0; group < output.length; group += 1) {
        const item = output[group];
        if (item.status !== 'ok' && item.status !== 'ok_zero') warningCount += 1;
        if (item.quality) flaggedCount += 1;
        callback({
          temperature,
          density,
          outputGroup: group,
          energyLow: query.energy.outputEdges[group],
          energyHigh: query.energy.outputEdges[group + 1],
          opacity: item.value,
          status: item.status,
          quality: describeQuality(item.quality),
        });
        emitted += 1;
        if (emitted >= limit) {
          return { emitted, warningCount, flaggedCount, truncated: emitted < query.rowCount };
        }
      }
    }
  }
  return { emitted, warningCount, flaggedCount, truncated: false };
}

function renderPreview(rows, query, warningCount, flaggedCount = 0) {
  elements.previewBody.replaceChildren();
  const fragment = document.createDocumentFragment();
  for (const row of rows) {
    const tr = document.createElement('tr');
    const values = [
      formatScientific(temperatureFromEV(row.temperature, query.temperatureOutputUnit)),
      formatScientific(row.density),
      String(row.outputGroup),
      formatScientific(row.energyLow),
      formatScientific(row.energyHigh),
      formatScientific(row.opacity),
      row.status,
      row.quality || '—',
    ];
    for (const value of values) {
      const td = document.createElement('td');
      td.textContent = value;
      tr.append(td);
    }
    fragment.append(tr);
  }
  elements.previewBody.append(fragment);
  const label = state.manifest.field_metadata[query.field].label;
  elements.previewTemperatureHeading.textContent = `Temperature [${temperatureUnitLabel(query.temperatureOutputUnit)}]`;
  elements.previewDescription.textContent = (
    `${label}: ${query.rowCount.toLocaleString()} values; ` +
    `${warningCount.toLocaleString()} warnings and ${flaggedCount.toLocaleString()} quality flags in the displayed rows.`
  );
  elements.previewLimit.textContent = rows.length < query.rowCount
    ? `Showing ${rows.length.toLocaleString()} of ${query.rowCount.toLocaleString()}`
    : `${rows.length.toLocaleString()} rows`;
  elements.previewSection.hidden = false;
  elements.previewSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function previewSelection() {
  clearError();
  try {
    const query = getQuery();
    setBusy(true, `Loading ${query.chunks.length} compressed chunks…`);
    const loadedChunks = await loadQueryChunks(query);
    const rows = [];
    const result = forEachResult(
      query,
      loadedChunks,
      (row) => rows.push(row),
      PREVIEW_ROW_LIMIT,
    );
    renderPreview(rows, query, result.warningCount, result.flaggedCount);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

const QUALITY_HEADER_LINES = [
  '# quality: flags of the 1024-group values used for each row (none if unflagged)',
  '#   lines_aliased: STAR\'s frequency grid holds a line strength that differs from STAR\'s line Planck sum by >10%;',
  '#     line-dominated groups were rescaled to that sum, but which group each line falls in is uncertain',
  '#   lines_aliased_x2: as lines_aliased, with the difference exceeding a factor of 2',
  '#   lines_unresolved: the grid holds <1/10 or >10x the line strength; groups carry the continuum only and the',
  '#     line contribution is included in the grey Planck mean but not in any group',
  '#   rosseland_lines_rescaled: STAR\'s Rosseland mean used point-sampled lines that are not stored; the stored',
  '#     bin-averaged lines were scaled to reproduce STAR\'s grey Rosseland mean, so the line blocking per group is approximate',
  '#   sub_cell: the group is narrower than STAR\'s frequency grid spacing (value interpolated)',
  '#   grey_rescaled: all groups of this cell were scaled by 2-30% so that they reproduce STAR\'s grey mean',
  '#     (STAR\'s quadrature and the exact group integrals differ on spiky spectra)',
];

function createTextHeader(query, warningCount) {
  const metadata = state.manifest.field_metadata[query.field];
  const energyOperation = query.energy.mode === 'native'
    ? 'native 1024-group values (no energy collapse)'
    : 'weighted group collapse';
  return [
    '# GLOW custom opacity query',
    `# field: ${query.field}`,
    `# field_label: ${metadata.label}`,
    `# opacity_units: ${metadata.units}`,
    '# source_energy_groups: 1024',
    '# temperature_interpolation: linear in log(T), log(kappa)',
    '# density_interpolation: linear in log(rho), log(kappa)',
    `# energy_operation: ${energyOperation}`,
    '# weight_evaluation: logarithmic quadrature with common-factor cancellation',
    '# planck_fields: kplanck',
    '# rosseland_total_field: krosseland',
    '# rosseland_harmonic_fields: krosseland',
    '# rosseland_absorption_collapse: sum(kappa_abs_i * W_i / kappa_total_i) / sum(W_i / kappa_total_i)',
    '# opacity_inside_native_group: piecewise constant for website re-collapse',
    '# partial_native_group_overlap: log-weight integral evaluated only over overlap',
    '# extrapolation: disabled',
    '# nonpositive_interpolation_policy: all-zero T-rho stencil returns zero; mixed zero/positive T-rho stencil returns nan',
    `# temperature_output_unit: ${temperatureUnitLabel(query.temperatureOutputUnit)}`,
    `# temperature_count: ${query.temperatures.length}`,
    `# density_count: ${query.densities.length}`,
    `# output_group_count: ${query.energy.outputBinCount}`,
    `# warning_count: ${warningCount}`,
    ...QUALITY_HEADER_LINES,
    '#',
    '# Columns:',
    `# temperature_${query.temperatureOutputUnit === 'K' ? 'K' : 'eV'} density_g_cm-3 output_group_index energy_low_eV energy_high_eV opacity_cm2_g-1 status quality`,
    '',
  ].join('\n');
}

async function downloadSelection() {
  clearError();
  try {
    const query = getQuery();
    setBusy(true, `Loading ${query.chunks.length} compressed chunks and building TXT…`);
    const loadedChunks = await loadQueryChunks(query);
    const batches = [];
    let lines = [];
    const batchSize = 20000;

    const result = forEachResult(query, loadedChunks, (row) => {
      lines.push([
        temperatureFromEV(row.temperature, query.temperatureOutputUnit).toExponential(16),
        row.density.toExponential(16),
        row.outputGroup,
        row.energyLow.toExponential(16),
        row.energyHigh.toExponential(16),
        Number.isFinite(row.opacity) ? row.opacity.toExponential(16) : 'nan',
        row.status,
        row.quality || 'none',
      ].join(' '));
      if (lines.length >= batchSize) {
        batches.push(`${lines.join('\n')}\n`);
        lines = [];
      }
    });
    if (lines.length) batches.push(`${lines.join('\n')}\n`);

    const blob = new Blob([createTextHeader(query, result.warningCount), ...batches], {
      type: 'text/plain;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = (
      `GLOW_${state.activeTable.id}_${query.field}_T${query.temperatures.length}_rho${query.densities.length}_` +
      `${query.energy.outputBinCount}groups_T${query.temperatureOutputUnit}.txt`
    );
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}


function showPlotError(message) {
  elements.plotFormError.textContent = message;
  elements.plotFormError.hidden = false;
}

function clearPlotError() {
  elements.plotFormError.textContent = '';
  elements.plotFormError.hidden = true;
}

function selectedPlotEnergyMode() {
  return document.querySelector('input[name="plot-energy-mode"]:checked').value;
}

function plotFixedDensity() {
  if (elements.plotDensityMode.value === 'native') {
    return state.axes.rho_gcc[Number.parseInt(elements.plotDensitySelect.value, 10)];
  }
  const value = Number(elements.plotDensityCustom.value);
  findLogBracket(state.axes.rho_gcc, value, 'Density');
  return value;
}

function plotFixedTemperature() {
  if (elements.plotTemperatureMode.value === 'native') {
    return state.axes.temp_eV[Number.parseInt(elements.plotTemperatureSelect.value, 10)];
  }
  const displayedValue = Number(elements.plotTemperatureCustom.value);
  const valueEv = glowTemperatureToEv(displayedValue, glowPlotTemperatureUnit());
  findLogBracket(state.axes.temp_eV, valueEv, 'Temperature');
  return valueEv;
}

function refreshPlotTemperatureLabels() {
  if (!state.axes) return;
  const unit = elements.plotTemperatureUnitSelect.value;
  const label = temperatureUnitLabel(unit);
  elements.plotTemperatureNativeLabelText.textContent = `Fixed temperature [${label}]`;
  elements.plotTemperatureCustomLabelText.textContent = `Fixed temperature [${label}]`;
  Array.from(elements.plotTemperatureSelect.options).forEach((option, index) => {
    const display = temperatureFromEV(state.axes.temp_eV[index], unit);
    option.textContent = `${index}: ${formatScientific(display, 5)} ${label}`;
  });
}

function updatePlotTemperatureUnit() {
  const previous = elements.plotTemperatureUnitSelect.dataset.previousUnit || 'eV';
  const next = elements.plotTemperatureUnitSelect.value;
  const current = Number(elements.plotTemperatureCustom.value);
  if (Number.isFinite(current) && current > 0 && previous !== next) {
    const valueEV = temperatureToEV(current, previous);
    elements.plotTemperatureCustom.value = String(temperatureFromEV(valueEV, next));
  }
  elements.plotTemperatureUnitSelect.dataset.previousUnit = next;
  refreshPlotTemperatureLabels();
  updatePlotPanels();
}

function invalidateLastPlot() {
  state.lastPlot = null;
  elements.plotDownloadButton.disabled = true;
}

function updatePlotPanels() {
  if (!state.manifest) return;
  const sweep = elements.plotSweepSelect.value;
  const spectrumMode = sweep === 'frequency';
  const energyMode = selectedPlotEnergyMode();

  elements.plotFixedDensityPanel.hidden = !(sweep === 'temperature' || spectrumMode);
  elements.plotFixedTemperaturePanel.hidden = !(sweep === 'density' || spectrumMode);
  elements.plotSpectralAxisPanel.hidden = !spectrumMode;
  elements.plotEnergyTreatment.hidden = spectrumMode;

  elements.plotDensityNativeLabel.hidden = elements.plotDensityMode.value !== 'native';
  elements.plotDensityCustomLabel.hidden = elements.plotDensityMode.value !== 'custom';
  elements.plotTemperatureNativeLabel.hidden = elements.plotTemperatureMode.value !== 'native';
  elements.plotTemperatureCustomLabel.hidden = elements.plotTemperatureMode.value !== 'custom';

  elements.plotGroupPanel.hidden = spectrumMode || energyMode !== 'group';
  elements.plotIntegratedHelp.hidden = spectrumMode || energyMode !== 'integrated';
  if (!spectrumMode && energyMode === 'integrated') {
    elements.plotIntegratedHelp.textContent = (
      'Uses the conservative full-range weighted mean for the selected field: '
      + 'Planck arithmetic for absorption and scattering, Rosseland harmonic for total opacity, '
      + 'or Rosseland transport-weighted absorption.'
    );
  }
  updatePlotSummary();
}

function groupEnergyLabel(group) {
  const low = state.axes.hnu_ev_edges[group];
  const high = state.axes.hnu_ev_edges[group + 1];
  return `${formatScientific(low, 4)} – ${formatScientific(high, 4)} eV`;
}

function makeSpecificGroupPlotQuery(field, sweep, fixedValue, group) {
  const temperatures = sweep === 'temperature' ? [...state.axes.temp_eV] : [fixedValue];
  const densities = sweep === 'density' ? [...state.axes.rho_gcc] : [fixedValue];
  const temperatureBrackets = temperatures.map((value) => (
    findLogBracket(state.axes.temp_eV, value, 'Temperature')
  ));
  const densityBrackets = densities.map((value) => (
    findLogBracket(state.axes.rho_gcc, value, 'Density')
  ));
  const energy = {
    mode: 'native',
    outputEdges: state.axes.hnu_ev_edges.slice(group, group + 2),
    nativeGroupStart: group,
    nativeGroupStopExclusive: group + 1,
    outputBinCount: 1,
  };
  const chunks = uniqueRequiredChunks(field, temperatureBrackets, energy);
  return {
    field,
    temperatures,
    densities,
    temperatureBrackets,
    densityBrackets,
    energy,
    spectra: temperatures.length * densities.length,
    rowCount: temperatures.length * densities.length,
    chunks,
    compressedBytes: chunks.reduce((sum, chunk) => sum + chunk.compressed_bytes, 0),
  };
}

function makeSpectrumPlotQuery(field, temperature, density) {
  const temperatures = [temperature];
  const densities = [density];
  const temperatureBrackets = [findLogBracket(state.axes.temp_eV, temperature, 'Temperature')];
  const densityBrackets = [findLogBracket(state.axes.rho_gcc, density, 'Density')];
  const groupCount = state.manifest.dimensions.groups;
  const energy = {
    mode: 'native',
    outputEdges: [...state.axes.hnu_ev_edges],
    nativeGroupStart: 0,
    nativeGroupStopExclusive: groupCount,
    outputBinCount: groupCount,
  };
  const chunks = uniqueRequiredChunks(field, temperatureBrackets, energy);
  return {
    field,
    temperatures,
    densities,
    temperatureBrackets,
    densityBrackets,
    energy,
    spectra: 1,
    rowCount: groupCount,
    chunks,
    compressedBytes: chunks.reduce((sum, chunk) => sum + chunk.compressed_bytes, 0),
  };
}

function getPlotDefinition() {
  const field = elements.plotFieldSelect.value;
  if (!(field in state.manifest.field_metadata)) {
    throw new Error(`Unknown opacity field: ${field}`);
  }

  const sweep = elements.plotSweepSelect.value;
  const temperatureUnit = glowPlotTemperatureUnit();

  if (sweep === 'frequency') {
    const temperature = plotFixedTemperature();
    const density = plotFixedDensity();
    const spectralUnit = elements.plotSpectralUnitSelect.value === 'Hz' ? 'Hz' : 'eV';
    const query = makeSpectrumPlotQuery(field, temperature, density);
    return {
      field,
      sweep,
      temperature,
      density,
      temperatureUnit,
      spectralUnit,
      energyMode: 'spectrum',
      pointCount: state.manifest.dimensions.groups,
      compressedBytes: query.compressedBytes,
      energyLabel: (
        `${state.manifest.dimensions.groups} native groups; `
        + `T = ${glowFormatPlotTemperature(temperature, temperatureUnit)}; `
        + `density = ${formatScientific(density)} g cm^-3`
      ),
      query,
    };
  }

  const fixedValue = sweep === 'temperature' ? plotFixedDensity() : plotFixedTemperature();
  const energyMode = selectedPlotEnergyMode();
  const pointCount = sweep === 'temperature'
    ? state.axes.temp_eV.length
    : state.axes.rho_gcc.length;

  if (energyMode === 'integrated') {
    const info = state.plotManifest.fields[field];
    if (!info) throw new Error(`Integrated plot data are missing for ${field}.`);
    return {
      field,
      sweep,
      fixedValue,
      temperatureUnit,
      energyMode,
      pointCount,
      compressedBytes: info.compressed_bytes,
      energyLabel: field === 'kplanck_scattering'
        ? 'Full published energy range -- Planck-weighted arithmetic mean'
        : 'Full published energy range -- weighted mean',
    };
  }

  const group = Number.parseInt(elements.plotGroupInput.value, 10);
  const groupCount = state.manifest.dimensions.groups;
  if (!Number.isInteger(group) || group < 0 || group >= groupCount) {
    throw new Error(`Native group must be an integer from 0 to ${groupCount - 1}.`);
  }
  const query = makeSpecificGroupPlotQuery(field, sweep, fixedValue, group);
  return {
    field,
    sweep,
    fixedValue,
    temperatureUnit,
    energyMode,
    group,
    pointCount,
    compressedBytes: query.compressedBytes,
    energyLabel: `Group ${group}: ${groupEnergyLabel(group)}`,
    query,
  };
}

function updatePlotSummary() {
  if (!state.plotManifest) return;
  clearPlotError();
  try {
    const definition = getPlotDefinition();
    elements.plotPointSummary.textContent = definition.pointCount.toLocaleString();
    elements.plotTransferSummary.textContent = formatBytes(definition.compressedBytes);
    elements.plotEnergySummaryLabel.textContent = definition.sweep === 'frequency' ? 'Spectrum' : 'Energy';
    elements.plotEnergySummary.textContent = definition.energyLabel;
    if (definition.sweep !== 'frequency') {
      const group = Math.max(
        0,
        Math.min(
          state.manifest.dimensions.groups - 1,
          Number.parseInt(elements.plotGroupInput.value, 10) || 0,
        ),
      );
      elements.plotGroupRange.textContent = groupEnergyLabel(group);
    }
  } catch (error) {
    elements.plotPointSummary.textContent = '--';
    elements.plotTransferSummary.textContent = '--';
    elements.plotEnergySummary.textContent = '--';
    showPlotError(error.message);
  }
}

async function loadPlotQuality(field) {
  const info = state.plotManifest?.quality?.[field];
  if (!info) return null;
  if (state.plotCache.has(info.file)) return state.plotCache.get(info.file);
  const promise = (async () => {
    const url = datasetUrl(`plot/${info.file}`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
    return decodeOpacityArray(await decompressGzip(response), info);
  })();
  state.plotCache.set(info.file, promise);
  try {
    return await promise;
  } catch (error) {
    state.plotCache.delete(info.file);
    throw error;
  }
}

function readGreyValue(values, rhoIndex, temperatureIndex) {
  const temperatureCount = state.axes.temp_eV.length;
  return values[rhoIndex * temperatureCount + temperatureIndex];
}

function integratedPlotPoints(definition, values, quality = null) {
  const greyQuality = (temperatureBracket, densityBracket) => (quality
    ? describeQuality(stencilQuality(
      (temperatureIndex, rhoIndex) => readGreyValue(quality, rhoIndex, temperatureIndex),
      temperatureBracket,
      densityBracket,
    ))
    : '');
  const points = [];
  if (definition.sweep === 'temperature') {
    const densityBracket = findLogBracket(
      state.axes.rho_gcc, definition.fixedValue, 'Density',
    );
    for (let ti = 0; ti < state.axes.temp_eV.length; ti += 1) {
      const temperatureBracket = {
        lowerIndex: ti,
        upperIndex: ti,
        fraction: 0,
        exact: true,
      };
      const result = interpolateOpacityLogLog(
        (temperatureIndex, rhoIndex) => readGreyValue(values, rhoIndex, temperatureIndex),
        temperatureBracket,
        densityBracket,
      );
      points.push({
        x: glowTemperatureFromEv(state.axes.temp_eV[ti], definition.temperatureUnit),
        y: result.value,
        status: result.status,
        quality: greyQuality(temperatureBracket, densityBracket),
      });
    }
  } else {
    const temperatureBracket = findLogBracket(
      state.axes.temp_eV, definition.fixedValue, 'Temperature',
    );
    for (let ri = 0; ri < state.axes.rho_gcc.length; ri += 1) {
      const densityBracket = {
        lowerIndex: ri,
        upperIndex: ri,
        fraction: 0,
        exact: true,
      };
      const result = interpolateOpacityLogLog(
        (temperatureIndex, rhoIndex) => readGreyValue(values, rhoIndex, temperatureIndex),
        temperatureBracket,
        densityBracket,
      );
      points.push({
        x: state.axes.rho_gcc[ri],
        y: result.value,
        status: result.status,
        quality: greyQuality(temperatureBracket, densityBracket),
      });
    }
  }
  return points;
}

async function specificGroupPlotPoints(definition) {
  const loadedChunks = await loadQueryChunks(definition.query);
  const points = [];
  forEachResult(definition.query, loadedChunks, (row) => {
    points.push({
      x: definition.sweep === 'temperature'
        ? glowTemperatureFromEv(row.temperature, definition.temperatureUnit)
        : row.density,
      y: row.opacity,
      status: row.status,
      quality: row.quality,
    });
  });
  return points;
}

async function spectrumPlotPoints(definition) {
  const query = definition.query;
  const start = query.energy.nativeGroupStart;
  const stop = query.energy.nativeGroupStopExclusive;
  const spectrum = {
    values: new Float64Array(stop - start),
    statuses: new Array(stop - start),
    quality: new Uint8Array(stop - start),
  };
  let loadedCompressedBytes = 0;
  let completedBatches = 0;

  await processSpectrumChunkBatches(
    query.chunks,
    fetchChunk,
    (chunkValues, _batchIndex, batchCount) => {
      const loadedChunks = new Map(chunkValues.map(({ chunk, values }) => (
        [chunk.file, values]
      )));
      const groupStart = Math.max(start, chunkValues[0].chunk.group_start);
      const groupStop = Math.min(stop, chunkValues[0].chunk.group_stop);
      const temperatureBracket = query.temperatureBrackets[0];
      const densityBracket = query.densityBrackets[0];

      for (let group = groupStart; group < groupStop; group += 1) {
        const result = interpolateOpacityLogLog(
          (temperatureIndex, densityIndex) => readNativeValue(
            query,
            loadedChunks,
            definition.field,
            group,
            temperatureIndex,
            densityIndex,
          ),
          temperatureBracket,
          densityBracket,
        );
        spectrum.values[group - start] = result.value;
        spectrum.statuses[group - start] = result.status;
        spectrum.quality[group - start] = stencilQuality(
          (temperatureIndex, densityIndex) => readNativeQuality(
            loadedChunks, definition.field, group, temperatureIndex, densityIndex,
          ),
          temperatureBracket,
          densityBracket,
        );
      }

      loadedCompressedBytes += chunkValues.reduce(
        (sum, { chunk }) => sum + chunk.compressed_bytes,
        0,
      );
      completedBatches += 1;
      elements.loadingMessage.textContent = [
        `Loading line-plot data: ${completedBatches}/${batchCount} energy blocks`,
        `(${formatBytes(loadedCompressedBytes)}/${formatBytes(definition.compressedBytes)})...`,
      ].join(' ');
    },
    MAX_CONCURRENT_SPECTRUM_CHUNKS,
  );

  return makeSpectrumPlotPoints(
    spectrum,
    state.axes.hnu_ev_edges,
    definition.spectralUnit,
    GLOW_HZ_PER_EV,
  ).map((point, index) => ({ ...point, quality: describeQuality(spectrum.quality[index]) }));
}

function svgElement(name, attributes = {}, text = '') {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  if (text) element.textContent = text;
  return element;
}

function logarithmicTicks(minimum, maximum, maximumTicks = 9) {
  const lowPower = Math.floor(Math.log10(minimum));
  const highPower = Math.ceil(Math.log10(maximum));
  const span = Math.max(1, highPower - lowPower);
  const step = Math.max(1, Math.ceil(span / maximumTicks));
  const ticks = [];
  for (let power = lowPower; power <= highPower; power += step) {
    const value = 10 ** power;
    if (value >= minimum / 1.000001 && value <= maximum * 1.000001) {
      ticks.push({ value, label: `10^${power}` });
    }
  }
  if (ticks.length === 0) ticks.push({ value: minimum, label: formatScientific(minimum, 2) });
  return ticks;
}

function linearTicks(minimum, maximum, count = 6) {
  if (minimum === maximum) return [{ value: minimum, label: formatScientific(minimum, 2) }];
  return Array.from({ length: count }, (_, index) => {
    const value = minimum + (index / (count - 1)) * (maximum - minimum);
    return { value, label: formatScientific(value, 2) };
  });
}

function renderLinePlot(definition, points) {
  const width = 920;
  const height = 540;
  const margin = { left: 92, right: 28, top: 28, bottom: 76 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const logX = elements.plotLogX.checked;
  const logY = elements.plotLogY.checked;
  const { drawable, xMin, xMax, yMin, yMax } = preparePlotDomain(points, logX, logY);

  const xProject = logX
    ? (value) => margin.left + ((Math.log(value) - Math.log(xMin)) / (Math.log(xMax) - Math.log(xMin))) * innerWidth
    : (value) => margin.left + ((value - xMin) / (xMax - xMin)) * innerWidth;
  const yProject = logY
    ? (value) => margin.top + innerHeight - ((Math.log(value) - Math.log(yMin)) / (Math.log(yMax) - Math.log(yMin))) * innerHeight
    : (value) => margin.top + innerHeight - ((value - yMin) / (yMax - yMin)) * innerHeight;

  const svg = svgElement('svg', {
    viewBox: `0 0 ${width} ${height}`,
    class: 'opacity-line-svg',
    preserveAspectRatio: 'xMidYMid meet',
  });
  svg.append(svgElement('rect', {
    x: margin.left,
    y: margin.top,
    width: innerWidth,
    height: innerHeight,
    class: 'plot-background',
  }));

  const xTicks = logX ? logarithmicTicks(xMin, xMax) : linearTicks(xMin, xMax);
  const yTicks = logY ? logarithmicTicks(yMin, yMax) : linearTicks(yMin, yMax);
  for (const tick of yTicks) {
    const y = yProject(tick.value);
    svg.append(svgElement('line', {
      x1: margin.left,
      x2: margin.left + innerWidth,
      y1: y,
      y2: y,
      class: 'plot-grid-line',
    }));
    svg.append(svgElement('text', {
      x: margin.left - 12,
      y: y + 5,
      class: 'plot-tick-label y-tick-label',
      'text-anchor': 'end',
    }, tick.label));
  }
  for (const tick of xTicks) {
    const x = xProject(tick.value);
    svg.append(svgElement('line', {
      x1: x,
      x2: x,
      y1: margin.top,
      y2: margin.top + innerHeight,
      class: 'plot-grid-line',
    }));
    svg.append(svgElement('text', {
      x,
      y: margin.top + innerHeight + 26,
      class: 'plot-tick-label',
      'text-anchor': 'middle',
    }, tick.label));
  }
  svg.append(svgElement('line', {
    x1: margin.left,
    x2: margin.left,
    y1: margin.top,
    y2: margin.top + innerHeight,
    class: 'plot-axis-line',
  }));
  svg.append(svgElement('line', {
    x1: margin.left,
    x2: margin.left + innerWidth,
    y1: margin.top + innerHeight,
    y2: margin.top + innerHeight,
    class: 'plot-axis-line',
  }));

  let pathData = '';
  let penDown = false;
  for (const point of points) {
    if (!isDrawablePoint(point, logX, logY)) {
      penDown = false;
      continue;
    }
    const x = xProject(point.x);
    const y = yProject(point.y);
    pathData += `${penDown ? ' L' : ' M'} ${x.toFixed(3)} ${y.toFixed(3)}`;
    penDown = true;
  }
  svg.append(svgElement('path', { d: pathData.trim(), class: 'plot-data-line' }));

  for (const point of drawable) {
    const circle = svgElement('circle', {
      cx: xProject(point.x),
      cy: yProject(point.y),
      r: definition.sweep === 'frequency' ? 1.8 : 2.6,
      class: point.quality ? 'plot-data-point plot-data-point-flagged' : 'plot-data-point',
    });
    const title = definition.sweep === 'frequency'
      ? `group=${point.group}; energy=[${point.energyLow.toExponential(8)}, ${point.energyHigh.toExponential(8)}] eV; x=${point.x.toExponential(8)}; opacity=${point.y.toExponential(8)}; status=${point.status}`
      : `x=${point.x.toExponential(8)}; opacity=${point.y.toExponential(8)}; status=${point.status}`;
    const qualityText = point.quality ? `; quality=${point.quality}` : '';
    circle.append(svgElement('title', {}, title + qualityText));
    svg.append(circle);
  }

  let xLabel;
  if (definition.sweep === 'temperature') {
    xLabel = `Temperature [${definition.temperatureUnit}]${logX ? ' -- log scale' : ' -- linear scale'}`;
  } else if (definition.sweep === 'density') {
    xLabel = `Density [g cm^-3]${logX ? ' -- log scale' : ' -- linear scale'}`;
  } else {
    xLabel = definition.spectralUnit === 'Hz'
      ? `Frequency [Hz]${logX ? ' -- log scale' : ' -- linear scale'}`
      : `Photon energy [eV]${logX ? ' -- log scale' : ' -- linear scale'}`;
  }
  svg.append(svgElement('text', {
    x: margin.left + innerWidth / 2,
    y: height - 20,
    class: 'plot-axis-label',
    'text-anchor': 'middle',
  }, xLabel));
  svg.append(svgElement('text', {
    x: 24,
    y: margin.top + innerHeight / 2,
    class: 'plot-axis-label',
    'text-anchor': 'middle',
    transform: `rotate(-90 24 ${margin.top + innerHeight / 2})`,
  }, `Opacity [cm^2 g^-1] -- ${logY ? 'log' : 'linear'} scale`));

  elements.plotChart.replaceChildren(svg);
  const metadata = state.manifest.field_metadata[definition.field];
  let fixedLabel;
  if (definition.sweep === 'temperature') {
    fixedLabel = `density = ${formatScientific(definition.fixedValue)} g cm^-3`;
  } else if (definition.sweep === 'density') {
    fixedLabel = `temperature = ${glowFormatPlotTemperature(definition.fixedValue, definition.temperatureUnit)}`;
  } else {
    fixedLabel = `temperature = ${glowFormatPlotTemperature(definition.temperature, definition.temperatureUnit)}; density = ${formatScientific(definition.density)} g cm^-3`;
  }
  elements.plotDescription.textContent = `${metadata.label}; ${fixedLabel}; ${definition.energyLabel}.`;

  const omitted = points.length - drawable.length;
  const warnings = points.filter((point) => point.status !== 'ok' && point.status !== 'ok_zero').length;
  const flagged = points.filter((point) => point.quality).length;
  const flaggedNote = flagged > 0
    ? ` ${flagged} points carry quality flags (hollow markers; hover for details, see Methods).`
    : '';
  elements.plotWarningBadge.textContent = `${drawable.length} plotted; ${omitted} omitted`;
  if (logY) {
    elements.plotZeroNote.textContent = omitted > 0
      ? `${omitted} zero, non-finite, negative, or invalid points cannot be represented on a logarithmic y-axis and are shown as gaps. ${warnings} points had interpolation warnings.${flaggedNote}`
      : `${warnings} points had interpolation warnings.${flaggedNote}`;
  } else {
    elements.plotZeroNote.textContent = omitted > 0
      ? `${omitted} non-finite, negative, or invalid points are shown as gaps. Zero opacities are displayed on the linear y-axis. ${warnings} points had interpolation warnings.${flaggedNote}`
      : `Zero opacities are displayed normally on the linear y-axis. ${warnings} points had interpolation warnings.${flaggedNote}`;
  }
  elements.plotSection.hidden = false;
  elements.plotSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function drawPlot() {
  clearPlotError();
  try {
    const definition = getPlotDefinition();
    setBusy(true, `Loading line-plot data (${formatBytes(definition.compressedBytes)})...`);
    const points = await loadPlotPoints(definition, {
      spectrum: spectrumPlotPoints,
      integrated: async (plotDefinition) => integratedPlotPoints(
        plotDefinition,
        await loadPlotField(plotDefinition.field),
        await loadPlotQuality(plotDefinition.field),
      ),
      specificGroup: specificGroupPlotPoints,
    });
    renderLinePlot(definition, points);
    state.lastPlot = { definition, points };
    elements.plotDownloadButton.disabled = false;
  } catch (error) {
    showPlotError(error.message);
  } finally {
    setBusy(false);
  }
}

function downloadPlotCsv() {
  if (!state.lastPlot) return;
  const { definition, points } = state.lastPlot;
  const xName = definition.sweep === 'temperature'
    ? (definition.temperatureUnit === 'K' ? 'temperature_K' : 'temperature_eV')
    : definition.sweep === 'density'
      ? 'density_g_cm-3'
      : (definition.spectralUnit === 'Hz' ? 'frequency_Hz' : 'photon_energy_eV');
  const spectrumMode = definition.sweep === 'frequency';
  const columnHeader = spectrumMode
    ? `${xName},group_index,energy_low_eV,energy_high_eV,opacity_cm2_g-1,status,quality`
    : `${xName},opacity_cm2_g-1,status,quality`;
  const lines = [
    '# GLOW line plot',
    `# field: ${definition.field}`,
    `# energy: ${definition.energyLabel}`,
    `# x_axis: ${xName}`,
    `# y_axis: opacity_cm2_g-1 (${elements.plotLogY.checked ? 'log display' : 'linear display'})`,
    ...QUALITY_HEADER_LINES,
    columnHeader,
  ];
  for (const point of points) {
    const values = spectrumMode
      ? [
        point.x.toExponential(16),
        point.group,
        point.energyLow.toExponential(16),
        point.energyHigh.toExponential(16),
        Number.isFinite(point.y) ? point.y.toExponential(16) : 'nan',
        point.status,
        point.quality || 'none',
      ]
      : [
        point.x.toExponential(16),
        Number.isFinite(point.y) ? point.y.toExponential(16) : 'nan',
        point.status,
        point.quality || 'none',
      ];
    lines.push(values.join(','));
  }
  const blob = new Blob([`${lines.join('\n')}\n`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `GLOW_${state.activeTable.id}_plot_${definition.field}_${definition.sweep}_${definition.energyMode}.csv`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function populateControls() {
  [
    elements.fieldSelect,
    elements.plotFieldSelect,
    elements.temperatureSelect,
    elements.densityMinSelect,
    elements.densityMaxSelect,
    elements.plotDensitySelect,
    elements.plotTemperatureSelect,
  ].forEach((select) => select.replaceChildren());

  for (const [field, metadata] of Object.entries(state.manifest.field_metadata)) {
    const option = document.createElement('option');
    option.value = field;
    option.textContent = metadata.label;
    elements.fieldSelect.append(option);
    elements.plotFieldSelect.append(option.cloneNode(true));
  }
  elements.fieldSelect.value = 'kplanck';
  elements.plotFieldSelect.value = 'kplanck';

  state.axes.temp_eV.forEach((temperature, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${index}: ${formatScientific(temperature, 5)} eV`;
    elements.temperatureSelect.append(option);
    const plotOption = option.cloneNode(true);
    plotOption.textContent = '';
    elements.plotTemperatureSelect.append(plotOption);
  });

  state.axes.rho_gcc.forEach((density, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${index}: ${formatScientific(density, 5)}`;
    elements.densityMinSelect.append(option);
    elements.densityMaxSelect.append(option.cloneNode(true));
    elements.plotDensitySelect.append(option.cloneNode(true));
  });
  elements.densityMinSelect.value = '0';
  elements.densityMaxSelect.value = String(Math.min(7, state.axes.rho_gcc.length - 1));

  const maximumGroup = state.manifest.dimensions.groups - 1;
  elements.groupMinInput.max = String(maximumGroup);
  elements.groupMaxInput.max = String(maximumGroup);
  elements.groupMaxInput.value = String(Math.min(127, maximumGroup));

  const nativeEdges = state.axes.hnu_ev_edges;
  elements.energyLogMin.value = String(nativeEdges[0]);
  elements.energyLogMax.value = String(nativeEdges[nativeEdges.length - 1]);

  elements.plotGroupInput.max = String(maximumGroup);
  elements.plotDensitySelect.value = String(Math.floor(state.axes.rho_gcc.length / 2));
  elements.plotTemperatureSelect.value = String(Math.floor(state.axes.temp_eV.length / 2));
  elements.plotDensityCustom.value = String(state.axes.rho_gcc[Math.floor(state.axes.rho_gcc.length / 2)]);
  state.plotTemperatureUnit = glowPlotTemperatureUnit();
  elements.plotTemperatureCustom.value = String(glowTemperatureFromEv(
    state.axes.temp_eV[Math.floor(state.axes.temp_eV.length / 2)],
    state.plotTemperatureUnit,
  ));
  glowRefreshPlotTemperatureControls();
  elements.plotTemperatureUnitSelect.dataset.previousUnit = elements.plotTemperatureUnitSelect.value;
  refreshPlotTemperatureLabels();
  elements.previewTemperatureHeading.textContent = 'Temperature [eV]';
}

function attachEvents() {
  elements.tableSelect.addEventListener('change', () => {
    selectTable(elements.tableSelect.value);
  });
  elements.plotTableSelect.addEventListener('change', () => {
    selectTable(elements.plotTableSelect.value);
  });

  document.querySelectorAll('input[name="temperature-mode"], input[name="density-mode"], input[name="energy-mode"]')
    .forEach((radio) => radio.addEventListener('change', updateModePanels));

  [
    elements.fieldSelect,
    elements.temperatureSelect,
    elements.temperatureList,
    elements.densityMinSelect,
    elements.densityMaxSelect,
    elements.densityList,
    elements.groupMinInput,
    elements.groupMaxInput,
    elements.energyEdgeList,
    elements.energyLogMin,
    elements.energyLogMax,
    elements.energyLogCount,
  ].forEach((control) => {
    control.addEventListener('input', updateSummary);
    control.addEventListener('change', updateSummary);
  });

  elements.previewButton.addEventListener('click', previewSelection);
  elements.downloadButton.addEventListener('click', downloadSelection);
  elements.temperatureOutputUnitSelect.addEventListener('change', () => {
    elements.previewTemperatureHeading.textContent = (
      `Temperature [${temperatureUnitLabel(elements.temperatureOutputUnitSelect.value)}]`
    );
    elements.previewSection.hidden = true;
  });

  [
    elements.plotFieldSelect,
    elements.plotSweepSelect,
    elements.plotSpectralUnitSelect,
    elements.plotLogX,
    elements.plotLogY,
    elements.plotDensityMode,
    elements.plotDensitySelect,
    elements.plotDensityCustom,
    elements.plotTemperatureMode,
    elements.plotTemperatureSelect,
    elements.plotTemperatureCustom,
    elements.plotGroupInput,
  ].forEach((control) => {
    control.addEventListener('input', () => {
      invalidateLastPlot();
      updatePlotPanels();
    });
    control.addEventListener('change', () => {
      invalidateLastPlot();
      updatePlotPanels();
    });
  });
  elements.plotTemperatureUnitSelect.addEventListener('change', () => {
    invalidateLastPlot();
    updatePlotTemperatureUnit();
  });
  elements.plotTemperatureUnitSelect.addEventListener(
    'change', glowHandlePlotTemperatureUnitChange,
  );
  document.querySelectorAll('input[name="plot-energy-mode"]')
    .forEach((radio) => radio.addEventListener('change', () => {
      invalidateLastPlot();
      updatePlotPanels();
    }));
  elements.plotButton.addEventListener('click', drawPlot);
  elements.plotDownloadButton.addEventListener('click', downloadPlotCsv);
}

function validateDataset(manifest, axes, plotManifest) {
  if (manifest.dimensions.groups !== 1024) {
    throw new Error(`Expected 1024 native groups, found ${manifest.dimensions.groups}.`);
  }
  if (manifest.storage.axis_order.join(',') !== 'group,rho,temp') {
    throw new Error('Unexpected browser-data axis order.');
  }
  if (axes.temp_eV.length !== manifest.dimensions.temperatures ||
      axes.rho_gcc.length !== manifest.dimensions.densities ||
      axes.hnu_ev_edges.length !== manifest.dimensions.groups + 1) {
    throw new Error('Axis lengths disagree with the manifest.');
  }
  if (plotManifest.dimensions.temperatures !== manifest.dimensions.temperatures ||
      plotManifest.dimensions.densities !== manifest.dimensions.densities ||
      plotManifest.axis_order.join(',') !== 'rho,temp') {
    throw new Error('Integrated plot data disagree with the main dataset axes.');
  }
}

function populateTableSelector() {
  const { tables } = state.tableCatalog;
  if (!Array.isArray(tables) || tables.length === 0) {
    throw new Error('The opacity-table catalog does not contain any tables.');
  }
  elements.tableSelect.replaceChildren();
  elements.plotTableSelect.replaceChildren();
  for (const table of tables) {
    if (!table.id || !table.data_root || !table.label) {
      throw new Error('The opacity-table catalog contains an incomplete entry.');
    }
    const option = document.createElement('option');
    option.value = table.id;
    option.textContent = table.label;
    elements.tableSelect.append(option);
    elements.plotTableSelect.append(option.cloneNode(true));
  }
}

async function selectTable(tableId) {
  const table = state.tableCatalog.tables.find((candidate) => candidate.id === tableId);
  if (!table) {
    throw new Error(`Unknown opacity table: ${tableId}.`);
  }

  setBusy(true, `Loading ${table.label}…`);
  elements.loadingMessage.classList.remove('error');
  let loadingError = null;
  try {
    const dataRoot = table.data_root.replace(/\/$/, '');
    const [manifest, axes, plotManifest] = await Promise.all([
      fetchJson(`${dataRoot}/manifest.json`),
      fetchJson(`${dataRoot}/axes.json`),
      fetchJson(`${dataRoot}/plot/manifest.json`),
    ]);
    validateDataset(manifest, axes, plotManifest);

    state.activeTable = table;
    state.dataRoot = dataRoot;
    state.manifest = manifest;
    state.axes = axes;
    state.plotManifest = plotManifest;
    state.chunkCache.clear();
    state.collapsePlanCache.clear();
    state.plotCache.clear();
    state.lastPlot = null;

    populateControls();
    elements.tableSelect.value = table.id;
    elements.plotTableSelect.value = table.id;
    elements.tableDescription.textContent = table.description || '';
    elements.previewSection.hidden = true;
    elements.plotSection.hidden = true;
    updateModePanels();
    updatePlotPanels();
  } catch (error) {
    loadingError = `Table loading failed: ${error.message}`;
    if (state.activeTable) {
      elements.tableSelect.value = state.activeTable.id;
      elements.plotTableSelect.value = state.activeTable.id;
    }
  } finally {
    setBusy(false);
    if (loadingError) {
      elements.loadingMessage.textContent = loadingError;
      elements.loadingMessage.classList.add('error');
      elements.loadingMessage.hidden = false;
    }
  }
}

async function initialize() {
  try {
    state.tableCatalog = await fetchJson(TABLE_CATALOG_URL);
    populateTableSelector();
    attachEvents();
    elements.form.hidden = false;
    elements.plotForm.hidden = false;
    const defaultTable = state.tableCatalog.default_table || state.tableCatalog.tables[0].id;
    await selectTable(defaultTable);
  } catch (error) {
    elements.loadingMessage.textContent = `Dataset initialization failed: ${error.message}`;
    elements.loadingMessage.classList.add('error');
  }
}

initialize();
