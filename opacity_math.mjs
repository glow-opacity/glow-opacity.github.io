export const COLLAPSE_KIND = Object.freeze({
  kplanck: 'planck',
  krosseland: 'rosseland_harmonic',
  krosseland_absorption: 'rosseland_absorption',
  kplanck_scattering: 'planck',
});

export function requiredFieldsForCollapse(field) {
  const kind = COLLAPSE_KIND[field];
  if (!kind) throw new Error(`Unknown opacity field: ${field}`);
  return kind === 'rosseland_absorption' ? [field, 'krosseland'] : [field];
}

const GL32_X = Object.freeze([
  0.0483076656877383162,
  0.144471961582796493,
  0.239287362252137075,
  0.331868602282127650,
  0.421351276130635345,
  0.506899908932229390,
  0.587715757240762329,
  0.663044266930215201,
  0.732182118740289680,
  0.794483795967942407,
  0.849367613732569970,
  0.896321155766052124,
  0.934906075937739689,
  0.964762255587506431,
  0.985611511545268335,
  0.997263861849481564,
]);

const GL32_W = Object.freeze([
  0.0965400885147278006,
  0.0956387200792748594,
  0.0938443990808045656,
  0.0911738786957638847,
  0.0876520930044038111,
  0.0833119242269467552,
  0.0781938957870703065,
  0.0723457941088485062,
  0.0658222227763618468,
  0.0586840934785355471,
  0.0509980592623761762,
  0.0428358980222266807,
  0.0342738629130214331,
  0.0253920653092620595,
  0.0162743947309056706,
  0.00701861000947009660,
]);

const EXACT_REL_TOL = 2e-13;
const ACTIVE_WEIGHT_EPS = 1e-15;

function assertFinitePositive(value, label) {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a finite positive number.`);
  }
}

export function parsePositiveList(text, label, { deduplicate = true } = {}) {
  const tokens = String(text)
    .trim()
    .split(/[\s,;]+/)
    .filter(Boolean);

  if (tokens.length === 0) {
    throw new Error(`${label} list is empty.`);
  }

  const values = tokens.map((token) => {
    const value = Number(token);
    assertFinitePositive(value, `${label} value "${token}"`);
    return value;
  });

  if (!deduplicate) {
    return values;
  }

  const seen = new Set();
  return values.filter((value) => {
    const key = value.toString();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function parseStrictlyIncreasingEdges(text, label = 'Energy edges') {
  const values = parsePositiveList(text, label, { deduplicate: false });
  if (values.length < 2) {
    throw new Error(`${label} must contain at least two values.`);
  }
  for (let i = 1; i < values.length; i += 1) {
    if (!(values[i] > values[i - 1])) {
      throw new Error(`${label} must be strictly increasing.`);
    }
  }
  return values;
}

export function makeLogEdges(minimum, maximum, numberOfBins) {
  assertFinitePositive(minimum, 'Minimum energy');
  assertFinitePositive(maximum, 'Maximum energy');
  if (!(maximum > minimum)) {
    throw new Error('Maximum energy must exceed minimum energy.');
  }
  if (!Number.isInteger(numberOfBins) || numberOfBins < 1 || numberOfBins > 1024) {
    throw new Error('Number of output groups must be an integer from 1 to 1024.');
  }

  const lo = Math.log(minimum);
  const hi = Math.log(maximum);
  return Array.from({ length: numberOfBins + 1 }, (_, index) => (
    Math.exp(lo + (index / numberOfBins) * (hi - lo))
  ));
}

function approximatelyEqual(a, b) {
  return Math.abs(a - b) <= EXACT_REL_TOL * Math.max(Math.abs(a), Math.abs(b));
}

export function findLogBracket(axis, requestedValue, label = 'Value') {
  assertFinitePositive(requestedValue, label);
  if (!Array.isArray(axis) && !(axis instanceof Float64Array)) {
    throw new Error(`${label} axis must be array-like.`);
  }
  if (axis.length < 1) {
    throw new Error(`${label} axis is empty.`);
  }

  const minimum = axis[0];
  const maximum = axis[axis.length - 1];
  if (requestedValue < minimum && !approximatelyEqual(requestedValue, minimum)) {
    throw new Error(`${label} ${requestedValue} is below the table minimum ${minimum}.`);
  }
  if (requestedValue > maximum && !approximatelyEqual(requestedValue, maximum)) {
    throw new Error(`${label} ${requestedValue} is above the table maximum ${maximum}.`);
  }
  if (approximatelyEqual(requestedValue, minimum)) {
    return { lowerIndex: 0, upperIndex: 0, fraction: 0, exact: true };
  }
  if (approximatelyEqual(requestedValue, maximum)) {
    const index = axis.length - 1;
    return { lowerIndex: index, upperIndex: index, fraction: 0, exact: true };
  }

  let low = 0;
  let high = axis.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (axis[middle] <= requestedValue) low = middle;
    else high = middle;
  }

  if (approximatelyEqual(requestedValue, axis[low])) {
    return { lowerIndex: low, upperIndex: low, fraction: 0, exact: true };
  }
  if (approximatelyEqual(requestedValue, axis[high])) {
    return { lowerIndex: high, upperIndex: high, fraction: 0, exact: true };
  }

  const fraction = (
    (Math.log(requestedValue) - Math.log(axis[low])) /
    (Math.log(axis[high]) - Math.log(axis[low]))
  );
  return { lowerIndex: low, upperIndex: high, fraction, exact: false };
}

export function interpolateOpacityLogLog(getValue, temperatureBracket, densityBracket) {
  const tTerms = temperatureBracket.lowerIndex === temperatureBracket.upperIndex
    ? [{ index: temperatureBracket.lowerIndex, weight: 1 }]
    : [
      { index: temperatureBracket.lowerIndex, weight: 1 - temperatureBracket.fraction },
      { index: temperatureBracket.upperIndex, weight: temperatureBracket.fraction },
    ];
  const rhoTerms = densityBracket.lowerIndex === densityBracket.upperIndex
    ? [{ index: densityBracket.lowerIndex, weight: 1 }]
    : [
      { index: densityBracket.lowerIndex, weight: 1 - densityBracket.fraction },
      { index: densityBracket.upperIndex, weight: densityBracket.fraction },
    ];

  const active = [];
  for (const t of tTerms) {
    for (const rho of rhoTerms) {
      const weight = t.weight * rho.weight;
      if (weight <= ACTIVE_WEIGHT_EPS) continue;
      const value = getValue(t.index, rho.index);
      active.push({ value, weight });
    }
  }

  if (active.length === 0) {
    return { value: Number.NaN, status: 'empty_interpolation_stencil' };
  }
  if (active.some(({ value }) => !Number.isFinite(value) || value < 0)) {
    return { value: Number.NaN, status: 'invalid_interpolation_stencil' };
  }
  if (active.every(({ value }) => value === 0)) {
    return { value: 0, status: 'ok_zero' };
  }
  if (active.some(({ value }) => value === 0)) {
    return { value: Number.NaN, status: 'nonpositive_interpolation_stencil' };
  }

  let logValue = 0;
  let totalWeight = 0;
  for (const term of active) {
    logValue += term.weight * Math.log(term.value);
    totalWeight += term.weight;
  }
  return { value: Math.exp(logValue / totalWeight), status: 'ok' };
}

function logExpm1Positive(x) {
  if (!(x > 0) || !Number.isFinite(x)) return Number.NEGATIVE_INFINITY;
  if (x < 50) return Math.log(Math.expm1(x));
  return x + Math.log1p(-Math.exp(-x));
}

export function logPlanckIntegrand(x) {
  if (!(x > 0) || !Number.isFinite(x)) return Number.NEGATIVE_INFINITY;
  return 3 * Math.log(x) - logExpm1Positive(x);
}

export function logRosselandIntegrand(x) {
  if (!(x > 0) || !Number.isFinite(x)) return Number.NEGATIVE_INFINITY;
  return 4 * Math.log(x) - x - 2 * Math.log(-Math.expm1(-x));
}

function logSumExp(values) {
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (value > maximum) maximum = value;
  }
  if (!Number.isFinite(maximum)) return Number.NEGATIVE_INFINITY;
  let sum = 0;
  for (const value of values) {
    if (Number.isFinite(value)) sum += Math.exp(value - maximum);
  }
  return maximum + Math.log(sum);
}

function gaussLegendre32LogIntegral(logIntegrand, lower, upper) {
  if (!(upper > lower)) return Number.NEGATIVE_INFINITY;
  const midpoint = 0.5 * (lower + upper);
  const halfWidth = 0.5 * (upper - lower);
  const terms = [];
  for (let i = 0; i < GL32_X.length; i += 1) {
    const offset = halfWidth * GL32_X[i];
    const logQuadratureWeight = Math.log(GL32_W[i]);
    terms.push(logQuadratureWeight + logIntegrand(midpoint - offset));
    terms.push(logQuadratureWeight + logIntegrand(midpoint + offset));
  }
  return Math.log(halfWidth) + logSumExp(terms);
}

export function integrateLogPlanckWeight(energyLow, energyHigh, temperature) {
  assertFinitePositive(energyLow, 'Lower energy');
  assertFinitePositive(energyHigh, 'Upper energy');
  assertFinitePositive(temperature, 'Temperature');
  if (!(energyHigh > energyLow)) return Number.NEGATIVE_INFINITY;
  return gaussLegendre32LogIntegral(
    logPlanckIntegrand,
    energyLow / temperature,
    energyHigh / temperature,
  );
}

export function integrateLogRosselandWeight(energyLow, energyHigh, temperature) {
  assertFinitePositive(energyLow, 'Lower energy');
  assertFinitePositive(energyHigh, 'Upper energy');
  assertFinitePositive(temperature, 'Temperature');
  if (!(energyHigh > energyLow)) return Number.NEGATIVE_INFINITY;
  return gaussLegendre32LogIntegral(
    logRosselandIntegrand,
    energyLow / temperature,
    energyHigh / temperature,
  );
}

// Compatibility helpers. The collapse engine itself never exponentiates an
// absolute weight; it operates on logarithmic weights so Wien-tail factors
// cancel before exponentiation.
export function integratePlanckWeight(energyLow, energyHigh, temperature) {
  return Math.exp(integrateLogPlanckWeight(energyLow, energyHigh, temperature));
}

export function integrateRosselandWeight(energyLow, energyHigh, temperature) {
  return Math.exp(integrateLogRosselandWeight(energyLow, energyHigh, temperature));
}

export function buildCollapsePlan(kind, nativeEdges, outputEdges, temperature) {
  if (!['planck', 'rosseland_harmonic', 'rosseland_absorption'].includes(kind)) {
    throw new Error(`Unknown collapse kind: ${kind}`);
  }
  if (nativeEdges.length < 2 || outputEdges.length < 2) {
    throw new Error('Collapse edges are incomplete.');
  }
  assertFinitePositive(temperature, 'Temperature');

  const nativeMin = nativeEdges[0];
  const nativeMax = nativeEdges[nativeEdges.length - 1];
  if (outputEdges[0] < nativeMin && !approximatelyEqual(outputEdges[0], nativeMin)) {
    throw new Error('Output energy range begins below the native table.');
  }
  if (outputEdges[outputEdges.length - 1] > nativeMax &&
      !approximatelyEqual(outputEdges[outputEdges.length - 1], nativeMax)) {
    throw new Error('Output energy range ends above the native table.');
  }

  const integrateLog = kind === 'planck'
    ? integrateLogPlanckWeight
    : integrateLogRosselandWeight;
  const bins = [];
  let nativeIndex = 0;

  for (let outputIndex = 0; outputIndex < outputEdges.length - 1; outputIndex += 1) {
    const low = outputEdges[outputIndex];
    const high = outputEdges[outputIndex + 1];
    if (!(high > low)) throw new Error('Output energy edges must be strictly increasing.');

    while (nativeIndex < nativeEdges.length - 1 && nativeEdges[nativeIndex + 1] <= low) {
      nativeIndex += 1;
    }

    const terms = [];
    let cursor = nativeIndex;
    while (cursor < nativeEdges.length - 1 && nativeEdges[cursor] < high) {
      const overlapLow = Math.max(low, nativeEdges[cursor]);
      const overlapHigh = Math.min(high, nativeEdges[cursor + 1]);
      if (overlapHigh > overlapLow) {
        const logWeight = integrateLog(overlapLow, overlapHigh, temperature);
        terms.push({ nativeIndex: cursor, logWeight, overlapLow, overlapHigh });
      }
      if (nativeEdges[cursor + 1] >= high) break;
      cursor += 1;
    }
    bins.push({ low, high, terms });
  }

  return { kind, temperature, bins };
}

function scaledTerms(terms, effectiveLogWeight = (term) => term.logWeight) {
  const finite = terms
    .map((term) => ({ term, logWeight: effectiveLogWeight(term) }))
    .filter(({ logWeight }) => Number.isFinite(logWeight));
  if (finite.length === 0) return [];
  const maximum = Math.max(...finite.map(({ logWeight }) => logWeight));
  return finite.map(({ term, logWeight }) => ({
    term,
    weight: Math.exp(logWeight - maximum),
  }));
}

export function applyCollapsePlan(
  plan,
  nativeOpacity,
  nativeStatus = null,
  { totalOpacity = null, totalStatus = null } = {},
) {
  const results = [];
  for (const bin of plan.bins) {
    if (bin.terms.length === 0) {
      results.push({ value: Number.NaN, status: 'no_native_overlap' });
      continue;
    }

    const badSelected = bin.terms.find((term) => {
      if (!nativeStatus) return false;
      const status = nativeStatus[term.nativeIndex];
      return status && status !== 'ok' && status !== 'ok_zero';
    });
    if (badSelected) {
      results.push({ value: Number.NaN, status: nativeStatus[badSelected.nativeIndex] });
      continue;
    }

    const weighted = scaledTerms(bin.terms);
    if (weighted.length === 0) {
      results.push({ value: Number.NaN, status: 'invalid_weight_integral' });
      continue;
    }

    if (plan.kind === 'planck') {
      let numerator = 0;
      let denominator = 0;
      let invalid = false;
      for (const { term, weight } of weighted) {
        const opacity = nativeOpacity[term.nativeIndex];
        if (!Number.isFinite(opacity) || opacity < 0) {
          invalid = true;
          break;
        }
        numerator += weight * opacity;
        denominator += weight;
      }
      if (invalid || !(denominator > 0)) {
        results.push({ value: Number.NaN, status: 'invalid_native_opacity' });
      } else {
        const value = numerator / denominator;
        results.push({ value, status: value === 0 ? 'ok_zero' : 'ok' });
      }
      continue;
    }

    if (plan.kind === 'rosseland_harmonic') {
      let numerator = 0;
      let denominator = 0;
      let hasZero = false;
      let invalid = false;
      for (const { term, weight } of weighted) {
        const opacity = nativeOpacity[term.nativeIndex];
        if (!Number.isFinite(opacity) || opacity < 0) {
          invalid = true;
          break;
        }
        if (opacity === 0) {
          hasZero = true;
          break;
        }
        numerator += weight;
        denominator += weight / opacity;
      }
      if (invalid) {
        results.push({ value: Number.NaN, status: 'invalid_native_opacity' });
      } else if (hasZero) {
        results.push({ value: 0, status: 'ok_zero' });
      } else if (!(denominator > 0) || !Number.isFinite(denominator)) {
        results.push({ value: Number.NaN, status: 'invalid_harmonic_denominator' });
      } else {
        results.push({ value: numerator / denominator, status: 'ok' });
      }
      continue;
    }

    if (!totalOpacity) {
      results.push({ value: Number.NaN, status: 'missing_total_rosseland_spectrum' });
      continue;
    }
    const badTotal = bin.terms.find((term) => {
      if (!totalStatus) return false;
      const status = totalStatus[term.nativeIndex];
      return status && status !== 'ok' && status !== 'ok_zero';
    });
    if (badTotal) {
      results.push({ value: Number.NaN, status: `total_${totalStatus[badTotal.nativeIndex]}` });
      continue;
    }

    const transportTerms = scaledTerms(bin.terms, (term) => {
      const total = totalOpacity[term.nativeIndex];
      return Number.isFinite(total) && total > 0
        ? term.logWeight - Math.log(total)
        : Number.NaN;
    });
    if (transportTerms.length !== bin.terms.length) {
      results.push({ value: Number.NaN, status: 'invalid_total_rosseland_stencil' });
      continue;
    }

    let numerator = 0;
    let denominator = 0;
    let invalid = false;
    for (const { term, weight } of transportTerms) {
      const component = nativeOpacity[term.nativeIndex];
      if (!Number.isFinite(component) || component < 0) {
        invalid = true;
        break;
      }
      numerator += weight * component;
      denominator += weight;
    }
    if (invalid || !(denominator > 0)) {
      results.push({ value: Number.NaN, status: 'invalid_rosseland_absorption_stencil' });
    } else {
      const value = numerator / denominator;
      results.push({ value, status: value === 0 ? 'ok_zero' : 'ok' });
    }
  }
  return results;
}

