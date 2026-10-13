import assert from 'node:assert/strict';
import {
  COLLAPSE_KIND,
  applyCollapsePlan,
  buildCollapsePlan,
  findLogBracket,
} from '../opacity_math.mjs';

assert.equal(COLLAPSE_KIND.kplanck, 'planck');
assert.equal(COLLAPSE_KIND.kplanck_scattering, 'planck');
assert.equal(COLLAPSE_KIND.krosseland, 'rosseland_harmonic');
assert.ok(!('kross_scattering' in COLLAPSE_KIND));

const edges = [1, 2, 4];
const plan = buildCollapsePlan(
  COLLAPSE_KIND.kplanck_scattering,
  edges,
  [1, 4],
  1,
);
const result = applyCollapsePlan(plan, [0, 2]);
assert.equal(result.length, 1);
assert.equal(result[0].status, 'ok');
assert.ok(result[0].value > 0);
assert.ok(result[0].value < 2);

const zeroResult = applyCollapsePlan(plan, [0, 0]);
assert.equal(zeroResult[0].value, 0);
assert.equal(zeroResult[0].status, 'ok_zero');

// Axis matching must be relative: densities far below 1 g/cc must not snap to the axis minimum.
const densityAxis = Array.from({ length: 128 }, (_, i) => 10 ** (-17 + (20 * i) / 127));
for (const index of [0, 3, 27, 64, 127]) {
  const bracket = findLogBracket(densityAxis, densityAxis[index], 'Density');
  assert.equal(bracket.lowerIndex, index);
  assert.equal(bracket.upperIndex, index);
}
const between = findLogBracket(densityAxis, 2.5e-13, 'Density');
assert.equal(between.lowerIndex, 27);
assert.equal(between.upperIndex, 28);

console.log('Planck-scattering opacity-math tests passed.');
