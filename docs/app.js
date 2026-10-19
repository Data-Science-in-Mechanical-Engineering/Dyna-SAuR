const u1Slider = document.getElementById('u1Slider');
const u2Slider = document.getElementById('u2Slider');
const normSlider = document.getElementById('normSlider');
const newPointsBtn = document.getElementById('newPointsBtn');
const valueReadout = document.getElementById('valueReadout');

const vectorPlotEl = document.getElementById('vectorPlot');
const mainPlotEl = document.getElementById('mainPlot');

// Visual theme (matches styles.css); only affects appearance.
const THEME = {
  font: { family: 'Inter, system-ui, sans-serif', size: 13, color: '#2b3a33' },
  grid: '#e8eeea',
  axis: '#9aa8a0',
  ink: '#0f1d17',
  vector: '#1e7a4c',
  ball: '#12a4b8',
  hyperplane: '#0f1d17',
  unsafe: '#d62728',
  safe: '#2ca02c',
  clipped: '#f08a24',
  projection: '#8a968f',
};

function axisStyle(title, extra) {
  return {
    title: { text: title, font: { family: 'Times New Roman, serif', size: 18, color: THEME.ink } },
    gridcolor: THEME.grid,
    zerolinecolor: THEME.axis,
    linecolor: THEME.axis,
    tickfont: { size: 12, color: '#5b6b63' },
    ...extra,
  };
}

let u = [0.2, 0.2];
let samplePoints = generateRandomPoints();

function generateRandomPoints() {
  const points = [];
  for (let i = 0; i < 5; i += 1) {
    const x = -0.99 + Math.random() * 1.98;
    const y = -0.99 + Math.random() * 1.98;
    points.push([x, y]);
  }
  return points;
}

function norm2(v) {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1]);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function enforceUnitNorm(v) {
  const n = norm2(v);
  if (n > 1) return [v[0] / n, v[1] / n];
  return v;
}

function computeHyperplane(v) {
  const vNorm = Math.max(norm2(v), 1e-12);
  const n = [v[0] / vNorm, v[1] / vNorm];
  const offset = (vNorm * 2 - 1) * (Math.abs(n[0]) + Math.abs(n[1]));

  const xs = [];
  const ys = [];
  const nPts = 300;

  if (Math.abs(n[1]) > 1e-12) {
    for (let i = 0; i < nPts; i += 1) {
      const x = -1.25 + (2.5 * i) / (nPts - 1);
      const y = (offset - n[0] * x) / n[1];
      xs.push(x);
      ys.push(y);
    }
  } else {
    const x = Math.abs(n[0]) > 1e-12 ? offset / n[0] : 0;
    xs.push(x, x);
    ys.push(-1.25, 1.25);
  }

  return { n, offset, xs, ys };
}

function projectPoints(v, points) {
  const hp = computeHyperplane(v);
  const { n, offset } = hp;

  const original = points.map((p) => [p[0], p[1]]);

  const oneStep = (pts) => {
    return pts.map((p) => {
      const dot = p[0] * n[0] + p[1] * n[1];
      if (dot < offset) {
        const adjust = dot - offset;
        return [
          clamp(p[0] - adjust * n[0], -1, 1),
          clamp(p[1] - adjust * n[1], -1, 1),
        ];
      }
      return [p[0], p[1]];
    });
  };

  const first = oneStep(original);
  const second = oneStep(first);

  const filtered = first.map((p1, i) => {
    const p2 = second[i];
    const numerator = offset - (p1[0] * n[0] + p1[1] * n[1]);
    const denominator = (p1[0] - p2[0]) * n[0] + (p1[1] - p2[1]) * n[1] + 1e-8;
    const t = numerator / denominator;
    return [
      p1[0] + (p1[0] - p2[0]) * t,
      p1[1] + (p1[1] - p2[1]) * t,
    ];
  });

  const wasProjected = original.map((p) => p[0] * n[0] + p[1] * n[1] < offset);
  const clippedMask = filtered.map((p, i) => {
    const dot = p[0] * n[0] + p[1] * n[1];
    return wasProjected[i] && dot < offset - 1e-4;
  });

  return { original, filtered, wasProjected, clippedMask, ...hp };
}

function syncControlsFromVector() {
  const n = norm2(u);
  u1Slider.value = u[0].toFixed(3);
  u2Slider.value = u[1].toFixed(3);
  normSlider.value = clamp(n, 0.001, 1).toFixed(3);
  valueReadout.textContent = `u = (${u[0].toFixed(3)}, ${u[1].toFixed(3)}), ‖u‖₂ = ${n.toFixed(3)}`;
}

function draw() {
  const projection = projectPoints(u, samplePoints);

  const vectorLayout = {
    font: THEME.font,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#ffffff',
    xaxis: axisStyle('u₁', { range: [-1.1, 1.1], scaleanchor: 'y', scaleratio: 1, constrain: 'domain' }),
    yaxis: axisStyle('u₂', { range: [-1.1, 1.1] }),
    margin: { l: 56, r: 16, t: 16, b: 120 },
    shapes: [
      {
        type: 'circle',
        xref: 'x',
        yref: 'y',
        x0: -1,
        x1: 1,
        y0: -1,
        y1: 1,
        fillcolor: 'rgba(18, 164, 184, 0.07)',
        line: { color: THEME.ball, width: 2 },
      },
    ],
    annotations: [
      {
        x: u[0],
        y: u[1],
        ax: 0,
        ay: 0,
        xref: 'x',
        yref: 'y',
        axref: 'x',
        ayref: 'y',
        showarrow: true,
        arrowhead: 3,
        arrowsize: 1,
        arrowwidth: 3,
        arrowcolor: THEME.vector,
        text: '',
      },
      {
        x: 0.02,
        y: 0.98,
        xref: 'paper',
        yref: 'paper',
        xanchor: 'left',
        yanchor: 'top',
        align: 'left',
        showarrow: false,
        bordercolor: '#e3e9e5',
        borderwidth: 1,
        borderpad: 6,
        bgcolor: 'rgba(255,255,255,0.92)',
        font: { family: 'JetBrains Mono, monospace', size: 12, color: THEME.ink },
        text: `u = (${u[0].toFixed(2)}, ${u[1].toFixed(2)})<br>‖u‖₂ = ${norm2(u).toFixed(2)}`,
      },
    ],
  };

  Plotly.react(
    vectorPlotEl,
    [
      {
        x: [0, u[0]],
        y: [0, u[1]],
        mode: 'lines',
        line: { color: THEME.vector, width: 3 },
        name: 'Current Vector',
        showlegend: false,
      },
    ],
    vectorLayout,
    { responsive: true, displayModeBar: false },
  );

  const unsafeX = [];
  const unsafeY = [];
  const safeX = [];
  const safeY = [];
  const clippedX = [];
  const clippedY = [];
  const connectorX = [];
  const connectorY = [];

  for (let i = 0; i < projection.original.length; i += 1) {
    const o = projection.original[i];
    const f = projection.filtered[i];

    safeX.push(f[0]);
    safeY.push(f[1]);

    if (projection.wasProjected[i]) {
      unsafeX.push(o[0]);
      unsafeY.push(o[1]);
      connectorX.push(o[0], f[0], null);
      connectorY.push(o[1], f[1], null);
    }

    if (projection.clippedMask[i]) {
      clippedX.push(f[0]);
      clippedY.push(f[1]);
    }
  }

  const n = projection.n;
  const d = projection.offset;
  const hyperplaneName = `Hyperplane <i>w</i><sup>T</sup><i>a</i> = ${d.toFixed(2)}, <i>w</i> = (${n[0].toFixed(2)}, ${n[1].toFixed(2)})`;

  const shadeN = 140;
  const shadeX = [];
  const shadeY = [];
  for (let i = 0; i < shadeN; i += 1) {
    shadeX.push(-1 + (2 * i) / (shadeN - 1));
    shadeY.push(-1 + (2 * i) / (shadeN - 1));
  }
  const shadeZ = shadeY.map((yy) => shadeX.map((xx) => n[0] * xx + n[1] * yy - d));
  const maxAbsZ = Math.max(
    1e-8,
    ...shadeZ.flat().map((v) => Math.abs(v)),
  );

  const unsafeTraceX = unsafeX.length > 0 ? unsafeX : [null];
  const unsafeTraceY = unsafeY.length > 0 ? unsafeY : [null];

  const mainTraces = [
    {
      x: shadeX,
      y: shadeY,
      z: shadeZ,
      type: 'heatmap',
      colorscale: [
        [0.0, 'rgba(220, 53, 69, 0.20)'],
        [0.499, 'rgba(220, 53, 69, 0.20)'],
        [0.501, 'rgba(40, 167, 69, 0.20)'],
        [1.0, 'rgba(40, 167, 69, 0.20)'],
      ],
      zmin: -maxAbsZ,
      zmax: maxAbsZ,
      showscale: false,
      hoverinfo: 'skip',
      name: 'Hyperplane side shading',
      showlegend: false,
    },
    {
      x: projection.xs,
      y: projection.ys,
      mode: 'lines',
      line: { color: THEME.hyperplane, width: 3 },
      name: hyperplaneName,
    },
    {
      x: connectorX,
      y: connectorY,
      mode: 'lines',
      line: { color: THEME.projection, width: 2, dash: 'dot' },
      name: 'Projection path',
      showlegend: false,
    },
    {
      x: unsafeTraceX,
      y: unsafeTraceY,
      mode: 'markers',
      marker: {
        color: THEME.unsafe,
        size: 15,
        line: { color: '#ffffff', width: 2 },
      },
      name: 'Proposed action (potentially unsafe)',
      showlegend: true,
    },
    {
      x: safeX,
      y: safeY,
      mode: 'markers',
      marker: {
        color: THEME.safe,
        opacity: 0.85,
        size: 15,
        line: { color: '#ffffff', width: 2 },
      },
      name: 'Filtered action (viable)',
    },
    {
      x: clippedX,
      y: clippedY,
      mode: 'markers',
      marker: { color: THEME.clipped, symbol: 'x-thin', size: 20, line: { color: THEME.clipped, width: 3 } },
      name: 'Clipped action',
    },
  ];

  const mainLayout = {
    font: THEME.font,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#ffffff',
    xaxis: axisStyle('a₁', { range: [-1.1, 1.1], scaleanchor: 'y', scaleratio: 1, constrain: 'domain' }),
    yaxis: axisStyle('a₂', { range: [-1.1, 1.1] }),
    margin: { l: 56, r: 16, t: 16, b: 120 },
    shapes: [
      {
        type: 'rect',
        x0: -1,
        x1: 1,
        y0: -1,
        y1: 1,
        line: { color: THEME.ink, width: 2 },
      },
    ],
    legend: {
      orientation: 'h',
      x: 0.5,
      y: 0.01,
      yref: 'container',
      xanchor: 'center',
      yanchor: 'bottom',
      font: { size: 12.5 },
      itemsizing: 'constant',
    },
  };

  Plotly.react(mainPlotEl, mainTraces, mainLayout, {
    responsive: true,
    displayModeBar: false,
  });
}

function onUComponentChange() {
  let next = [Number(u1Slider.value), Number(u2Slider.value)];
  next = enforceUnitNorm(next);
  u = next;
  syncControlsFromVector();
  draw();
}

function onNormChange() {
  const targetNorm = Number(normSlider.value);
  const currentNorm = norm2(u);

  let direction;
  if (currentNorm > 1e-12) {
    direction = [u[0] / currentNorm, u[1] / currentNorm];
  } else {
    direction = [1, 0];
  }

  u = [direction[0] * targetNorm, direction[1] * targetNorm];
  u = enforceUnitNorm(u);
  syncControlsFromVector();
  draw();
}

u1Slider.addEventListener('input', onUComponentChange);
u2Slider.addEventListener('input', onUComponentChange);
normSlider.addEventListener('input', onNormChange);
newPointsBtn.addEventListener('click', () => {
  samplePoints = generateRandomPoints();
  draw();
});

syncControlsFromVector();
draw();
