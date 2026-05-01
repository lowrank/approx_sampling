# Results

Interactive convergence curves showing \(L^2\) error vs number of sampling points.
**Click legend items to toggle methods on/off**; double-click to isolate a single method.

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<div id="charts_container"></div>

<script>
const COLORS = {
  'adaptive_residual':'#000000',
  'uniform':'#1f77b4','chebyshev':'#ff7f0e','qmc_sobol':'#2ca02c',
  'qmc_halton':'#17becf',
  'iter_uniform':'#1f77b4','iter_chebyshev':'#ff7f0e',
  'adversarial':'#8c564b','importance_sampling':'#e377c2',
  'normalizing_flow':'#bcbd22','mdn':'#9467bd',
  'svgd':'#7f7f7f','ensemble':'#d62728','gp_ucb':'#2ca02c',
  'policy':'#e377c2','neural_process':'#17becf','diffusion':'#ff7f0e'
};
const WIDTH = {'adaptive_residual':2.5};
const DASH = {'iter_uniform':'dash','iter_chebyshev':'dash'};

fetch('figures/charts_data.json')
  .then(r => r.json())
  .then(data => {
    const container = document.getElementById('charts_container');
    const fns = Object.keys(data).sort();

    // Group by class
    const groups = {smooth:[], oscillatory:[], sharp:[], other:[]};
    for (const fn of fns) {
      if (fn.startsWith('smooth_')) groups.smooth.push(fn);
      else if (fn.startsWith('oscillatory_')) groups.oscillatory.push(fn);
      else if (fn.startsWith('sharp_')) groups.sharp.push(fn);
      else groups.other.push(fn);
    }

    for (const [label, funcs] of [['Smooth',groups.smooth],['Oscillatory',groups.oscillatory],
                                   ['Sharp',groups.sharp],['Other',groups.other]]) {
      if (!funcs.length) continue;
      const h2 = document.createElement('h2');
      h2.textContent = label;
      container.appendChild(h2);

      for (const fn of funcs) {
        const div = document.createElement('div');
        div.id = 'c_' + fn;
        div.style.cssText = 'width:100%;height:450px';
        container.appendChild(div);

        if (!data[fn]) continue;
        const traces = [];
        for (const [alg, rec] of Object.entries(data[fn])) {
          traces.push({
            x: rec.budgets, y: rec.errors,
            mode: 'lines+markers',
            name: alg === 'adaptive_residual' ? alg + ' (baseline)' : alg,
            line: { color: COLORS[alg]||'#333', width: WIDTH[alg]||1.2, dash: DASH[alg]||'solid' },
          marker: { size: alg === 'adaptive_residual' ? 6 : 4 }
        });
      }
      Plotly.newPlot('c_' + fn, traces, {
        title: fn.replace(/_/g, ' '),
        xaxis: { title: '# samples', type: 'log', dtick: 'D2' },
        yaxis: { title: 'L² error', type: 'log' },
        legend: { font: { size: 9 }, itemclick: 'toggle', itemdoubleclick: 'toggleothers' },
        hovermode: 'x unified',
        margin: { t: 35, b: 55, l: 70, r: 15 }
      }, { responsive: true, displaylogo: false });
    }
  }

  // ---- Class-average charts ----
  for (const [cls, prefix] of [['Smooth','smooth_'],['Oscillatory','oscillatory_'],['Sharp','sharp_']]) {
    const clsFns = Object.keys(data).filter(fn => fn.startsWith(prefix));
    if (clsFns.length < 2) continue;
    const div = document.createElement('div');
    div.id = 'class_' + cls.toLowerCase();
    div.style.cssText = 'width:100%;height:500px';
    container.appendChild(document.createElement('h2')).textContent = cls + ' — class average ± std';
    container.appendChild(div);

    // Aggregate per algorithm across functions
    const algMap = {};
    for (const fn of clsFns) {
      for (const [alg, rec] of Object.entries(data[fn])) {
        if (!algMap[alg]) algMap[alg] = [];
        algMap[alg].push(rec.errors);
      }
    }
    const traces2 = [];
    for (const [alg, errLists] of Object.entries(algMap)) {
      if (errLists.length < 2) continue;
      const maxLen = Math.max(...errLists.map(e => e.length));
      const arr = [];
      for (const e of errLists) {
        const padded = e.slice();
        while (padded.length < maxLen) padded.push(padded[padded.length-1]);
        arr.push(padded);
      }
      const mean = arr[0].map((_,i) => arr.reduce((s,v)=>s+v[i],0)/arr.length);
      const std  = arr[0].map((_,i) => Math.sqrt(arr.reduce((s,v)=>s+(v[i]-mean[i])**2,0)/arr.length));
      const b = data[clsFns[0]][alg]?.budgets?.slice(0,maxLen) || [];
      traces2.push({
        x: b, y: mean, mode: 'lines', name: alg,
        line: { color: COLORS[alg]||'#333', width: WIDTH[alg]||2 },
        showlegend: true
      });
      traces2.push({
        x: b.concat(b.slice().reverse()),
        y: mean.map((m,i)=>m+std[i]).concat(mean.map((m,i)=>m-std[i]).reverse()),
        fill: 'toself', fillcolor: (COLORS[alg]||'#333')+'20',
        line: { width: 0 }, showlegend: false, name: alg+'_band'
      });
    }
    Plotly.newPlot('class_' + cls.toLowerCase(), traces2, {
      title: cls + ' class — mean ± std (' + clsFns.length + ' functions)',
      xaxis: { title: '# samples', type: 'log', dtick: 'D2' },
      yaxis: { title: 'L² error', type: 'log' },
      legend: { font: { size: 9 }, itemclick: 'toggle', itemdoubleclick: 'toggleothers' },
      hovermode: 'x unified',
      margin: { t: 35, b: 55, l: 70, r: 15 }
    }, { responsive: true, displaylogo: false });
  }
  })
  .catch(() => {
    document.getElementById('charts_container').innerHTML =
      '<p style="color:#888;text-align:center;padding:100px 0;font-style:italic">'
      + 'No data yet. Run <code>python scripts/run_full_experiment.py --quick</code></p>';
  });
</script>
