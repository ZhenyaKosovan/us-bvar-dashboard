(() => {
  const chartNodes = (root) => {
    const nodes = [];
    if (root.matches?.('.bvar-echart')) nodes.push(root);
    root.querySelectorAll?.('.bvar-echart').forEach((node) => nodes.push(node));
    return nodes;
  };

  const destroyCharts = (root) => {
    chartNodes(root).forEach((node) => {
      node.bvarResizeObserver?.disconnect();
      node.bvarChart?.dispose();
      node.bvarChart = null;
    });
  };

  const escapeHtml = (value) => String(value).replace(
    /[&<>'"]/g,
    (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
    }[character]),
  );

  const prepareChartOption = (raw) => {
    const bands = raw.bvarBands || [];
    const decimals = raw.bvarValueDecimals ?? 2;
    const units = raw.bvarUnits || '';
    delete raw.bvarBands;
    delete raw.bvarValueDecimals;
    delete raw.bvarUnits;
    const bandSeries = bands.map((band) => ({
      name: band.name,
      type: 'custom',
      data: band.data,
      dimensions: ['month', 'lower', 'upper'],
      encode: {x: 0, y: [1, 2]},
      silent: true,
      tooltip: {show: false},
      z: 1,
      renderItem: (params, api) => {
        const current = band.data[params.dataIndex];
        const next = band.data[params.dataIndex + 1];
        if (!next) return null;
        return {
          type: 'polygon',
          shape: {
            points: [
              api.coord([current[0], current[1]]),
              api.coord([next[0], next[1]]),
              api.coord([next[0], next[2]]),
              api.coord([current[0], current[2]]),
            ],
          },
          style: {fill: band.color, stroke: 'none'},
        };
      },
    }));
    raw.series = [...bandSeries, ...raw.series];
    raw.xAxis.axisLabel.formatter = (value) => new Intl.DateTimeFormat(
      undefined,
      {month: 'short', year: '2-digit', timeZone: 'UTC'},
    ).format(new Date(value));
    raw.tooltip.formatter = (parameters) => {
      const visible = parameters.filter((item) => item.seriesType !== 'custom');
      if (!visible.length) return '';
      const month = new Intl.DateTimeFormat(
        undefined,
        {month: 'long', year: 'numeric', timeZone: 'UTC'},
      ).format(new Date(visible[0].value[0]));
      const rows = visible.map((item) => (
        `${item.marker}${escapeHtml(item.seriesName)}: ` +
        `<strong>${Number(item.value[1]).toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        })}</strong>`
      ));
      return `<strong>${month}</strong><br>${rows.join('<br>')}<br>` +
        `<span class="chart-tooltip-units">${escapeHtml(units)}</span>`;
    };
    return raw;
  };

  const renderCharts = (root = document) => {
    chartNodes(root).forEach((node) => {
      const configNode = node.querySelector('script.chart-config');
      const target = node.querySelector('.chart-target');
      if (!configNode || !target || typeof echarts === 'undefined') return;
      const signature = configNode.textContent;
      if (node.dataset.signature === signature) return;
      node.bvarResizeObserver?.disconnect();
      node.bvarChart?.dispose();
      node.bvarChart = echarts.init(target, null, {renderer: 'svg'});
      node.bvarChart.setOption(prepareChartOption(JSON.parse(signature)));
      let lastWidth = target.clientWidth;
      let lastHeight = target.clientHeight;
      node.bvarResizeObserver = new ResizeObserver((entries) => {
        const rectangle = entries[0]?.contentRect;
        if (!rectangle || (
          rectangle.width === lastWidth && rectangle.height === lastHeight
        )) {
          return;
        }
        lastWidth = rectangle.width;
        lastHeight = rectangle.height;
        requestAnimationFrame(() => node.bvarChart?.resize());
      });
      node.bvarResizeObserver.observe(target);
      node.dataset.signature = signature;
    });
  };

  const updateConstraintCount = () => {
    const editor = document.querySelector('.scenario-modal');
    if (!editor) return;
    const dialog = editor.closest('.modal-content') || editor;
    const count = [...editor.querySelectorAll('.scenario-value')]
      .filter((field) => field.value.trim()).length;
    const badge = editor.querySelector('.constraint-count');
    const runButton = dialog.querySelector('#run_scenario');
    if (badge) badge.textContent = count === 1 ? '1 constraint' : `${count} constraints`;
    if (runButton) runButton.disabled = count === 0;
  };

  const updateScenarioRow = (select) => {
    const row = select.closest('[data-scenario-row]');
    if (!row) return;
    const unitsByScale = JSON.parse(row.dataset.unitsByScale);
    const transformation = Object.hasOwn(unitsByScale, select.value)
      ? select.value
      : 'level';
    row.querySelector('.scenario-row-units').textContent = unitsByScale[transformation];
    row.querySelectorAll('[data-values-by-scale]').forEach((cell) => {
      const values = JSON.parse(cell.dataset.valuesByScale);
      cell.textContent = values[transformation];
    });
    row.querySelectorAll('.scenario-value').forEach((field) => {
      const placeholders = JSON.parse(field.dataset.placeholdersByScale);
      field.placeholder = placeholders[transformation] ?? placeholders.level;
      field.setAttribute(
        'aria-label',
        `${field.dataset.seriesLabel}, ${field.dataset.month}, ` +
          unitsByScale[transformation],
      );
    });
  };

  const initializeScenarioEditors = (root = document) => {
    const editors = [];
    if (root.matches?.('.scenario-modal')) editors.push(root);
    root.querySelectorAll?.('.scenario-modal').forEach((node) => editors.push(node));
    editors.forEach((editor) => {
      editor.querySelectorAll('[id^="sc_transform_"]').forEach(updateScenarioRow);
      filterScenarioRows(editor);
      updateConstraintCount();
    });
  };

  const filterScenarioRows = (editor) => {
    const query = (editor.querySelector('#scenario-variable-search')?.value || '')
      .trim().toLocaleLowerCase();
    const group = editor.querySelector('#scenario-group-filter')?.value || 'All';
    editor.querySelectorAll('[data-scenario-row]').forEach((row) => {
      const matchesQuery = !query || row.dataset.seriesSearch.includes(query);
      const matchesGroup = group === 'All' || row.dataset.seriesGroup === group;
      row.hidden = !(matchesQuery && matchesGroup);
    });
  };

  document.addEventListener('change', (event) => {
    if (event.target.matches('[id^="sc_transform_"]')) updateScenarioRow(event.target);
    if (event.target.matches('.scenario-value')) updateConstraintCount();
  });
  document.addEventListener('input', (event) => {
    if (event.target.matches('.scenario-value')) updateConstraintCount();
    if (event.target.matches('#scenario-variable-search')) {
      filterScenarioRows(event.target.closest('.scenario-modal'));
    }
  });
  document.addEventListener('change', (event) => {
    if (event.target.matches('#scenario-group-filter')) {
      filterScenarioRows(event.target.closest('.scenario-modal'));
    }
  });
  document.addEventListener('DOMContentLoaded', () => {
    renderCharts();
    initializeScenarioEditors();
    new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.removedNodes.forEach((node) => {
          if (node.nodeType === 1) destroyCharts(node);
        });
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1) {
            renderCharts(node);
            initializeScenarioEditors(node);
          }
        });
      });
    }).observe(document.body, {childList: true, subtree: true});
  });
})();