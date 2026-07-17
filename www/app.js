(() => {
	const chartNodes = (root) => {
		const nodes = [];
		if (root.matches?.(".bvar-echart")) nodes.push(root);
		root.querySelectorAll?.(".bvar-echart").forEach((node) => nodes.push(node));
		return nodes;
	};

	const destroyCharts = (root) => {
		if (root.isConnected) return;
		chartNodes(root).forEach((node) => {
			node.bvarResizeObserver?.disconnect();
			node.bvarChart?.dispose();
			node.bvarChart = null;
			delete node.dataset.signature;
		});
	};

	const escapeHtml = (value) =>
		String(value).replace(
			/[&<>'"]/g,
			(character) =>
				({
					"&": "&amp;",
					"<": "&lt;",
					">": "&gt;",
					"'": "&#39;",
					'"': "&quot;",
				})[character],
		);

	const parseJson = (value, fallback) => {
		try {
			return JSON.parse(value);
		} catch (_error) {
			return fallback;
		}
	};

	const WORKSPACE_STORAGE_KEY = "us-bvar-workspace-v1";
	const WORKSPACE_VERSION = 1;
	const MAX_WORKSPACE_CARDS = 8;
	let workspaceRestored = false;
	let workspaceDrag = null;
	let workspaceDropTarget = null;

	const emptyWorkspaceState = () => ({
		version: WORKSPACE_VERSION,
		selection: [],
		cards: {},
		transforms: {},
	});

	const readWorkspaceState = () => {
		try {
			const parsed = JSON.parse(
				localStorage.getItem(WORKSPACE_STORAGE_KEY) || "null",
			);
			if (!parsed || parsed.version !== WORKSPACE_VERSION)
				return emptyWorkspaceState();
			return {
				version: WORKSPACE_VERSION,
				selection: Array.isArray(parsed.selection) ? parsed.selection : [],
				cards:
					parsed.cards && typeof parsed.cards === "object" ? parsed.cards : {},
				transforms:
					parsed.transforms && typeof parsed.transforms === "object"
						? parsed.transforms
						: {},
			};
		} catch (_error) {
			return emptyWorkspaceState();
		}
	};

	const workspaceState = readWorkspaceState();

	const writeWorkspaceState = () => {
		try {
			localStorage.setItem(
				WORKSPACE_STORAGE_KEY,
				JSON.stringify(workspaceState),
			);
			const status = document.querySelector(".canvas-save-status");
			if (status) status.textContent = "Workspace saved in this browser.";
		} catch (_error) {
			const status = document.querySelector(".canvas-save-status");
			if (status) status.textContent = "Workspace could not be saved.";
		}
	};

	const announceWorkspace = (message) => {
		const announcer = document.querySelector("#workspace-announcer");
		if (!announcer) return;
		announcer.textContent = "";
		requestAnimationFrame(() => {
			announcer.textContent = message;
		});
	};

	const validSeriesIds = () =>
		new Set(
			[
				...document.querySelectorAll(".variable-library-item[data-series-id]"),
			].map((item) => item.dataset.seriesId),
		);

	const normalizeSelection = (seriesIds) => {
		const valid = validSeriesIds();
		const normalized = [];
		seriesIds.forEach((seriesId) => {
			if (valid.has(seriesId) && !normalized.includes(seriesId))
				normalized.push(seriesId);
		});
		return normalized.slice(0, MAX_WORKSPACE_CARDS);
	};

	const selectionControl = () =>
		document.querySelector("#visible_variables")?.selectize || null;

	const currentSelection = () => {
		const control = selectionControl();
		if (control) return normalizeSelection([...control.items]);
		return normalizeSelection(
			[...document.querySelectorAll(".chart-card[data-series-id]")].map(
				(card) => card.dataset.seriesId,
			),
		);
	};

	const refreshLibraryState = () => {
		const selected = new Set(currentSelection());
		document
			.querySelectorAll(".variable-library-item[data-series-id]")
			.forEach((item) => {
				const isSelected = selected.has(item.dataset.seriesId);
				item.classList.toggle("is-selected", isSelected);
				const button = item.querySelector(".variable-add-button");
				const label =
					item.querySelector(".library-item-label")?.textContent || "chart";
				if (button) {
					button.setAttribute("aria-pressed", String(isSelected));
					button.title = isSelected
						? `Remove ${label} from the canvas`
						: `Add ${label} to the canvas`;
					const accessibleLabel = button.querySelector(".visually-hidden");
					if (accessibleLabel) {
						accessibleLabel.textContent = isSelected
							? `Remove ${label}`
							: `Add ${label}`;
					}
				}
			});
	};

	const setWorkspaceSelection = (seriesIds, message = "") => {
		const control = selectionControl();
		if (!control) return false;
		const selection = normalizeSelection(seriesIds);
		if (!selection.length) {
			announceWorkspace("Keep at least one chart on the canvas.");
			return false;
		}
		control.setValue(selection, true);
		control.$input.trigger("change");
		workspaceState.selection = selection;
		writeWorkspaceState();
		refreshLibraryState();
		if (message) announceWorkspace(message);
		return true;
	};

	const restoreWorkspaceSelection = () => {
		const control = selectionControl();
		if (!control || workspaceRestored) return Boolean(control);
		workspaceRestored = true;
		const saved = normalizeSelection(workspaceState.selection);
		if (saved.length) {
			setWorkspaceSelection(saved);
		} else {
			workspaceState.selection = currentSelection();
			writeWorkspaceState();
		}
		return true;
	};

	const cardPreference = (seriesId) => {
		const saved = workspaceState.cards[seriesId];
		return saved && typeof saved === "object" ? saved : {};
	};

	const applyCardPreferences = (root = document) => {
		const cards = [];
		if (root.matches?.(".chart-card")) cards.push(root);
		root
			.querySelectorAll?.(".chart-card[data-series-id]")
			.forEach((card) => cards.push(card));
		cards.forEach((card) => {
			const preference = cardPreference(card.dataset.seriesId);
			let cardSize = "standard";
			if (preference.wide && preference.tall) cardSize = "focus";
			else if (preference.wide) cardSize = "wide";
			else if (preference.tall) cardSize = "tall";
			card.dataset.cardSize = cardSize;
			card.classList.toggle("chart-card-wide", Boolean(preference.wide));
			card.classList.toggle("chart-card-tall", Boolean(preference.tall));
			card.querySelectorAll(".chart-size-option").forEach((option) => {
				const selected = option.dataset.cardSize === cardSize;
				option.classList.toggle("is-active", selected);
				option.setAttribute("aria-checked", String(selected));
			});
			const transform = workspaceState.transforms[card.dataset.seriesId];
			const select = card.querySelector('[id^="plot_transform_"]');
			if (select && transform && select.value !== transform) {
				select.value = transform;
				select.dispatchEvent(new Event("change", { bubbles: true }));
			}
		});
	};

	const filterVariableLibrary = () => {
		const query = (
			document.querySelector("#variable-library-search")?.value || ""
		)
			.trim()
			.toLocaleLowerCase();
		const activeFilter =
			document.querySelector(".library-filter.is-active")?.dataset.group ||
			"All";
		let visible = 0;
		document.querySelectorAll(".variable-library-item").forEach((item) => {
			const matchesQuery = !query || item.dataset.seriesSearch.includes(query);
			const matchesGroup =
				activeFilter === "All" || item.dataset.seriesGroup === activeFilter;
			item.hidden = !(matchesQuery && matchesGroup);
			if (!item.hidden) visible += 1;
		});
		const empty = document.querySelector(".library-empty");
		if (empty) empty.hidden = visible !== 0;
	};

	const setCardSize = (button) => {
		const card = button.closest(".chart-card");
		if (!card) return;
		const size = button.dataset.cardSize || "standard";
		workspaceState.cards[card.dataset.seriesId] = {
			wide: size === "wide" || size === "focus",
			tall: size === "tall" || size === "focus",
		};
		writeWorkspaceState();
		applyCardPreferences(card);
		card.closest(".chart-menu")?.removeAttribute("open");
		requestAnimationFrame(() =>
			card.querySelector(".bvar-echart")?.bvarChart?.resize(),
		);
		announceWorkspace(
			`${card.querySelector("h2")?.textContent || "Chart"} changed to ${size} size.`,
		);
	};

	const moveWorkspaceCard = (seriesId, direction) => {
		const selection = currentSelection();
		const currentIndex = selection.indexOf(seriesId);
		const targetIndex = currentIndex + direction;
		if (currentIndex < 0 || targetIndex < 0 || targetIndex >= selection.length)
			return;
		[selection[currentIndex], selection[targetIndex]] = [
			selection[targetIndex],
			selection[currentIndex],
		];
		setWorkspaceSelection(
			selection,
			`${seriesId} moved ${direction < 0 ? "earlier" : "later"} in the matrix.`,
		);
	};

	const clearWorkspaceDragStyles = () => {
		document.body.classList.remove("workspace-is-dragging");
		document
			.querySelector("#workspace-canvas")
			?.classList.remove("is-drop-ready");
		document.querySelectorAll(".chart-card.is-drop-target").forEach((card) => {
			card.classList.remove("is-drop-target", "drop-after");
		});
		workspaceDrag = null;
		workspaceDropTarget = null;
	};

	const initializeWorkspace = (root = document) => {
		applyCardPreferences(root);
		refreshLibraryState();
		filterVariableLibrary();
		if (!restoreWorkspaceSelection()) {
			window.setTimeout(() => initializeWorkspace(), 100);
		}
	};

	const prepareChartOption = (raw) => {
		const bands = raw.bvarBands || [];
		const decimals = raw.bvarValueDecimals ?? 2;
		const units = raw.bvarUnits || "";
		delete raw.bvarBands;
		delete raw.bvarValueDecimals;
		delete raw.bvarUnits;
		const bandSeries = bands.map((band) => ({
			name: band.name,
			type: "custom",
			data: band.data,
			dimensions: ["month", "lower", "upper"],
			encode: { x: 0, y: [1, 2] },
			silent: true,
			tooltip: { show: false },
			z: 1,
			renderItem: (params, api) => {
				const current = band.data[params.dataIndex];
				const next = band.data[params.dataIndex + 1];
				if (!next) return null;
				return {
					type: "polygon",
					shape: {
						points: [
							api.coord([current[0], current[1]]),
							api.coord([next[0], next[1]]),
							api.coord([next[0], next[2]]),
							api.coord([current[0], current[2]]),
						],
					},
					style: { fill: band.color, stroke: "none" },
				};
			},
		}));
		raw.series = [...bandSeries, ...raw.series];
		raw.xAxis.axisLabel.formatter = (value) =>
			new Intl.DateTimeFormat(undefined, {
				month: "short",
				year: "2-digit",
				timeZone: "UTC",
			}).format(new Date(value));
		raw.tooltip.formatter = (parameters) => {
			const visible = parameters.filter((item) => item.seriesType !== "custom");
			if (!visible.length) return "";
			const month = new Intl.DateTimeFormat(undefined, {
				month: "long",
				year: "numeric",
				timeZone: "UTC",
			}).format(new Date(visible[0].value[0]));
			const rows = visible.map(
				(item) =>
					`${item.marker}${escapeHtml(item.seriesName)}: ` +
					`<strong>${Number(item.value[1]).toLocaleString(undefined, {
						minimumFractionDigits: decimals,
						maximumFractionDigits: decimals,
					})}</strong>`,
			);
			return (
				`<strong>${month}</strong><br>${rows.join("<br>")}<br>` +
				`<span class="chart-tooltip-units">${escapeHtml(units)}</span>`
			);
		};
		return raw;
	};

	const renderCharts = (root = document) => {
		chartNodes(root).forEach((node) => {
			const configNode = node.querySelector("script.chart-config");
			const target = node.querySelector(".chart-target");
			if (!configNode || !target || typeof echarts === "undefined") return;
			const signature = configNode.textContent;
			if (
				node.dataset.signature === signature &&
				node.bvarChart &&
				!node.bvarChart.isDisposed?.()
			)
				return;
			const option = parseJson(signature, null);
			if (!option) return;
			node.bvarResizeObserver?.disconnect();
			node.bvarChart?.dispose();
			node.bvarChart = echarts.init(target, null, { renderer: "svg" });
			node.bvarChart.setOption(prepareChartOption(option));
			let lastWidth = target.clientWidth;
			let lastHeight = target.clientHeight;
			node.bvarResizeObserver = new ResizeObserver((entries) => {
				const rectangle = entries[0]?.contentRect;
				if (
					!rectangle ||
					(rectangle.width === lastWidth && rectangle.height === lastHeight)
				) {
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
		const editor = document.querySelector(".scenario-modal");
		if (!editor) return;
		const dialog = editor.closest(".modal-content") || editor;
		const count = [...editor.querySelectorAll(".scenario-value")].filter(
			(field) => field.value.trim(),
		).length;
		const badge = editor.querySelector(".constraint-count");
		const runButton = dialog.querySelector("#run_scenario");
		if (badge)
			badge.textContent = count === 1 ? "1 constraint" : `${count} constraints`;
		if (runButton) runButton.disabled = count === 0;
	};

	const updateScenarioRow = (select) => {
		const row = select.closest("[data-scenario-row]");
		if (!row) return;
		const unitsByScale = parseJson(row.dataset.unitsByScale, {});
		const transformation = Object.hasOwn(unitsByScale, select.value)
			? select.value
			: "level";
		row.querySelector(".scenario-row-units").textContent =
			unitsByScale[transformation] || "";
		row.querySelectorAll("[data-values-by-scale]").forEach((cell) => {
			const values = parseJson(cell.dataset.valuesByScale, {});
			cell.textContent = values[transformation] || "";
		});
		row.querySelectorAll(".scenario-value").forEach((field) => {
			const placeholders = parseJson(field.dataset.placeholdersByScale, {});
			field.placeholder =
				placeholders[transformation] ?? placeholders.level ?? "";
			field.setAttribute(
				"aria-label",
				`${field.dataset.seriesLabel}, ${field.dataset.month}, ` +
					(unitsByScale[transformation] || ""),
			);
		});
	};

	const initializeScenarioEditors = (root = document) => {
		const editors = [];
		if (root.matches?.(".scenario-modal")) editors.push(root);
		root
			.querySelectorAll?.(".scenario-modal")
			.forEach((node) => editors.push(node));
		editors.forEach((editor) => {
			editor
				.querySelectorAll('[id^="sc_transform_"]')
				.forEach(updateScenarioRow);
			filterScenarioRows(editor);
			updateConstraintCount();
		});
	};

	const filterScenarioRows = (editor) => {
		const query = (
			editor.querySelector("#scenario-variable-search")?.value || ""
		)
			.trim()
			.toLocaleLowerCase();
		const group =
			editor.querySelector("#scenario-group-filter")?.value || "All";
		editor.querySelectorAll("[data-scenario-row]").forEach((row) => {
			const matchesQuery = !query || row.dataset.seriesSearch.includes(query);
			const matchesGroup = group === "All" || row.dataset.seriesGroup === group;
			row.hidden = !(matchesQuery && matchesGroup);
		});
	};

	document.addEventListener("click", (event) => {
		const addButton = event.target.closest(".variable-add-button");
		if (addButton) {
			const selection = currentSelection();
			const seriesId = addButton.dataset.seriesId;
			if (selection.includes(seriesId)) {
				if (selection.length === 1) {
					announceWorkspace("Keep at least one chart on the canvas.");
				} else {
					setWorkspaceSelection(
						selection.filter((candidate) => candidate !== seriesId),
						`${seriesId} removed from the matrix.`,
					);
				}
			} else if (selection.length >= MAX_WORKSPACE_CARDS) {
				announceWorkspace(
					`The canvas is limited to ${MAX_WORKSPACE_CARDS} charts.`,
				);
			} else {
				setWorkspaceSelection(
					[...selection, seriesId],
					`${seriesId} added to the matrix.`,
				);
			}
			return;
		}

		const filterButton = event.target.closest(".library-filter");
		if (filterButton) {
			document.querySelectorAll(".library-filter").forEach((button) => {
				const active = button === filterButton;
				button.classList.toggle("is-active", active);
				button.setAttribute("aria-pressed", String(active));
			});
			filterVariableLibrary();
			return;
		}

		const presetButton = event.target.closest(".workspace-preset");
		if (presetButton) {
			const seriesIds = parseJson(presetButton.dataset.seriesIds, []);
			setWorkspaceSelection(
				seriesIds,
				`${presetButton.dataset.workspacePreset} workspace applied.`,
			);
			return;
		}

		const action = event.target.closest(".chart-action, .chart-menu-option");
		if (!action) return;
		const seriesId = action.dataset.seriesId;
		if (action.matches(".chart-remove-button")) {
			const selection = currentSelection();
			if (selection.length === 1) {
				announceWorkspace("Keep at least one chart on the canvas.");
				return;
			}
			setWorkspaceSelection(
				selection.filter((candidate) => candidate !== seriesId),
				`${seriesId} removed from the matrix.`,
			);
			window.setTimeout(() => {
				document
					.querySelector(
						`.variable-add-button[data-series-id="${CSS.escape(seriesId)}"]`,
					)
					?.focus();
			}, 250);
		} else if (action.matches(".chart-move-earlier")) {
			moveWorkspaceCard(seriesId, -1);
		} else if (action.matches(".chart-move-later")) {
			moveWorkspaceCard(seriesId, 1);
		} else if (action.matches(".chart-size-option")) {
			setCardSize(action);
		}
		action.closest(".chart-menu")?.removeAttribute("open");
	});

	document.addEventListener("change", (event) => {
		if (event.target.matches('[id^="sc_transform_"]'))
			updateScenarioRow(event.target);
		if (event.target.matches(".scenario-value")) updateConstraintCount();
		if (event.target.matches("#scenario-group-filter")) {
			filterScenarioRows(event.target.closest(".scenario-modal"));
		}
		if (event.target.matches('[id^="plot_transform_"]')) {
			const card = event.target.closest(".chart-card");
			if (card) {
				workspaceState.transforms[card.dataset.seriesId] = event.target.value;
				writeWorkspaceState();
			}
		}
		if (event.target.matches("#visible_variables") && workspaceRestored) {
			workspaceState.selection = currentSelection();
			writeWorkspaceState();
			refreshLibraryState();
		}
	});

	document.addEventListener("input", (event) => {
		if (event.target.matches(".scenario-value")) updateConstraintCount();
		if (event.target.matches("#scenario-variable-search")) {
			filterScenarioRows(event.target.closest(".scenario-modal"));
		}
		if (event.target.matches("#variable-library-search"))
			filterVariableLibrary();
	});

	document.addEventListener("dragstart", (event) => {
		const handle = event.target.closest(".chart-drag-handle");
		const libraryItem = event.target.closest(".variable-library-item");
		if (!handle && !libraryItem) return;
		const source = handle || libraryItem;
		workspaceDrag = {
			seriesId: source.dataset.seriesId,
			type: handle ? "card" : "library",
		};
		event.dataTransfer.effectAllowed = handle ? "move" : "copyMove";
		event.dataTransfer.setData("text/plain", workspaceDrag.seriesId);
		document.body.classList.add("workspace-is-dragging");
	});

	document.addEventListener("dragover", (event) => {
		const canvas = event.target.closest("#workspace-canvas");
		if (!canvas || !workspaceDrag) return;
		event.preventDefault();
		event.dataTransfer.dropEffect =
			workspaceDrag.type === "card" ? "move" : "copy";
		canvas.classList.add("is-drop-ready");
		document.querySelectorAll(".chart-card.is-drop-target").forEach((card) => {
			card.classList.remove("is-drop-target", "drop-after");
		});
		const target = event.target.closest(".chart-card");
		workspaceDropTarget = null;
		if (!target || target.dataset.seriesId === workspaceDrag.seriesId) return;
		const rectangle = target.getBoundingClientRect();
		const after =
			event.clientY > rectangle.top + rectangle.height / 2 ||
			(Math.abs(event.clientY - (rectangle.top + rectangle.height / 2)) <
				rectangle.height / 3 &&
				event.clientX > rectangle.left + rectangle.width / 2);
		target.classList.add("is-drop-target");
		target.classList.toggle("drop-after", after);
		workspaceDropTarget = { seriesId: target.dataset.seriesId, after };
	});

	document.addEventListener("drop", (event) => {
		if (!event.target.closest("#workspace-canvas") || !workspaceDrag) return;
		event.preventDefault();
		const selection = currentSelection();
		const alreadySelected = selection.includes(workspaceDrag.seriesId);
		if (!alreadySelected && selection.length >= MAX_WORKSPACE_CARDS) {
			announceWorkspace(
				`The canvas is limited to ${MAX_WORKSPACE_CARDS} charts.`,
			);
			clearWorkspaceDragStyles();
			return;
		}
		const reordered = selection.filter(
			(seriesId) => seriesId !== workspaceDrag.seriesId,
		);
		if (workspaceDropTarget) {
			const targetIndex = reordered.indexOf(workspaceDropTarget.seriesId);
			reordered.splice(
				targetIndex < 0
					? reordered.length
					: targetIndex + Number(workspaceDropTarget.after),
				0,
				workspaceDrag.seriesId,
			);
		} else {
			reordered.push(workspaceDrag.seriesId);
		}
		setWorkspaceSelection(
			reordered,
			`${workspaceDrag.seriesId} ${alreadySelected ? "reordered" : "added"} in the matrix.`,
		);
		clearWorkspaceDragStyles();
	});

	document.addEventListener("dragend", clearWorkspaceDragStyles);

	document.addEventListener("DOMContentLoaded", () => {
		renderCharts();
		initializeScenarioEditors();
		initializeWorkspace();
		new MutationObserver((mutations) => {
			mutations.forEach((mutation) => {
				mutation.removedNodes.forEach((node) => {
					if (node.nodeType === 1) destroyCharts(node);
				});
				mutation.addedNodes.forEach((node) => {
					if (node.nodeType === 1) {
						renderCharts(node);
						initializeScenarioEditors(node);
						initializeWorkspace(node);
					}
				});
			});
		}).observe(document.body, { childList: true, subtree: true });
	});
})();
