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
		const feedback = document.querySelector("#workspace-feedback");
		if (!feedback) return;
		feedback.textContent = "";
		requestAnimationFrame(() => {
			feedback.textContent = message;
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
		const atLimit = selected.size >= MAX_WORKSPACE_CARDS;
		document
			.querySelectorAll(".variable-library-item[data-series-id]")
			.forEach((item) => {
				const isSelected = selected.has(item.dataset.seriesId);
				item.classList.toggle("is-selected", isSelected);
				const button = item.querySelector(".variable-add-button");
				const label =
					item.querySelector(".library-item-label")?.textContent || "chart";
				if (button) {
					button.disabled = !isSelected && atLimit;
					button.setAttribute("aria-pressed", String(isSelected));
					let title = `Add ${label} to the canvas`;
					let accessibleText = `Add ${label}`;
					if (isSelected) {
						title = `Remove ${label} from the canvas`;
						accessibleText = `Remove ${label}`;
					} else if (atLimit) {
						title = `Remove a chart before adding ${label}`;
						accessibleText = `Chart limit reached; remove a chart before adding ${label}`;
					}
					button.title = title;
					const accessibleLabel = button.querySelector(".visually-hidden");
					if (accessibleLabel) accessibleLabel.textContent = accessibleText;
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

	const resetWorkspace = (button) => {
		workspaceState.selection = [];
		workspaceState.cards = {};
		workspaceState.transforms = {};
		document.querySelectorAll(".chart-card[data-series-id]").forEach((card) => {
			const select = card.querySelector('[id^="plot_transform_"]');
			const defaultTransform = card.dataset.defaultTransform;
			if (select && defaultTransform && select.value !== defaultTransform) {
				select.value = defaultTransform;
				select.dispatchEvent(new Event("change", { bubbles: true }));
			}
		});
		const defaults = parseJson(button.dataset.seriesIds, []);
		setWorkspaceSelection(
			defaults,
			"Workspace restored to the default overview.",
		);
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
			const cardSize = preference.wide ? "wide" : "standard";
			card.dataset.cardSize = cardSize;
			card.classList.toggle("chart-card-wide", Boolean(preference.wide));
			card.classList.remove("chart-card-tall");
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
		const size = button.dataset.cardSize === "wide" ? "wide" : "standard";
		workspaceState.cards[card.dataset.seriesId] = {
			wide: size === "wide",
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
			if (!configNode || !target || window.echarts === undefined) return;
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

	const scenarioFields = (editor) => [
		...editor.querySelectorAll(".scenario-value"),
	];

	const writeScenarioField = (field, value) => {
		field.value = value;
		field.dispatchEvent(new Event("input", { bubbles: true }));
		field.dispatchEvent(new Event("change", { bubbles: true }));
	};

	const scenarioNumberPattern =
		/^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

	const parsedScenarioValue = (field) => {
		const entered = field.value.trim();
		if (!entered) return null;
		if (!scenarioNumberPattern.test(entered)) return Number.NaN;
		return Number(entered.replaceAll(",", ""));
	};

	const scenarioFieldError = (field) => {
		if (!field.value.trim()) return "";
		const value = parsedScenarioValue(field);
		if (!Number.isFinite(value)) return "Enter a valid number.";
		if (Math.abs(value) > 1_000_000_000)
			return "Values cannot exceed one billion in absolute magnitude.";
		const row = field.closest("[data-scenario-row]");
		const transformation = row?.querySelector(".scenario-transform")?.value;
		if (row?.dataset.modelTransform === "log") {
			if (transformation === "level" && value <= 0)
				return "Levels must be greater than zero.";
			if (transformation !== "level" && value <= -100)
				return "Growth assumptions must be greater than −100%.";
		}
		return "";
	};

	const setScenarioValidation = (editor, message, field = null) => {
		const output = editor.querySelector("#scenario_validation");
		if (!output) return;
		let box = output.querySelector(".scenario-validation-message");
		if (!box) {
			box = document.createElement("div");
			box.className = "scenario-validation-message";
			box.setAttribute("role", "alert");
			box.setAttribute("aria-live", "assertive");
			output.replaceChildren(box);
		}
		box.textContent = message;
		box.hidden = !message;
		if (message && field) {
			field.setAttribute("aria-invalid", "true");
			field.classList.add("scenario-input-invalid");
		}
	};

	const validateScenarioEditor = (editor, { showMessage = false } = {}) => {
		const nameField = editor.querySelector("#scenario_name");
		const name = nameField?.value.trim() || "";
		const entered = scenarioFields(editor).filter((field) =>
			field.value.trim(),
		);
		const maxConstraints = Number(editor.dataset.maxConstraints) || 60;
		let firstInvalid = null;
		let error = "";
		for (const field of entered) {
			field.removeAttribute("aria-invalid");
			field.classList.remove("scenario-input-invalid");
			field.removeAttribute("title");
			const fieldError = scenarioFieldError(field);
			if (fieldError) {
				field.setAttribute("aria-invalid", "true");
				field.classList.add("scenario-input-invalid");
				field.title = fieldError;
			}
			if (!error && fieldError) {
				error = `${field.dataset.seriesShortLabel}, ${field.dataset.month}: ${fieldError}`;
				firstInvalid = field;
			}
		}
		if (!name) {
			error = "Give this scenario a name.";
			firstInvalid = nameField;
		} else if (name.length > 60) {
			error = "Scenario names must be 60 characters or fewer.";
			firstInvalid = nameField;
		} else if (!entered.length) {
			error = "Enter at least one scenario value.";
		} else if (entered.length > maxConstraints) {
			error = `Use at most ${maxConstraints} scenario assumptions per run.`;
		}
		const runButton = editor
			.closest(".modal-content")
			?.querySelector("#run_scenario");
		if (runButton) {
			runButton.disabled = Boolean(error);
			runButton.classList.toggle("disabled", Boolean(error));
			runButton.setAttribute("aria-disabled", String(Boolean(error)));
		}
		const actionHelp = editor
			.closest(".modal-content")
			?.querySelector("#scenario-action-help");
		if (actionHelp) {
			let help = `${entered.length} valid ${entered.length === 1 ? "assumption" : "assumptions"} ready to calculate.`;
			if (!name) {
				help = "Add a name and at least one valid assumption.";
			} else if (!entered.length) {
				help = "Enter at least one valid assumption.";
			} else if (error) {
				help = "Correct the highlighted assumption before calculating.";
			}
			actionHelp.textContent = help;
		}
		if (showMessage && error) {
			setScenarioValidation(editor, error, firstInvalid);
			firstInvalid?.focus();
		} else if (!error) {
			setScenarioValidation(editor, "");
		}
		return !error;
	};

	const filterScenarioRows = (editor) => {
		const query = (
			editor.querySelector("#scenario-variable-search")?.value || ""
		)
			.trim()
			.toLocaleLowerCase();
		const group =
			editor.querySelector("#scenario-group-filter")?.value || "All";
		const selectedOnly =
			editor
				.querySelector(".scenario-selected-only")
				?.getAttribute("aria-pressed") === "true";
		editor.querySelectorAll("[data-scenario-row]").forEach((row) => {
			const matchesQuery = !query || row.dataset.seriesSearch.includes(query);
			const matchesGroup = group === "All" || row.dataset.seriesGroup === group;
			const hasValues = [...row.querySelectorAll(".scenario-value")].some(
				(field) => field.value.trim(),
			);
			row.hidden = !(
				matchesQuery &&
				matchesGroup &&
				(!selectedOnly || hasValues)
			);
		});
	};

	const updateAssumptionSummary = (editor) => {
		const fields = scenarioFields(editor).filter((field) => field.value.trim());
		const list = editor.querySelector(".scenario-assumption-list");
		const empty = editor.querySelector(".scenario-assumption-empty");
		if (!list || !empty) return;
		list.replaceChildren();
		empty.hidden = Boolean(fields.length);
		fields.forEach((field) => {
			const row = field.closest("[data-scenario-row]");
			const transformation = row?.querySelector(".scenario-transform");
			const item = document.createElement("div");
			item.className = "scenario-assumption-item";
			const focusButton = document.createElement("button");
			focusButton.type = "button";
			focusButton.className = "scenario-assumption-chip";
			focusButton.dataset.fieldId = field.id;
			const [month, year] = field.dataset.month.split(" ");
			const transformationLabel =
				transformation?.selectedOptions?.[0]?.text ||
				transformation?.value ||
				"Level";
			focusButton.textContent =
				`${field.dataset.seriesShortLabel} · ${month.slice(0, 3)} ${year} · ` +
				`${transformationLabel} ${field.value}`;
			const removeButton = document.createElement("button");
			removeButton.type = "button";
			removeButton.className = "scenario-assumption-remove";
			removeButton.dataset.fieldId = field.id;
			removeButton.setAttribute(
				"aria-label",
				`Remove ${field.dataset.seriesShortLabel}, ${field.dataset.month}`,
			);
			removeButton.textContent = "×";
			item.append(focusButton, removeButton);
			list.append(item);
		});
	};

	const updateConstraintCount = (
		editor = document.querySelector(".scenario-modal"),
	) => {
		if (!editor) return;
		const count = scenarioFields(editor).filter((field) =>
			field.value.trim(),
		).length;
		const badge = editor.querySelector(".constraint-count");
		const selectedOnly = editor.querySelector(".scenario-selected-only");
		if (badge)
			badge.textContent = count === 1 ? "1 constraint" : `${count} constraints`;
		if (selectedOnly) {
			selectedOnly.disabled = count === 0;
			if (!count) selectedOnly.setAttribute("aria-pressed", "false");
		}
		updateAssumptionSummary(editor);
		validateScenarioEditor(editor);
		filterScenarioRows(editor);
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
				`${field.dataset.seriesLabel}, ${field.dataset.month}, ${unitsByScale[transformation] || ""}`,
			);
		});
		select.dataset.appliedTransformation = transformation;
	};

	let pendingScenarioTransformation = null;

	const finishScenarioTransformation = (action) => {
		if (!pendingScenarioTransformation) return;
		const { select, next, previous } = pendingScenarioTransformation;
		const row = select.closest("[data-scenario-row]");
		const fields = row ? [...row.querySelectorAll(".scenario-value")] : [];
		if (action === "keep") {
			select.value = next;
		} else if (action === "clear") {
			fields.forEach((field) => writeScenarioField(field, ""));
			select.value = next;
		} else {
			select.value = previous;
		}
		updateScenarioRow(select);
		select.dispatchEvent(new Event("change", { bubbles: true }));
		pendingScenarioTransformation = null;
		document.querySelector("#scenario-transform-dialog")?.close();
		updateConstraintCount(row?.closest(".scenario-modal"));
	};

	const changeScenarioTransformation = (select) => {
		const row = select.closest("[data-scenario-row]");
		if (!row) return;
		const previous = select.dataset.appliedTransformation;
		const next = select.value;
		const hasValues = [...row.querySelectorAll(".scenario-value")].some(
			(field) => field.value.trim(),
		);
		if (previous && previous !== next && hasValues) {
			select.value = previous;
			const dialog = row
				.closest(".scenario-modal")
				?.querySelector("#scenario-transform-dialog");
			if (!dialog?.showModal) {
				if (
					window.confirm(
						"Keep entered numbers and reinterpret them on the new scale?",
					)
				) {
					select.value = next;
				}
				updateScenarioRow(select);
				return;
			}
			pendingScenarioTransformation = { select, next, previous };
			dialog.querySelector("#scenario-transform-dialog-copy").textContent =
				"Choose whether to keep the entered numbers on the new scale, clear them, or cancel the change.";
			dialog.showModal();
			return;
		}
		updateScenarioRow(select);
		updateConstraintCount(row.closest(".scenario-modal"));
	};

	const initializeScenarioEditors = (root = document) => {
		const editors = [];
		if (root.matches?.(".scenario-modal")) editors.push(root);
		root
			.querySelectorAll?.(".scenario-modal")
			.forEach((node) => editors.push(node));
		editors.forEach((editor) => {
			editor.querySelectorAll(".scenario-transform").forEach(updateScenarioRow);
			updateConstraintCount(editor);
		});
	};

	const requestScenarioEditorClose = () => {
		const editor = document.querySelector(".scenario-modal");
		if (!editor) return false;
		const hasValues = scenarioFields(editor).some((field) =>
			field.value.trim(),
		);
		const hasName = Boolean(
			editor.querySelector("#scenario_name")?.value.trim(),
		);
		if (
			(hasValues || hasName) &&
			!window.confirm(
				"Discard the name and assumptions entered in this editor?",
			)
		) {
			return false;
		}
		if (window.Shiny?.modal?.remove) {
			window.Shiny.modal.remove();
		} else {
			const modal = editor.closest(".modal");
			if (modal) window.bootstrap?.Modal.getOrCreateInstance(modal).hide();
		}
		return true;
	};

	const applyScenarioRowAction = (button) => {
		const row = button.closest("[data-scenario-row]");
		const editor = row?.closest(".scenario-modal");
		if (!row || !editor) return;
		const fields = [...row.querySelectorAll(".scenario-value")];
		const entered = fields.filter((field) => field.value.trim());
		const action = button.dataset.rowAction;
		if (action === "clear") {
			if (
				entered.length &&
				!window.confirm("Clear every assumption in this row?")
			)
				return;
			fields.forEach((field) => writeScenarioField(field, ""));
		} else if (action === "hold") {
			const firstIndex = fields.findIndex((field) => field.value.trim());
			if (firstIndex < 0) {
				setScenarioValidation(
					editor,
					"Enter a starting value before using Hold from first.",
				);
				return;
			}
			if (
				fields.slice(firstIndex + 1).some((field) => field.value.trim()) &&
				!window.confirm(
					"Replace later assumptions with the first entered value?",
				)
			)
				return;
			fields
				.slice(firstIndex)
				.forEach((field) =>
					writeScenarioField(field, fields[firstIndex].value),
				);
		} else if (action === "interpolate") {
			if (entered.length < 2) {
				setScenarioValidation(
					editor,
					"Enter two endpoints before interpolating a path.",
				);
				return;
			}
			const first = fields.indexOf(entered[0]);
			const last = fields.indexOf(entered.at(-1));
			const start = parsedScenarioValue(fields[first]);
			const end = parsedScenarioValue(fields[last]);
			if (!Number.isFinite(start) || !Number.isFinite(end) || first === last) {
				setScenarioValidation(
					editor,
					"Interpolation endpoints must be valid numbers.",
				);
				return;
			}
			for (let index = first; index <= last; index += 1) {
				const share = (index - first) / (last - first);
				const value = start + (end - start) * share;
				writeScenarioField(fields[index], String(Number(value.toFixed(4))));
			}
		}
		updateConstraintCount(editor);
	};

	const applyScenarioStarter = (button) => {
		const editor = button.closest(".scenario-modal");
		const row = editor?.querySelector(
			`[data-scenario-row="${CSS.escape(button.dataset.seriesId)}"]`,
		);
		if (!editor || !row) return;
		const fields = [...row.querySelectorAll(".scenario-value")];
		if (
			fields.some((field) => field.value.trim()) &&
			!window.confirm("Replace the existing assumptions for this variable?")
		)
			return;
		const select = row.querySelector(".scenario-transform");
		select.value = button.dataset.transformation;
		updateScenarioRow(select);
		select.dispatchEvent(new Event("change", { bubbles: true }));
		const adjustment = Number(button.dataset.adjustment);
		fields.forEach((field) => {
			const placeholders = parseJson(field.dataset.placeholdersByScale, {});
			const baseline = Number(
				String(placeholders[button.dataset.transformation] || "").replaceAll(
					",",
					"",
				),
			);
			writeScenarioField(
				field,
				String(Number((baseline + adjustment).toFixed(4))),
			);
		});
		const name = editor.querySelector("#scenario_name");
		if (name && !name.value.trim())
			writeScenarioField(name, button.dataset.scenarioName);
		updateConstraintCount(editor);
		row.scrollIntoView({
			block: "center",
			inline: "start",
			behavior: "smooth",
		});
	};

	const sendScenarioRequest = (inputId) => {
		if (!window.Shiny?.setInputValue) return false;
		window.Shiny.setInputValue(
			inputId,
			`${Date.now()}-${Math.random().toString(16).slice(2)}`,
			{ priority: "event" },
		);
		return true;
	};

	const initializeScenarioGuards = (root = document) => {
		const guarded = [];
		if (root.matches?.("#run_scenario, #reset_modal, #delete_scenario"))
			guarded.push(root);
		root
			.querySelectorAll?.("#run_scenario, #reset_modal, #delete_scenario")
			.forEach((button) => guarded.push(button));
		guarded.forEach((button) => {
			if (button.dataset.scenarioGuarded) return;
			button.dataset.scenarioGuarded = "true";
			button.addEventListener(
				"click",
				(event) => {
					const editor = document.querySelector(".scenario-modal");
					if (button.id === "run_scenario" && editor) {
						if (!validateScenarioEditor(editor, { showMessage: true })) {
							event.preventDefault();
							event.stopImmediatePropagation();
							return;
						}
						if (sendScenarioRequest("run_scenario_request")) {
							event.preventDefault();
							event.stopImmediatePropagation();
						}
					} else if (button.id === "reset_modal" && editor) {
						const hasValues = scenarioFields(editor).some((field) =>
							field.value.trim(),
						);
						if (
							hasValues &&
							!window.confirm("Reset every assumption in this editor?")
						) {
							event.preventDefault();
							event.stopImmediatePropagation();
							return;
						}
						if (sendScenarioRequest("reset_modal_request")) {
							event.preventDefault();
							event.stopImmediatePropagation();
						}
					} else if (button.id === "delete_scenario") {
						const activeName =
							document.querySelector("#active_scenario")?.selectedOptions[0]
								?.text;
						if (
							!window.confirm(
								`Delete “${activeName || "this scenario"}”? This cannot be undone.`,
							)
						) {
							event.preventDefault();
							event.stopImmediatePropagation();
						}
					}
				},
				true,
			);
		});
	};

	document.addEventListener(
		"keydown",
		(event) => {
			if (event.key !== "Escape" || !document.querySelector(".scenario-modal"))
				return;
			event.preventDefault();
			event.stopPropagation();
			requestScenarioEditorClose();
		},
		true,
	);

	document.addEventListener("click", (event) => {
		const exitButton = event.target.closest(".scenario-exit");
		if (exitButton) {
			requestScenarioEditorClose();
			return;
		}

		const transformAction = event.target.closest("[data-transform-action]");
		if (transformAction) {
			finishScenarioTransformation(transformAction.dataset.transformAction);
			return;
		}

		const selectedOnly = event.target.closest(".scenario-selected-only");
		if (selectedOnly) {
			const active = selectedOnly.getAttribute("aria-pressed") !== "true";
			selectedOnly.setAttribute("aria-pressed", String(active));
			selectedOnly.textContent = active
				? "Show all variables"
				: "Show assumptions only";
			filterScenarioRows(selectedOnly.closest(".scenario-modal"));
			return;
		}

		const assumptionChip = event.target.closest(".scenario-assumption-chip");
		if (assumptionChip) {
			const field = document.querySelector(
				`#${CSS.escape(assumptionChip.dataset.fieldId)}`,
			);
			field?.scrollIntoView({
				block: "center",
				inline: "center",
				behavior: "smooth",
			});
			field?.focus({ preventScroll: true });
			return;
		}

		const assumptionRemove = event.target.closest(
			".scenario-assumption-remove",
		);
		if (assumptionRemove) {
			const field = document.querySelector(
				`#${CSS.escape(assumptionRemove.dataset.fieldId)}`,
			);
			if (field) writeScenarioField(field, "");
			updateConstraintCount(assumptionRemove.closest(".scenario-modal"));
			return;
		}

		const rowAction = event.target.closest(".scenario-row-action");
		if (rowAction) {
			applyScenarioRowAction(rowAction);
			return;
		}

		const starter = event.target.closest(".scenario-starter");
		if (starter) {
			applyScenarioStarter(starter);
			return;
		}

		const exportButton = event.target.closest(".scenario-export");
		if (exportButton) {
			const payload = parseJson(exportButton.dataset.scenarioExport, null);
			if (!payload) return;
			const blob = new Blob([JSON.stringify(payload, null, 2)], {
				type: "application/json",
			});
			const link = document.createElement("a");
			const safeName = (exportButton.dataset.scenarioName || "scenario")
				.toLocaleLowerCase()
				.replace(/[^a-z0-9]+/g, "-")
				.replace(/^-|-$/g, "");
			link.href = URL.createObjectURL(blob);
			link.download = `${safeName || "scenario"}.json`;
			link.click();
			URL.revokeObjectURL(link.href);
			return;
		}

		const forecastLink = event.target.closest(
			'.section-navigation a[href="#forecast-data"]',
		);
		if (forecastLink) {
			const forecastData = document.querySelector("#forecast-data");
			if (forecastData) forecastData.open = true;
		}

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

		const resetButton = event.target.closest(".workspace-reset");
		if (resetButton) {
			resetWorkspace(resetButton);
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
		if (event.target.matches(".scenario-transform"))
			changeScenarioTransformation(event.target);
		if (event.target.matches(".scenario-value"))
			updateConstraintCount(event.target.closest(".scenario-modal"));
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
		if (event.target.matches(".scenario-value, #scenario_name"))
			updateConstraintCount(event.target.closest(".scenario-modal"));
		if (event.target.matches("#scenario-variable-search")) {
			filterScenarioRows(event.target.closest(".scenario-modal"));
		}
		if (event.target.matches("#variable-library-search"))
			filterVariableLibrary();
	});

	document.addEventListener("paste", (event) => {
		const field = event.target.closest(".scenario-value");
		const text = event.clipboardData?.getData("text") || "";
		if (!field || !/[\t\r\n]/.test(text)) return;
		event.preventDefault();
		const rowFields = [
			...field
				.closest("[data-scenario-row]")
				.querySelectorAll(".scenario-value"),
		];
		const start = rowFields.indexOf(field);
		text
			.trim()
			.split(/[\t\r\n]+/)
			.slice(0, rowFields.length - start)
			.forEach((value, index) =>
				writeScenarioField(rowFields[start + index], value.trim()),
			);
		updateConstraintCount(field.closest(".scenario-modal"));
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
		initializeScenarioGuards();
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
						initializeScenarioGuards(node);
						initializeWorkspace(node);
					}
				});
			});
		}).observe(document.body, { childList: true, subtree: true });
	});
})();
