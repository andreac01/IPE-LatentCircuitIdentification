document.addEventListener('DOMContentLoaded', function() {
	// --- DOM Element Selection ---
	const runButton = document.getElementById('run-button');
	const modeSwitch = document.getElementById('input-mode-switch');
	const modelTaskView = document.getElementById('model-task-view');
	const promptTargetView = document.getElementById('prompt-target-view');
	const promptInput = document.getElementById('prompt-input');
	const targetTokenInput = document.getElementById('target-token-input');
	const numPathsSlider = document.getElementById('num-paths-slider');
	const numPathsValue = document.getElementById('num-paths-value');
	const divideHeadsSwitch = document.getElementById('divide-heads-switch');
	const colorSchemeSelect = document.getElementById('color-scheme-select');
	const primaryPlotDiv = document.getElementById('primary-plot');
	const secondaryPlotWrapper = document.getElementById('secondary-plot-wrapper');
	const secondaryPlotDiv = document.getElementById('secondary-plot');
	const loader = document.getElementById('loader');
	const showSecondaryPlotSwitch = document.getElementById('show-secondary-plot-switch');
	const edgePrioritySwitch = document.getElementById('edge-priority-switch');
	const allEdgesSwitch = document.getElementById('all-edges-switch');

	// --- State ---
	let fullPlotData = null; // Will store all data from the backend
	let currentPrimaryPlotData = null;
	let currentSecondaryPlotPathIdx = null;
	let sortedEdgesCache = [];
	let pathSlider = null; // Will be created dynamically
	let currentSecondaryPlotDetails = null;




	// --- Event Listeners ---
	numPathsSlider.addEventListener('input', () => {
		numPathsValue.textContent = numPathsSlider.value;
	});

	// initialize display
	numPathsValue.textContent = numPathsSlider.value;
	runButton.addEventListener('click', runVisualization);
	
	// Re-run visualization when controls are changed
	// Backend-dependent controls
	[divideHeadsSwitch].forEach(el => {
		el.addEventListener('change', runVisualization);
	});

	// Mode switch listener
	modeSwitch.addEventListener('change', function() {
		if (this.checked) {
			// Custom Prompt Mode
			modelTaskView.style.display = 'none';
			promptTargetView.style.display = 'flex';
			runButton.textContent = 'Compute and Visualize';
			runButton.title = 'Find the top-200 paths using a Best First Search approach. If a target is provided the percentage of change in its probability will be used as metric. If target is empty use KL divergence. The search will cut off after 3 minutes.';
		} else {
			// Model & Task Mode (Default)
			modelTaskView.style.display = 'flex';
			promptTargetView.style.display = 'none';
			runButton.textContent = 'Visualize Circuit';
			runButton.title = 'Visualize the pre-computed paths.'
		}
	});

	// Download paths button
	const downloadButton = document.getElementById('download-button');
	downloadButton.addEventListener('click', async () => {
		// Ensure we have data to send
		if (!fullPlotData || !fullPlotData.graphData) {
			alert('No path data available to download. Run a visualization first.');
			return;
		}

		loader.style.display = 'block';
		downloadButton.disabled = true;

		try {
			// Reconstruct the exact parameters used for the last "run" request from the current UI state
			let lastRunParams = {};
			if (modeSwitch.checked) {
				lastRunParams = {
					model_name: document.getElementById('model-select-run').value,
					precomputed: false,
					prompt: promptInput.value,
					target: targetTokenInput.value,
					divide_heads: divideHeadsSwitch.checked,
					uuid: fullPlotData.uuid,
				};
			} else {
				lastRunParams = {
					model_name: document.getElementById('model-select').value,
					precomputed: true,
					task_name: document.getElementById('task-select').value,
					task: document.getElementById('mode-select').value,
					patch: document.getElementById('patch-select').value,
					divide_heads: divideHeadsSwitch.checked,
					uuid: fullPlotData.uuid,
				};
			}
			// Also include the current num_paths selection to make intent explicit
			lastRunParams.num_paths = parseInt(numPathsSlider.value, 10);

			// Send the data to backend for packaging (adjust endpoint if needed)
			const resp = await fetch('/api/download_paths', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					params: lastRunParams,
				})
			});

			if (!resp.ok) throw new Error(`Server responded with ${resp.status}`);

			const blob = await resp.blob();

			// Prefer a .pkl filename for this endpoint; try to extract from headers first
			let filename = 'paths.pkl';
			const cd = resp.headers.get('Content-Disposition') || '';
			const m = cd.match(/filename\*=UTF-8''([^;]+)|filename="([^"]+)"|filename=([^;]+)/);
			if (m) {
				try {
					// pick the first non-undefined capture group and decode safely
					const rawName = m[1] || m[2] || m[3];
					filename = decodeURIComponent(rawName.trim().replace(/^["']|["']$/g, ''));
				} catch (e) {
					// fallback to default if decode fails
					console.warn('Failed to decode filename from header, using default paths.pkl', e);
					filename = 'paths.pkl';
				}
			} else {
				// fallback based on content-type
				const ct = resp.headers.get('Content-Type') || '';
				if (ct.includes('zip')) filename = 'paths.zip';
				else if (ct.includes('json')) filename = 'paths.json';
				else filename = 'paths.pkl'; // explicit default per requirement
			}

			// Trigger download
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = filename;
			document.body.appendChild(a);
			a.click();
			a.remove();
			URL.revokeObjectURL(url);
		} catch (err) {
			console.error('Download failed:', err);
			primaryPlotDiv.innerHTML = `<div class="alert alert-danger">Download failed: ${err.message}</div>`;
		} finally {
			loader.style.display = 'none';
			downloadButton.disabled = false;
		}
	});

	// Frontend-only controls
	[numPathsSlider, colorSchemeSelect, edgePrioritySwitch, allEdgesSwitch].forEach(el => {
		el.addEventListener('change', filterAndDraw);
	});

	// Theme change listener (listens for custom event from HTML)
	window.addEventListener('themeChanged', () => {
		setTimeout(handleResizeOrThemeChange, 50); // Delay to allow DOM to update
	});

	// Window resize listener
	window.addEventListener('resize', debounce(handleResizeOrThemeChange, 150));

	// The switch's direct visibility toggle is handled in the HTML.
	// This listener ensures the plot is drawn/highlighted if needed.
	showSecondaryPlotSwitch.addEventListener('change', (event) => {
		const isVisible = event.currentTarget.checked;
		if (isVisible) {
			// Show the secondary plot container
			secondaryPlotWrapper.style.display = 'block';

			let pathIdxToLoad = 0; // Default to the first path
			if (currentSecondaryPlotPathIdx !== null) {
				// If a path was already selected, use that one
				pathIdxToLoad = currentSecondaryPlotPathIdx;
			}
			// Load the secondary plot and highlight the path in the primary plot
			loadSecondaryPlot(pathIdxToLoad);
			// Reload the primary plot to fill the space
			if (currentPrimaryPlotData) {
				drawPrimaryPlot(currentPrimaryPlotData);
				currentSecondaryPlotPathIdx = null; // Reset the secondary plot path index
				currentSecondaryPlotDetails = null; // Reset the secondary plot details
			}
			highlightPrimaryPlotPath(pathIdxToLoad);
		} else {
			// Hide the secondary plot container
			secondaryPlotWrapper.style.display = 'none';

			// When hiding the secondary plot, reset the highlight on the primary plot
			if (primaryPlotDiv.innerHTML.trim() && sortedEdgesCache.length > 0) {
				const opacities = Array(sortedEdgesCache.length).fill(0.5);
				const traceIndices = Array.from({ length: sortedEdgesCache.length }, (_, i) => i);
				Plotly.restyle(primaryPlotDiv, { opacity: opacities }, traceIndices);
			}
			// Reload the primary plot to fill the space
			if (currentPrimaryPlotData) {
				drawPrimaryPlot(currentPrimaryPlotData);
				currentSecondaryPlotPathIdx = null; // Reset the secondary plot path index
				currentSecondaryPlotDetails = null; // Reset the secondary plot details
			}
		}
	});

	// Add titles for tooltips on sliders
	numPathsSlider.title = "Controls the number of most important paths to calculate and display.";

	/**
	 * Debounce function to limit the rate at which a function gets called.
	 */
	function debounce(func, delay) {
		let timeout;
		return function(...args) {
			const context = this;
			clearTimeout(timeout);
			timeout = setTimeout(() => func.apply(context, args), delay);
		};
	}

	/**
	 * Gets Plotly layout properties based on the current theme.
	 */
	function getPlotlyThemeLayout() {
		const isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
		return {
			plot_bgcolor: isDark ? '#2a2a2a' : 'white',
			paper_bgcolor: isDark ? '#2a2a2a' : 'white',
			font: {
				color: isDark ? '#e9ecef' : '#212529'
			},
			gridcolor: isDark ? '#444' : '#f0f0f0',
			edgeBorderColor: isDark ? '#000' : '#333333' // Custom property for edge borders
		};
	}

	/**
	 * Redraws existing plots with the current theme or on resize.
	 */
	function handleResizeOrThemeChange() {
		if (currentPrimaryPlotData) {
			drawPrimaryPlot(currentPrimaryPlotData);
			if (currentSecondaryPlotPathIdx !== null) {
				highlightPrimaryPlotPath(currentSecondaryPlotPathIdx);
			}
		}
		if (currentSecondaryPlotDetails && currentSecondaryPlotPathIdx !== null) {
			drawSecondaryPlot(currentSecondaryPlotDetails, currentSecondaryPlotPathIdx);
		}
	}

	/**
	 * Generates an array of distinct colors.
	 */
	function generateCategoryColors(count) {
		const colors = [];
		const isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
		
		for (let i = 0; i < count; i++) {
			const hue = (i / count) * 360;
			if (isDark) {
				// Brighter, more saturated colors for dark theme to be visible
				colors.push(`hsl(${hue}, 100%, 80%)`);
			} else {
				// Vivid but not too bright colors for light theme
				colors.push(`hsl(${hue}, 100%, 60%)`);
			}
		}
		return colors;
	}

	/**
	 * Fetches data and triggers the primary plot rendering.
	 */
	async function runVisualization() {
		loader.style.display = 'block';
		primaryPlotDiv.innerHTML = '';
		secondaryPlotWrapper.style.display = 'none';
		showSecondaryPlotSwitch.checked = false;
		pathSlider = null;
		fullPlotData = null;
		currentPrimaryPlotData = null;
		currentSecondaryPlotPathIdx = null;
		currentSecondaryPlotDetails = null;
		sortedEdgesCache = [];

		try {
			let body = {};
			let suffix = '';
			let delta_time = 1000;
			if (modeSwitch.checked) {
				body = {
					model_name: document.getElementById('model-select-run').value,
					precomputed: false,
					prompt: promptInput.value,
					target: targetTokenInput.value,
					divide_heads: divideHeadsSwitch.checked,
				};
				suffix = '-run';
				delta_time = 10000;
			} else {
				body = {
					model_name: document.getElementById('model-select').value,
					precomputed: true,
					task_name: document.getElementById('task-select').value,
					task: document.getElementById('mode-select').value,
					patch: document.getElementById('patch-select').value,
					divide_heads: divideHeadsSwitch.checked,
				};
				delta_time = 1000;
			}

			// Start the job
			const runResponse = await fetch('/api/run_model', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ ...body })
			});

			if (!runResponse.ok) throw new Error(`HTTP error! status: ${runResponse.status}`);
			const runData = await runResponse.json();
			const { job_id } = runData;

			if (!job_id) throw new Error('Failed to start computation job.');

			// Poll for the result
			let data;
			while (true) {
				await new Promise(resolve => setTimeout(resolve, delta_time)); // Wait 1.5 seconds between polls
				const resultResponse = await fetch(`/api/get_result/${job_id}`);
				if (!resultResponse.ok) throw new Error(`Error polling for result: ${resultResponse.status}`);
				
				const resultData = await resultResponse.json();
				if (resultData.status === 'complete') {
					data = resultData;
					break;
				} else if (resultData.status === 'error') {
					throw new Error(`Computation failed: ${resultData.message}`);
				}
				// If status is 'processing', the loop continues
			}
			console.log("Received data:", data);
			fullPlotData = data['result']; // Store full data { graphData: {...}, pathDetails: {...}, predicted_tokens: [...], predicted_probabilities: [...] }

			// Update predicted token and probability fields
			if (fullPlotData.predicted_tokens && fullPlotData.predicted_probabilities) {
				updatePredictedTokenAndProbability(fullPlotData.predicted_tokens, fullPlotData.predicted_probabilities, suffix);
			}

			// Access num_paths from the nested graphData object
			const numPathsAvailable = fullPlotData.graphData ? fullPlotData.graphData.num_paths : 0;
			numPathsSlider.max = numPathsAvailable > 0 ? numPathsAvailable : 0;
			if (parseInt(numPathsSlider.value) > parseInt(numPathsSlider.max)) {
				numPathsSlider.value = numPathsSlider.max;
			}
			numPathsValue.textContent = numPathsSlider.value;
			
			filterAndDraw();

		} catch (error) {
			console.error("Error fetching model data:", error);
			primaryPlotDiv.innerHTML = `<div class="alert alert-danger">Failed to load visualization: ${error.message}</div>`;
		} finally {
			loader.style.display = 'none';
			window.dispatchEvent(new Event('resize'));
		}
	}

	/**
	 * Updates the predicted token and probability fields in the UI.
	 */
	function updatePredictedTokenAndProbability(predictedTokens, predictedProbabilities, el_id_suffix='') {
		const predictedTokenInput = document.getElementById(`predicted-token${el_id_suffix}`);
		const predictedProbabilityInput = document.getElementById(`predicted-probability${el_id_suffix}`);

		if (predictedTokens && predictedTokens.length > 0 && predictedProbabilities && predictedProbabilities.length > 0) {
			// Set the top predicted token and its probability
			predictedTokenInput.value = predictedTokens[0];
			predictedProbabilityInput.value = predictedProbabilities[0].toFixed(4);

			// Update tooltips with the top 10 predictions
			const tooltipContent = predictedTokens
				.map((token, index) => `${token.replace(/\s/g, '_')}: ${predictedProbabilities[index].toFixed(4)}\n`)
				.join('\n');
			predictedTokenInput.setAttribute('title', tooltipContent);
			predictedProbabilityInput.setAttribute('title', tooltipContent);

			// Reinitialize Bootstrap tooltips to reflect the updated content
			new bootstrap.Tooltip(predictedTokenInput);
			new bootstrap.Tooltip(predictedProbabilityInput);
		} else {
			// Clear fields if no predictions are available
			predictedTokenInput.value = '';
			predictedProbabilityInput.value = '';
			predictedTokenInput.setAttribute('title', 'No predictions available.');
			predictedProbabilityInput.setAttribute('title', 'No predictions available.');
		}
	}

	/**
	 * Filters the full data based on UI controls and triggers a redraw.
	 */
	function filterAndDraw() {
		// Check for fullPlotData and the nested graphData object
		if (!fullPlotData || !fullPlotData.graphData) {
			console.log("No data available to draw.");
			return;
		}

		const numPathsToShow = parseInt(numPathsSlider.value);
		
		// Filter edges from the nested graphData object
		const filteredEdges = fullPlotData.graphData.edges.filter(edge => edge.path_idx < numPathsToShow);

		const involvedNodeIds = new Set();
		filteredEdges.forEach(edge => {
			involvedNodeIds.add(edge.source);
			involvedNodeIds.add(edge.target);
		});

		// Update nodes from the nested graphData object
		fullPlotData.graphData.nodes.forEach(node => {
			node.involved = involvedNodeIds.has(node.id);
		});

		// Create a new data object for plotting using the nested graphData
		currentPrimaryPlotData = {
			...fullPlotData.graphData,
			nodes: fullPlotData.graphData.nodes,
			edges: filteredEdges,
			num_paths: numPathsToShow,
		};

		secondaryPlotWrapper.style.display = 'none';
		showSecondaryPlotSwitch.checked = false;
		currentSecondaryPlotPathIdx = null;
		currentSecondaryPlotDetails = null;

		drawPrimaryPlot(currentPrimaryPlotData);

		if (currentPrimaryPlotData.num_paths > 0) {
			showSecondaryPlotSwitch.checked = true;
			secondaryPlotWrapper.style.display = 'block';
			loadSecondaryPlot(0);
			highlightPrimaryPlotPath(0);
			window.dispatchEvent(new Event('resize'));
		}
	}

	/**
	 * Creates the path slider control and injects it into a given container.
	 */
	function createPathSlider(container, numPaths, initialValue) {
		container.innerHTML = ''; // Clear previous

		const wrapper = document.createElement('div');
		wrapper.className = 'd-flex justify-content-between align-items-center mb-2';

		const sliderGroup = document.createElement('div');
		sliderGroup.className = 'd-flex align-items-center flex-grow-1';

		const label = document.createElement('label');
		label.htmlFor = 'path-slider';
		label.className = 'form-label small fw-bold mb-0 me-2';
		label.textContent = 'Path index:';

		pathSlider = document.createElement('input');
		pathSlider.type = 'range';
		pathSlider.className = 'form-range flex-grow-1';
		pathSlider.id = 'path-slider';
		pathSlider.min = 0;
		pathSlider.max = numPaths > 0 ? numPaths - 1 : 0;
		pathSlider.value = initialValue;
		pathSlider.title = "Select a specific path to view its details in the secondary plot below.";

		const textInput = document.createElement('input');
		textInput.type = 'number';
		textInput.className = 'form-control form-control-sm ms-2';
		textInput.style.width = '60px';
		textInput.min = 0;
		textInput.max = numPaths > 0 ? numPaths - 1 : 0;
		textInput.value = initialValue;
		textInput.title = "Enter path index directly.";


		sliderGroup.append(label, pathSlider, textInput);
		wrapper.append(sliderGroup);
		container.append(wrapper);

		const debouncedUpdate = debounce((pathIdx) => {
			if (pathIdx !== currentSecondaryPlotPathIdx) {
				loadSecondaryPlot(pathIdx);
				highlightPrimaryPlotPath(pathIdx);
			}
		}, 500); // 500ms delay

		// Add event listener to the newly created slider
		pathSlider.addEventListener('input', (event) => {
			const pathIdx = parseInt(event.target.value, 10);
			textInput.value = pathIdx; // Sync text input
			debouncedUpdate(pathIdx);
		});

		// Add event listener to the text input
		textInput.addEventListener('input', (event) => {
			let pathIdx = parseInt(event.target.value, 10);
			if (isNaN(pathIdx) || pathIdx < 0) pathIdx = 0;
			if (pathIdx > parseInt(pathSlider.max)) pathIdx = parseInt(pathSlider.max);
			pathSlider.value = pathIdx;
			debouncedUpdate(pathIdx);
		});
	}

	/**
	* Renders the primary plot using Plotly.js.
	*/
	function drawPrimaryPlot(data) {
		const { nodes, edges, max_abs_weight, num_paths, n_positions, n_layers, n_heads } = data;
		const colorScheme = colorSchemeSelect.value;
		const themeLayout = getPlotlyThemeLayout();
		const divideHeads = divideHeadsSwitch.checked;
		const strongestEdgesOnTop = edgePrioritySwitch.checked;
		const plotWidth = primaryPlotDiv.clientWidth || 800;
		const plotHeight = primaryPlotDiv.clientHeight || 750;
		const cellWidth = plotWidth / (n_positions + 1);
		const cellHeight = plotHeight / (n_layers + 3);
		const baseNodeSize = Math.min(cellWidth, cellHeight) * 0.3;
		const attnNodeSize = divideHeads ? Math.max(4, Math.min(baseNodeSize, 1.3*baseNodeSize / Math.ceil(n_heads/4))) : baseNodeSize;

				const yCoords = nodes.map(n => n.y);
				const minY = Math.min(...yCoords);
				const maxY = Math.max(...yCoords);
				const maxX = Math.max(...nodes.map(n => n.x));
				const minX = Math.min(...nodes.map(n => n.x));

				const nodesByComponent = {};
				nodes.forEach(node => {
					const componentType = node.cmpt === 'sa' ? 'attn' : node.cmpt;
					if (!nodesByComponent[componentType]) nodesByComponent[componentType] = [];
					nodesByComponent[componentType].push(node);
				});

		// xTicks are in between the embedding nodes
		const xTickVals = [];
		const sortedEmbedNodeCoords = nodes.filter(n => n.cmpt === 'emb').map(n => n.x).sort((a, b) => a - b);
		if (sortedEmbedNodeCoords.length > 1) {
			for (let i = 0; i < sortedEmbedNodeCoords.length - 1; i++) {
				xTickVals.push((sortedEmbedNodeCoords[i] + sortedEmbedNodeCoords[i+1]) / 2);
			}
		}
		const positionWidth = sortedEmbedNodeCoords.length > 1 ? (sortedEmbedNodeCoords[1] - sortedEmbedNodeCoords[0]) : cellWidth;
		
		const xAxis = {
			showgrid: true,
			gridcolor: themeLayout.gridcolor,
			zeroline: false,
			gridwidth: 2,
			tickmode: 'array',
			tickvals: xTickVals,
			showticklabels: false,
		};
		
		const yTickVals = [];
		const sortedMLPYCoords = [...new Set(nodes.filter(n => n.cmpt === 'mlp').map(n => n.y))].sort((a, b) => a - b);
		layerHeight = sortedMLPYCoords.length > 0 ? (sortedMLPYCoords[1] - sortedMLPYCoords[0]) : cellHeight;
		const yTickTexts = [];
		const lmhNode = nodes.find(n => n.cmpt === 'lmh');
		if (lmhNode) {
			yTickVals.push(lmhNode.y - layerHeight*0.5);
			yTickTexts.push('LMH');
		}
		for (let i = n_layers - 1; i >= 0; i--) {
			const layerNode = nodes.find(n => n.layer === i && n.cmpt === 'mlp');
			if (layerNode) {
				yTickVals.push(layerNode.y - layerHeight*0.2);
				yTickTexts.push(`L${i}`);
			}
		}
		const embNode = nodes.find(n => n.cmpt === 'emb');
		if (embNode) {
			yTickVals.push(embNode.y - layerHeight*0.5);
			yTickTexts.push('EMB');
		}
		console.log(`Y Tick Vals: ${yTickVals}, layerHeight: ${layerHeight}, cellHeight: ${cellHeight}`);
		const yAxis = {
			showgrid: true,
			gridcolor: themeLayout.gridcolor,
			gridwidth: 2,
			zeroline: false,
			tickmode: 'array',
			tickvals: yTickVals,
			ticktext: yTickTexts,
			range: [maxY + layerHeight, minY - layerHeight],
			tickson: 'boundaries',
			ticklabelposition: 'outside bottom' // Show tick labels below the axis
		};

		// --- 1. Create Node Traces (split by involvement) ---
		const baseNodeSizes = {};
		const uninvolvedNodeTraces = [];
		const involvedNodeTraces = [];
		
		Object.entries(nodesByComponent).forEach(([cmpt, groupNodes]) => {
			const involvedNodes = groupNodes.filter(n => n.involved);
			const uninvolvedNodes = groupNodes.filter(n => !n.involved);
		
			let style = { symbol: 'circle', size: baseNodeSize };
			switch (cmpt) {
				case 'emb': style = { symbol: 'square', size: baseNodeSize * 2 }; break;
				case 'lmh': style = { symbol: 'square', size: baseNodeSize * 2 }; break;
				case 'mlp': style = { symbol: 'square', size: baseNodeSize }; break;
				case 'attn': style = { symbol: 'circle', size: attnNodeSize }; break;
				default: style = { symbol: 'diamond', size: baseNodeSize }; break;
			}
			baseNodeSizes[cmpt] = style.size;
		
			// --- A. Uninvolved Node Trace (drawn first, in the background) ---
			if (uninvolvedNodes.length > 0) {
				let groupedNodes = uninvolvedNodes;
				if (!divideHeads && cmpt === 'attn') {
					// Group nodes by layer and position if not dividing heads
					const groupedMap = new Map();
					uninvolvedNodes.forEach(node => {
						const key = `L${node.layer}_P${node.position}`;
						if (!groupedMap.has(key)) {
							groupedMap.set(key, { ...node, head: undefined, count: 0 });
						}
						groupedMap.get(key).count += 1;
					});
					groupedNodes = Array.from(groupedMap.values());
				}

				let textContent;
				switch (cmpt) {
					case 'emb': textContent = groupedNodes.map(n => (n.label || 'test').replace(/ /g, ' ')); break;
					default: textContent = groupedNodes.map(() => ''); break;
				}
				uninvolvedNodeTraces.push({
					x: groupedNodes.map(n => n.x),
					y: groupedNodes.map(n => n.y),
					text: textContent,
					mode: 'markers+text',
					type: 'scatter',
					textfont: {
						size: Math.min(style.size, baseNodeSize) * 0.7,
						color: `rgba(${themeLayout.font.color}, 0.6)`
					},
					marker: {
						...style,
						color: `rgba(${themeLayout.font.color === '#e9ecef' ? '233, 236, 239' : '66, 74, 82'}, 0.25)`,
						line: { width: 1.5, color: themeLayout.font.color },
						opacity: 0.2,
					},
					hoverinfo: 'text',
					hovertext: groupedNodes.map(n => {
						const { layer, position, count } = n;
						if (cmpt === 'attn') return `ATTN L${layer} P${position} (${count} heads)`.trim();
						if (cmpt === 'mlp') return `MLP L${layer} P${position}`;
						if (cmpt === 'emb') return `Embedding P${position}`;
						if (cmpt === 'lmh') return `Output P${position}`;
						return `${cmpt.toUpperCase()} L${layer} P${position}`;
					}),
					meta: { type: 'node', cmpt: cmpt, involved: false }
				});
			}
		
			// --- B. Involved Node Trace (drawn last, on top) ---
			if (involvedNodes.length > 0) {
				let textContent;
				switch (cmpt) {
					case 'emb': textContent = involvedNodes.map(n => (n.label || '').replace(/ /g, ' ')); break;
					case 'lmh': textContent = involvedNodes.map(n => (n.label || '').replace(/ /g, ' ')); break;
					default: textContent = involvedNodes.map(n => n.label); break;
				}
		
				involvedNodeTraces.push({
					x: involvedNodes.map(n => n.x),
					y: involvedNodes.map(n => n.y),
					text: textContent,
					mode: 'markers+text',
					type: 'scatter',
					textfont: {
						size: Math.min(style.size, baseNodeSize) * 0.7,
						color: themeLayout.font.color
					},
					marker: {
						...style,
						color: themeLayout.plot_bgcolor,
						line: { width: 1.5, color: themeLayout.font.color },
						opacity: 1.0,
					},
					hoverinfo: 'text',
					hovertext: involvedNodes.map(n => {
						const { layer, position, head } = n;
						if (cmpt === 'attn') return `ATTN L${layer} ${divideHeads ? `H${head}` : ''} P${position}`.trim();
						if (cmpt === 'mlp') return `MLP L${layer} P${position}`;
						if (cmpt === 'emb') return `Embedding P${position}`;
						if (cmpt === 'lmh') return `Output P${position}`;
						return `${cmpt.toUpperCase()} L${layer} P${position}`;
					}),
					meta: { type: 'node', cmpt: cmpt, involved: true }
				});
			}
		});

		// --- 2. Create Edge Traces (drawn between node layers) ---
		const edgeTraces = [];
		const pathIndexColors = generateCategoryColors(num_paths);
		const inputPosColors = generateCategoryColors(n_positions);

		sortedEdgesCache = [...edges].sort((a, b) => {
			const weightA = Math.abs(a.weight);
			const weightB = Math.abs(b.weight);
			return strongestEdgesOnTop ? weightA - weightB : weightB - weightA;
		});
		
		const nodeMap = new Map(nodes.map(n => [n.id, n]));

		drawPrimaryPlot.parallelEdgeCounts = {};
		sortedEdgesCache.forEach(edge => {
			const sourceNode = nodeMap.get(edge.source);
			const targetNode = nodeMap.get(edge.target);
			
			let key;
			if (!divideHeadsSwitch.checked && (sourceNode?.cmpt === 'attn' || targetNode?.cmpt === 'attn')) {
				// When not dividing heads and dealing with attention nodes create a key that ignores head information
				const sourceId = sourceNode ? `${sourceNode.cmpt}_l${sourceNode.layer}_p${sourceNode.position}` : edge.source;
				const targetId = targetNode ? `${targetNode.cmpt}_l${targetNode.layer}_p${targetNode.position}` : edge.target;
				key = `${sourceId}-${targetId}`;
			} else {
				// Otherwise use the original edge IDs
				key = `${edge.source}-${edge.target}`;
			}
			
			drawPrimaryPlot.parallelEdgeCounts[key] = (drawPrimaryPlot.parallelEdgeCounts[key] || 0) + 1;
		});
		console.log("Parallel edge counts:", drawPrimaryPlot.parallelEdgeCounts);
	
		drawPrimaryPlot.parallelEdgeDrawn = {};

		const baseEdgeWidths = [];
		
		sortedEdgesCache.forEach((edge, index) => {
			const sourceNode = nodeMap.get(edge.source);
			const targetNode = nodeMap.get(edge.target);
			if (!sourceNode || !targetNode) return;

			let color;
			const normWeight = 0.5 + 0.5 * (edge.weight / max_abs_weight);
			switch (colorScheme) {
				case 'path_weight':
					let r, g, b;
					if (edge.weight >= 0) {
						// Positive weights: iceblue (rgba(205, 255, 180, 1)) to bright blue (rgba(205, 255, 0, 1))
						r = Math.round(205 * (1 - normWeight) + 205 * normWeight);
						g = Math.round(255 * (1 - normWeight) + 255 * normWeight);
						b = Math.round(180 * (1 - normWeight) + 0 * normWeight);
					} else {
						// Negative weights: light orange (rgba(255, 100, 100, 1)) to dark red (rgba(180, 100, 0, 1))
						r = Math.round(255 * (1 - normWeight) + 180 * normWeight);
						g = Math.round(100 * (1 - normWeight) + 100 * normWeight);
						b = Math.round(100 * (1 - normWeight) + 0 * normWeight);
					}

					color = `rgba(${r}, ${g}, ${b}, 1)`;
					break;
				case 'input_position':
					color = inputPosColors[edge.start_pos % n_positions];
					break;
				case 'path_index':
				default:
					color = pathIndexColors[edge.path_idx % num_paths];
					break;
			}
			const minWidth = 1;
			const maxWidth = 30;
			const edgeWidth = minWidth + 2 * (Math.abs(normWeight-0.5) * (maxWidth - minWidth));
			baseEdgeWidths.push(edgeWidth);

			const parallelEdgeDrawn = drawPrimaryPlot.parallelEdgeDrawn;
			const parallelEdgeCounts = drawPrimaryPlot.parallelEdgeCounts;
			let edgeKey;
			if (!divideHeadsSwitch.checked && (sourceNode?.cmpt === 'attn' || targetNode?.cmpt === 'attn')) {
				// When not dividing heads and dealing with attention nodes create a key that ignores head information
				const sourceId = sourceNode ? `${sourceNode.cmpt}_l${sourceNode.layer}_p${sourceNode.position}` : edge.source;
				const targetId = targetNode ? `${targetNode.cmpt}_l${targetNode.layer}_p${targetNode.position}` : edge.target;
				edgeKey = `${sourceId}-${targetId}`;
			} else {
				// Otherwise use the original edge IDs
				edgeKey = `${edge.source}-${edge.target}`;
			}
			if (!(edgeKey in parallelEdgeDrawn)) parallelEdgeDrawn[edgeKey] = 0;
			const count = parallelEdgeDrawn[edgeKey];
			const totalEdges = parallelEdgeCounts[edgeKey] || 1;

			const dx = targetNode.x - sourceNode.x;
			const dy = targetNode.y - sourceNode.y;
			const mx = (sourceNode.x + targetNode.x) / 2;
			const my = (sourceNode.y + targetNode.y) / 2;

			let radMagnitude = 0;
			if (totalEdges > 1) {
				if (count > 0) {
					const direction = count % 2 === 0 ? -1 : 1;
					radMagnitude = direction * (Math.ceil(count / 2));
				}
			}
			parallelEdgeDrawn[edgeKey] += 1;

			let perpX = -dy;
			let perpY = dx*(maxY - minY)/(maxX - minX);
			const offset = 0.0025 * radMagnitude * Math.min(positionWidth, layerHeight);
			const ctrlX = mx + perpX * offset;
			const ctrlY = my + perpY * offset;

			const isQueryEdge = targetNode.in_type === 'query' && (targetNode.cmpt === 'sa' || targetNode.cmpt === 'attn');
			const lineShape = { shape: 'spline', smoothing: 1, dash: isQueryEdge ? 'dot' : 'solid' };
			const borderLineShape = { shape: 'spline', smoothing: 1, dash: 'solid' };
			const borderWidth = 2;

			edgeTraces.push({
				x: count > 1? [sourceNode.x, ctrlX, targetNode.x] : [sourceNode.x, (sourceNode.x + ctrlX)/2, ctrlX, (targetNode.x + ctrlX)/2, targetNode.x],
				y: count > 1? [sourceNode.y, ctrlY, targetNode.y] : [sourceNode.y, (sourceNode.y + ctrlY)/2, ctrlY, (targetNode.y + ctrlY)/2, targetNode.y],
				mode: 'lines',
				line: { ...borderLineShape, color: themeLayout.edgeBorderColor, width: edgeWidth + borderWidth * 2 },
				type: 'scatter', hoverinfo: 'none',
				opacity: 0.2,
				meta: { type: 'border' }
			});

			edgeTraces.push({
				x: count > 1? [sourceNode.x, ctrlX, targetNode.x] : [sourceNode.x, (sourceNode.x + ctrlX)/2, ctrlX, (targetNode.x + ctrlX)/2, targetNode.x],
				y: count > 1? [sourceNode.y, ctrlY, targetNode.y] : [sourceNode.y, (sourceNode.y + ctrlY)/2, ctrlY, (targetNode.y + ctrlY)/2, targetNode.y],
				mode: 'lines',
				line: { ...lineShape, color: color, width: edgeWidth },
				type: 'scatter', hoverinfo: 'text',
				hovertext: `Path: ${edge.path_idx}<br>Weight: ${edge.weight.toFixed(4)}<br>Click to view details`,
				customdata: Array(3).fill(edge.path_idx),
				opacity: 0.95,
				meta: { type: 'edge' }
			});
		});
		
		// --- 3. Assemble All Traces in Order ---
		const traces = [
			...uninvolvedNodeTraces,
			...edgeTraces,
			...involvedNodeTraces
		];
		
		const layout = {
			...themeLayout,
			showlegend: false,
			xaxis: xAxis,
			yaxis: yAxis,
			margin: { t: 20, b: 40, l: 80, r: 20 },
			hovermode: 'closest',
			autosize: true,
		};

		Plotly.newPlot(primaryPlotDiv, traces, layout, {responsive: true});

		// Add zoom event listener to scale elements
		primaryPlotDiv.on('plotly_relayout', function(eventData) {
			if (eventData['xaxis.range[0]'] !== undefined || eventData['yaxis.range[0]'] !== undefined) {
				const xRange = primaryPlotDiv.layout.xaxis.range;
				const yRange = primaryPlotDiv.layout.yaxis.range;
				const initialXRange = drawPrimaryPlot.initialXRange;
				const initialYRange = drawPrimaryPlot.initialYRange;
				
				const xZoom = (initialXRange[1] - initialXRange[0]) / (xRange[1] - xRange[0]);
				const yZoom = Math.abs((initialYRange[1] - initialYRange[0]) / (yRange[1] - yRange[0]));
				const zoomFactor = Math.min(xZoom, yZoom);

				const updateData = {};
				const traceIndices = [];
				
				primaryPlotDiv.data.forEach((trace, i) => {
					if (trace.meta?.type === 'node') {
						const baseSize = drawPrimaryPlot.baseNodeSizes[trace.meta.cmpt];
						const scaledSize = baseSize * zoomFactor;
						if (!updateData['marker.size']) updateData['marker.size'] = [];
						if (!updateData['textfont.size']) updateData['textfont.size'] = [];
						updateData['marker.size'][i] = scaledSize;
						if (trace.meta.involved) {
							updateData['textfont.size'][i] = scaledSize * 0.7;
						}
						traceIndices.push(i);
					} else if (trace.meta?.type === 'edge' || trace.meta?.type === 'border') {
						const edgeIndex = Math.floor((i - uninvolvedNodeTraces.length) / 2);
						const baseWidth = drawPrimaryPlot.baseEdgeWidths[edgeIndex];
						const scaledWidth = baseWidth * zoomFactor;
						
						if (!updateData['line.width']) updateData['line.width'] = [];
						if (trace.meta.type === 'border') {
							updateData['line.width'][i] = scaledWidth + 2;
						} else {
							updateData['line.width'][i] = scaledWidth;
						}
						traceIndices.push(i);
					}
				});

				if (traceIndices.length > 0) {
					Plotly.restyle(primaryPlotDiv, updateData, traceIndices);
				}
			}
		});

		primaryPlotDiv.on('plotly_click', function(eventData) {
			const point = eventData.points[0];
			const pathIdx = point?.customdata;
			console.log("Clicked on path index:", pathIdx);
			if (pathIdx !== undefined) {
				showSecondaryPlotSwitch.checked = true;
				if (pathSlider) {
					pathSlider.value = pathIdx;
					pathSlider.dispatchEvent(new Event('input', { bubbles: true }));
					pathSlider.dispatchEvent(new Event('change', { bubbles: true }));
				} else {
					highlightPrimaryPlotPath(pathIdx);
					loadSecondaryPlot(pathIdx);
				}
			} else {
				const opacities = Array(sortedEdgesCache.length).fill(0.5);
				const traceIndices = Array.from({ length: sortedEdgesCache.length }, (_, i) => i);
				Plotly.restyle(primaryPlotDiv, { opacity: opacities }, traceIndices);
			}
			window.dispatchEvent(new Event('resize'));
		});

		primaryPlotDiv.on('plotly_hover', function(eventData) {
			const point = eventData.points[0];
			if (point?.customdata !== undefined || point?.hovertext) {
				primaryPlotDiv.style.cursor = 'pointer';
			}
		});

		primaryPlotDiv.on('plotly_unhover', function() {
			primaryPlotDiv.style.cursor = 'default';
		});

		drawPrimaryPlot.baseEdgeWidths = baseEdgeWidths;
		drawPrimaryPlot.baseNodeSizes = baseNodeSizes;
		drawPrimaryPlot.base = baseNodeSize;
		drawPrimaryPlot.positionWidth = positionWidth;
		drawPrimaryPlot.layerHeight = layerHeight;
		drawPrimaryPlot.initialXRange = [Math.min(...nodes.map(n => n.x)) - positionWidth, Math.max(...nodes.map(n => n.x)) + positionWidth];
		drawPrimaryPlot.initialYRange = [maxY + layerHeight, minY - layerHeight];
	}
	/**
	* Highlights a specific path on the primary plot by adjusting opacities.
	*/
	function highlightPrimaryPlotPath(pathIdx) {
		if (!primaryPlotDiv.innerHTML.trim() || sortedEdgesCache.length === 0) return;
		const themeLayout = getPlotlyThemeLayout();
		const showAllEdges = allEdgesSwitch.checked;

		const borderOpacities = [];
		const mainOpacities = [];
		const borderColors = [];
		const borderWidths = [];
		const numUninvolvedNodeTraces = primaryPlotDiv.data.filter(t => t.meta?.type === 'node' && !t.meta.involved).length;

		sortedEdgesCache.forEach(edge => {
			const isSelected = edge.path_idx === pathIdx;
			
			mainOpacities.push(isSelected ? 0.95 : (showAllEdges ? 0.05 : 0.5));
			borderOpacities.push(isSelected ? 0.7 : (showAllEdges ? 0.025 : 0.3));
			borderColors.push(isSelected ? '#8B0000' : themeLayout.edgeBorderColor);
			borderWidths.push(isSelected ? 5 : 1);
		});

		const borderTraceIndices = Array.from({ length: sortedEdgesCache.length }, (_, i) => numUninvolvedNodeTraces + 2 * i);
		const mainTraceIndices = Array.from({ length: sortedEdgesCache.length }, (_, i) => numUninvolvedNodeTraces + 2 * i + 1);

		Plotly.restyle(primaryPlotDiv, {
			'opacity': borderOpacities,
			'line.color': borderColors,
			'line.width': borderWidths.map((w, i) => (drawPrimaryPlot.baseEdgeWidths[i] + w * 2))
		}, borderTraceIndices);

		Plotly.restyle(primaryPlotDiv, {
			'opacity': mainOpacities
		}, mainTraceIndices);
	}

	/**
	 * Loads data for the secondary plot when a path is clicked or selected.
	 */
	function loadSecondaryPlot(pathIdx) { // The function is no longer async
		currentSecondaryPlotPathIdx = pathIdx;
		if (pathSlider) {
			pathSlider.value = pathIdx;
		}
		secondaryPlotWrapper.style.display = 'block';

		// --- ENTIRE LOGIC REPLACED ---
		// No more fetching. Get details directly from the stored data.
		if (fullPlotData && fullPlotData.pathDetails && fullPlotData.pathDetails[pathIdx]) {
			const details = {
				path: fullPlotData.pathDetails[pathIdx].path,
				weight: fullPlotData.pathDetails[pathIdx].weight,
			};
			currentSecondaryPlotDetails = details; // Store details
			drawSecondaryPlot(details, pathIdx);
		} else {
			// This case might happen if data is inconsistent, which is unlikely but good to handle.
			const errorMessage = `Failed to load path details: Data not found for path index ${pathIdx}.`;
			console.error(errorMessage);
			secondaryPlotDiv.innerHTML = `<div class="alert alert-danger">${errorMessage}</div>`;
			currentSecondaryPlotDetails = null; // Clear on error
		}
	}
	/**
	 * Renders the secondary plot with path and token details.
	 */
	function drawSecondaryPlot(details, pathIdx) {
		const path_data = details.path;
		const weight = details.weight;
		secondaryPlotDiv.innerHTML = '';
	
		const container = document.createElement('div');
		container.className = 'd-flex h-100';
	
		const mainContent = document.createElement('div');
		mainContent.className = 'flex-grow-1 d-flex flex-column';
		mainContent.style.overflow = 'hidden';
	
		const pathSliderContainer = document.createElement('div');
		pathSliderContainer.className = 'px-2 pt-2';
		createPathSlider(pathSliderContainer, currentPrimaryPlotData.num_paths, pathIdx);

		const pathHeader = document.createElement('h6');
		pathHeader.textContent = `Path ${pathIdx} Components (w: ${weight.toFixed(3)})`;
		pathHeader.className = 'px-2 pt-2 text-center border-top';
	
		const pathVizContainer = document.createElement('div');
		pathVizContainer.className = 'd-flex flex-grow-1';
		pathVizContainer.style.height = 'calc(50% - 20px)';
	
		const pathViz = document.createElement('div');
		pathViz.id = 'path-visualization';
		pathViz.className = 'flex-grow-1';
	
		const pathNodeSlider = document.createElement('input');
		pathNodeSlider.type = 'range';
		pathNodeSlider.min = 0;
		pathNodeSlider.max = path_data.length > 0 ? path_data.length - 1 : 0;
		pathNodeSlider.value = 0;
		pathNodeSlider.step = 1;
		pathNodeSlider.className = 'path-node-slider';
		pathNodeSlider.style.writingMode = 'vertical-lr';
		pathNodeSlider.title = "Slide to inspect the token distribution at different nodes in this path.";

		// Inject CSS for the slider track and thumb since it requires pseudo-selectors
		// which cannot be styled inline.
		const styleId = 'path-node-slider-styles';
		if (!document.getElementById(styleId)) {
			const style = document.createElement('style');
			style.id = styleId;
			// Styles inspired by the provided image, adapted for a vertical slider.
			style.textContent = `
				.path-node-slider {
					-webkit-appearance: none;
					appearance: none;
					width: 8px; /* Width of the track area */
					background: transparent; /* Remove default background */
					cursor: pointer;
					padding: 0;
					margin: 0 10px; /* Add some margin */
				}

				/* Track */
				.path-node-slider::-webkit-slider-runnable-track {
					width: 8px;
					height: 100%;
					background: #adb5bd; /* Grey track color from image */
					border-radius: 4px;
				}
				.path-node-slider::-moz-range-track {
					width: 8px;
					height: 100%;
					background: #adb5bd;
					border-radius: 4px;
				}

				/* Thumb */
				.path-node-slider::-webkit-slider-thumb {
					-webkit-appearance: none;
					appearance: none;
					height: 20px; /* Thumb height */
					width: 20px;  /* Thumb width */
					background: #0d6efd; /* Blue thumb color from image */
					border-radius: 50%;
					border: none;
					margin-top: -6px; /* Center thumb on the track */
				}
				.path-node-slider::-moz-range-thumb {
					height: 20px;
					width: 20px;
					background: #0d6efd;
					border-radius: 50%;
					border: none;
				}
			`;
			document.head.appendChild(style);
		}
	
		pathVizContainer.append(pathViz, pathNodeSlider);
	
		const tokenHeader = document.createElement('h6');
		tokenHeader.id = 'token-header';
		tokenHeader.textContent = 'Top Tokens at Node';
		tokenHeader.className = 'px-2 pt-2 text-center border-top';
	
		const tokenChart = document.createElement('div');
		tokenChart.id = 'token-chart';
		tokenChart.style.height = 'calc(50% - 20px)';
	
		mainContent.append(pathSliderContainer, pathHeader, pathVizContainer, tokenHeader, tokenChart);
		container.append(mainContent);
		secondaryPlotDiv.append(container);
	
		const reversedPathData = [...path_data].reverse();
		reversedPathData.forEach((node, index) => {
			node.id = `node_${index}_${node.cmpt}_l${node.layer}_p${node.position}`;
		});
	
		const tokenMap = {};
		reversedPathData.forEach(node => {
			tokenMap[node.id] = {
				topk_strtokens: node.tokens,
				topk_probs: node.probs ?? []
			};
		});
		console.log("Token Map:", tokenMap);

	
		const { nodeStyles, edgeTraces } = drawPathVisualization(pathViz, reversedPathData, tokenChart, tokenHeader, tokenMap, pathNodeSlider);
	
		if (reversedPathData.length > 0) {
			const firstNode = reversedPathData[0];
			tokenHeader.textContent = `Top Tokens at ${getNodeLabel(firstNode)}`;
			drawTokenChart(tokenChart, tokenMap[firstNode.id]);
			// Initial highlight
			const initialColors = nodeStyles.map((s, i) => i === 0 ? '#FF0000' : s.color);
			Plotly.restyle(pathViz, { 'marker.color': [initialColors] }, edgeTraces.length);
		}
	
		pathNodeSlider.addEventListener('input', (event) => {
			const nodeIndex = parseInt(event.target.value, 10);
			const selectedNode = reversedPathData[nodeIndex];
			tokenHeader.textContent = `Top Tokens at ${getNodeLabel(selectedNode)}`;
			drawTokenChart(tokenChart, tokenMap[selectedNode.id]);
	
			const highlightColor = '#FF0000'; // Bright Red
			const nodeColors = nodeStyles.map((s, i) => i === nodeIndex ? highlightColor : s.color);
			Plotly.restyle(pathViz, { 'marker.color': [nodeColors] }, edgeTraces.length);
		});
	}

	/**
	 * Generates a descriptive label for a node.
	 */
	function getNodeLabel(node) {
		const cmpt = node.cmpt || node.class_name;
		if (cmpt === 'attn' || cmpt === 'sa') return `ATTN L${node.layer ?? '?'} H${node.head ?? '?'} P${node.position ?? '?'}`;
		if (cmpt === 'mlp') return `MLP L${node.layer ?? '?'} P${node.position ?? '?'}`;
		if (cmpt === 'emb') return `EMB P${node.position ?? '?'}`;
		if (cmpt === 'lmh') return `LMH P${node.position ?? '?'}`;
		return `${cmpt.toUpperCase()} P${node.position ?? '?'}`;
	}
	
	function drawPathVisualization(container, pathData, tokenChartContainer, tokenHeader, allTokens, slider) {
		const nodeLabels = pathData.map(getNodeLabel);
		const yPositions = Array.from({length: pathData.length}, (_, i) => i * 2);
		const themeLayout = getPlotlyThemeLayout();
		
		let edgeTraces = [];
		for (let i = 1; i < pathData.length; i++) {
			const targetNode = pathData[i-1];
			const isQueryEdge = targetNode.in_type === 'query' && (targetNode.cmpt === 'sa' || targetNode.cmpt === 'attn');

			edgeTraces.push({
				x: [0, 0], y: [yPositions[i-1], yPositions[i]], mode: 'lines',
				line: { 
					color: '#607D8B', 
					width: 2,
					dash: isQueryEdge ? 'dot' : 'solid'
				}, 
				hoverinfo: 'none', type: 'scatter'
			});
		}
		
		const nodeStyles = pathData.map(d => {
			const cmpt = d.cmpt || d.class_name;
			switch(cmpt) {
				case 'attn':
				case 'sa':
					return { symbol: 'circle', color: '#B0C4DE' }; // Light Steel Blue
				case 'mlp': return { symbol: 'square', color: '#90EE90' }; // Light Green
				case 'emb': return { symbol: 'square', color: '#FFDAB9' }; // PeachPuff
				case 'lmh': return { symbol: 'square', color: '#FFA07A' }; // LightSalmon
				default: return { symbol: 'diamond', color: '#D3D3D3' }; // Light Grey
			}
		});

		const nodeTrace = {
			x: Array(pathData.length).fill(0), y: yPositions, mode: 'markers+text',
			marker: {
				size: 20,
				symbol: nodeStyles.map(s => s.symbol),
				color: nodeStyles.map(s => s.color),
				line: { color: '#555555', width: 2 }
			},
			text: nodeLabels, textposition: 'middle right', hoverinfo: 'text', type: 'scatter'
		};
		
		const layout = {
			...themeLayout,
			showlegend: false,
			margin: { t: 10, b: 10, l: 10, r: 10 },
			autosize: true,
			xaxis: {
				visible: false,
				range: [-0.5, 4.5]
			},
			yaxis: {
				visible: false,
				range: [-1, Math.max(0, yPositions[yPositions.length - 1] + 1)],
				autorange: 'reversed'
			}
		};
		
		Plotly.newPlot(container, [...edgeTraces, nodeTrace], layout, {responsive: true, displayModeBar: false});

		container.on('plotly_click', (eventData) => {
			const point = eventData.points[0];
			if (point && point.curveNumber === edgeTraces.length) {
				const clickedNodeData = pathData[point.pointNumber];
				tokenHeader.textContent = `Top Tokens at ${getNodeLabel(clickedNodeData)}`;
				drawTokenChart(tokenChartContainer, allTokens[clickedNodeData.id]);
				slider.value = point.pointNumber;
				
				const highlightColor = '#FF0000';
				const currentColors = nodeStyles.map((s, i) => i === point.pointNumber ? highlightColor : s.color);
				Plotly.restyle(container, { 'marker.color': [currentColors] }, edgeTraces.length);
			}
		});

		container.on('plotly_hover', (eventData) => {
			if (eventData.points[0]?.curveNumber === edgeTraces.length) container.style.cursor = 'pointer';
		});
		container.on('plotly_unhover', () => { container.style.cursor = 'default'; });

		return { nodeStyles, edgeTraces };
	}
	
	/**
	 * Draws the token chart part of the secondary plot
	 */
	function drawTokenChart(container, tokens) {
		if (tokens?.topk_strtokens?.length > 0 && tokens?.topk_probs?.length > 0) {
			const themeLayout = getPlotlyThemeLayout();
			const tokenData = {
				y: [...tokens.topk_strtokens].reverse(),
				x: [...tokens.topk_probs].reverse(),
				type: 'bar', orientation: 'h', marker: { color: '#4682B4' },
				text: [...tokens.topk_probs].reverse().map(p => p.toFixed(4)),
				textposition: 'auto'
			};
			
			const layout = {
				...themeLayout,
				margin: { t: 10, b: 40, l: 100, r: 20 },
				xaxis: { title: 'Probability', range: [0, Math.max(...tokens.topk_probs) * 1.1] },
				yaxis: { title: '', automargin: true }
			};
			
			Plotly.newPlot(container, [tokenData], layout, {responsive: true, displayModeBar: false});
		} else {
			container.innerHTML = '<div class="alert alert-info p-2 m-2">No token data available for this node.</div>';
		}
	}
	
	// --- Page Setup ---
	// Initialize Bootstrap Tooltips after the window loads to ensure bootstrap is defined
	window.addEventListener('load', () => {
		const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
		tooltipTriggerList.forEach(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
	});

	// Dark Mode Toggle
	const themeToggle = document.getElementById('theme-toggle');
	const htmlEl = document.documentElement;
	
	themeToggle.addEventListener('click', () => {
		const currentTheme = htmlEl.getAttribute('data-bs-theme');
		const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
		htmlEl.setAttribute('data-bs-theme', newTheme);
		themeToggle.querySelector('i').className = newTheme === 'dark' ? 'bi bi-moon-fill' : 'bi bi-sun-fill';
		window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme: newTheme } }));
	});

	// Settings Panel Logic
	const settingsPanel = document.getElementById('settings-panel');
	const settingsToggle = document.getElementById('settings-toggle');
	const resizeHandle = document.getElementById('resize-handle');
	const toggleIcon = settingsToggle.querySelector('i');

	// Collapse by default
	settingsPanel.classList.add('collapsed');
	toggleIcon.className = 'bi bi-gear-fill';
	settingsToggle.setAttribute('data-bs-original-title', 'Show Settings');

	const toggleSettings = () => {
		const isCollapsed = settingsPanel.classList.toggle('collapsed');
		toggleIcon.className = isCollapsed ? 'bi bi-gear-fill' : 'bi bi-x-lg';
		settingsToggle.setAttribute('data-bs-original-title', isCollapsed ? 'Show Settings' : 'Hide Settings');
		
		// Trigger resize to adjust plots after transition
		setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
	};

	settingsToggle.addEventListener('click', toggleSettings);

	// Resizing logic
	let isResizing = false;
	resizeHandle.addEventListener('mousedown', () => {
		isResizing = true;
		document.body.style.cursor = 'col-resize';
		document.body.style.userSelect = 'none';
	});

	document.addEventListener('mousemove', (e) => {
		if (!isResizing) return;
		const newWidth = e.clientX;
		const minWidth = parseInt(getComputedStyle(settingsPanel).minWidth);
		const maxWidth = parseInt(getComputedStyle(settingsPanel).maxWidth);
		if (newWidth >= minWidth && newWidth <= maxWidth) {
			settingsPanel.style.width = `${newWidth}px`;
			window.dispatchEvent(new Event('resize'));
		}
	});

	document.addEventListener('mouseup', () => {
		isResizing = false;
		document.body.style.cursor = 'default';
		document.body.style.userSelect = 'auto';
	});

	// Initial run on page load
	runVisualization();
});
