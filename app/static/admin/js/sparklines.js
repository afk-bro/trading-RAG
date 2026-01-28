/**
 * Sparkline rendering for backtest tables.
 * Uses uPlot for fast, lightweight charts.
 */

(function() {
    'use strict';

    // Configuration
    const SPARKLINE_WIDTH = 96;
    const SPARKLINE_HEIGHT = 28;
    const LINE_COLOR = '#3fb950';  // Green for positive, matches --success
    const LINE_COLOR_NEG = '#f85149';  // Red for negative, matches --danger
    const LINE_WIDTH = 1.5;

    // In-memory cache to avoid refetching on scroll/filter
    const cache = new Map();

    // Batch queue for progressive loading
    let pendingRenders = [];
    let renderScheduled = false;

    /**
     * Fetch sparkline data from API with caching.
     */
    async function fetchSparkline(runId) {
        if (cache.has(runId)) {
            return cache.get(runId);
        }

        try {
            const response = await fetch(`/backtests/runs/${runId}/sparkline`);
            if (!response.ok) {
                cache.set(runId, { y: [], status: 'error' });
                return cache.get(runId);
            }
            const data = await response.json();
            cache.set(runId, data);
            return data;
        } catch (e) {
            console.warn(`Failed to fetch sparkline for ${runId}:`, e);
            cache.set(runId, { y: [], status: 'error' });
            return cache.get(runId);
        }
    }

    /**
     * Create uPlot options for sparkline.
     */
    function createSparklineOpts(data) {
        const y = data.y || [];
        const isPositive = y.length >= 2 && y[y.length - 1] >= y[0];
        const color = isPositive ? LINE_COLOR : LINE_COLOR_NEG;

        return {
            width: SPARKLINE_WIDTH,
            height: SPARKLINE_HEIGHT,
            cursor: { show: false },
            legend: { show: false },
            axes: [
                { show: false },  // x-axis
                { show: false },  // y-axis
            ],
            series: [
                {},  // x series (implicit index)
                {
                    stroke: color,
                    width: LINE_WIDTH,
                    fill: null,
                },
            ],
            scales: {
                x: { time: false },
            },
        };
    }

    /**
     * Render sparkline into container element.
     */
    function renderSparkline(container, data) {
        // Clear any existing content
        container.innerHTML = '';

        if (!data || data.status !== 'ok' || !data.y || data.y.length < 2) {
            // Show dash for empty/error
            container.innerHTML = '<span class="sparkline-empty">-</span>';
            return;
        }

        const y = data.y;
        const x = Array.from({ length: y.length }, (_, i) => i);

        try {
            new uPlot(createSparklineOpts(data), [x, y], container);
        } catch (e) {
            console.warn('Failed to render sparkline:', e);
            container.innerHTML = '<span class="sparkline-empty">-</span>';
        }
    }

    /**
     * Process pending renders in batches.
     */
    async function processPendingRenders() {
        renderScheduled = false;

        // Take a batch of pending renders
        const batch = pendingRenders.splice(0, 10);
        if (batch.length === 0) return;

        // Fetch all in parallel
        const results = await Promise.all(
            batch.map(async ({ container, runId }) => {
                const data = await fetchSparkline(runId);
                return { container, data };
            })
        );

        // Render all
        results.forEach(({ container, data }) => {
            renderSparkline(container, data);
        });

        // Schedule next batch if more pending
        if (pendingRenders.length > 0) {
            scheduleRenders();
        }
    }

    /**
     * Schedule batch processing.
     */
    function scheduleRenders() {
        if (renderScheduled) return;
        renderScheduled = true;
        requestAnimationFrame(processPendingRenders);
    }

    /**
     * Queue a sparkline for rendering.
     */
    function queueSparkline(container, runId) {
        // Check if already rendered
        if (container.dataset.rendered === 'true') return;
        container.dataset.rendered = 'true';

        // Add loading indicator
        container.innerHTML = '<span class="sparkline-loading"></span>';

        // Queue for batch processing
        pendingRenders.push({ container, runId });
        scheduleRenders();
    }

    /**
     * Initialize all sparklines on the page.
     */
    function initSparklines() {
        const containers = document.querySelectorAll('.sparkline[data-run-id]');
        containers.forEach(container => {
            const runId = container.dataset.runId;
            if (runId) {
                queueSparkline(container, runId);
            }
        });
    }

    /**
     * Re-initialize after HTMX swap (for dynamic content).
     */
    function handleHtmxSwap(event) {
        // Find sparklines in the swapped content
        const target = event.detail.target || event.target;
        const containers = target.querySelectorAll('.sparkline[data-run-id]');
        containers.forEach(container => {
            container.dataset.rendered = '';  // Reset rendered flag
            const runId = container.dataset.runId;
            if (runId) {
                queueSparkline(container, runId);
            }
        });
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSparklines);
    } else {
        initSparklines();
    }

    // Re-initialize after HTMX swaps
    document.addEventListener('htmx:afterSwap', handleHtmxSwap);

    // Expose for manual triggering if needed
    window.initSparklines = initSparklines;
})();
