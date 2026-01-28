/**
 * Compare Tray - Global selection management for comparing backtest runs.
 *
 * Features:
 * - Persistent selection via localStorage
 * - Checkbox sync across pages
 * - Sticky bottom tray with actions
 * - HTMX swap compatibility
 */

(function() {
    'use strict';

    const STORAGE_KEY = 'admin.compare.v1';
    const COLLAPSED_KEY = 'admin.compare.trayCollapsed';
    const MAX_SELECTIONS = 12;
    const MIN_TO_COMPARE = 2;

    // State
    let selection = {
        version: 1,
        runs: [],
        updated_at: Date.now()
    };

    // ==========================================================================
    // Storage
    // ==========================================================================

    function loadSelection() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (parsed.version === 1 && Array.isArray(parsed.runs)) {
                    selection = parsed;
                }
            }
        } catch (e) {
            console.warn('Failed to load compare selection:', e);
        }
    }

    function saveSelection() {
        selection.updated_at = Date.now();
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));
        } catch (e) {
            console.warn('Failed to save compare selection:', e);
        }
    }

    function isCollapsed() {
        return localStorage.getItem(COLLAPSED_KEY) === '1';
    }

    function setCollapsed(collapsed) {
        localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0');
    }

    // ==========================================================================
    // Selection Management
    // ==========================================================================

    function getSelection() {
        return [...selection.runs];
    }

    function isSelected(runId) {
        return selection.runs.includes(runId);
    }

    function addToSelection(runId) {
        if (selection.runs.includes(runId)) return false;
        if (selection.runs.length >= MAX_SELECTIONS) {
            showMaxWarning();
            return false;
        }
        selection.runs.push(runId);
        saveSelection();
        dispatchChangeEvent();
        return true;
    }

    function removeFromSelection(runId) {
        const idx = selection.runs.indexOf(runId);
        if (idx === -1) return false;
        selection.runs.splice(idx, 1);
        saveSelection();
        dispatchChangeEvent();
        return true;
    }

    function toggleSelection(runId) {
        if (isSelected(runId)) {
            removeFromSelection(runId);
        } else {
            addToSelection(runId);
        }
    }

    function clearSelection() {
        selection.runs = [];
        saveSelection();
        dispatchChangeEvent();
        syncCheckboxes();
    }

    function showMaxWarning() {
        // Brief visual feedback
        const tray = document.getElementById('compare-tray');
        if (tray) {
            tray.classList.add('tray-shake');
            setTimeout(() => tray.classList.remove('tray-shake'), 300);
        }
    }

    // ==========================================================================
    // Events
    // ==========================================================================

    function dispatchChangeEvent() {
        window.dispatchEvent(new CustomEvent('compare:changed', {
            detail: { runs: getSelection() }
        }));
    }

    // ==========================================================================
    // Checkbox Sync
    // ==========================================================================

    function syncCheckboxes() {
        const checkboxes = document.querySelectorAll('.js-compare-toggle');
        checkboxes.forEach(cb => {
            const runId = cb.dataset.id;
            if (runId) {
                cb.checked = isSelected(runId);
            }
        });
    }

    function attachCheckboxListeners() {
        const checkboxes = document.querySelectorAll('.js-compare-toggle');
        checkboxes.forEach(cb => {
            // Remove existing listener to avoid duplicates
            cb.removeEventListener('change', handleCheckboxChange);
            cb.addEventListener('change', handleCheckboxChange);
        });
    }

    function handleCheckboxChange(e) {
        const runId = e.target.dataset.id;
        if (!runId) return;

        if (e.target.checked) {
            if (!addToSelection(runId)) {
                // Revert checkbox if at limit
                e.target.checked = false;
            }
        } else {
            removeFromSelection(runId);
        }
        renderTray();
    }

    // ==========================================================================
    // Tray Rendering
    // ==========================================================================

    function renderTray() {
        const tray = document.getElementById('compare-tray');
        if (!tray) return;

        const count = selection.runs.length;
        const collapsed = isCollapsed();

        // Update visibility
        if (count === 0) {
            tray.classList.add('tray-hidden');
            return;
        }
        tray.classList.remove('tray-hidden');
        tray.classList.toggle('tray-collapsed', collapsed);

        // Update count
        const countEl = tray.querySelector('.tray-count');
        if (countEl) {
            countEl.textContent = count;
        }

        // Update chips (show first 5)
        const chipsEl = tray.querySelector('.tray-chips');
        if (chipsEl) {
            const displayed = selection.runs.slice(0, 5);
            const more = count - displayed.length;

            chipsEl.innerHTML = displayed.map(id =>
                `<span class="tray-chip" title="${id}">
                    ${id.slice(0, 8)}
                    <button class="chip-remove" data-id="${id}">&times;</button>
                </span>`
            ).join('');

            if (more > 0) {
                chipsEl.innerHTML += `<span class="tray-more">+${more} more</span>`;
            }

            // Attach remove handlers
            chipsEl.querySelectorAll('.chip-remove').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const id = btn.dataset.id;
                    removeFromSelection(id);
                    syncCheckboxes();
                    renderTray();
                });
            });
        }

        // Update buttons state
        const compareBtn = tray.querySelector('.tray-btn-compare');
        if (compareBtn) {
            compareBtn.disabled = count < MIN_TO_COMPARE;
        }
    }

    function toggleTrayCollapse() {
        const collapsed = !isCollapsed();
        setCollapsed(collapsed);
        renderTray();
    }

    // ==========================================================================
    // Actions
    // ==========================================================================

    function openCompare() {
        if (selection.runs.length < MIN_TO_COMPARE) return;

        // Get workspace_id from current URL or first visible element
        const params = new URLSearchParams(window.location.search);
        let workspaceId = params.get('workspace_id');

        if (!workspaceId) {
            // Try to find from page
            const wsInput = document.querySelector('input[name="workspace_id"]');
            if (wsInput) workspaceId = wsInput.value;
        }

        // Build compare URL
        const runIds = selection.runs.join(',');
        let url = `/admin/backtests/compare-runs?run_ids=${encodeURIComponent(runIds)}`;
        if (workspaceId) {
            url += `&workspace_id=${encodeURIComponent(workspaceId)}`;
        }

        window.location.href = url;
    }

    // ==========================================================================
    // Initialization
    // ==========================================================================

    function init() {
        loadSelection();
        syncCheckboxes();
        attachCheckboxListeners();
        renderTray();

        // Attach tray button handlers
        const tray = document.getElementById('compare-tray');
        if (tray) {
            const compareBtn = tray.querySelector('.tray-btn-compare');
            const clearBtn = tray.querySelector('.tray-btn-clear');
            const collapseBtn = tray.querySelector('.tray-collapse');

            if (compareBtn) compareBtn.addEventListener('click', openCompare);
            if (clearBtn) clearBtn.addEventListener('click', clearSelection);
            if (collapseBtn) collapseBtn.addEventListener('click', toggleTrayCollapse);
        }
    }

    function syncPage() {
        syncCheckboxes();
        attachCheckboxListeners();
        renderTray();
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Re-sync after HTMX swaps
    document.body.addEventListener('htmx:afterSwap', syncPage);

    // Expose API
    window.CompareTray = {
        getSelection,
        isSelected,
        addToSelection,
        removeFromSelection,
        toggleSelection,
        clearSelection,
        syncPage
    };
})();
