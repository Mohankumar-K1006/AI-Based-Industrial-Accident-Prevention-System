/**
 * admin.js — Client-Side Logic for Admin Dashboard
 * ==================================================
 * Handles:
 *   1. Real-time worker status polling (every 1s)
 *   2. System stats polling (every 2s)
 *   3. Alert log polling (every 1.5s) with auto-scroll
 *   4. Zone list loading and display
 *   5. Interactive zone drawing on canvas overlay
 *   6. Zone CRUD operations via API calls
 */

// ──────────────────────────────────────────────
//  STATE
// ──────────────────────────────────────────────

const AppState = {
    isDrawing: false,
    currentPoints: [],      // points being drawn: [{x, y}, ...]
    existingZones: [],      // zones loaded from server
    lastAlertCount: 0,      // for detecting new alerts
    pollTimers: {},         // interval IDs for cleanup
};

// ──────────────────────────────────────────────
//  DOM ELEMENTS
// ──────────────────────────────────────────────

const DOM = {
    // Video
    videoStream: () => document.getElementById('video-stream'),
    videoContainer: () => document.getElementById('video-container'),
    zoneCanvas: () => document.getElementById('zone-canvas'),
    overlayMode: () => document.getElementById('overlay-mode'),

    // Stats
    fpsValue: () => document.getElementById('fps-value'),
    workersCount: () => document.getElementById('workers-count'),
    uptimeValue: () => document.getElementById('uptime-value'),

    // Zone controls
    zoneTypeSelect: () => document.getElementById('zone-type-select'),
    zoneNameInput: () => document.getElementById('zone-name-input'),
    btnStartDrawing: () => document.getElementById('btn-start-drawing'),
    btnFinishZone: () => document.getElementById('btn-finish-zone'),
    btnCancelDrawing: () => document.getElementById('btn-cancel-drawing'),
    btnUndoPoint: () => document.getElementById('btn-undo-point'),
    zoneInstructions: () => document.getElementById('zone-instructions'),
    pointsCounter: () => document.getElementById('points-counter'),
    pointCount: () => document.getElementById('point-count'),
    zonesList: () => document.getElementById('zones-list'),

    // Workers
    workersTbody: () => document.getElementById('workers-tbody'),
    dangerBadge: () => document.getElementById('danger-badge'),

    // Alerts
    alertsScroll: () => document.getElementById('alerts-scroll'),
    alertEmpty: () => document.getElementById('alert-empty'),
    alertTotal: () => document.getElementById('alert-total'),

    // System
    sysTotalWorkers: () => document.getElementById('sys-total-workers'),
    sysFrames: () => document.getElementById('sys-frames'),
    sysTotalAlerts: () => document.getElementById('sys-total-alerts'),
    sysActiveZones: () => document.getElementById('sys-active-zones'),
};


// ──────────────────────────────────────────────
//  INITIALIZATION
// ──────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initZoneCanvas();
    initZoneControls();
    loadZones();
    startPolling();
});


// ──────────────────────────────────────────────
//  POLLING (Real-Time Data Fetching)
// ──────────────────────────────────────────────

function startPolling() {
    // Workers: every 1 second
    pollWorkers();
    AppState.pollTimers.workers = setInterval(pollWorkers, 1000);

    // Stats: every 2 seconds
    pollStats();
    AppState.pollTimers.stats = setInterval(pollStats, 2000);

    // Alerts: every 1.5 seconds
    pollAlerts();
    AppState.pollTimers.alerts = setInterval(pollAlerts, 1500);
}


async function pollWorkers() {
    try {
        const response = await fetch('/api/workers');
        const data = await response.json();
        renderWorkersTable(data.workers);
        updateWorkerCount(data.count);
    } catch (err) {
        console.error('Error polling workers:', err);
    }
}


async function pollStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        DOM.fpsValue().textContent = data.fps || '—';
        DOM.uptimeValue().textContent = data.uptime_str || '00:00:00';
        DOM.sysTotalWorkers().textContent = data.total_workers_tracked || 0;
        DOM.sysFrames().textContent = formatNumber(data.frame_count || 0);
        DOM.sysTotalAlerts().textContent = data.total_alerts || 0;
    } catch (err) {
        console.error('Error polling stats:', err);
    }
}


async function pollAlerts() {
    try {
        const response = await fetch('/api/logs?limit=50');
        const data = await response.json();
        renderAlertLog(data.alerts);
        DOM.alertTotal().textContent = `${data.count} alerts`;
    } catch (err) {
        console.error('Error polling alerts:', err);
    }
}


// ──────────────────────────────────────────────
//  WORKERS TABLE RENDERING
// ──────────────────────────────────────────────

function renderWorkersTable(workers) {
    const tbody = DOM.workersTbody();
    const dangerBadge = DOM.dangerBadge();

    if (!workers || workers.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="6">No workers detected</td></tr>`;
        dangerBadge.style.display = 'none';
        return;
    }

    let hasDanger = false;
    let html = '';

    for (const w of workers) {
        const statusClass = w.status.toLowerCase();
        const statusIcon = w.status === 'DANGER' ? '🔴' :
                           w.status === 'WARNING' ? '🟡' : '🟢';

        if (w.status === 'DANGER') hasDanger = true;

        const vehicleHtml = w.in_vehicle
            ? `<span class="vehicle-badge">🚗 ${w.vehicle_info?.class_name || 'Vehicle'}</span>`
            : '<span style="color: var(--text-muted)">—</span>';

        html += `
            <tr>
                <td><strong style="color: var(--text-primary)">Worker ${w.worker_id}</strong></td>
                <td><span class="status-badge ${statusClass}">${statusIcon} ${w.status}</span></td>
                <td><span class="coord-text">(${w.centroid[0]}, ${w.centroid[1]})</span></td>
                <td><span class="coord-text">${w.edge_points_count}/17</span></td>
                <td><span class="coord-text">${w.time_in_zone || '—'}</span></td>
                <td>${vehicleHtml}</td>
            </tr>`;
    }

    tbody.innerHTML = html;
    dangerBadge.style.display = hasDanger ? 'flex' : 'none';
}


function updateWorkerCount(count) {
    DOM.workersCount().textContent = count;
}


// ──────────────────────────────────────────────
//  ALERT LOG RENDERING
// ──────────────────────────────────────────────

function renderAlertLog(alerts) {
    const container = DOM.alertsScroll();
    const emptyEl = DOM.alertEmpty();

    if (!alerts || alerts.length === 0) {
        if (emptyEl) emptyEl.style.display = 'block';
        return;
    }

    if (emptyEl) emptyEl.style.display = 'none';

    // Only re-render if there are new alerts
    if (alerts.length === AppState.lastAlertCount) return;
    AppState.lastAlertCount = alerts.length;

    // Render most recent first
    const reversed = [...alerts].reverse();
    let html = '';

    for (const alert of reversed) {
        const statusClass = alert.status.toLowerCase();
        const icon = alert.status === 'DANGER' ? '🚨' : '⚠️';

        html += `
            <div class="alert-entry ${statusClass}">
                <span class="alert-time">${alert.time_str}</span>
                <span class="alert-message">${icon} ${alert.message}</span>
            </div>`;
    }

    container.innerHTML = html;
}


// ──────────────────────────────────────────────
//  ZONE MANAGEMENT
// ──────────────────────────────────────────────

async function loadZones() {
    try {
        const response = await fetch('/api/zones');
        const data = await response.json();
        AppState.existingZones = data.zones || [];
        renderZonesList();
        DOM.sysActiveZones().textContent = AppState.existingZones.length;
    } catch (err) {
        console.error('Error loading zones:', err);
    }
}


function renderZonesList() {
    const container = DOM.zonesList();

    if (AppState.existingZones.length === 0) {
        container.innerHTML = '<div class="zone-empty">No zones defined. Draw one above!</div>';
        return;
    }

    let html = '';
    for (const zone of AppState.existingZones) {
        const dotClass = zone.type === 'DANGER' ? 'danger' : 'warning';
        const vertexCount = zone.points ? zone.points.length : 0;

        html += `
            <div class="zone-item" data-zone-id="${zone.id}">
                <div class="zone-item-info">
                    <div class="zone-dot ${dotClass}"></div>
                    <span class="zone-item-name">${zone.name || zone.id}</span>
                    <span class="zone-item-type">${zone.type}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="zone-item-vertices">${vertexCount} pts</span>
                    <button class="btn btn-danger-sm" onclick="deleteZone('${zone.id}')">
                        🗑️ Delete
                    </button>
                </div>
            </div>`;
    }

    container.innerHTML = html;
}


async function deleteZone(zoneId) {
    if (!confirm(`Delete zone "${zoneId}"? This cannot be undone.`)) return;

    try {
        const response = await fetch(`/api/zones/${zoneId}`, { method: 'DELETE' });
        const data = await response.json();

        if (response.ok) {
            await loadZones();
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (err) {
        console.error('Error deleting zone:', err);
        alert('Failed to delete zone. Check console for details.');
    }
}


// ──────────────────────────────────────────────
//  ZONE CANVAS DRAWING
// ──────────────────────────────────────────────

function initZoneCanvas() {
    const canvas = DOM.zoneCanvas();
    const container = DOM.videoContainer();

    // Resize canvas to match video container
    function resizeCanvas() {
        const rect = container.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        redrawCanvas();
    }

    window.addEventListener('resize', resizeCanvas);
    // Initial resize after video loads
    const videoEl = DOM.videoStream();
    videoEl.addEventListener('load', resizeCanvas);
    setTimeout(resizeCanvas, 500);

    // Click handler for placing points
    canvas.addEventListener('click', (e) => {
        if (!AppState.isDrawing) return;

        const rect = canvas.getBoundingClientRect();
        const x = Math.round(e.clientX - rect.left);
        const y = Math.round(e.clientY - rect.top);

        AppState.currentPoints.push({ x, y });
        DOM.pointCount().textContent = AppState.currentPoints.length;

        // Enable finish button if we have 3+ points
        if (AppState.currentPoints.length >= 3) {
            DOM.btnFinishZone().disabled = false;
        }

        redrawCanvas();
    });

    // Mouse move for preview line
    canvas.addEventListener('mousemove', (e) => {
        if (!AppState.isDrawing || AppState.currentPoints.length === 0) return;

        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        redrawCanvas(mx, my);
    });
}


function redrawCanvas(mouseX, mouseY) {
    const canvas = DOM.zoneCanvas();
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const points = AppState.currentPoints;
    if (points.length === 0) return;

    const zoneType = DOM.zoneTypeSelect().value;
    const color = zoneType === 'DANGER'
        ? 'rgba(239, 68, 68, 0.7)'
        : 'rgba(245, 158, 11, 0.7)';
    const fillColor = zoneType === 'DANGER'
        ? 'rgba(239, 68, 68, 0.15)'
        : 'rgba(245, 158, 11, 0.15)';

    // Draw filled polygon preview
    if (points.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();
    }

    // Draw lines between points
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }

    // Preview line to mouse position
    if (mouseX !== undefined && mouseY !== undefined) {
        ctx.lineTo(mouseX, mouseY);
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw closing line (faint)
    if (points.length >= 3 && mouseX === undefined) {
        ctx.beginPath();
        ctx.moveTo(points[points.length - 1].x, points[points.length - 1].y);
        ctx.lineTo(points[0].x, points[0].y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw vertex dots
    for (let i = 0; i < points.length; i++) {
        ctx.beginPath();
        ctx.arc(points[i].x, points[i].y, 5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? '#10b981' : color;
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Point label
        ctx.fillStyle = 'white';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(`${i + 1}`, points[i].x + 8, points[i].y - 8);
    }
}


// ──────────────────────────────────────────────
//  ZONE CONTROL BUTTONS
// ──────────────────────────────────────────────

function initZoneControls() {
    DOM.btnStartDrawing().addEventListener('click', startDrawing);
    DOM.btnFinishZone().addEventListener('click', finishZone);
    DOM.btnCancelDrawing().addEventListener('click', cancelDrawing);
    DOM.btnUndoPoint().addEventListener('click', undoPoint);
}


function startDrawing() {
    AppState.isDrawing = true;
    AppState.currentPoints = [];

    const canvas = DOM.zoneCanvas();
    canvas.classList.add('drawing-active');

    DOM.btnStartDrawing().disabled = true;
    DOM.btnFinishZone().disabled = true;
    DOM.btnCancelDrawing().disabled = false;
    DOM.btnUndoPoint().disabled = false;
    DOM.pointsCounter().style.display = 'block';
    DOM.pointCount().textContent = '0';

    DOM.overlayMode().textContent = 'DRAWING MODE';
    DOM.overlayMode().classList.add('drawing');

    DOM.zoneInstructions().innerHTML = `
        <p>🎯 <strong>Click on the video</strong> to place polygon vertices.
        Place at least <strong>3 points</strong> to form a zone.
        Click <strong>"Finish & Save"</strong> when done.</p>`;

    redrawCanvas();
}


async function finishZone() {
    if (AppState.currentPoints.length < 3) {
        alert('You need at least 3 points to create a zone!');
        return;
    }

    const zoneType = DOM.zoneTypeSelect().value;
    const zoneName = DOM.zoneNameInput().value.trim() || `${zoneType} Zone ${Date.now()}`;

    // Convert canvas coordinates to video-native coordinates
    // The canvas matches the displayed video size, but the server
    // needs coordinates in the original video resolution.
    // For now we use canvas coords (the server will handle them).
    const points = AppState.currentPoints.map(p => [p.x, p.y]);

    const newZone = {
        id: `zone_${Date.now()}`,
        type: zoneType,
        name: zoneName,
        points: points,
        created_at: new Date().toISOString(),
    };

    // Add to existing zones and save all
    const allZones = [...AppState.existingZones, newZone];

    try {
        const response = await fetch('/api/zones', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ zones: allZones }),
        });

        const data = await response.json();

        if (response.ok) {
            cancelDrawing();
            await loadZones();
            DOM.zoneNameInput().value = '';
        } else {
            alert(`Error saving zone: ${data.error}`);
        }
    } catch (err) {
        console.error('Error saving zone:', err);
        alert('Failed to save zone. Check console for details.');
    }
}


function cancelDrawing() {
    AppState.isDrawing = false;
    AppState.currentPoints = [];

    const canvas = DOM.zoneCanvas();
    canvas.classList.remove('drawing-active');

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    DOM.btnStartDrawing().disabled = false;
    DOM.btnFinishZone().disabled = true;
    DOM.btnCancelDrawing().disabled = true;
    DOM.btnUndoPoint().disabled = true;
    DOM.pointsCounter().style.display = 'none';

    DOM.overlayMode().textContent = 'MONITORING';
    DOM.overlayMode().classList.remove('drawing');

    DOM.zoneInstructions().innerHTML = `
        <p>Click <strong>"Start Drawing Zone"</strong> then click on the video feed
        to place polygon vertices. Minimum 3 points required.</p>`;
}


function undoPoint() {
    if (AppState.currentPoints.length > 0) {
        AppState.currentPoints.pop();
        DOM.pointCount().textContent = AppState.currentPoints.length;

        if (AppState.currentPoints.length < 3) {
            DOM.btnFinishZone().disabled = true;
        }

        redrawCanvas();
    }
}


// ──────────────────────────────────────────────
//  UTILITY FUNCTIONS
// ──────────────────────────────────────────────

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}
