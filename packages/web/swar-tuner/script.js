import { Dhwani } from 'dhwani';

const sampleWindowMs = 10000;
const smoothingBufferSize = 10;
const sampleCapacity = 1200;
const pitchBuffer = [];
const samples = new Array(sampleCapacity);

let sampleStart = 0;
let sampleCount = 0;
let audioContext = null;
let stream = null;
let tuner = new Dhwani();
let isRunning = false;
let analyser = null;
let animationFrameId = null;
let canvasContext = null;
let canvasWidth = 0;
let canvasHeight = 0;
let canvasColors = {};
let currentNote = null;
let currentPitchCents = null;
let viewCents = 0;
let steadyKey = null;
let steadySince = null;

const canvas = document.getElementById('pitch-canvas');
const startBtn = document.getElementById('start-btn');
const swarDisplay = document.getElementById('current-swar');
const octaveDisplay = document.getElementById('current-octave');
const freqDisplay = document.getElementById('frequency-display');
const bottomFreqDisplay = document.getElementById('bottom-frequency');
const centsDisplay = document.getElementById('cents-display');
const centsThumb = document.getElementById('cents-thumb');
const statusDisplay = document.getElementById('tuner-status');

function getSmoothedFrequency(newFreq) {
    if (!newFreq) {
        pitchBuffer.length = 0;
        return null;
    }

    pitchBuffer.push(newFreq);
    if (pitchBuffer.length > smoothingBufferSize) pitchBuffer.shift();
    const sorted = [...pitchBuffer].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

function setupCanvas() {
    canvasContext = canvas.getContext('2d');
    const resizeObserver = new ResizeObserver(() => {
        // Scale the backing canvas for crisp lines on high-density screens.
        const ratio = window.devicePixelRatio || 1;
        canvasWidth = canvas.clientWidth;
        canvasHeight = canvas.clientHeight;
        canvas.width = Math.round(canvasWidth * ratio);
        canvas.height = Math.round(canvasHeight * ratio);
        canvasContext.setTransform(ratio, 0, 0, ratio, 0, 0);
        render(performance.now());
    });
    resizeObserver.observe(canvas);

    const themeObserver = new MutationObserver(updateCanvasColors);
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    updateCanvasColors();
}

function updateCanvasColors() {
    const styles = getComputedStyle(document.querySelector('.st-page'));
    canvasColors = {
        accent: styles.getPropertyValue('--st-accent').trim(),
        text: styles.getPropertyValue('--text-color').trim(),
        grid: styles.getPropertyValue('--st-grid').trim(),
        gridSa: styles.getPropertyValue('--st-grid-sa').trim(),
        muted: styles.getPropertyValue('--st-muted').trim(),
        trace: styles.getPropertyValue('--st-trace').trim()
    };
    if (canvasContext) render(performance.now());
}

function pushSample(timestamp, frequency, noteData) {
    let steady = false;
    let cents = null;
    let swar = null;
    let octave = null;

    if (frequency) {
        cents = 1200 * Math.log2(frequency / 240);
        if (noteData) {
            swar = noteData.swar;
            octave = noteData.octave;
            const key = `${swar}:${octave}`;

            if (key !== steadyKey) {
                steadyKey = key;
                steadySince = null;
            }
            if (Math.abs(noteData.cents) <= 15) {
                if (steadySince === null) steadySince = timestamp;
                steady = timestamp - steadySince >= 150;
            } else {
                steadySince = null;
            }
        } else {
            steadyKey = null;
            steadySince = null;
        }
    } else {
        steadyKey = null;
        steadySince = null;
    }

    while (sampleCount && timestamp - samples[sampleStart].t > sampleWindowMs) {
        samples[sampleStart] = undefined;
        sampleStart = (sampleStart + 1) % sampleCapacity;
        sampleCount -= 1;
    }
    if (sampleCount === sampleCapacity) {
        samples[sampleStart] = undefined;
        sampleStart = (sampleStart + 1) % sampleCapacity;
        sampleCount -= 1;
    }
    samples[(sampleStart + sampleCount) % sampleCapacity] = { t: timestamp, cents, swar, octave, steady };
    sampleCount += 1;
}

function drawGrid(now) {
    const ctx = canvasContext;
    const left = 42;
    const right = canvasWidth - 12;
    const top = 14;
    const bottom = canvasHeight - 42;
    const plotWidth = right - left;
    const plotHeight = bottom - top;
    const centsPerPixel = 1320 / plotHeight;
    const centerY = top + plotHeight / 2;
    const targetCents = currentPitchCents === null ? viewCents : currentPitchCents;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (currentPitchCents !== null) {
        if (reducedMotion) viewCents = targetCents;
        else viewCents += (targetCents - viewCents) * 0.12;
    }

    const yForCents = cents => centerY + (viewCents - cents) / centsPerPixel;
    const lanes = [];
    for (const octave of tuner.octaves) {
        for (const swar of tuner.swarOrder) {
            // Convert each octave's base frequency into cents relative to Sa = 240 Hz.
            const frequency = tuner.baseFrequencies[swar] * Math.pow(2, octave);
            lanes.push({ swar, octave, cents: 1200 * Math.log2(frequency / 240) });
        }
    }
    lanes.sort((a, b) => a.cents - b.cents);

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.textBaseline = 'middle';

    const activeLane = currentNote
        ? lanes.find(lane => lane.swar === currentNote.swar && lane.octave === currentNote.octave)
        : null;
    if (activeLane) {
        const laneY = yForCents(activeLane.cents);
        ctx.fillStyle = canvasColors.accent + '20';
        ctx.fillRect(left, laneY - 5, plotWidth, 10);
    }

    for (const lane of lanes) {
        const y = yForCents(lane.cents);
        if (y < top || y > bottom) continue;

        const isSa = lane.swar === 'Sa';
        const isActive = activeLane === lane;
        ctx.beginPath();
        ctx.setLineDash([3, 4]);
        ctx.lineWidth = isActive ? 1.8 : 1;
        ctx.strokeStyle = isActive ? canvasColors.accent : isSa ? canvasColors.gridSa : canvasColors.grid;
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = isActive ? canvasColors.accent : canvasColors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(lane.swar, 5, y);
    }

    ctx.strokeStyle = canvasColors.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, bottom);
    ctx.lineTo(right, bottom);
    ctx.stroke();

    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.fillStyle = canvasColors.muted;
    ctx.textAlign = 'center';
    for (let second = 9; second >= 1; second -= 1) {
        const x = right - (second / 10) * plotWidth;
        ctx.beginPath();
        ctx.moveTo(x, bottom);
        ctx.lineTo(x, bottom + 4);
        ctx.stroke();
        ctx.fillText(`-${second}`, x, bottom + 15);
    }
    ctx.fillText('seconds', left + plotWidth / 2, canvasHeight - 7);
    return { left, right, top, bottom, plotWidth, centerY, centsPerPixel, yForCents, now };
}

function drawTrace(plot) {
    const ctx = canvasContext;
    const { left, right, top, bottom, plotWidth, yForCents, now } = plot;
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, plotWidth, bottom - top);
    ctx.clip();

    for (let index = 1; index < sampleCount; index += 1) {
        const previous = samples[(sampleStart + index - 1) % sampleCapacity];
        const sample = samples[(sampleStart + index) % sampleCapacity];
        if (previous.cents === null || sample.cents === null || sample.t - previous.t > 150) continue;

        const x1 = right - ((now - previous.t) / sampleWindowMs) * plotWidth;
        const x2 = right - ((now - sample.t) / sampleWindowMs) * plotWidth;
        if (x2 < left || x1 > right) continue;

        const isSteady = previous.steady && sample.steady
            && previous.swar === sample.swar && previous.octave === sample.octave;
        ctx.beginPath();
        ctx.setLineDash(isSteady ? [] : [7, 5]);
        ctx.lineWidth = isSteady ? 3 : 2;
        ctx.strokeStyle = isSteady ? canvasColors.accent : canvasColors.trace;
        ctx.lineCap = 'round';
        ctx.moveTo(x1, yForCents(previous.cents));
        ctx.lineTo(x2, yForCents(sample.cents));
        ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.restore();
}

function render(now) {
    if (!canvasContext || !canvasWidth || !canvasHeight) return;
    const plot = drawGrid(now);
    drawTrace(plot);
}

function clearSamples() {
    samples.fill(undefined);
    sampleStart = 0;
    sampleCount = 0;
}

function updatePanel(noteData, frequency) {
    const displayFrequency = frequency ? `${frequency.toFixed(1)} Hz` : '--';
    freqDisplay.textContent = displayFrequency;
    bottomFreqDisplay.textContent = displayFrequency;

    if (!noteData) {
        currentNote = null;
        swarDisplay.textContent = '--';
        swarDisplay.classList.add('st-is-silent');
        octaveDisplay.textContent = '';
        centsDisplay.textContent = '--';
        centsThumb.style.left = '50%';
        centsThumb.classList.add('st-is-muted');
        statusDisplay.textContent = '--';
        statusDisplay.dataset.status = 'silent';
        return;
    }

    currentNote = noteData;
    const cents = noteData.cents;
    const roundedCents = Math.round(cents);
    swarDisplay.textContent = noteData.swar;
    swarDisplay.classList.remove('st-is-silent');
    octaveDisplay.textContent = `Oct ${noteData.octave}`;
    centsDisplay.textContent = `${roundedCents > 0 ? '+' : ''}${roundedCents} cents`;
    centsThumb.style.left = `${Math.max(0, Math.min(100, (cents + 50) * 2))}%`;
    centsThumb.classList.remove('st-is-muted');

    if (Math.abs(cents) < 15) {
        statusDisplay.textContent = 'IN TUNE';
        statusDisplay.dataset.status = 'tuned';
    } else if (cents >= 15) {
        statusDisplay.textContent = 'SHARP';
        statusDisplay.dataset.status = 'sharp';
    } else {
        statusDisplay.textContent = 'FLAT';
        statusDisplay.dataset.status = 'flat';
    }
}

function loop() {
    if (!isRunning) return;

    const buffer = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(buffer);
    let frequency = tuner.getPitch(buffer);
    if (frequency && (frequency < 70 || frequency > 1200)) frequency = null;

    const smoothed = getSmoothedFrequency(frequency);
    const noteData = tuner.getNote(smoothed);
    const timestamp = performance.now();
    currentPitchCents = smoothed ? 1200 * Math.log2(smoothed / 240) : null;

    pushSample(timestamp, smoothed, noteData);
    updatePanel(noteData, smoothed);
    render(timestamp);
    animationFrameId = requestAnimationFrame(loop);
}

async function start() {
    if (isRunning) return;
    startBtn.disabled = true;

    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (audioContext.state === 'suspended') await audioContext.resume();
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;

        const filter = audioContext.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.value = 400;
        filter.Q.value = 0.5;

        source.connect(filter);
        filter.connect(analyser);

        tuner = new Dhwani({ sampleRate: audioContext.sampleRate });
        pitchBuffer.length = 0;
        clearSamples();
        isRunning = true;
        startBtn.textContent = 'Stop Listening';
        startBtn.disabled = false;
        loop();
    } catch (error) {
        console.error(error);
        if (stream) stream.getTracks().forEach(track => track.stop());
        stream = null;
        if (audioContext && audioContext.state !== 'closed') {
            try {
                await audioContext.close();
            } catch (closeError) {
                console.error('Unable to close the audio context.', closeError);
            }
        }
        audioContext = null;
        startBtn.textContent = 'Start Listening';
        startBtn.disabled = false;
        alert('Microphone access is required for the tuner.');
    }
}

async function stop() {
    isRunning = false;
    if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
    animationFrameId = null;

    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    analyser = null;
    pitchBuffer.length = 0;
    clearSamples();
    currentPitchCents = null;
    currentNote = null;
    steadyKey = null;
    steadySince = null;
    updatePanel(null, null);
    render(performance.now());
    startBtn.textContent = 'Start Listening';

    const context = audioContext;
    audioContext = null;
    if (context && context.state !== 'closed') {
        try {
            await context.close();
        } catch (error) {
            console.error('Unable to close the audio context.', error);
        }
    }
}

function handleListenClick() {
    if (isRunning) stop();
    else start();
}

setupCanvas();
startBtn.addEventListener('click', handleListenClick);
