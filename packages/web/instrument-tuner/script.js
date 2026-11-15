import { ChromaticTuner } from 'chromatic-tuner';

let audioContext = null;
let tuner = null;
let isRunning = false;
let analyser = null;
let stream = null;
let animationFrameId = null;
let selectedInstrument = 'chromatic';
let currentNoteData = null;
let currentFrequency = null;
let stringLampElements = [];

const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const instrumentStrings = {
    guitar: [
        { note: 'E', octave: 2 }, { note: 'A', octave: 2 }, { note: 'D', octave: 3 },
        { note: 'G', octave: 3 }, { note: 'B', octave: 3 }, { note: 'E', octave: 4 }
    ],
    bass: [
        { note: 'E', octave: 1 }, { note: 'A', octave: 1 }, { note: 'D', octave: 2 },
        { note: 'G', octave: 2 }
    ],
    violin: [
        { note: 'G', octave: 3 }, { note: 'D', octave: 4 }, { note: 'A', octave: 4 },
        { note: 'E', octave: 5 }
    ],
    ukulele: [
        { note: 'G', octave: 4 }, { note: 'C', octave: 4 }, { note: 'E', octave: 4 },
        { note: 'A', octave: 4 }
    ]
};

const noteDisplay = document.getElementById('current-note');
const octaveDisplay = document.getElementById('current-octave');
const freqDisplay = document.getElementById('frequency-display');
const centsDisplay = document.getElementById('cents-display');
const tuningStatus = document.getElementById('tuning-status');
const ledStrip = document.getElementById('led-strip');
const noteRuler = document.getElementById('note-ruler');
const noteTape = document.getElementById('note-tape');
const stringLamps = document.getElementById('string-lamps');
const startBtn = document.getElementById('start-btn');
const startLabel = document.getElementById('start-label');
const instrumentButtons = document.querySelectorAll('.it-instrument-button');

const smoothingBufferSize = 24;
const pitchBuffer = [];
const leds = [];

function getSmoothedFrequency(newFreq) {
    if (!newFreq) {
        if (pitchBuffer.length > 0) pitchBuffer.shift();
        return null;
    }
    pitchBuffer.push(newFreq);
    if (pitchBuffer.length > smoothingBufferSize) pitchBuffer.shift();
    const sorted = [...pitchBuffer].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

function createLedStrip() {
    for (let index = 0; index <= 20; index += 1) {
        const led = document.createElement('span');
        const distance = Math.abs(index - 10);
        const zone = distance === 0 ? 'green' : distance <= 2 ? 'lime' : distance <= 5 ? 'amber' : 'red';
        led.className = 'it-led';
        led.dataset.zone = zone;
        ledStrip.append(led);
        leds.push(led);
    }
}

function createNoteTape() {
    for (let repeat = 0; repeat < 24; repeat += 1) {
        for (const note of noteNames) {
            const slot = document.createElement('span');
            slot.className = 'it-tape-slot';
            slot.textContent = note;
            noteTape.append(slot);
        }
    }

    positionRulerTape({ note: 'A', octave: 4, cents: 0 });
    requestAnimationFrame(() => noteTape.classList.add('it-is-ready'));
}

function positionRulerTape(noteData) {
    const noteIndex = noteNames.indexOf(noteData.note);
    const slotIndex = 12 * 12 + noteData.octave * 12 + noteIndex;
    const slot = noteTape.children[slotIndex];
    if (!slot) return;

    const offset = noteRuler.clientWidth / 2
        - slot.offsetLeft
        - slot.offsetWidth / 2
        - (noteData.cents / 100) * slot.offsetWidth;
    noteTape.style.transform = `translateX(${offset}px)`;
}

function updateLedStrip(noteData) {
    leds.forEach(led => led.classList.remove('it-is-lit'));
    if (!noteData) return;

    const index = Math.max(0, Math.min(20, Math.round(noteData.cents / 5) + 10));
    leds[index].classList.add('it-is-lit');
}

function updateRuler(noteData) {
    if (!noteData) {
        noteRuler.classList.add('it-is-silent');
        return;
    }

    noteRuler.classList.remove('it-is-silent');
    positionRulerTape(noteData);
}

function stringFrequency(string) {
    const midiNote = 12 * (string.octave + 1) + noteNames.indexOf(string.note);
    return 440 * Math.pow(2, (midiNote - 69) / 12);
}

function updateStringLamps(frequency, noteData) {
    const strings = instrumentStrings[selectedInstrument];
    if (!strings) return;

    let nearestIndex = -1;
    let nearestCents = Infinity;
    if (frequency && noteData) {
        strings.forEach((string, index) => {
            const cents = 1200 * Math.log2(frequency / stringFrequency(string));
            if (Math.abs(cents) < Math.abs(nearestCents)) {
                nearestCents = cents;
                nearestIndex = index;
            }
        });
    }

    stringLampElements.forEach((lamp, index) => {
        lamp.classList.remove('it-is-lit', 'it-is-tuned');
        if (index === nearestIndex) {
            lamp.classList.add('it-is-lit');
            if (Math.abs(nearestCents) < 5) lamp.classList.add('it-is-tuned');
        }
    });
}

function renderStringLamps() {
    const strings = instrumentStrings[selectedInstrument];
    stringLamps.hidden = !strings;
    stringLampElements = [];
    stringLamps.replaceChildren();
    if (!strings) return;

    strings.forEach(string => {
        const lamp = document.createElement('span');
        lamp.className = 'it-string-lamp';
        const bulb = document.createElement('span');
        bulb.className = 'it-string-bulb';
        const label = document.createElement('span');
        label.className = 'it-string-label';
        label.textContent = `${string.note}${string.octave}`;
        lamp.append(bulb, label);
        stringLamps.append(lamp);
        stringLampElements.push(lamp);
    });
}

function updateUI(noteData) {
    currentNoteData = noteData;
    currentFrequency = noteData ? noteData.frequency : null;
    updateLedStrip(noteData);
    updateRuler(noteData);
    updateStringLamps(currentFrequency, noteData);

    if (!noteData) {
        noteDisplay.textContent = '- -';
        octaveDisplay.textContent = '';
        freqDisplay.textContent = '-- Hz';
        centsDisplay.textContent = '--';
        tuningStatus.className = 'it-tuning-status';
        noteDisplay.dataset.tone = 'silent';
        return;
    }

    noteDisplay.textContent = noteData.note;
    octaveDisplay.textContent = noteData.octave;
    freqDisplay.textContent = `${noteData.frequency.toFixed(1)} Hz`;
    centsDisplay.textContent = `${noteData.cents > 0 ? '+' : ''}${noteData.cents.toFixed(0)}`;

    const cents = Math.abs(noteData.cents);
    tuningStatus.className = 'it-tuning-status';
    if (cents < 5) {
        tuningStatus.classList.add('it-is-tuned');
        noteDisplay.dataset.tone = 'tuned';
    } else if (cents < 15) {
        tuningStatus.classList.add('it-is-close');
        noteDisplay.dataset.tone = 'close';
    } else {
        tuningStatus.classList.add('it-is-off');
        noteDisplay.dataset.tone = 'off';
    }
}

function setInstrument(instrument) {
    selectedInstrument = instrument;
    instrumentButtons.forEach(button => {
        const isSelected = button.dataset.instrument === instrument;
        button.setAttribute('aria-pressed', String(isSelected));
    });
    renderStringLamps();
    updateStringLamps(currentFrequency, currentNoteData);
}

function loop() {
    if (!isRunning) return;

    const buffer = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(buffer);

    let frequency = tuner.getPitch(buffer);
    if (frequency && (frequency < 20 || frequency > 4000)) frequency = null;

    const smoothed = getSmoothedFrequency(frequency);
    const noteData = tuner.getNote(smoothed);
    updateUI(noteData);
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
        source.connect(analyser);

        tuner = new ChromaticTuner({ sampleRate: audioContext.sampleRate });
        pitchBuffer.length = 0;
        isRunning = true;
        startBtn.classList.add('it-is-running');
        startBtn.setAttribute('aria-pressed', 'true');
        startLabel.textContent = 'STOP';
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
        startBtn.disabled = false;
        alert('Microphone access is required for the tuner.');
    }
}

async function stop() {
    isRunning = false;
    startBtn.disabled = true;
    if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    analyser = null;
    pitchBuffer.length = 0;
    updateUI(null);

    const context = audioContext;
    audioContext = null;
    if (context && context.state !== 'closed') {
        try {
            await context.close();
        } catch (error) {
            console.error('Unable to close the audio context.', error);
        }
    }

    startBtn.classList.remove('it-is-running');
    startBtn.setAttribute('aria-pressed', 'false');
    startLabel.textContent = 'START';
    startBtn.disabled = false;
}

function handleFootswitch() {
    if (isRunning) stop();
    else start();
}

createLedStrip();
createNoteTape();
new ResizeObserver(() => {
    positionRulerTape(currentNoteData || { note: 'A', octave: 4, cents: 0 });
}).observe(noteRuler);
renderStringLamps();
updateStringLamps(null, null);
instrumentButtons.forEach(button => {
    button.addEventListener('click', () => setInstrument(button.dataset.instrument));
});
startBtn.addEventListener('click', handleFootswitch);
