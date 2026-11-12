# Swara-Aware user guide

Swara-Aware listens while you sing and shows which swara you are on and how far off
you are, measured from **your own Sa**. Glides (meend) and ornaments (gamaka) are shown
but never counted as mistakes. This guide walks through one practice session.

## Before you start

- **Use headphones** (wired is best; Bluetooth adds delay). The tanpura drone must go
  into your ears, not into the microphone. If the drone plays on the laptop speakers,
  the app may "hear" the drone's Sa and Pa instead of your voice.
- **Microphone placement.** Use the laptop's built-in mic or a headset mic. Sit about an
  arm's length (40–60 cm) from a laptop mic, or keep a headset mic 3–5 cm from your
  mouth, slightly to the side so your breath does not hit it. Keep the distance the same
  throughout the session.
- **A quiet room.** Turn off fans and close windows if you can. Steady noise like a fan
  is mostly fine, but loud noise makes the app go "silent" or wobble.
- **Microphone permission.** Windows: Settings → Privacy & security → Microphone →
  allow desktop apps. macOS: System Settings → Privacy & Security → Microphone → allow
  your terminal.

Start the app from the `swara-aware` folder:

```
python scripts/run_app.py
```

## Step 1: set your Sa

Your Sa is the base note that everything is measured from. Choose one of these:

- **Find my Sa:** click it and hum a comfortable, steady note for 3 seconds. The app
  takes the steadiest part and sets Sa (the status bar shows the frequency and the
  nearest note name). If it says it could not find a steady note, hum a little louder
  and keep the note still.
- **Pick a note:** choose a note name and octave from the dropdowns (for example C#
  and 3), the way you would choose a harmonium key.
- **Type Hz:** enter an exact frequency in the Hz box and press Enter.

Changing Sa starts a new session, because every pitch is measured from Sa.

Also check the other controls in the top bar:
- **Tuning:** *equal* (like a harmonium) or *just* (pure ratios, like a tanpura's
  harmonics).
- **Method:** *YIN* is the default and the best choice for singing.
- **Input:** your microphone.

## Step 2: start the drone

Put your headphones on, tick **Drone** at the bottom left, and set the volume slider so
the drone sits comfortably under your voice. The first time, a reminder about
headphones appears. The drone repeats Pa–Sa–Sa–low Sa like a tanpura, tuned to your Sa.

## Step 3: start listening and sing

Click **Start listening** and sing. Begin with long, steady notes: Sa, then Re, Ga, and
so on.

## Step 4: read the display

**The pitch trace (left).** The last 10 seconds of your voice on a ladder of swara
lines. Lines with a dot before the name are the lower octave (`.Pa`); lines with an
apostrophe are the upper octave (`Sa'`).
- The line of the swara you are nearest is highlighted in saffron, with a shaded
  **in-tune band** (±15 cents) around it. Keep your line inside the band.
- **Solid saffron line:** a held note. These are judged.
- **Dashed line:** a glide or meend. Not judged.
- **Dotted line:** an ornament or gamaka. Not judged.

**The readout (right).**
- **Big name:** the swara you are singing (grey while you are gliding).
- **"+8 cents":** how far you are from it. Plus means sharp (too high), minus means
  flat (too low). 100 cents is one semitone.
- **Meter:** the needle shows the same number. Green zone = in tune (±15), yellow zone =
  close (±30).
- **Status chip:** IN TUNE, CLOSE, SHARP, FLAT, GLIDING, ORNAMENT or SILENT.
- **Hint line:** says when a note starts counting ("Held 0.6 s · counted in your
  score"). A note must be held for a quarter of a second before it is judged.

Tips while singing:
- If the needle keeps sitting on one side, adjust slowly towards the centre and hold.
- Slides between notes are fine; they will never lower your score.
- If your voice is very quiet, the app shows SILENT. Sing a little louder, or move
  closer to the microphone.

## Analysing a recording instead of live singing

To check a recording, for example a Sa Re Ga Ma exercise saved as a WAV file:

1. Click **Open audio file…** in the top bar and choose the file. WAV is best; FLAC and
   OGG work too.
2. Set Sa to the singer's Sa. If the recording starts with a held Sa, click **Find my Sa**
   while it plays.
3. The file starts by itself. Leave **Play aloud** ticked to hear it, and tick **Loop**
   to repeat it.
4. Read the display and the **End session** summary exactly as for live singing.

Solo voice works best. Accompaniment (tanpura, harmonium, tabla) can pull the pitch
tracker towards the instruments. To go back to live singing, choose your microphone in
the **Input** dropdown.

To follow audio that is playing on the computer (a video, say) without saving a file,
choose **Stereo Mix** in the **Input** dropdown if your computer has it.

## Step 5: record (optional)

Click **● Record** to keep a copy of the session. When you click it again (or stop
listening), the audio (WAV) and the pitch track (CSV) are saved in
`results/sessions/`.

## Step 6: review the summary

Click **End session**. The summary shows:
- the overall **% of held time in tune**, how many notes you held, and your average error
- how many seconds you spent on held notes, glides, ornaments and silence
- a **bar chart per swara**: % in tune and the average error ("−12¢ avg" means you tend
  to sing that swara 12 cents flat, which is a useful thing to practise)

**Save CSV…** stores every frame (time, frequency, cents, swara, state). **Save
plots…** stores a pitch-trace picture and the accuracy chart. After closing the
summary, a fresh session begins.

## Troubleshooting

| What you see | What to do |
|---|---|
| "No microphone found" | Connect or enable a microphone and restart, or try another **Input**. |
| "Cannot open the microphone" | Allow microphone access (see above); close other apps using the mic. |
| Always SILENT | Sing louder or move closer; check the right **Input** is selected. |
| Pitch jumps to Sa/Pa when the drone is on | The mic hears the drone: use headphones. |
| Swara is one octave off | Your Sa is in a different octave: use **Find my Sa** or change the octave. |
| Display feels slow | Normal: about 0.1–0.2 s of delay; a note is judged after 0.25 s of holding. |

Swara-Aware is a practice aid. It checks pitch only, not voice quality, rhythm or
expression, and it does not replace a teacher.
