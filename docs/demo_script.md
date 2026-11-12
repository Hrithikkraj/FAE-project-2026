# Five-minute demo script

**Setup before the demo (not timed).**
- Laptop on charge, headphones plugged in, external speakers *off* (the drone must
  reach only the headphones).
- Any free chromatic tuner app open on a phone.
- Run `python scripts/make_demo_wav.py` once. As a fallback, keep a second terminal
  ready with `python scripts/run_app.py --wav data/demo_phrase.wav --loop`, in case the
  room is noisy or the singer is unavailable.
- Start the app with `python scripts/run_app.py`. Sa = D3 (or the singer's Sa), method
  = YIN.
- Have `results/RESULTS.md` and `results/e5_ornament.png` open in a second window.

| Time | What to do | What to say |
|---|---|---|
| 0:00–0:40 | **The problem.** Sing a slow meend Ga→Pa into the phone tuner. | "A normal tuner shows Western note names like E or F#, and during this deliberate slide it flashes 'wrong' the whole time. It doesn't know my Sa and it marks meend as a mistake." |
| 0:40–1:20 | **Set Sa.** Click *Find my Sa* and hum Sa for 3 s. Point at the status bar and the ladder. | "Everything is measured from the singer's own Sa. I hum it, and the swara ladder redraws around my voice." |
| 1:20–1:50 | **Drone.** Tick *Drone*, accept the headphone reminder, set the volume. | "A tanpura-style drone made by additive synthesis: Pa, Sa, Sa, low Sa. It plays in headphones only. If the mic hears it, the pitch tracker follows the drone instead of the voice." |
| 1:50–2:50 | **Sing a phrase with a meend:** Sa, Re, Ga, then a slow slide up to Pa, and hold Pa. | "Held notes are solid saffron and judged; the slide is dashed and labelled GLIDING. It is never counted as an error. Watch the needle and the in-tune band on Pa." |
| 2:50–3:30 | **A flat note gets flagged.** Hold Pa deliberately low (about a quarter of a semitone). | "Now I sing Pa about 40 cents flat: the chip says FLAT and the meter needle moves left. Slides are forgiven, but held notes are checked strictly." |
| 3:30–4:10 | **Switch method live.** Keep singing an 'aa' on Sa; change Method to *FFT peak*, then *Autocorrelation*, then back to *YIN*. | "We wrote four pitch detectors from scratch. FFT peak picks the loudest harmonic and can jump an octave on some vowels; YIN and autocorrelation track the repetition period. Each costs about 0.2 ms per frame, under 2% of the real-time budget." |
| 4:10–5:00 | **Results.** Click *End session* and show the per-swara chart. Then switch to `results/e5_ornament.png` and the E1/E2 tables. | "The summary shows in-tune time per swara. In our evaluation, a naive judge-every-frame tuner flagged 70% of meend frames and 76% of gamaka frames as errors; ours flagged 1.8% and 1.0%, while still catching 95% of notes sung 30–45 cents off. Limits: pitch only, synthetic tests mostly, and slow meends under 200 cents per second still get judged." |

**If something goes wrong.** Switch to the fallback terminal (`--wav data/demo_phrase.wav
--loop`). It contains a hum on Sa, held Sa/Re/Ga, a meend to a flat Pa, and a gamaka, so
every step above can still be shown.
