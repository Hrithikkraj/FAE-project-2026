# Swara

Swara is a browser-based musical toolkit for music practice. It includes:

- A Hindustani swar tuner
- A chromatic instrument tuner
- A metronome
- A tanpura drone

## How it works

The tuners listen to audio from your device's microphone. They analyze the sound to estimate its pitch, then show the closest swar or musical note and whether it is sharp or flat.

The swar tuner uses a Just Intonation scale based on Sa. The instrument tuner uses standard chromatic notes. The metronome plays beats at the selected tempo, and the tanpura tool generates a drone for practice.

Your browser may ask for microphone permission when you use a tuner. Allow access for the tuner to hear your instrument or voice.

## Run Swara on your computer

1. Install Node.js version 18 or newer.
2. Open a terminal in the Swara project folder, where the main `package.json` file is located.
3. Install the project packages:

   ```bash
   npm install
   ```

4. Start the development server:

   ```bash
   npm run dev
   ```

5. Open the local address shown in the terminal. It is usually `http://localhost:5173/`.

To stop the server, press `Ctrl+C` in the terminal.
