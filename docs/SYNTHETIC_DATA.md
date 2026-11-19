# Synthetic dual-microphone data

`highspl.data` renders what the two microphones of the pod would deliver in a
battlefield / aviation scene, together with an exact **clean target** (the
talker's speech at the primary mic, through only the *linear* parts of the
hardware chain). Everything is seedable: a scene is determined by
`(base_seed, index)`, and `data/manifests/<set>.json` stores a SHA-256 per audio
file so any drift in the generator or its libraries is detected
(`scripts/generate_mixtures.py --set eval_v1 --verify`).

```
uv run python scripts/download_data.py                       # LibriSpeech test-clean + dev-clean
uv run python scripts/generate_mixtures.py --set eval_v1 --workers 8
uv run python scripts/generate_mixtures.py --set tiny        # 6 scenes, no downloads (CI)
```

## Chain

speech (Pa at the primary mic) + noise sources (plane waves / diffuse / wind /
impulses) -> **pod geometry** (`data/rir.py`) -> **hardware model**
(`data/clipping.py`) -> `mic.flac` (2 ch, 24-bit) and `clean.flac`.

Levels are calibrated in dB SPL at the primary mic. Digital full scale equals the
microphone's acoustic overload point (AOP): a sine at `aop_db` rms peaks at 1.0.

## Noise sources (`data/noise_synth.py`)

Values are literature-typical *ranges*; every scene draws inside them (domain
randomisation), so no single number is critical.

| class | what is modelled | frequency content |
|---|---|---|
| jet | jet-mixing noise (two log-frequency humps, per-octave), fan blade-pass tone + harmonics (blades x RPM/60), buzz-saw shaft-order tones, turbine whine, afterburner crackle | hump 100 Hz-1 kHz; BPF 1-5 kHz; whine 3-10 kHz |
| helicopter | main-rotor BPF harmonics (10-30 Hz), blade slap (impulsive, ~BPF), tail rotor, turbine whine, LF broadband | mostly < 1 kHz, tones to 8 kHz |
| tank | track-pitch harmonics (10-100 Hz), diesel firing-frequency harmonics (50-260 Hz, drifting RPM), clank transients, intake LF | mostly < 1 kHz |
| missile | rocket-plume rumble (hump 30-500 Hz), crackle (positively skewed pulses), launch onset | 20-500 Hz + crackle |
| gunfire | Friedlander muzzle blast (0.3-3 ms), supersonic N-wave crack, reverberant tail; single shots or 600-900 rpm bursts | 0.2-5 kHz, crest > 20 dB |
| artillery | long Friedlander blast low-passed by distance, rumble tail | < 300 Hz |
| wind | per-mic **independent** LF turbulence with gust modulation | < 500 Hz, incoherent between mics |
| babble (optional) | competing far-field speech | speech band |

Spatial model: far-field sources are plane waves at a random polar angle
(0 deg = in front of the user, 180 deg = behind), arriving `d*cos(theta)/c`
earlier at the front mic; a diffuse component has the theoretical two-mic
coherence `sinc(2*pi*f*d/c)`; the housing shadows the far-side mic at high
frequencies.

## Speech (`data/speech_synth.py`, `data/mixture.py`)

| tier | source | purpose |
|---|---|---|
| **S1** | parametric source-filter voice: glottal pulses with F0 contour (declination, accents, jitter/shimmer), formant resonators moving between Peterson-Barney vowel targets, fricatives (/s sh f/), plosives (closure + burst + aspiration), syllable/word/phrase timing with 20-40 % pauses, Lombard mode | **exact ground truth** (F0, voicing, formants, phones) for property checks and a harmonic-noise stress test |
| **S2** | real speech: LibriSpeech (test-clean for evaluation, dev-clean for training), 16 kHz upsampled to 48 kHz | headline quality |
| **S3** | S2 plus an approximate Lombard transform (pitch/formant raise, HF tilt flattening) | speech under effort |

Speech level follows the Lombard rule used by the composer: talker level at 1 m
is `55 + 0.6*max(noise_dBA - 50, 0)` dB capped at 92 dB, plus the near-field gain
`20*log10(1 m / r1)` (~+25 dB at 5.5 cm), i.e. **~105-118 dB SPL at the mic**:
close to the INMP441's 120 dB overload point on its own.

Known limitations: S1 vowels are stationary harmonic stacks and a model trained
on real speech may treat them like engine tones (reported separately, never
averaged with S2/S3); real-speech tiers are band-limited to 8 kHz;
`aop_db`-matched self-noise is white, not the INMP441's shaped floor; the
vehicle-cabin (`pyroomacoustics`) environment is not generated.

## Hardware model (`data/clipping.py`)

Sensitivity (+/-1 dB per-mic mismatch) -> 60 Hz HPF -> soft-knee overload then
hard clip at full scale -> self-noise (33 dB SPL equivalent) -> 15 kHz LPF ->
**I2S L/R offset** (0-1 sample on the reference) -> 24-bit quantisation ->
optional software DC blocker (off) -> **ALSA overrun** (lost periods, both
channels, rare). There is no microcontroller stage: the mics feed the Pi's I2S
directly.

## Stratification of `eval_v1`

240 scenes x 10 s: 6 noise groups (jet, helicopter, tank, missile, gunfire,
artillery) x 4 SPL bands at the mic (90-105, 105-120, 120-130, 130-138 dB) x 3
speech tiers (the 24-scene group x band cycle rotates S1, S2, S3, so 96 S1, 72 S2
and 72 S3 scenes). Hardware parameters (mismatch, I2S offset,
geometry) are drawn per scene.
