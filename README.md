# Exoplanet Swarm 🚀🌟

> A production-ready multi-agent CrewAI pipeline that autonomously ingests raw NASA telescope data, processes the photometric signal, runs Box-fitting Least Squares (BLS) transit detection, and produces a public-facing science summary — all with no manual data download required.

---

## How It Works

The pipeline connects directly to NASA's [Mikulski Archive for Space Telescopes (MAST)](https://mast.stsci.edu/) via the [`lightkurve`](https://docs.lightkurve.org/) Python package. No dataset download needed — agents pull live FITS data on demand.

```
Space Scraper → Signal Processor → Astrophysicist → Science Communicator
     ↓                ↓                  ↓                   ↓
  MAST API       SG filter +         BLS periodogram     2-paragraph
  lightkurve     3σ clipping         (astropy BLS)       public summary
```

### Four Agents

| Agent | Tool Used | Responsibility |
|-------|-----------|----------------|
| **Space Scraper** | `fetch_lightcurve_tool` | Queries MAST, stitches Kepler/TESS quarters, normalizes flux |
| **Signal Processor** | `clean_signal_tool` | Savitzky-Golay detrending + 3σ outlier removal |
| **Astrophysicist** | `bls_periodogram_tool` | BLS periodogram over 5,000 trial periods, outputs period / depth / SNR / probability |
| **Science Communicator** | *(LLM only)* | Translates raw numbers into an accessible 2-paragraph public summary |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

> **Swap LLMs**: Replace `ChatOpenAI` in `exoplanet_swarm.py` with any LangChain-compatible provider (Anthropic, Gemini, Ollama, etc.).

### 3. Run the pipeline

```bash
# Default target: Kepler-186
python exoplanet_swarm.py

# Override target star via CLI
python exoplanet_swarm.py "TOI 700"
python exoplanet_swarm.py "Kepler-442"
```

### 4. Generate visualizations

```bash
# Fetch live from MAST + plot
python visualize.py "Kepler-186"

# Use cached fixtures (fast, offline)
python visualize.py --cached
```

Outputs a 4-panel PNG:
- Raw normalized light curve
- Detrended & cleaned light curve
- BLS power spectrum with period annotations
- Phase-folded light curve at best-fit orbital period

---

## Running Tests

```bash
pip install pytest
```

```bash
# Fast unit tests (fully mocked, no network)
pytest tests/ -v -m unit

# Integration tests against REAL Kepler-186 MAST data
# (downloads ~50k cadences on first run, caches to tests/fixtures/)
pytest tests/ -v -m integration

# Everything
pytest tests/ -v
```

### What the integration tests validate

The integration suite (`TestRealKepler186Data`) fetches actual Kepler-186 photometry and asserts:

- `records > 10,000` cadences retrieved (4 years of Kepler data)
- Flux is normalized near 1.0
- Cleaning reduces flux scatter vs raw
- **BLS best period falls within 10% of a known confirmed orbit:**

| Planet | Period (days) |
|--------|--------------|
| Kepler-186 b | 3.887 |
| Kepler-186 c | 7.267 |
| Kepler-186 d | 13.342 |
| Kepler-186 e | 22.408 |
| Kepler-186 f | 129.945 ← habitable zone |

- Transit depth is physically plausible (100–10,000 ppm)
- Planet probability > 30% for real Kepler data

Test fixtures are cached to `tests/fixtures/` so MAST is only hit once per machine. Delete that folder to force a fresh download.

---

## Architecture Details

### `fetch_lightcurve_tool`
- Tries Kepler PDCSAP_FLUX first, falls back to TESS SPOC
- Downloads up to 8 quarters/sectors and stitches them
- Normalizes by median flux; removes NaN cadences

### `clean_signal_tool`
- **Stage 1**: Savitzky-Golay filter (window ≈ 1% of series, always odd, cubic polynomial) divides out slow stellar variability
- **Stage 2**: 3σ sigma-clipping removes cosmic rays and momentum dumps
- Window is intentionally narrow to preserve short-duration transit dips

### `bls_periodogram_tool`
- Tests 5,000 log-spaced trial periods from 0.5 days to baseline/3
- Tests 4 duration hypotheses (0.05, 0.10, 0.15, 0.20 days) simultaneously
- SNR = peak\_power / median\_power
- Planet probability: `P = 1 − exp(−SNR / 10)` (heuristic proxy, not Bayesian posterior)
- Detection quality: `Strong (SNR≥15)` / `Moderate (≥7)` / `Weak (≥3)` / `Noise (<3)`

---

## Project Structure

```
zerve-ai/
├── exoplanet_swarm.py     # Main CrewAI pipeline (agents, tasks, tools, crew)
├── visualize.py           # 4-panel diagnostic plot (dark theme, matplotlib)
├── requirements.txt       # Python dependencies
├── tests/
│   ├── conftest.py        # Real Kepler-186 data fixtures (MAST-cached)
│   └── test_exoplanet_swarm.py  # Unit + integration test suites
└── tests/fixtures/        # Auto-created on first integration test run
    ├── kepler186_raw.json
    ├── kepler186_clean.json
    └── kepler186_bls.json
```

---

## Target Stars to Try

| Star | Why it's interesting |
|------|---------------------|
| `Kepler-186` | 5 planets; Earth-size planet in habitable zone (186f) |
| `Kepler-442` | Super-Earth in habitable zone, high habitability score |
| `TOI 700` | TESS habitable zone Earth-size planet |
| `Kepler-62` | Two habitable zone planets (62e, 62f) |
| `TRAPPIST-1` | 7 Earth-size planets; 3 in habitable zone |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `crewai` | Multi-agent orchestration |
| `langchain-openai` | LLM interface |
| `lightkurve` | NASA MAST data retrieval |
| `astropy` | BLS periodogram, units |
| `scipy` | Savitzky-Golay filter |
| `numpy` | Numerical operations |
| `pandas` | DataFrame handling |
| `matplotlib` | Visualization |
