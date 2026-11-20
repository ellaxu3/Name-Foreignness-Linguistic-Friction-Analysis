# Name Foreignness & Linguistic Friction Analysis (`NameForiegnness.py`)

`NameForiegnness.py` is a quantitative linguistics and demographic analysis script that evaluates **50 first names** (25 Asian ethnic names across 7 heritage groups and 25 traditional U.S. baseline names) across two U.S. birth cohorts (**1995 vs. 2025**).

It measures how demographic concentration ("Name Foreignness") interacts with spoken sound rarity (phonotactics) and spelling-to-sound mismatch (orthographic opacity), exporting three publication-ready visualizations.

---

## Core Metrics & Methodology

1. **Name Foreignness Index ($F\text{-Index}$):**
   * Measures the relative concentration of a name among foreign-born vs. U.S.-born individuals on a `0–100` scale (`0` = exclusively U.S.-born, `50` = demographic parity, `100` = exclusively foreign-born).
   * Tracks longitudinal shifts ($\Delta = F_{2025} - F_{1995}$) to distinguish **2nd-generation diffusion** ($\Delta < 0$, names becoming more common among U.S.-born cohorts) from **cohort aging shifts** ($\Delta > 0$).

2. **ARPAbet Phoneme Bigram Surprisal (`mean_bigram_bits`):**
   * Evaluates spoken sound rarity using an add-$k$ smoothed ($k = 0.5$) ARPAbet phoneme bigram language model trained on the CMU Pronouncing Dictionary (`nltk.corpus.cmudict`), with a built-in fallback corpus if NLTK data is unavailable.
   * Outputs the mean information-theoretic surprisal in **bits** ($-\log_2 P(p_i \mid p_{i-1})$), where higher values indicate rarer phoneme transitions in American English.

3. **Orthographic Opacity Score (`orthographic_opacity`):**
   * A rule-based heuristic (`0–100`) scoring spelling-to-sound friction for English readers.
   * Penalizes non-English orthographic patterns (e.g., initial `X`, `ZH`, `NG`, `SR`, unfollowed `Q`), unfamiliar vowel/consonant clusters (`EO`, `HY`, `YI`, `NH$`), rare character bigrams, and multi-syllable length.

---

## Dataset Coverage

* **Ethnic Cohorts (25 names):** Chinese (4), Indian (6), Korean (4), Vietnamese (4), Pakistani (4), Japanese (2), and Filipino (1).
* **U.S. Baseline (25 names):** 12 male and 13 female high-frequency traditional U.S. names (e.g., *James*, *Michael*, *Mary*, *Jennifer*).
* **Embedded Data:** Cohort counts (`PERWT`) for 1995 and 2025 and target ARPAbet transcriptions are self-contained inside the script—no external CSV files are required.

---

## Generated Outputs

Running the script generates three figures, each exported in **4 formats** (`.png` at 300 DPI, `.pdf`, `.svg`, and serialized `.fig.pkl` matplotlib objects):

| Base Filename | Visualization Type | Description |
| :--- | :--- | :--- |
| `chart1_f_index_shift` | Dumbbell Plot | Compares 1995 vs. 2025 $F\text{-Index}$ across all 50 names relative to the $F = 50$ parity line. |
| `chart2_foreignness_vs_phoneme_rarity` | Scatter Plot | Plots 2025 $F\text{-Index}$ against mean ARPAbet bigram surprisal (bits) for all 50 names. |
| `chart3_spelling_vs_phoneme_rarity` | Quadrant Scatter Plot | Maps orthographic opacity vs. phonotactic rarity across the 25 Asian names, dividing them into *High Fluency*, *Sound-Rare Only*, *Spelling Trap Only*, and *Double Friction* quadrants. |

---

## Installation & Usage

### Prerequisites
Install the required Python libraries:
```bash
pip install pandas numpy matplotlib nltk
```

*(Optional)* Download the CMU Pronouncing Dictionary so the bigram model uses the full lexicon instead of the built-in fallback list:
```bash
python -c "import nltk; nltk.download('cmudict')"
```

### Running the Script
```bash
python NameForiegnness.py
```
All 12 output files (`chart1_*`, `chart2_*`, `chart3_*`) will be saved directly to the current working directory.