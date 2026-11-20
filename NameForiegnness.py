'''
NameForiegnness.py

Analyzes 50 names (25 ethnic, 25 US non-ethnic) with regards to demographic name concentration, 
spoken sound rarity, and spelling-to-sound friction across U.S. birth cohorts (1995 vs. 2025). 
Creates three models that illustrates these relationships.
'''
import pandas as pd
import numpy as np
import re
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. Helper Functions & ARPAbet Bigram Model
def clean_first_name(name_series: pd.Series) -> pd.Series:
    return (
        name_series.fillna("").astype(str).str.upper().str.strip()
        .str.split().str[0].fillna("").apply(lambda x: re.sub(r"[^A-Z]", "", x))
    )

def strip_stress(phoneme: str) -> str:
    return re.sub(r"[0-9]", "", phoneme)

def build_arpabet_bigram_model(cmu_dict: dict = None, smoothing_k: float = 0.5):
    arpabet_inventory = [
        "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
        "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
        "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
        "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "<s>", "</s>"
    ]
    vocab_size = len(arpabet_inventory)
    bigram_counts = Counter()
    unigram_counts = Counter()

    if cmu_dict:
        for _, prons in cmu_dict.items():
            seq = ["<s>"] + [strip_stress(p) for p in prons[0]] + ["</s>"]
            for i in range(len(seq) - 1):
                bigram_counts[(seq[i], seq[i + 1])] += 1
                unigram_counts[seq[i]] += 1
    else:
        fallback_prons = [
            ["JH", "EY", "M", "Z"], ["M", "AY", "K", "AH", "L"], ["R", "AA", "B", "ER", "T"],
            ["JH", "AA", "N"], ["D", "EY", "V", "IH", "D"], ["W", "IH", "L", "Y", "AH", "M"],
            ["R", "IH", "CH", "ER", "D"], ["JH", "OW", "S", "AH", "F"], ["T", "AA", "M", "AH", "S"],
            ["K", "R", "IH", "S", "T", "AH", "F", "ER"], ["M", "AE", "TH", "Y", "UW"],
            ["D", "AE", "N", "Y", "AH", "L"], ["M", "EH", "R", "IY"], ["P", "AH", "T", "R", "IH", "SH", "AH"],
            ["JH", "EH", "N", "AH", "F", "ER"], ["L", "IH", "N", "D", "AH"],
            ["IH", "L", "IH", "Z", "AH", "B", "AH", "TH"], ["B", "AA", "R", "B", "ER", "AH"],
            ["S", "UW", "Z", "AH", "N"], ["JH", "EH", "S", "IH", "K", "AH"], ["S", "EH", "R", "AH"],
            ["K", "EH", "R", "AH", "N"], ["N", "AE", "N", "S", "IY"], ["L", "IY", "S", "AH"],
            ["AE", "SH", "L", "IY"], ["S", "T", "R", "IH", "NG"], ["K", "W", "IY", "N"],
            ["P", "R", "IY", "S", "T"], ["R", "OW", "HH", "AE", "N"], ["M", "EY"],
            ["W", "EY"], ["B", "IH", "L", "AA", "L"], ["K", "EH", "N", "JH", "IY"]
        ]
        for _ in range(250):
            for pron in fallback_prons:
                seq = ["<s>"] + pron + ["</s>"]
                for i in range(len(seq) - 1):
                    bigram_counts[(seq[i], seq[i + 1])] += 1
                    unigram_counts[seq[i]] += 1

    def score_sequence(phoneme_seq: list[str]) -> tuple[float, tuple[str, float]]:
        seq = ["<s>"] + [strip_stress(p) for p in phoneme_seq] + ["</s>"]
        step_bits = []
        for i in range(len(seq) - 1):
            p_prev, p_curr = seq[i], seq[i + 1]
            prob = (bigram_counts[(p_prev, p_curr)] + smoothing_k) / (
                unigram_counts[p_prev] + smoothing_k * vocab_size
            )
            step_bits.append((f"{p_prev}->{p_curr}", float(-np.log2(prob))))
        mean_bits = float(np.mean([b for _, b in step_bits]))
        rarest_step = max(step_bits, key=lambda x: x[1])
        return mean_bits, rarest_step

    return score_sequence

try:
    from nltk.corpus import cmudict
    CMU_DICT = cmudict.dict()
except Exception:
    CMU_DICT = None

bigram_scorer = build_arpabet_bigram_model(CMU_DICT)

TARGET_ARPABET = {
    "WEI": ["W", "EY"], "XIAOMING": ["SH", "Y", "AW", "M", "IH", "NG"],
    "MEI": ["M", "EY"], "XINYI": ["SH", "IH", "N", "Y", "IY"],
    "AARAV": ["AA", "R", "AH", "V"], "ARJUN": ["AA", "R", "JH", "UH", "N"],
    "SRIRAM": ["SH", "R", "IY", "R", "AA", "M"], "PRIYA": ["P", "R", "IY", "Y", "AH"],
    "ANANYA": ["AH", "N", "AH", "N", "Y", "AH"], "DEEPIKA": ["D", "IY", "P", "IH", "K", "AH"],
    "JIHOON": ["JH", "IY", "HH", "UW", "N"], "HYUNWOO": ["HH", "Y", "AH", "N", "W", "UW"],
    "MINJI": ["M", "IH", "N", "JH", "IY"], "SEOYEON": ["S", "AH", "Y", "AH", "N"],
    "MINH": ["M", "IH", "N", "Y"], "QUANG": ["K", "W", "AA", "NG"],
    "THANH": ["T", "AE", "N", "Y"], "LINH": ["L", "IH", "N", "Y"],
    "TARIQ": ["T", "AA", "R", "IH", "K"], "BILAL": ["B", "IH", "L", "AA", "L"],
    "AYESHA": ["AY", "IY", "SH", "AH"], "ZOYA": ["Z", "OY", "AH"],
    "KENJI": ["K", "EH", "N", "JH", "IY"], "HARUKA": ["HH", "AA", "R", "UW", "K", "AH"],
    "LILIBETH": ["L", "IH", "L", "IH", "B", "EH", "TH"],
    "JAMES": ["JH", "EY", "M", "Z"], "MICHAEL": ["M", "AY", "K", "AH", "L"],
    "ROBERT": ["R", "AA", "B", "ER", "T"], "JOHN": ["JH", "AA", "N"],
    "DAVID": ["D", "EY", "V", "IH", "D"], "WILLIAM": ["W", "IH", "L", "Y", "AH", "M"],
    "RICHARD": ["R", "IH", "CH", "ER", "D"], "JOSEPH": ["JH", "OW", "S", "AH", "F"],
    "THOMAS": ["T", "AA", "M", "AH", "S"], "CHRISTOPHER": ["K", "R", "IH", "S", "T", "AH", "F", "ER"],
    "MATTHEW": ["M", "AE", "TH", "Y", "UW"], "DANIEL": ["D", "AE", "N", "Y", "AH", "L"],
    "MARY": ["M", "EH", "R", "IY"], "PATRICIA": ["P", "AH", "T", "R", "IH", "SH", "AH"],
    "JENNIFER": ["JH", "EH", "N", "AH", "F", "ER"], "LINDA": ["L", "IH", "N", "D", "AH"],
    "ELIZABETH": ["IH", "L", "IH", "Z", "AH", "B", "AH", "TH"], "BARBARA": ["B", "AA", "R", "B", "ER", "AH"],
    "SUSAN": ["S", "UW", "Z", "AH", "N"], "JESSICA": ["JH", "EH", "S", "IH", "K", "AH"],
    "SARAH": ["S", "EH", "R", "AH"], "KAREN": ["K", "EH", "R", "AH", "N"],
    "NANCY": ["N", "AE", "N", "S", "IY"], "LISA": ["L", "IY", "S", "AH"],
    "ASHLEY": ["AE", "SH", "L", "IY"],
}

def score_orthographic_opacity(w: str) -> float:
    rules = [
        (r"^X", 24), (r"X", 10), (r"^Q(?!U)", 26), (r"QU", 12),
        (r"^ZH", 22), (r"^NG", 25), (r"NH$", 16), (r"TH$", 12),
        (r"HY", 16), (r"EO", 16), (r"YI", 14), (r"AO|IAO|IU|UI", 14), (r"^AA", 12),
        (r"^SR", 18),
    ]
    pts = 0.0
    for pat, wt in rules:
        if re.search(pat, w):
            if pat == r"X" and w.startswith("X") and "X" not in w[1:]:
                continue
            pts += wt
    syllables = max(1, len(re.findall(r"[AEIOUY]+", w)))
    pts += max(0, (syllables - 2)) * 5.0
    rare_bg = {"XI", "YI", "HY", "EO", "NG", "NH", "QU", "AA", "JI", "KA", "YU", "WO", "SR"}
    pts += sum(6.0 for bg in [w[i:i+2] for i in range(len(w)-1)] if bg in rare_bg)
    return min(100.0, pts)

# 2. Build 50-Name Longitudinal DataFrame
name_specs_longitudinal = [
    ("Wei", "M", "Chinese", 420, 45, 380, 110), ("Xiaoming", "M", "Chinese", 310, 5, 295, 18),
    ("Mei", "F", "Chinese", 380, 95, 360, 290), ("Xinyi", "F", "Chinese", 290, 12, 340, 45),
    ("Aarav", "M", "Indian", 350, 140, 520, 890), ("Arjun", "M", "Indian", 410, 185, 580, 940),
    ("Sriram", "M", "Indian", 345, 70, 410, 230), ("Priya", "F", "Indian", 460, 210, 510, 690),
    ("Ananya", "F", "Indian", 340, 115, 490, 640), ("Deepika", "F", "Indian", 280, 25, 320, 95),
    ("Jihoon", "M", "Korean", 275, 20, 260, 55), ("Hyunwoo", "M", "Korean", 260, 10, 245, 30),
    ("Minji", "F", "Korean", 295, 30, 280, 85), ("Seoyeon", "F", "Korean", 240, 15, 275, 45),
    ("Minh", "M", "Vietnamese", 390, 110, 340, 310), ("Quang", "M", "Vietnamese", 315, 22, 260, 65),
    ("Thanh", "M", "Vietnamese", 340, 40, 290, 115), ("Linh", "F", "Vietnamese", 405, 130, 370, 350),
    ("Tariq", "M", "Pakistani", 310, 85, 350, 240), ("Bilal", "M", "Pakistani", 295, 75, 340, 220),
    ("Ayesha", "F", "Pakistani", 360, 120, 410, 380), ("Zoya", "F", "Pakistani", 270, 90, 360, 450),
    ("Kenji", "M", "Japanese", 250, 65, 230, 210), ("Haruka", "F", "Japanese", 230, 25, 220, 75),
    ("Lilibeth", "F", "Filipino", 265, 35, 240, 95),
    ("James", "M", "US_Baseline", 85, 4800, 110, 4600), ("Michael", "M", "US_Baseline", 95, 5100, 105, 3400),
    ("Robert", "M", "US_Baseline", 60, 4200, 65, 2100), ("John", "M", "US_Baseline", 75, 4500, 85, 2900),
    ("David", "M", "US_Baseline", 110, 4400, 130, 3300), ("William", "M", "US_Baseline", 55, 3900, 75, 4100),
    ("Richard", "M", "US_Baseline", 40, 3100, 45, 1200), ("Joseph", "M", "US_Baseline", 70, 3600, 80, 2800),
    ("Thomas", "M", "US_Baseline", 50, 3400, 65, 2600), ("Christopher", "M", "US_Baseline", 65, 4100, 70, 1900),
    ("Matthew", "M", "US_Baseline", 45, 3800, 60, 2500), ("Daniel", "M", "US_Baseline", 120, 3950, 145, 3500),
    ("Mary", "F", "US_Baseline", 70, 4100, 75, 1800), ("Patricia", "F", "US_Baseline", 45, 3200, 40, 850),
    ("Jennifer", "F", "US_Baseline", 60, 4400, 55, 950), ("Linda", "F", "US_Baseline", 35, 3000, 30, 650),
    ("Elizabeth", "F", "US_Baseline", 80, 3900, 95, 3600), ("Barbara", "F", "US_Baseline", 30, 2700, 25, 600),
    ("Susan", "F", "US_Baseline", 40, 2900, 35, 700), ("Jessica", "F", "US_Baseline", 55, 4200, 50, 1100),
    ("Sarah", "F", "US_Baseline", 75, 4050, 85, 2400), ("Karen", "F", "US_Baseline", 50, 2850, 35, 450),
    ("Nancy", "F", "US_Baseline", 35, 2600, 30, 550), ("Lisa", "F", "US_Baseline", 45, 3150, 40, 750),
    ("Ashley", "F", "US_Baseline", 25, 3700, 30, 900),
]

records = []
for name, sex, group, f95, n95, f25, n25 in name_specs_longitudinal:
    records.extend([
        {"first_name": name, "sex": sex, "birth_cohort": 1995, "is_foreign_born": 1, "PERWT": f95},
        {"first_name": name, "sex": sex, "birth_cohort": 1995, "is_foreign_born": 0, "PERWT": n95},
        {"first_name": name, "sex": sex, "birth_cohort": 2025, "is_foreign_born": 1, "PERWT": f25},
        {"first_name": name, "sex": sex, "birth_cohort": 2025, "is_foreign_born": 0, "PERWT": n25},
    ])

df_raw = pd.DataFrame(records)
df_raw["clean_name"] = clean_first_name(df_raw["first_name"])
grouped = (
    df_raw.groupby(["birth_cohort", "sex", "clean_name", "is_foreign_born"])["PERWT"]
    .sum().unstack(fill_value=0).rename(columns={0: "n_native", 1: "n_foreign"}).reset_index()
)
totals = grouped.groupby(["birth_cohort", "sex"])[["n_native", "n_foreign"]].transform("sum")
grouped["f_index"] = 100.0 * (grouped["n_foreign"] / totals["n_foreign"]) / (
    (grouped["n_foreign"] / totals["n_foreign"]) + (grouped["n_native"] / totals["n_native"])
)

comp = grouped.pivot(index=["clean_name", "sex"], columns="birth_cohort", values="f_index").reset_index()
comp.columns = ["clean_name", "sex", "f_index_1995", "f_index_2025"]
comp["f_index_change"] = comp["f_index_2025"] - comp["f_index_1995"]

meta = pd.DataFrame(name_specs_longitudinal, columns=["name_raw", "sex", "group", "_", "__", "___", "____"])
meta["clean_name"] = clean_first_name(meta["name_raw"])
comp = comp.merge(meta[["clean_name", "sex", "group"]], on=["clean_name", "sex"])

arpabet_res = comp["clean_name"].apply(lambda n: bigram_scorer(TARGET_ARPABET[n]))
comp["mean_bigram_bits"] = arpabet_res.apply(lambda x: x[0])
comp["orthographic_opacity"] = comp["clean_name"].apply(score_orthographic_opacity)

plt.rcParams.update({"axes.edgecolor": "#CCCCCC", "axes.linewidth": 0.8})
palette = {
    "Chinese": "#D95F02", "Indian": "#1B9E77", "Korean": "#7570B3",
    "Vietnamese": "#E7298A", "Pakistani": "#66A61E", "Japanese": "#E6AB02",
    "Filipino": "#A6761D", "US_Baseline": "#5C768D",
}

# 3. Zoomable Annotation Helper (Uniform styling for ALL points)
def annotate_all_zoomable(ax, df_sub, x_col, y_col):
    x_vals = df_sub[x_col].values
    y_vals = df_sub[y_col].values
    x_span = max(1e-6, x_vals.max() - x_vals.min())
    y_span = max(1e-6, y_vals.max() - y_vals.min())

    placed = []
    df_ordered = df_sub.sort_values([x_col, y_col])

    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for _, row in df_ordered.iterrows():
        x, y = row[x_col], row[y_col]
        nx, ny = x / x_span, y / y_span

        neighbors = sum(1 for px, py in placed if np.hypot(nx - px, ny - py) < 0.045)
        placed.append((nx, ny))

        angle = angles[neighbors % len(angles)]
        radius = 4.5 + 3.0 * (neighbors // len(angles))
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)

        ax.annotate(
            row["clean_name"],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.2,
            fontweight="normal",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
            zorder=4,
        )

def save_figure_outputs(fig: plt.Figure, base_filename: str, dpi: int = 300):
    fig.savefig(f"{base_filename}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{base_filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_filename}.svg", bbox_inches="tight")
    with open(f"{base_filename}.fig.pkl", "wb") as f:
        pickle.dump(fig, f)

# =========================================================================
# FIGURE 1: ALL 50 Names — 30-Year Shift in Name Foreignness (1995 -> 2025)
# =========================================================================
fig1, ax1 = plt.subplots(figsize=(10, 13), dpi=300)
dumb_df = comp.sort_values("f_index_2025").reset_index(drop=True)

y_pos = np.arange(len(dumb_df))
for idx, row in dumb_df.iterrows():
    color = "#2B8CBE" if row["f_index_change"] < 0 else "#E34A33"
    ax1.plot([row["f_index_1995"], row["f_index_2025"]], [idx, idx], color=color, alpha=0.65, lw=1.8)

ax1.scatter(dumb_df["f_index_1995"], y_pos, color="#999999", s=40, zorder=2, edgecolor="white", lw=0.4)
ax1.scatter(dumb_df["f_index_2025"], y_pos, color="#1F4E78", s=48, zorder=3, edgecolor="white", lw=0.4)
ax1.axvline(50, color="#AAAAAA", ls="--", lw=1.0)
ax1.text(51.5, 1.5, "Parity (F = 50)", color="#666666", fontsize=9, style="italic")

ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{r['clean_name']} ({r['group'][:3]})" for _, r in dumb_df.iterrows()], fontsize=8)

ax1.set_xlim(0, 105)
ax1.set_ylim(-1, len(dumb_df))
ax1.set_xlabel("Name Foreignness Index (0 = U.S.-Born Only, 100 = Foreign-Born Only)", fontsize=10)
ax1.set_title("30-Year Shift in Name Foreignness (1995 → 2025)", fontsize=12.5, fontweight="bold", pad=12, loc="left")
ax1.grid(axis="x", ls=":", alpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)

legend_elems = [
    Line2D([0], [0], marker="o", color="w", label="1995 F-Index", markerfacecolor="#999999", markersize=7),
    Line2D([0], [0], marker="o", color="w", label="2025 F-Index", markerfacecolor="#1F4E78", markersize=7.5),
    Line2D([0], [0], color="#2B8CBE", lw=2, label="2nd-Gen Diffusion (Δ < 0)"),
    Line2D([0], [0], color="#E34A33", lw=2, label="Cohort Aging Shift (Δ > 0)"),
]
ax1.legend(handles=legend_elems, loc="lower right", fontsize=9, frameon=True, facecolor="white")
plt.tight_layout()
save_figure_outputs(fig1, "chart1_f_index_shift")

# =========================================================================
# FIGURE 2: 2025 F-Index vs. ARPAbet Phoneme Bigram Surprisal (All 50 Labeled)
# =========================================================================
fig2, ax2 = plt.subplots(figsize=(10.5, 7.5), dpi=300)
for grp, grp_df in comp.groupby("group"):
    ax2.scatter(
        grp_df["mean_bigram_bits"], grp_df["f_index_2025"],
        label=grp.replace("_", " "), color=palette[grp], s=60, alpha=0.88, edgecolor="white", lw=0.6
    )

annotate_all_zoomable(ax2, comp, "mean_bigram_bits", "f_index_2025")

ax2.axhline(50, color="#BBBBBB", ls="--", lw=1.0)
ax2.set_xlim(1.6, 8.75)
ax2.set_ylim(4, 105)
ax2.set_xlabel("Mean ARPAbet Phoneme Bigram Surprisal (Bits — Higher = Rarer Sound Sequence)", fontsize=10)
ax2.set_ylabel("2025 Foreignness Index (F-Index)", fontsize=10)
ax2.set_title("2025 Foreignness Index vs. ARPAbet Phoneme Rarity", fontsize=12.5, fontweight="bold", pad=12, loc="left")
ax2.grid(True, ls=":", alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(ncol=2, fontsize=8.5, loc="lower right", frameon=True, facecolor="white")
plt.tight_layout()
save_figure_outputs(fig2, "chart2_foreignness_vs_phoneme_rarity")

# =========================================================================
# FIGURE 3: Orthographic Opacity vs. Phonotactic Rarity (All 25 Asian Labeled)
# =========================================================================
fig3, ax3 = plt.subplots(figsize=(10.5, 7.5), dpi=300)
eth = comp[comp["group"] != "US_Baseline"].copy()
for grp, grp_df in eth.groupby("group"):
    ax3.scatter(grp_df["orthographic_opacity"], grp_df["mean_bigram_bits"],
                label=grp, color=palette[grp], s=70, alpha=0.9, edgecolor="white", lw=0.6)

ax3.axvline(eth["orthographic_opacity"].median(), color="#BBBBBB", ls="--", lw=1.0)
ax3.axhline(eth["mean_bigram_bits"].median(), color="#BBBBBB", ls="--", lw=1.0)

annotate_all_zoomable(ax3, eth, "orthographic_opacity", "mean_bigram_bits")

ax3.text(1.5, 8.35, "Sound-Rare Only\n(Easy spelling, rare phonemes)", fontsize=8, color="#555555", style="italic")
ax3.text(31, 8.35, "Double Friction\n(Spelling trap + rare phonemes)", fontsize=8, color="#990000", style="italic")
ax3.text(1.5, 5.45, "High Fluency\n(Transparent spelling + common sounds)", fontsize=8, color="#1B7837", style="italic")
ax3.text(31, 3.3, "Spelling Trap Only\n(Opaque spelling, familiar sounds)", fontsize=8, color="#555555", style="italic")

ax3.set_xlim(-3.5, 58)
ax3.set_ylim(1.9, 8.75)
ax3.set_xlabel("Orthographic Opacity Score (Spelling-to-Sound Mismatch)", fontsize=10)
ax3.set_ylabel("Mean ARPAbet Bigram Surprisal (Bits)", fontsize=10)
ax3.set_title("Separating Spelling Traps vs. Phoneme Rarity (All 25 Asian Names Labeled)", fontsize=12.5, fontweight="bold", pad=12, loc="left")
ax3.grid(True, ls=":", alpha=0.5)
ax3.spines[["top", "right"]].set_visible(False)
ax3.legend(ncol=2, fontsize=8, loc="lower right", frameon=True, facecolor="white")
plt.tight_layout()
save_figure_outputs(fig3, "chart3_spelling_vs_phoneme_rarity")

figures = {
    "fig1_f_index_shift": fig1,
    "fig2_foreignness_vs_phoneme_rarity": fig2,
    "fig3_spelling_vs_phoneme_rarity": fig3,
}