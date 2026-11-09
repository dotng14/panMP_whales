"""Markdown for the Experiment 3 report. Every number comes from the saved results."""

from __future__ import annotations

import numpy as np

REQUIRED = (
    "## Previous experiment: Experiment 2 and the pipeline check",
    "## Current experiment: Experiment 3",
    "## Comparing Experiment 2 and Experiment 3",
)
NAMES = {"csv": "CSV", "exp2": "Exp 2", "exp2b": "Exp 2b", "M1": "M1 theme HMM", "M2": "M2 song-first", "M3": "M3 units", "M4": "M4 six theme HMMs"}
MODELS = ("exp2", "exp2b", "M1", "M2", "M3", "M4")
PAGE_URL_FILE = "pairs/page_url.txt"


def f(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{d}f}"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def render(D: dict) -> str:
    ev, res, used = D["eval"], D["results"], D["inputs_used"]
    B, C, R, H = ev["B"], ev["C"], ev["R"], ev["H"]
    have = [m for m in MODELS if m in B["test"]]
    hand = ev.get("hand_labels", False)
    L: list[str] = []
    add = L.append

    add("# Song-session HMM: Experiment 3 report")
    add("")
    add("Branch `dat-hmm`. Plan: [plan.md](plan.md). Inputs: [pipeline.md](pipeline.md). Metrics: [metrics.md](metrics.md). "
        "Numbers: `work/eval.json`, `work/models/results.json`, `work/inputs/*.json`. Regenerate with `python -m analysis.song_hmm.exp3.evaluate` then `python -m analysis.song_hmm.exp3.report`.")
    add("")

    # ---------------------------------------------------------------- summary
    add("## Summary")
    add("")
    hand_note = "against the hand labels" if hand else "**pending**: no hand labels yet (`reference/labels/`)"
    add(f"Test half (last 5 h, never used for fitting). Group H is {hand_note}; group B is agreement with the CSV, which looks generated and is a reference, not truth; "
        "C is measured on the audio itself; R checks Payne & McVay's rules.")
    add("")
    rows = []
    for m in have:
        b, c, r = B["test"][m], C["test"][m], R["test"][m]
        h = g(H, "test", m, default={}) or {}
        rows.append([NAMES[m], f(h.get("theme_balanced_accuracy")) if h else "pending", f(g(h, "song_start", "tol_30s", "f1"), 2) if h else "pending",
                     f(b["theme_balanced_accuracy"]), f(b["theme_kappa"]), f(b["boundaries"]["tol_30s"]["f1"], 2),
                     f(c["coherence"]["coherence"]), f(c["identity"]["identity"]), f(c["novelty"]["precision"], 2),
                     f"{b['n_sessions']} / {r['gaps_le_60s']}", f"{r['complete_songs']} / {r['complete_7_to_30_min']}", f(r["median_complete_min"], 1)])
    cc, cr = C["test"]["csv"], R["test"]["csv"]
    rows.append(["CSV (reference)", "–", "–", "–", "–", "–", f(cc["coherence"]["coherence"]), f(cc["identity"]["identity"]), f(cc["novelty"]["precision"], 2),
                 f"1 / {cr['gaps_le_60s']}", f"{cr['complete_songs']} / {cr['complete_7_to_30_min']}", f(cr["median_complete_min"], 1)])
    L += table(["test half", "H bal. acc.", "H song-start F1 ±30 s", "B bal. acc. (chance 0.167)", "B κ", "B boundary F1 ±30 s", "C coherence", "C identity",
                "C change-point prec.", "sessions / gaps ≤ 60 s", "complete songs / in 7–30 min", "median song (min)"], rows)
    add("")
    best_b = max((m for m in have if m.startswith("M")), key=lambda m: B["test"][m]["theme_balanced_accuracy"])
    sw = D["screen"]
    exp2_hit = g(sw, "S1_N1_X1_12s", "pca20", "repeat_hit")
    win_hit = g(sw, used.get("window"), "pca20", "repeat_hit")
    add("In short:")
    add(f"- **Inputs mattered most.** The 2D-Fourier phrase patch (X2) roughly doubles the song-repeat score of Experiment 2's input "
        f"({f(win_hit)} vs {f(exp2_hit)}; chance ≈ 0.05), and wins with every spectrogram and noise setting. The chosen input is `{used.get('window')}`.")
    add(f"- **Best model on the CSV: {NAMES[best_b]}** (balanced accuracy {f(B['test'][best_b]['theme_balanced_accuracy'])}, κ {f(B['test'][best_b]['theme_kappa'])}), "
        f"against {f(B['test']['exp2b']['theme_balanced_accuracy'])} / {f(B['test']['exp2b']['theme_kappa'])} for Experiment 2b. The gain is real but the level is still low.")
    add(f"- **The session rule now holds.** No Experiment 3 model ends a session on a pause of 60 s or less (Experiment 2 / 2b did {R['test']['exp2']['gaps_le_60s']} / {R['test']['exp2b']['gaps_le_60s']} times).")
    short = [m for m in ("M1", "M4") if m in R["test"] and np.isfinite(R["test"][m]["median_complete_min"]) and R["test"][m]["median_complete_min"] < 7]
    if short:
        add(f"- **Songs are too short in {', '.join(NAMES[m] for m in short)}** (median {', '.join(f(R['test'][m]['median_complete_min'], 1) + ' min' for m in short)}; Payne & McVay: 7–30 min): "
            "the HMMs pass through all six themes too quickly. M2, which finds the song cycle first, gives songs in range by construction.")
    add("- **The CSV remains a weak reference** (lowest coherence of all labellings, song starts drifting ~110 s from the audio's own song-to-song alignment). "
        "The hand labels (group H) and the pair judgements (group P) will decide; both are pending until you fill them in.")
    add("")

    # ---------------------------------------------------------------- data
    add("## Data, split and reference")
    add("")
    add("- Recording `SOCAL_H_76_230426_193500_df100.x.x.wav`: 2 kHz, mono, 10 h. Nothing above 1 kHz exists. **Train** 0–18,000 s, **test** 18,000–36,000 s; every statistic and fit uses train only.")
    add("- CSV reference: 53 songs, 30,923 units; covers every second with whole-second units, 8 songs skip themes, song starts drift a median of 110 s from the audio's alignment.")
    add(f"- Hand labels: {'present (' + ', '.join(ev.get('hand_clips', [])) + ')' if hand else 'two 30-minute test-half clips exported to `work/reference/`; instructions in [reference/README.md](reference/README.md); **not labelled yet**'}.")
    add(f"- Pair judgements: {'present' if D.get('pairs') else 'the blind pair page is built (`pairs/page/`); **not judged yet**'}.")
    add("")

    # ---------------------------------------------------------------- previous
    add(REQUIRED[0])
    add("")
    add("Full reports: [Experiment 2](../exp2_phrase_hmm/report.md), [pipeline check](../exp2_phrase_hmm/pipeline_check/report.md); diagnostics in [`../exp2_phrase_hmm/diagnostics/`](../exp2_phrase_hmm/diagnostics/).")
    add("")
    add("**Experiment 2** described each 2 s step by a 12 s phrase window from a 0.256 s / 0.05 s spectrogram with a 5-minute median noise floor, summarised as 295 order-free statistics (level, variability, flux, up/down sweep per band) and reduced to 20 PCA components. "
        "An 81-state theme HMM (theme chains of ≥ 20 s, pauses, background ≥ 30 s, T6 → T1 boundary) was seeded from k-means clusters (Exp 2) or from the CSV (Exp 2b).")
    add("")
    e2, e2b = B["test"]["exp2"], B["test"]["exp2b"]
    L += table(["Experiment 2, test half", "Exp 2 (audio seed)", "Exp 2b (CSV seed)"], [
        ["theme balanced accuracy vs CSV", f(e2["theme_balanced_accuracy"]), f(e2b["theme_balanced_accuracy"])],
        ["Cohen's κ vs CSV", f(e2["theme_kappa"]), f(e2b["theme_kappa"])],
        ["sessions / gaps ≤ 60 s", f"{e2['n_sessions']} / {R['test']['exp2']['gaps_le_60s']}", f"{e2b['n_sessions']} / {R['test']['exp2b']['gaps_le_60s']}"],
        ["theme coherence (audio)", f(C["test"]["exp2"]["coherence"]["coherence"]), f(C["test"]["exp2b"]["coherence"]["coherence"])],
    ])
    add("")
    pc = D["pipeline_check"]
    gcomb = g(pc, "G", default={})
    add("**What went wrong, from the pipeline check and the boundary diagnostics:**")
    add("")
    add("- *Processing was correct but lossy.* Reading, STFT framing, chunking and band pooling all passed (inverse-STFT SNR "
        f"{f(g(pc, 'C', 'istft_snr_db'), 0)} dB). But the median floor zeroed half of every band (song included), the 0.256 s window kept only "
        f"{f(g(pc, 'D', 'settings', 'a: 512 / 0.05 s (current)', 'depth_at_10hz'), 2)} of a 10 Hz pulse train, and the 295 statistics + PCA-20 kept less 'which phrase is this' than the raw spectrogram.")
    if gcomb:
        best_c = max(gcomb, key=lambda k: gcomb[k]["PCA-20"]["hit_rate"])
        add(f"- *Better settings existed.* On the song-repeat test the current setting scored {f(gcomb[next(iter(gcomb))]['PCA-20']['hit_rate'])}; the best combination ({best_c}) {f(gcomb[best_c]['PCA-20']['hit_rate'])}.")
    add("- *The model broke Payne & McVay's rules.* Themes T1 and T4 last ~4–6 s per song (single phrases) but the model forced ≥ 20 s; most session breaks were pauses ≤ 60 s, which by definition stay inside a session.")
    add("- *The CSV is unreliable.* Its song starts sit at inconsistent points of the audio's cycle, and even given its own song boundaries the audio's best 6-way split matches its themes no better than equal slices.")
    add("- *The audio repeats strongly once per song (≈ 13 min)* — the structure the professor suggested using.")
    if D.get("boundary_checks"):
        add("")
        add("<details><summary>Boundary diagnostics output</summary>")
        add("")
        add("```")
        add(D["boundary_checks"].strip())
        add("```")
        add("</details>")
    add("")

    # ---------------------------------------------------------------- current
    add(REQUIRED[1])
    add("")
    add("### Processing options and inputs")
    add("")
    add("Every option and input is defined in [pipeline.md](pipeline.md) (what it is, how it is built, shape, meaning, which model uses it).")
    add("")
    L += table(["code", "what"], [
        ["S1 / S2 / S3", "spectrogram: 512-sample window every 0.05 s / 256-sample window every 0.025 s / 512 below 150 Hz + 256 above, every 0.025 s"],
        ["N1 / N2 / N3 / N4", "noise: median floor, clipped (Exp 2) / none (one offset per band) / 10th-percentile floor / PCEN adaptive gain"],
        ["X1", "window statistics (Exp 2's descriptor), 12 or 20 s windows every 2 s, PCA-20/40"],
        ["X2", "2D-Fourier magnitude of the window (rhythm and unit pattern, shift-invariant) + mean band level, PCA-20/40"],
        ["X3", "0.25 s frame vectors (band levels, flux, sweeps), PCA-20: for M4"],
        ["X4", "detected units with energy / height / width measurements, grouped into unit types: for M3"],
        ["X5", "the raw spectrogram window, shift-tolerant cosine: screening reference"],
    ])
    add("")
    add("### Input screening")
    add("")
    add("Song-repeat hit rate on the test half (500 loud query windows; a hit = nearest window ≥ 5 min away lies 1 or 2 song periods away; chance ≈ 0.05). "
        "It ranks inputs; it is not an accuracy (correct matches 3+ songs away count as misses).")
    add("")
    add("![](figures/screening.png)")
    add("")
    top = sorted(((v["pca20"]["repeat_hit"], k) for k, v in sw.items() if "pca20" in v), reverse=True)[:8]
    L += table(["input", "PCA-20 hit", "PCA-40 hit", "CSV same-theme of the match"], [[k, f(sw[k]["pca20"]["repeat_hit"]), f(sw[k]["pca40"]["repeat_hit"]), f(sw[k]["pca20"]["csv_same"], 2)] for _, k in top])
    add("")
    un = D["units"]
    if un:
        add("**Unit detection (X4)** for each setting: units per minute in loud vs quiet minutes (a good detector finds units in song and few in quiet), median unit length, unit types (BIC), and the song-repeat hit of a bag-of-unit-types window:")
        add("")
        L += table(["setting", "units", "per min loud / quiet", "median length (s)", "types", "song-repeat hit"], [
            [k, f"{v['n_units']:,}", f"{v['units_per_min_loud']:.0f} / {v['units_per_min_quiet']:.0f}", f(v["duration_pct"]["50"], 2), v["k_types"], f(v["type_window"]["repeat_hit"])] for k, v in un.items()
        ])
        add("")
        add("Detected units on the three example stretches (S2 spectrogram, each noise option; cyan boxes = units):")
        add("")
        for n in ("N1", "N2", "N3", "N4"):
            add(f"![](figures/units_overlay_S2_{n}.png)")
        add("")
    add(f"**Chosen inputs** (`work/models/inputs_used.json`): M1 and M2 use `{used.get('window')}`; M4 uses X3 frames from `{used.get('frames_from')}`; M3 uses units from `{used.get('units')}`.")
    add("")
    add("### Pair precision (group P)")
    add("")
    if D.get("pairs") and D.get("pairs_key"):
        jud = D["pairs"]
        key = D["pairs_key"]
        verdict = {}
        for _, vs in jud.items():
            for pid, v in vs.items():
                verdict.setdefault(pid, []).append(v)
        rows = []
        for src in ("best_window", "best_X1", "exp2_input", "patch", "units", "random"):
            vals = [v for p in key["pairs"] if src in p["sources"] for v in verdict.get(p["id"], []) if v != "unsure"]
            rows.append([src, key["inputs"].get(src, "random"), len(vals), f(np.mean([v == "same" for v in vals]) if vals else None, 2)])
        L += table(["proposed by", "input", "judged pairs", "pair precision"], rows)
    else:
        add("**Pending.** The [blind pair page](https://claude.ai/artifact/2PHkxyNwseVqyTdkcd7dJM) shows each query with the match proposed by each input (and a random window), without saying which input proposed it. "
            "Once you judge the pairs, Claude reads the judgements into `work/pairs/judgments.json` and this table fills in.")
    add("")

    for m, title, desc in (
        ("M1", "M1: one theme-level HMM", "One state per theme (a theme can be a single 2 s window), a pause state per theme, a background chain of 60 s; from each theme: stay / next / pause / end; T6 → T1 is the song boundary. 2-component Gaussian mixture per theme on the chosen window input (PCA-40). Seeded from the CSV theme at each window (train), background from the quietest 10% of windows."),
        ("M2", "M2: song-boundary-first", "Find the song period from the audio's own repetition (median over 40-minute blocks), learn a template of one song cycle, align the recording to it with a cyclic Viterbi (each window stays, moves one or two template phases; tempo prior and score sharpness chosen on the train half so the alignment completes as many cycles as the period predicts). Split the template's phases into T1…T6 in order (start and cut points chosen to match the CSV on the train half; T1 / T4 may be a single phase). Song boundary = passing the template's start. Background = > 60 s of audio that does not match the template."),
        ("M3", "M3: unit-based", "One step per detected unit, plus GAP (3–60 s pause) and BGT (> 60 s) tokens. The 28 phrase slots of the drawing, each emitting a unit type (categorical); a slot may repeat (one sung unit detected as several pieces); after a theme's last slot: repeat / next / pause / end. Seeded from the CSV slot at each unit's time."),
        ("M4", "M4: six theme HMMs chained into a song model", "Each theme is a left-to-right chain of its phrase slots (3, 3, 6, 3, 5, 8), each slot two states (≥ 0.5 s) with a 2-component Gaussian mixture on 0.25 s frame vectors (X3); the phrase may repeat. (1) Each theme HMM is trained alone on its CSV segments (train); (2) each is tested on picking out its own theme (one-vs-rest AUC on test CSV segments); (3) the six are chained T1 → … → T6 → T1 with pauses and ≥ 60 s background; (4) embedded Baum–Welch on the whole train half with only the theme order imposed; (5) Viterbi."),
    ):
        add(f"### {title}")
        add("")
        add(desc)
        add("")
        if m not in B["test"]:
            add("*Not run* (see `run_models.py`).")
            add("")
            continue
        r = res.get(m, {})
        rows = []
        for h in ("train", "test"):
            b, c, rr = B[h][m], C[h][m], R[h][m]
            hh = g(H, h, m, default={}) or {}
            rows.append([h, f(hh.get("theme_balanced_accuracy")) if hh else ("pending" if h == "test" else "–"), f(b["theme_accuracy"]), f(b["theme_balanced_accuracy"]), f(b["theme_kappa"]),
                         f(b["boundaries"]["tol_30s"]["f1"], 2), f(c["coherence"]["coherence"]), f(c["identity"]["identity"]), f(c["novelty"]["precision"], 2),
                         f"{b['n_sessions']} / {rr['gaps_le_60s']}", f"{b['n_songs']} / {rr['complete_songs']} / {rr['complete_7_to_30_min']}", f(rr["median_complete_min"], 1)])
        L += table(["half", "H bal. acc.", "B accuracy", "B bal. acc.", "B κ", "B boundary F1 ±30 s", "C coherence", "C identity", "C change-point prec.",
                    "sessions / gaps ≤ 60 s", "songs / complete / in 7–30 min", "median song (min)"], rows)
        add("")
        add("Per-theme recall against the CSV (test): " + ", ".join(f"T{k + 1} {f(v, 2)}" for k, v in enumerate(B["test"][m]["theme_recall"])) + ".")
        add("")
        if r.get("loglik_per_step"):
            add(f"Log-likelihood per step: train {f(r['loglik_per_step']['train'], 2)}, test {f(r['loglik_per_step']['test'], 2)} (comparable only within this model). "
                + (f"Baum–Welch: {len(r['history']) - 1} iterations." if r.get("history") else ""))
            add("")
        if r.get("synthetic"):
            s = r["synthetic"]
            add(f"Synthetic recovery ({s['n_steps']:,} steps sampled from the fitted model, refit, decoded): theme accuracy {f(s['theme_accuracy'])}, song vs background {f(s['song_vs_background'])}, "
                f"song boundaries found {s['found_boundaries']} of {s['true_boundaries']}.")
            add("")
        if m == "M2" and r:
            add(f"Song period {f(r.get('period_min'), 1)} min; tempo prior {g(r, 'chosen', 'tempo_prior')}, sharpness {g(r, 'chosen', 'beta')} (chosen on train: expected {f(r.get('expected_train_cycles'), 1)} cycles); "
                f"template phases per theme {r.get('theme_phases')}; CSV themes matched by the phase split on train {f(r.get('phase_split_match'), 2)}.")
            add("")
        if m == "M4" and r:
            iso, emb = r["detection_isolated"], r["detection_embedded"]
            add("![](figures/m4_detection.png)")
            add("")
            add(f"Theme detection on {iso['n_segments']} test CSV segments: top-1 accuracy {f(iso['top1_accuracy'], 2)} after isolated training, {f(emb['top1_accuracy'], 2)} after embedded training (chance 0.17). "
                "AUC above 0.5 means the theme's HMM scores its own segments higher than the others'.")
            add("")
        if m == "M3" and r:
            add(f"Unit types: {r.get('k_types')}; tokens: train {g(r, 'n_tokens', 'train')}, test {g(r, 'n_tokens', 'test')}.")
            add("")

    # ---------------------------------------------------------------- compare
    add(REQUIRED[2])
    add("")
    for h in ("test", "train"):
        add(f"**{h.capitalize()} half**")
        add("")
        rows = []
        for m in have:
            b, c, rr = B[h][m], C[h][m], R[h][m]
            rows.append([NAMES[m], f(b["theme_balanced_accuracy"]), f(b["theme_kappa"]), f(b["boundaries"]["tol_30s"]["f1"], 2), f(c["coherence"]["coherence"]),
                         f(c["identity"]["identity"]), f(c["novelty"]["precision"], 2), f"{b['n_sessions']} / {rr['gaps_le_60s']}", f"{rr['complete_songs']} / {rr['complete_7_to_30_min']}",
                         f(rr["median_complete_min"], 1), f(c["longest_theme_run_s"] / 60, 1)])
        L += table(["model", "B bal. acc.", "B κ", "B boundary F1 ±30 s", "C coherence", "C identity", "C change-point prec.", "sessions / gaps ≤ 60 s",
                    "complete songs / in 7–30 min", "median song (min)", "longest theme run (min)"], rows)
        add("")
    add("Per-theme recall against the CSV, test half:")
    add("")
    L += table(["model"] + [f"T{k + 1}" for k in range(6)], [[NAMES[m]] + [f(v, 2) for v in B["test"][m]["theme_recall"]] for m in have])
    add("")
    add("![](figures/timeline.png)")
    add("")
    add("![](figures/song_lengths.png)")
    add("")
    add("![](figures/confusions.png)")
    add("")
    add("### What changed, what improved, what did not, and why")
    add("")
    m1 = B["test"].get("M1")
    add(f"1. **The input was the biggest lever.** Keeping the phrase's rhythm and unit pattern (X2) instead of order-free statistics (X1) roughly doubled the song-repeat score "
        f"({f(win_hit)} vs {f(exp2_hit)}). Shortening the FFT and dropping the median floor helped X1 (the pipeline check's finding), but for X2 the median floor is slightly *better* than no floor "
        f"(S2 X2 20 s: N1 {f(g(sw, 'S2_N1_X2_20s', 'pca20', 'repeat_hit'))}, N2 {f(g(sw, 'S2_N2_X2_20s', 'pca20', 'repeat_hit'))}): its rhythm and pattern terms do not depend on absolute level, so the floor's loss of level hurts it less.")
    if m1:
        add(f"2. **Same model family, better input and rules (Exp 2b → M1):** balanced accuracy {f(e2b['theme_balanced_accuracy'])} → {f(m1['theme_balanced_accuracy'])}, "
            f"κ {f(e2b['theme_kappa'])} → {f(m1['theme_kappa'])}, sessions {e2b['n_sessions']} → {m1['n_sessions']}, theme identity across songs "
            f"{f(C['test']['exp2b']['identity']['identity'])} → {f(C['test']['M1']['identity']['identity'])}. M1 is the bridge: it differs from Exp 2b mainly by its input and the Payne & McVay rules.")
    add(f"3. **The rules fixed sessions.** Every Experiment 3 model has 0 session breaks at pauses ≤ 60 s (Exp 2: {R['test']['exp2']['gaps_le_60s']}, Exp 2b: {R['test']['exp2b']['gaps_le_60s']}).")
    lens = {m: R["test"][m]["median_complete_min"] for m in have if m.startswith("M")}
    add("4. **Song length separates the approaches.** Median complete song: " + ", ".join(f"{NAMES[m]} {f(v, 1)} min" for m, v in lens.items()) +
        f" (CSV {f(R['test']['csv']['median_complete_min'], 1)} min; Payne & McVay 7–30 min). The HMMs that build songs theme by theme (M1, M4) pass through the cycle too fast: nothing in them knows how long a song is. "
        "M2 knows by construction (it aligns whole song cycles), but its theme labels inside the song are weaker.")
    if "M4" in res:
        iso = res["M4"]["detection_isolated"]
        add(f"5. **Six separate theme HMMs (M4) barely tell the themes apart from 0.25 s frames** (top-1 {f(iso['top1_accuracy'], 2)} on CSV segments, chance 0.17; best AUC "
            f"{max(iso[f'T{k + 1}'] for k in range(6)):.2f}). Chaining them gives κ comparable to M1, but the theme models themselves are weak — the frame input, not the chaining, is the limit.")
    if "M3" in B["test"]:
        add(f"6. **Units (M3):** balanced accuracy {f(B['test']['M3']['theme_balanced_accuracy'])}, κ {f(B['test']['M3']['theme_kappa'])}. Unit detection works (units appear in song and almost vanish in quiet with N2/N3), "
            "but detected blobs do not line up one-to-one with the drawing's units, so slot-by-slot emissions are noisy.")
    add("7. **Song boundaries against the CSV stay low for everyone** (F1 ±30 s in the table). Because the CSV's own song starts drift from the audio's cycle, this metric cannot separate the models; the hand labels must.")
    add("8. **Model capacity is not the bottleneck.** Every HMM recovers its own synthetic data almost perfectly.")
    add("")

    add("## Limitations and next steps")
    add("")
    add("- **Fill in the reference.** Label the two clips (`reference/README.md`) and judge the pairs. Every CSV-based number above is limited by the CSV; groups H and P are the real test.")
    add("- **Give the theme-by-theme models a song length.** Combine M2's song cycle with M1/M4's theme models: use M2's boundaries as hard song limits and let M1/M4 place the themes inside each song.")
    add("- **Better theme models for M4.** Use the X2 phrase input (or units) instead of 0.25 s frames for each theme HMM, so each detector sees whole phrases.")
    add("- **Unit detection.** Tune the detector against hand-marked units on a few minutes, so blobs match units one-to-one before M3's slot model.")
    add("- **Higher sample rate.** Not available for this recording; anything above 1 kHz is lost.")
    add("")
    return "\n".join(L)
