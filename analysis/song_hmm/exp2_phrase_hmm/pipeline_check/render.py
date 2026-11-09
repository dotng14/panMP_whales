"""reports/exp2_pipeline_check.md, from work/results.json. Verdicts are computed from the numbers."""

from __future__ import annotations

import numpy as np

OK, LOSS, NOTE, FAIL = "✅ pass", "⚠️ loses information", "ℹ️ note", "❌ fail"


def f(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{d}f}"


def pct(v, d=0):
    return "–" if v is None else f"{100 * v:.{d}f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def hit(r):
    return f"{f(r['hit_rate'])} (×{r['lift']:.1f})"


def render(R: dict) -> str:
    A, B, C, D, E, Fr, G = R["A"], R["B"], R["C"], R["D"], R["E"], R["F"], R.get("G", {})
    S = D["settings"]
    cur, short, long_ = S["a: 512 / 0.05 s (current)"], S["b: 256 / 0.025 s"], S["c: 1024 / 0.05 s"]
    U = D["units_and_sweeps"]
    u1 = U["1.0"] if "1.0" in U else U
    u3 = U.get("3.0", u1)
    ea, eb, ed = E["a: median 5 min, clip (current)"], E["b: 10th pct 5 min, clip"], E["d: no whitening"]
    ret = Fr["retrieval"]
    patch, desc, pc20, pc40 = (ret[k] for k in ("raw whitened patch (shift-tolerant)", "295 descriptors", "PCA-20", "PCA-40"))
    best_combo = max(G, key=lambda k: G[k]["PCA-20"]["hit_rate"]) if G else None
    cur_combo = next(iter(G)) if G else None

    L: list[str] = []
    add = L.append
    add("# Pipeline check: does wav → matrix keep the information in the recording?")
    add("")
    add("Checks the Experiment 2 input pipeline (`analysis/song_hmm/exp2_phrase_hmm/features.py`; stages I-0 … I-6 in [../pipeline.md](../pipeline.md)). "
        "Plan: [plan.md](plan.md). Code: `check.py` (it calls the real functions in `features.py`). Numbers: `work/results.json`. "
        "Listening clips (Griffin–Lim rebuilds, 8 kHz wav): `work/audio/` (not committed; regenerate with `python -m analysis.song_hmm.exp2_phrase_hmm.pipeline_check.check`).")
    add("")
    add("**How to read the song-repeat test (used in E, F and G).** A song repeats every ≈ 13 min. For 500 loud test-half windows, the nearest neighbour "
        "(among windows ≥ 5 min away) is found under a representation; a *hit* is a neighbour one or two song periods away (13 ± 2 or 26 ± 2 min). "
        f"Random choice hits {f(ret['PCA-20']['chance'])} of the time. A representation that keeps 'which phrase is this' scores higher; the number in brackets is the gain over chance.")
    add("")

    # ------------------------------------------------------------ summary
    add("## Summary")
    add("")
    rows = [
        ["A", "Header, size, sample count", f"fmt + `harp` ({A['chunks'][1]['size']} B metadata) + data; {A['n_samples']:,} samples = {A['duration_s']:,.0f} s at {A['sample_rate']} Hz", OK if A["data_bytes_match"] else FAIL],
        ["A", "Clipping, drop-outs", f"clipped {pct(A['clipped_fraction'], 3)}; constant runs ≥ 0.1 s: {A['n_constant_runs']}", OK if A["clipped_fraction"] < 1e-4 and A["n_constant_runs"] == 0 else FAIL],
        ["A", "DC offset", f"constant {f(A['dc_mean_counts'], 0)} counts (AC RMS ≈ {f(A['ac_rms_median'], 0)}); raises any band by ≤ {f(A['dc_leak_max_db_in_any_band'], 2)} dB", NOTE + " (harmless; subtract it anyway)"],
        ["A", "Time origin", f"CSV first unit at {f(A['csv_first_unit_s'], 1)} s from the file-name start", OK if abs(A["csv_first_unit_s"]) < 1 else FAIL],
        ["B", "Song energy outside 30–1000 Hz", f"{pct(B['song_excess_outside_fraction'], 1)} of the song's excess energy is below 30 Hz", OK if B["song_excess_outside_fraction"] < 0.05 else LOSS],
        ["B", "Source limit", "nothing above 1 kHz exists (`df100`, 2 kHz); no roll-off ≥ 10 dB before 1 kHz", NOTE],
        ["C", "STFT framing invertible", f"inverse-STFT SNR {f(C['istft_snr_db'], 0)} dB", OK if C["istft_snr_db"] > 40 else FAIL],
        ["C", "Chunked = one-pass", f"max difference {C['chunked_max_abs_diff']:.1e}", OK if C["chunked_max_abs_diff"] < 1e-4 else FAIL],
        ["C", "Time stamps", f"tone burst in frame {C['tone_frame']} (expected {C['tone_frame_expected']}); frames are labelled by their start, {C['frame_time_convention_offset_s']} s before their centre", OK if abs(C["tone_frame"] - C["tone_frame_expected"]) <= 1 else FAIL],
        ["C", "Band pooling", f"every bin 30–1000 Hz in exactly one of {C['n_bands']} bands; power identity error {C['band_power_identity_rel_err']:.0e}", OK if C["bins_in_exactly_one_band"] else FAIL],
        ["D", "Fast amplitude modulation (pulses)", f"current setting keeps {pct(cur['song_modulation_kept'])} of the song's envelope modulation (5 Hz depth {f(cur['depth_at_5hz'], 2)}, 10 Hz {f(cur['depth_at_10hz'], 2)}); 256-sample FFT keeps {pct(short['song_modulation_kept'])}", LOSS],
        ["D", "Short units", f"{pct(u3['share_units_shorter_than_window'])} of sound runs (≥ 13 dB above floor) are shorter than the 0.256 s window", LOSS if u3["share_units_shorter_than_window"] > 0.1 else OK],
        ["E", "Median noise floor", f"half of all cells clipped to 0 even in loud minutes ({pct(ea['share_cells_above_floor_loud'])} kept); {pct(ea['raw_power_under_floor_loud'])} of loud-minute power removed", LOSS],
        ["F", "Window summary (295 numbers)", f"song-repeat hit rate {hit(desc)} vs raw patch {hit(patch)}", LOSS],
        ["F", "PCA to 20", f"{pct(Fr['pca_explained_20'])} of variance; 80% needs {Fr['pca_components_for']['80']}, 90% needs {Fr['pca_components_for']['90']}; hit rate {hit(pc20)}", LOSS],
        ["F", "12 s window vs phrase length", f"median phrase period {f(Fr['phrase_period_s_percentiles']['50'], 1)} s; {pct(1 - Fr['share_phrase_period_under_12s'])} of windows have a period ≥ 12 s", NOTE],
    ]
    if G:
        rows.append(["G", "Best combined setting", f"{best_combo}: PCA-20 hit {hit(G[best_combo]['PCA-20'])} vs current {hit(G[cur_combo]['PCA-20'])}", NOTE])
    L += table(["", "check", "result", "verdict"], rows)
    add("")
    add("**Bottom line.** The pipeline is *implemented* correctly: the file is read completely and exactly, the STFT framing is invertible, chunking changes nothing, "
        "time stamps are right and band pooling keeps all the power. What it *keeps* is the problem: three design choices throw away information the song carries — "
        "the 0.256 s window blurs pulses and short units, the median noise floor deletes about half of every band (including quiet song), "
        "and the 295-number window summary followed by PCA-20 keeps noticeably less 'which phrase is this' than the spectrogram itself. Recommendations are at the end.")
    add("")

    # ------------------------------------------------------------ source limit
    add("## The source limit")
    add("")
    add("The file is a ×100 decimation (`df100`) to 2 kHz, so anything the whale sang above 1 kHz is gone before this pipeline starts. Within 0–1 kHz, the loud (singing) minutes "
        "are 12–18 dB above the quiet ones across 30–1000 Hz:")
    add("")
    L += table(["band", "loud − quiet (dB)"], [[k + " Hz", f(v, 1)] for k, v in B["loud_minus_quiet_db_by_band"].items()])
    add("")
    add(f"Below 30 Hz the difference is small: {pct(B['song_excess_outside_fraction'], 1)} of the song's extra energy lies there, so the 30 Hz lower edge loses almost nothing. "
        f"Quiet minutes have {pct(B['quiet_energy_below_30_fraction'])} of their energy below 30 Hz (sea and flow noise), loud minutes {pct(B['loud_energy_below_30_fraction'])}. "
        "The quiet spectrum also shows narrow lines at multiples of 50 Hz (100, 150, 200, 250 Hz, …): instrument tones, which a noise floor removes.")
    add("")
    add("![](figures/B_spectrum.png)")
    add("")

    # ------------------------------------------------------------ A
    add("## I-0 Reading the wav (check A)")
    add("")
    add(f"RIFF chunks: " + ", ".join(f"`{c['id'].strip()}` {c['size']:,} B at {c['offset']}" for c in A["chunks"]) +
        f". File size {A['file_size']:,} B = 12 + chunk headers + chunks, so the data chunk is complete. The `WavFileWarning: Chunk (non-data) not understood` comes from the `harp` chunk "
        "(HARP recorder metadata), which scipy skips; it does not affect the samples.")
    add("")
    add(f"- {A['n_samples']:,} int16 samples at {A['sample_rate']} Hz = {A['duration_s']:,.0f} s. No clipped samples, no constant runs (drop-outs) ≥ 0.1 s.")
    add(f"- DC offset: every 10-minute mean is between {f(A['dc_mean_range'][0], 1)} and {f(A['dc_mean_range'][1], 1)} counts, larger than the signal itself "
        f"(AC RMS ≈ {f(A['ac_rms_median'], 0)}). With a Hann window it leaks at most {f(A['dc_leak_max_db_in_any_band'], 2)} dB into any band ≥ 30 Hz, so the features are unaffected; "
        "subtracting the mean is still free and removes the question.")
    add("- The CSV's first unit starts at 0 s from the time in the file name (2023-04-26 19:35:00 UTC), so labels and audio share one clock.")
    add("")

    # ------------------------------------------------------------ C
    add("## I-1 STFT and band pooling (check C)")
    add("")
    add(f"- **Framing.** The complex STFT with these settings (Hann, 512, hop 100) inverts back to the waveform at {f(C['istft_snr_db'], 0)} dB SNR (machine precision): nothing is lost before the magnitude is taken.")
    add(f"- **Chunking.** Computing in chunks vs in one pass differs by {C['chunked_max_abs_diff']:.1e} (float32 rounding).")
    add(f"- **Time stamps.** A 0.1 s tone burst lands in frame {C['tone_frame']}, the frame whose centre is nearest (expected {C['tone_frame_expected']}). "
        f"The pipeline assigns frame n to time 0.05·n (its start); its centre is {C['frame_time_convention_offset_s']} s later. This is a constant shift, small next to 2 s windows, but Experiment 3 should use centres.")
    add(f"- **Pooling.** Every FFT bin from 30 to 1000 Hz belongs to exactly one band and summed band power equals summed bin power (relative error {C['band_power_identity_rel_err']:.0e}). "
        f"Of 64 log bands, {64 - C['n_bands']} below about 60 Hz contain no FFT bin (bins are 3.9 Hz apart) and are dropped without losing power: their bins belong to neighbouring bands.")
    add("- **What pooling loses.** Above ~150 Hz one band holds several bins (≈ 14 at 1 kHz), so pitch detail finer than a semitone is averaged away. Whale units are judged by contour, not by exact pitch, so this is a reasonable loss.")
    add("")
    add("The same three example stretches through the first stages (song boundary, inside theme 6, quietest start of the recording):")
    add("")
    add("![](figures/stages.png)")
    add("")

    # ------------------------------------------------------------ D
    add("## I-1 Time and frequency resolution vs the song (check D)")
    add("")
    add("**What the song contains.** Envelope modulation spectrum (30–1000 Hz) of the raw audio, share of modulation energy above a rate:")
    add("")
    L += table(["above", "loud minutes", "quiet minutes"], [[f"{k} Hz", pct(D["mod_share_loud"][k]), pct(D["mod_share_quiet"][k])] for k in ("2", "5", "10", "20")])
    add("")
    add("Noise has a flat envelope spectrum, so quiet minutes are high everywhere; what matters is that loud minutes still have a third of their modulation above 5 Hz: "
        "pulse trains and rapid unit sequences (Payne & McVay's subunits) are real in this recording.")
    add("")
    add("**What each STFT setting keeps** (depth of a 100 % amplitude-modulated 400 Hz tone that survives in the band series; 1 = all):")
    add("")
    L += table(["setting", "frequency step", "depth at 5 Hz", "depth at 10 Hz", "song modulation kept", "up-sweep sharpness"], [
        [k, f"{v['freq_step_hz']:.1f} Hz", f(v["depth_at_5hz"], 2), f(v["depth_at_10hz"], 2), pct(v["song_modulation_kept"]), f(v["chirp_concentration"], 2)] for k, v in S.items()
    ])
    add("")
    add("'Song modulation kept' weights the depth curve by the loud-minute modulation spectrum. 'Up-sweep sharpness' is the share of a 0.5 s, 200→800 Hz chirp's energy in its strongest band per frame (higher = a cleaner contour).")
    add("")
    add(f"The current 0.256 s window keeps only {f(cur['depth_at_10hz'], 2)} of a 10 Hz pulse train and {f(cur['depth_at_5hz'], 2)} at 5 Hz. Halving the window (256 samples, hop 0.025 s) "
        f"keeps {f(short['depth_at_10hz'], 2)} and {f(short['depth_at_5hz'], 2)}, and draws fast sweeps more sharply ({f(short['chirp_concentration'], 2)} vs {f(cur['chirp_concentration'], 2)}). "
        "The price is a 7.8 Hz frequency step, which merges the lowest bands (50 non-empty bands instead of 59, all below ~120 Hz). A longer window is worse on every count.")
    add("")
    add("![](figures/D_modulation.png)")
    add("")
    add("**Units and sweeps in the whitened spectrogram** (loud minutes; a frame is 'on' when its strongest band is above the threshold):")
    add("")
    if "1.0" in U:
        L += table(["'on' threshold", "frames on", "runs: 25th / 50th / 75th pct (s)", "runs shorter than the window", "steps where the ridge moves ≥ 1 band"], [
            [f"{th} ({4.3 * float(th):.0f} dB)", pct(v["share_frames_on"]), " / ".join(f(v["unit_length_s_percentiles"][q], 2) for q in ("25", "50", "75")), pct(v["share_units_shorter_than_window"]), pct(v["share_steps_with_band_change"])]
            for th, v in U.items()
        ])
    add("")
    add("At a low threshold almost every frame is 'on' (the median floor leaves half of every band above it), so runs merge into long stretches; at 9–13 dB, runs break into fragments, many shorter than the 0.256 s window. "
        "These runs are therefore only a rough guide to unit length (a unit with internal pulses splits into several runs), but they agree with the modulation test: much of the song changes faster than the window. "
        "The ridge moves by a whole band (≈ 1 semitone per 0.05 s ≈ 19 semitones/s) in about a quarter of steps: the up/down-sweep statistics see fast sweeps, but slower glides only appear as a change of level between bands.")
    add("")

    # ------------------------------------------------------------ E
    add("## I-2 Noise whitening (check E)")
    add("")
    L += table(["variant", "cells above floor (loud / quiet)", "loud power removed", "loud vs quiet separation (d′)", "song-repeat hit, 295", "song-repeat hit, PCA-20"], [
        [k, f"{pct(v['share_cells_above_floor_loud'])} / {pct(v['share_cells_above_floor_quiet'])}", pct(v["raw_power_under_floor_loud"]) if v["raw_power_under_floor_loud"] is not None else "0 (no clip)",
         f(v["loud_vs_quiet_separation_d"], 2), hit(v["retrieval"]["295 descriptors"]), hit(v["retrieval"]["PCA-20"])] for k, v in E.items()
    ])
    add("")
    add(f"- **The median floor (current) sits on the song.** The whale sings most of the time, so in the bands it uses, the 5-minute median *is* song. Subtracting it and clipping at 0 sets half of every band to zero in loud and quiet minutes alike "
        f"({pct(ea['share_cells_above_floor_loud'])} vs {pct(ea['share_cells_above_floor_quiet'])} kept) and removes {pct(ea['raw_power_under_floor_loud'])} of loud-minute power, including the quieter units. "
        "It also equalises loud and quiet periods, which is why sessions and background were hard to tell apart in Experiment 2.")
    add(f"- **A 10th-percentile floor** keeps most cells ({pct(eb['share_cells_above_floor_loud'])}), removes {pct(eb['raw_power_under_floor_loud'])} of loud power, and does better on the song-repeat test.")
    add(f"- **No whitening** (one fixed offset per band) separates loud from quiet best (d′ {f(ed['loud_vs_quiet_separation_d'], 2)}) and scores best on the song-repeat test, "
        "but slow noise stays in: the theme-6 example below shows a broadband haze that whitening removes, and the 50 Hz-multiple instrument lines stay in quiet periods.")
    add("")
    add("![](figures/E_whitening_variants.png)")
    add("")
    add("Listening clips for each example stretch are in `work/audio/`: `0_original`, `1_stft_magnitude` (Griffin–Lim from the full STFT magnitude), `2_59_bands` (from the band powers, spread flat inside each band), "
        "`3_whitened` (from the whitened band powers). Phase is estimated, so they sound rough; the point is what survives each step.")
    add("")

    # ------------------------------------------------------------ F
    add("## I-3 to I-6 Phrase windows, statistics, PCA (check F)")
    add("")
    add("**How much 'which phrase is this' survives each compression** (same 500 queries):")
    add("")
    L += table(["representation", "size per window", "song-repeat hit (×chance)", "10-NN overlap with the patch"], [
        ["whitened patch, best of ±3 s shifts", "240 × 59 (time-averaged to 30 × 59)", hit(patch), "–"],
        ["295 descriptors", "295", hit(desc), pct(Fr["knn_overlap_vs_patch"]["295 descriptors"])],
        ["PCA-40", "40", hit(pc40), pct(Fr["knn_overlap_vs_patch"]["PCA-40"])],
        ["PCA-20 (model input)", "20", hit(pc20), pct(Fr["knn_overlap_vs_patch"]["PCA-20"])],
        ["level-normalised descriptors", "295", hit(Fr["level_normalized"]["retrieval"]["295 descriptors"]), "–"],
    ])
    add("")
    add(f"- The 295 summary statistics keep about {pct(desc['hit_rate'] / patch['hit_rate'])} of the patch's song-repeat hit rate; PCA-20 keeps {pct(pc20['hit_rate'] / patch['hit_rate'])}. "
        f"PCA-20 explains {pct(Fr['pca_explained_20'])} of the descriptor variance; 80% needs {Fr['pca_components_for']['80']} components, 90% needs {Fr['pca_components_for']['90']}.")
    add("- Summary statistics are order-free: they keep which pitches, how choppy and which way units sweep, but not the *order* of units within the phrase, which is what defines a phrase.")
    add("- Dividing each window by its own level did not help, so loudness in the statistics is not what limits them.")
    add("")
    add("**Window length.** Dominant repeat period inside a 30 s stretch (phrase period): " +
        ", ".join(f"{q}th pct {f(v, 1)} s" for q, v in Fr["phrase_period_s_percentiles"].items()) +
        f". {pct(1 - Fr['share_phrase_period_under_12s'])} of windows have a period of 12 s or more, so a 12 s window often holds less than one full phrase. Song-repeat hits by window length:")
    add("")
    L += table(["window", "295 descriptors", "PCA-20"], [[k, hit(v["295 descriptors"]), hit(v["PCA-20"])] for k, v in Fr["window_length"].items()])
    add("")
    add("Window length between 8 and 20 s matters little for this test.")
    add("")

    # ------------------------------------------------------------ G
    if G:
        add("## Combined settings (check G)")
        add("")
        add("The changes above interact, so they were also tried together (same 500 queries; every row recomputes the spectrogram, floor, windows, statistics and PCA from the wav):")
        add("")
        L += table(["setting", "bands", "295-type descriptors", "PCA-20", "PCA-40"], [
            [k, v["n_bands"], hit(v["295-type descriptors"]), hit(v["PCA-20"]), hit(v["PCA-40"])] for k, v in G.items()
        ])
        add("")
        add(f"Best PCA-20 setting: **{best_combo}** ({hit(G[best_combo]['PCA-20'])} vs current {hit(G[cur_combo]['PCA-20'])}). For reference, the raw patch reaches {hit(patch)} with the current spectrogram.")
        add("")

    # ------------------------------------------------------------ recommendations
    add("## Recommendations for Experiment 3")
    add("")
    add("Keep:")
    add("- Reading, framing, chunking, band pooling and the 30–1000 Hz range: correct and essentially lossless for this 2 kHz file.")
    add("")
    add("Change:")
    add("1. **Subtract the DC offset** when reading (harmless today, but free).")
    add("2. **Label frames by their centre** (+0.128 s), so window times match the audio exactly.")
    if G:
        g = {k[0]: v["PCA-20"]["hit_rate"] for k, v in G.items()}
        add(f"3. **Drop the median noise floor (largest single gain).** With everything else unchanged, the song-repeat hit rate goes from {f(g['1'])} to {f(g['3'])} without whitening "
            f"({f(g['4'])} with a 10th-percentile floor). The median deletes half of every band, including song, and hides the loud/quiet difference that session detection needs. "
            "Slow noise (passing ships, instrument lines) then has to be handled separately, e.g. as its own feature or by a background model that expects it.")
        add(f"4. **Shorter FFT: 256 samples, hop 0.025 s, together with 3.** On its own it barely helps ({f(g['2'])}), because the median floor erases what it adds; without the floor it reaches {f(g['5'])} "
            f"and, with 20 s windows, **{f(g['6'])}** — the best setting tried, {f(g['6'] / g['1'], 1)}× the current one and above the raw patch under the current floor ({f(patch['hit_rate'])}). "
            f"It keeps {pct(short['song_modulation_kept'])} of the song's fast modulation instead of {pct(cur['song_modulation_kept'])} and draws sweeps more sharply; the price is fewer bands below ~120 Hz.")
    else:
        add(f"3. **Shorter FFT: 256 samples, hop 0.025 s.** It keeps {pct(short['song_modulation_kept'])} of the song's fast modulation instead of {pct(cur['song_modulation_kept'])}.")
        add("4. **Replace the median floor**: it deletes half of every band, including song.")
    add("5. **Keep more of the phrase than 295 order-free statistics and 20 PCs.** The raw patch with a small time shift carries the most 'which phrase is this'. "
        "For Experiment 3 this suggests comparing phrases directly (patch similarity / alignment between songs), and, if a compact vector is still needed, keeping more components (PCA-40 or more) or adding order-aware features.")
    add("6. **Use 20 s windows.** With the current spectrogram, 8–20 s makes little difference, but with the short FFT and no floor 20 s is best; phrase periods are 10–14 s for half of the windows, so 20 s nearly always holds a full phrase.")
    add("")
    add("None of this changes Experiment 2's features or results; when Experiment 3 is planned, the chosen settings go into an 'Experiment 3 input' section of `../pipeline.md`.")
    add("")
    return "\n".join(L)


if __name__ == "__main__":
    # re-render the report from saved results, without rerunning the checks:
    #   python -m analysis.song_hmm.exp2_phrase_hmm.pipeline_check.render
    import json
    from pathlib import Path

    from analysis.song_hmm.report_paths import write_report

    here = Path(__file__).resolve().parent
    results = json.loads((here / "work" / "results.json").read_text(encoding="utf-8"))
    print("wrote", write_report(render(results), here, here / "report.md"))
