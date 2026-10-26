"""Write the training report for the six-theme soft HMM."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from analysis.HMM.topology import THEME_N_SLOTS

HMM = Path(__file__).resolve().parent
MODEL = HMM / "song_soft_hmm_t6.npz"
FIG = HMM / "work" / "figures"
JSON = HMM / "work" / "train_report.json"
OUT = HMM / "song_soft_hmm_t6_report.pdf"

REPEAT_INIT_PRINT = (0.008, 0.984, 0.943, 0.585, 0.922, 0.950)


def _wrap(text: str, width: int = 92) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return "\n".join(lines)


class Report:
    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.page = 0

    def new_page(self, title: str) -> plt.Figure:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        self.page += 1
        fig.text(0.5, 0.02, str(self.page), ha="center", va="center", fontsize=9, color="#555555")
        fig.text(0.08, 0.955, title, fontsize=16, fontweight="bold", va="top", ha="left")
        self.y = 0.91
        self.fig = fig
        return fig

    def text(self, body: str, *, size: float = 10, gap: float = 0.012, family: str | None = None) -> None:
        wrapped = body if family else _wrap(body)
        n_lines = wrapped.count("\n") + 1
        height = n_lines * size * 1.35 / 72 / 11 + gap
        kwargs = {"fontfamily": family} if family else {}
        self.fig.text(
            0.08, self.y, wrapped, fontsize=size, va="top", ha="left", linespacing=1.35, **kwargs
        )
        self.y -= height

    def heading(self, title: str) -> None:
        self.fig.text(0.08, self.y, title, fontsize=12, fontweight="bold", va="top")
        self.y -= 0.028

    def image(self, path: Path, height: float = 0.28) -> None:
        ax = self.fig.add_axes([0.14, self.y - height, 0.74, height])
        ax.imshow(plt.imread(path))
        ax.axis("off")
        self.y -= height + 0.02

    def close(self) -> None:
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def _largest_branch_sentence(branch: np.ndarray) -> str:
    increases = [(theme + 1, float(branch[2 * theme])) for theme in range(len(branch) // 2)]
    first, second = sorted(increases, key=lambda item: item[1], reverse=True)[:2]
    return (
        f"The two largest drops are the +0.05 nudges on theme {first[0]} ({first[1]:.1f}) "
        f"and theme {second[0]} ({second[1]:.1f})."
    )


def main() -> None:
    model = np.load(MODEL, allow_pickle=True)
    report_json = json.loads(JSON.read_text(encoding="utf-8"))
    repeat = model["repeat_prob"]
    history = np.asarray(model["loglik_history"], dtype=np.float64)
    branch = model["branch_drop"]
    mean_drop = model["mean_drop"]
    saved_ll = float(report_json["loglik"])
    per_second = float(report_json["loglik_per_second"])
    normal = float(report_json["normal_per_second"])
    on_label = np.asarray(report_json["posterior_on_label_by_theme"], dtype=np.float64)
    labeled_s = np.asarray(report_json["theme_labeled_seconds"], dtype=np.float64)
    expected_s = np.asarray(report_json["theme_expected_seconds"], dtype=np.float64)
    runs = np.asarray(report_json["labeled_phrases_per_song"], dtype=np.float64)

    with PdfPages(OUT) as pdf:
        report = Report(pdf)
        report.new_page("Six-theme soft HMM")
        report.text(
            "Training report for the humpback song model in song_soft_hmm_t6.dot. "
            "Baum–Welch fits the slot Gaussians, the duration exits, the theme-end branches, and the start distribution. "
            "The motif scores are the observations. This report does not decode a state path."
        )
        report.heading("Recording and features")
        report.text(
            "File: SOCAL_H_76_230426_193500_df100.x.x_10kHz.wav. "
            "10,000 Hz, mono, 36,000 seconds. The file was upsampled from 2 kHz, "
            "so GPL was limited to 10–1000 Hz. GPL wrote 480 whitened windows of 75 s "
            "under analysis/HMM/work/gpl_193500/."
        )
        report.text(
            "Each window was summed over frequency and averaged to 10 Hz (360,000 samples). "
            "panMP kept the 12 strongest motif peaks, all 1 second long. "
            "Each second of the recording is the mean z-normalized correlation with each prototype, "
            "then each score is z-scored across the 10 hours. "
            "The motif CSV cuts the 10 hours into 53 songs and initializes the Gaussians. It is not the observation."
        )
        report.heading("Sequence log-likelihood")
        report.text(
            f"Baum–Welch shared one parameter set across the 53 songs and ran for {len(history)} iterations. "
            f"The E-step log-likelihood moved from {history[0]:,.1f} to {history[-1]:,.1f}. "
            f"The saved parameters score the 36,000 seconds at {saved_ll:,.1f}, "
            f"which is {per_second:.3f} nats per second. "
            f"Twelve independent standard normals, the distribution the z-score produces when every slot mean stays at zero, "
            f"score {normal:.3f} nats per second. "
            f"The fit is {per_second - normal:+.3f} nats per second relative to that baseline. "
            f"The initialization scored {float(report_json['initial_loglik']):,.1f}."
        )
        report.image(FIG / "loglik.png", 0.28)
        report.close()

        report.new_page("Where the likelihood comes from")
        report.text(
            f"Per song, the rate averages {report_json['song_loglik_per_second_mean']:.3f} nats per second "
            f"(median {report_json['song_loglik_per_second_median']:.3f}, "
            f"lowest {report_json['song_loglik_per_second_min']:.3f}, "
            f"highest {report_json['song_loglik_per_second_max']:.3f}). "
            "A song far below the others is one the shared parameters explain poorly."
        )
        report.image(FIG / "song_loglik.png", 0.26)
        report.heading("Complete-data split")
        report.text(
            "The posterior-weighted complete-data log-likelihood on the saved parameters splits into emission, transition, and start. "
            f"Emission {float(report_json['emission_complete']):,.1f}. "
            f"Transitions {float(report_json['transition_complete']):,.1f}. "
            f"Start {float(report_json['start_complete']):,.1f}. "
            f"The last training E-step, on the parameters before the final update, was {float(model['complete_loglik']):,.1f}."
        )
        report.heading("Posterior mass on the labeled theme")
        report.text(
            "At each second the forward–backward pass gives a probability for each theme. "
            "The number below is the average of the probability on the theme the CSV wrote for that second. "
            f"Across all seconds it is {100 * float(report_json['posterior_on_label']):.1f} percent. "
            "It is a property of the fitted distribution, not a chosen path."
        )
        mass_lines = ["Theme    labeled s    posterior s    mass on label"]
        for theme in range(6):
            mass_lines.append(
                f"{theme + 1:5d}    {labeled_s[theme]:9.0f}    {expected_s[theme]:11.0f}    {100 * on_label[theme]:13.1f}%"
            )
        report.text("\n".join(mass_lines), size=10, family="monospace")
        report.image(FIG / "theme_occupancy.png", 0.26)
        report.close()

        report.new_page("The fitted parameters")
        report.text(
            "The drawing has 28 phrase slots: 3, 3, 6, 3, 5, and 8 across themes 1 through 6. "
            "Each slot expands to five duration states, for 140 states, sharing one diagonal Gaussian over the 12 motif scores. "
            "A slot may end after 1–4 seconds and must end at 5. Only the last slot of a theme chooses repeat versus next. "
            "After theme 6, next returns to theme 1. A sequence may end in any theme. "
            "A song may also open on the first slot of any theme. "
            "Slots that share a unit name do not share a Gaussian."
        )
        report.heading("Theme-end probabilities")
        report.text(
            "Repeat is the probability of returning to the first slot of the same theme when a phrase ends. "
            "The drawing supplied the starting values. Labeled phrases per song is how many times the CSV repeats that theme's slot chain."
        )
        lines = ["Theme    slots    start repeat    fitted repeat    labeled phrases/song"]
        for theme, n_slots in enumerate(THEME_N_SLOTS):
            lines.append(
                f"{theme + 1:5d}    {n_slots:5d}    {REPEAT_INIT_PRINT[theme]:12.3f}    "
                f"{float(repeat[theme]):13.3f}    {runs[theme]:20.2f}"
            )
        report.text("\n".join(lines), size=9, family="monospace")
        start = model["start_prob"]
        start_line = "    ".join(f"{float(p):.3f}" for p in start)
        report.text("Fitted probability of opening on themes 1 through 6: " + start_line)
        report.text(
            f"The least-used slot carries {float(report_json['slot_occupancy_min']):.1f} expected seconds. "
            f"{int(report_json['slots_below_one_second'])} slots have less than one expected second."
        )
        report.heading("Parameter sensitivity")
        report.text(
            "Each free theme-end probability was nudged by 0.05, and each slot mean was shifted by a tenth of its scale on the first motif score. "
            "Nothing was refit. The drop is how far the sequence log-likelihood of all 53 songs fell."
        )
        sens_lines = []
        for theme in range(len(branch) // 2):
            sens_lines.append(
                f"Theme {theme + 1}: +0.05 drop {branch[2 * theme]:.1f}, "
                f"−0.05 drop {branch[2 * theme + 1]:.1f}."
            )
        report.text("\n".join(sens_lines))
        report.text(
            f"The median branch drop is {float(np.median(branch)):.1f}. "
            + _largest_branch_sentence(branch)
            + " "
            f"Across the 28 slot-mean shifts, the median drop is {float(np.median(mean_drop)):.1f} "
            f"and the largest is {float(np.max(mean_drop)):.1f}."
        )
        report.image(FIG / "branch_sensitivity.png", 0.22)
        report.close()

        report.new_page("The parameter file")
        report.text(
            "song_soft_hmm_t6.npz is the handoff. It holds the slot means and variances, the duration-exit probabilities, "
            "the theme-repeat probabilities, the start distribution, the 12 prototypes, and the score mean and standard deviation. "
            "A later decoder loads that file. This folder does not choose a state path."
        )
        report.heading("Where the pieces are")
        report.text(
            "Parameter file: analysis/HMM/song_soft_hmm_t6.npz.\n"
            "Topology: analysis/HMM/topology.py, matching song_soft_hmm_t6.dot.\n"
            "Baum–Welch: analysis/HMM/hmm.py.\n"
            "GPL collapse, panMP prototypes, and scoring: analysis/HMM/observations.py.\n"
            "Training entry: analysis/HMM/run_train.py.\n"
            "Notebook: analysis/HMM/train_song_soft_hmm_t6.ipynb.\n"
            "Metrics: analysis/HMM/work/train_report.json.\n"
            "This report: analysis/HMM/song_soft_hmm_t6_report.pdf."
        )
        report.text(
            "Retraining is python -m analysis.HMM.run_train from the repository root. "
            "It reuses the GPL windows and the motif-score file when they are already on disk, then overwrites the parameter file. "
            "python -c \"from analysis.HMM.run_train import refresh_metrics; refresh_metrics()\" "
            "recomputes the metrics for the saved parameters and does not run Baum–Welch again."
        )
        report.close()
    print(OUT)


if __name__ == "__main__":
    main()
