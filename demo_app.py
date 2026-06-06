"""Interface graphique de démonstration du pipeline (par étapes)."""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, ttk

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

COL_BG          = "#eef2f7"
COL_PANEL       = "#ffffff"
COL_CARD        = "#f8fafc"
COL_CARD_HOVER  = "#eef2f7"
COL_BORDER      = "#e2e8f0"
COL_BADGE       = "#e2e8f0"
COL_ACCENT      = "#0d9488"
COL_ACCENT_2    = "#4f46e5"
COL_OK          = "#059669"
COL_RUN         = "#d97706"
COL_PENDING     = "#cbd5e1"
COL_ERR         = "#dc2626"
COL_TXT         = "#1e293b"
COL_TXT_DIM     = "#64748b"
COL_SEL         = "#e0f2f1"

FONT_FAMILY = "Segoe UI"

PHASES = [
    ("1", "Génération copule",     "Méthode 1 — copule gaussienne",    COL_ACCENT),
    ("2", "Génération CTGAN",      "Méthode 2 — réseau génératif",     COL_ACCENT_2),
    ("3", "Sauvegarde datasets",   "Export des fichiers CSV",          "#0891b2"),
    ("4", "Analyses statistiques", "Descriptives • corrélations",      "#7c3aed"),
    ("5", "Visualisations",        "Distributions • heatmaps",         "#db2777"),
    ("6", "Évaluation ML",         "3 modèles • validation croisée",   "#ea580c"),
]

CLASS_FR = {
    "sain": "Sain", "diabete": "Diabète", "dyslipidemie": "Dyslipidémie",
    "hypertension": "Hypertension", "obesite": "Obésité",
    "risque_cardiovasculaire": "Risque cardiovasc.",
}

MODEL_FR = {
    "logistic_regression": "Régression logistique",
    "random_forest": "Forêt aléatoire", "mlp": "Réseau de neurones (MLP)",
}

FIG_FR = {
    "histograms_copula": "Histogrammes — Copule",
    "histograms_ctgan": "Histogrammes — CTGAN",
    "boxplots_copula": "Boîtes à moustaches — Copule",
    "scatter_copula": "Nuages de points — Copule",
    "scatter_ctgan": "Nuages de points — CTGAN",
    "heatmap_copula": "Corrélations — Copule",
    "heatmap_ctgan": "Corrélations — CTGAN",
    "heatmap_comparison": "Corrélations — Comparaison",
    "confusion_copula": "Matrices de confusion",
    "metrics_copula": "Performances ML",
    "cv_copula": "Validation croisée 5-fold",
}


class _StreamTee:
    _PCT = re.compile(r"(\d{1,3})%\|")

    def __init__(self, q, original):
        self.q = q
        self.original = original
        self._buf = ""

    def write(self, s):
        try:
            self.original.write(s)
        except Exception:
            pass
        self._buf += s
        while True:
            idx = max(self._buf.find("\r"), self._buf.find("\n"))
            if idx == -1:
                break
            line, self._buf = self._buf[:idx], self._buf[idx + 1:]
            m = self._PCT.search(line)
            if m:
                self.q.put(("progress", min(int(m.group(1)) / 100.0, 1.0)))

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass


def _phase_step_methods(pipe):
    return {
        "2": pipe._step_generate_ctgan,
        "3": pipe._step_save_datasets,
        "4": pipe._step_run_analysis,
        "5": pipe._step_run_visualizations,
        "6": pipe._step_run_ml_evaluation,
    }


def phase_worker(app, idx, params, q):
    """Exécute une étape du pipeline dans un thread."""
    import logging

    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        from clinical_synthetic_data.logging_setup import setup_logging

        setup_logging()
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

        class _H(logging.Handler):
            def emit(self_inner, record):
                try:
                    msg = record.getMessage().strip()
                except Exception:
                    msg = ""
                if msg and not msg.startswith("="):
                    q.put(("status", msg))

        root.addHandler(_H())
        root.setLevel(logging.INFO)
        sys.stdout = _StreamTee(q, orig_out)
        sys.stderr = _StreamTee(q, orig_err)

        if idx == "1":
            from clinical_synthetic_data.pipeline import (
                PipelineConfig, SyntheticDataPipeline)
            cfg = PipelineConfig(
                m_per_class=params["m_per_class"], seed=params["seed"],
                output_dir=Path(params["output_dir"]),
                ctgan_epochs=params["ctgan_epochs"],
                ctgan_batch_size=params["ctgan_batch_size"],
            )
            pipe = SyntheticDataPipeline(cfg)
            pipe._setup_directories()
            app.pipeline = pipe
            pipe._step_generate_copula()
        else:
            pipe = app.pipeline
            if pipe is None or pipe.df_copula is None:
                raise RuntimeError(
                    "L'étape 1 (génération copule) doit être exécutée d'abord.")
            _phase_step_methods(pipe)[idx]()
            if idx in ("3", "5", "6"):
                try:
                    pipe._step_save_summary()
                except Exception:
                    pass

        q.put(("phase_result", idx))
    except Exception:
        q.put(("phase_error", (idx, traceback.format_exc())))
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err


class PhaseItem(ctk.CTkFrame):
    def __init__(self, master, idx, title, subtitle, accent,
                 on_select, on_run):
        super().__init__(master, fg_color=COL_PANEL, corner_radius=12,
                         border_width=1, border_color=COL_BORDER)
        self.idx = idx
        self.accent = accent
        self.on_select = on_select
        self.on_run = on_run
        self.state = "pending"
        self.grid_columnconfigure(1, weight=1)

        self.dot = ctk.CTkLabel(self, text="○", width=22,
                                font=(FONT_FAMILY, 18, "bold"),
                                text_color=COL_PENDING)
        self.dot.grid(row=0, column=0, rowspan=2, padx=(12, 4), pady=12)

        self.title = ctk.CTkLabel(self, text=f"{idx}.  {title}", anchor="w",
                                  font=(FONT_FAMILY, 14, "bold"),
                                  text_color=COL_TXT)
        self.title.grid(row=0, column=1, sticky="ew", pady=(10, 0))
        self.sub = ctk.CTkLabel(self, text=subtitle, anchor="w",
                                font=(FONT_FAMILY, 11), text_color=COL_TXT_DIM)
        self.sub.grid(row=1, column=1, sticky="ew", pady=(0, 10))

        self.run_btn = ctk.CTkButton(
            self, text="▶", width=40, height=40, corner_radius=10,
            font=(FONT_FAMILY, 15, "bold"), fg_color=accent,
            hover_color=COL_TXT, text_color="#ffffff",
            command=lambda: self.on_run(self.idx))
        self.run_btn.grid(row=0, column=2, rowspan=2, padx=(6, 12), pady=10)

        for w in (self, self.title, self.sub, self.dot):
            w.bind("<Button-1>", lambda e: self.on_select(self.idx))

    def set_selected(self, selected: bool):
        self.configure(border_color=self.accent if selected else COL_BORDER,
                       border_width=2 if selected else 1,
                       fg_color=COL_SEL if selected else COL_PANEL)

    def set_state(self, state: str):
        self.state = state
        if state == "done":
            self.dot.configure(text="✓", text_color=COL_OK)
        elif state == "running":
            self.dot.configure(text="◉", text_color=COL_RUN)
        elif state == "error":
            self.dot.configure(text="✕", text_color=COL_ERR)
        else:
            self.dot.configure(text="○", text_color=COL_PENDING)

    def set_enabled(self, enabled: bool):
        self.run_btn.configure(state="normal" if enabled else "disabled",
                               fg_color=self.accent if enabled else COL_PENDING)


class DemoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Données Cliniques Synthétiques — Démonstration")
        self.geometry("1340x820")
        self.minsize(1160, 700)
        self.configure(fg_color=COL_BG)

        self.queue: "queue.Queue" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.pipeline = None
        self.items: dict[str, PhaseItem] = {}
        self.done: dict[str, bool] = {p[0]: False for p in PHASES}
        self.selected = "1"
        self.running_idx: str | None = None
        self.output_dir = Path("demo_output")
        self.start_time = 0.0
        self._fig_ref = None
        self._cur_fig: Path | None = None
        self._summary_cache: dict | None = None

        self._setup_ttk_style()
        self._build_layout()
        self._select_phase("1")
        self._refresh_enabled()
        self._poll_queue()

        for cand in ("results", "test_run", "outputs"):
            if (Path(cand) / "reports").exists() or (Path(cand) / "figures").exists():
                self.output_entry.delete(0, "end")
                self.output_entry.insert(0, cand)
                break

    def _setup_ttk_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Clinic.Treeview", background="#ffffff",
                        foreground=COL_TXT, fieldbackground="#ffffff",
                        rowheight=27, borderwidth=0, font=(FONT_FAMILY, 10))
        style.configure("Clinic.Treeview.Heading", background=COL_ACCENT,
                        foreground="#ffffff", font=(FONT_FAMILY, 10, "bold"),
                        borderwidth=0, relief="flat", padding=(6, 6))
        style.map("Clinic.Treeview", background=[("selected", COL_ACCENT_2)],
                  foreground=[("selected", "#ffffff")])
        style.map("Clinic.Treeview.Heading", background=[("active", "#0b7d72")])

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_navigator()
        self._build_output()

    def _build_sidebar(self):
        bar = ctk.CTkScrollableFrame(self, width=300, fg_color=COL_PANEL,
                                     corner_radius=0)
        bar.grid(row=0, column=0, sticky="nsew")

        head = ctk.CTkFrame(bar, fg_color="transparent")
        head.pack(fill="x", padx=20, pady=(18, 6))
        ctk.CTkLabel(head, text="🧬", font=(FONT_FAMILY, 34)).pack(anchor="w")
        ctk.CTkLabel(head, text="Données Cliniques\nSynthétiques",
                     font=(FONT_FAMILY, 18, "bold"), text_color=COL_TXT,
                     justify="left", anchor="w").pack(anchor="w", pady=(2, 0))
        ctk.CTkLabel(head, text="Copule gaussienne  •  CTGAN",
                     font=(FONT_FAMILY, 11, "bold"), text_color=COL_ACCENT,
                     anchor="w").pack(anchor="w")

        ctk.CTkFrame(bar, height=1, fg_color=COL_BORDER).pack(
            fill="x", padx=18, pady=10)

        ctk.CTkLabel(bar, text="CONFIGURATION", font=(FONT_FAMILY, 12, "bold"),
                     text_color=COL_TXT_DIM, anchor="w").pack(
            fill="x", padx=22, pady=(2, 4))

        self.m_entry = self._field(bar, "Patients par classe (M)", "1000",
                                   "Total = 6 × M patients")
        self.seed_entry = self._field(bar, "Graine (seed)", "42",
                                      "Reproductibilité")
        self.epochs_entry = self._field(bar, "Epochs CTGAN", "600",
                                        "Plus élevé = meilleur mais plus long")
        self.batch_entry = self._field(bar, "Batch size CTGAN", "500")

        out_box = ctk.CTkFrame(bar, fg_color="transparent")
        out_box.pack(fill="x", padx=20, pady=(10, 2))
        ctk.CTkLabel(out_box, text="Dossier de sortie / résultats",
                     font=(FONT_FAMILY, 12, "bold"), text_color=COL_TXT,
                     anchor="w").pack(fill="x", pady=(0, 4))
        row = ctk.CTkFrame(out_box, fg_color="transparent")
        row.pack(fill="x")
        row.grid_columnconfigure(0, weight=1)
        self.output_entry = ctk.CTkEntry(row, fg_color=COL_CARD,
                                         border_color=COL_BORDER, border_width=1,
                                         text_color=COL_TXT, height=34)
        self.output_entry.insert(0, "demo_output")
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(row, text="📁", width=40, height=34, fg_color=COL_CARD,
                      border_width=1, border_color=COL_BORDER,
                      text_color=COL_TXT, hover_color=COL_CARD_HOVER,
                      command=self._browse_output).grid(row=0, column=1)

        ctk.CTkFrame(bar, height=1, fg_color=COL_BORDER).pack(
            fill="x", padx=18, pady=10)

        actions = ctk.CTkFrame(bar, fg_color="transparent")
        actions.pack(fill="x", padx=18, pady=(2, 18))
        self.runall_btn = ctk.CTkButton(
            actions, text="⏩   Tout exécuter (1→6)", height=42,
            font=(FONT_FAMILY, 13, "bold"), fg_color=COL_ACCENT,
            text_color="#ffffff", hover_color="#0b7d72",
            command=self._run_all)
        self.runall_btn.pack(fill="x", pady=(4, 8))
        self.load_btn = ctk.CTkButton(
            actions, text="📂   Charger des résultats existants", height=38,
            font=(FONT_FAMILY, 12), fg_color="#ffffff", border_width=1,
            border_color=COL_BORDER, hover_color=COL_CARD_HOVER,
            text_color=COL_TXT, command=self._on_load_existing)
        self.load_btn.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(actions, text="🗂   Ouvrir le dossier", height=34,
                      font=(FONT_FAMILY, 12), fg_color="transparent",
                      hover_color=COL_CARD_HOVER, text_color=COL_TXT_DIM,
                      command=self._open_folder).pack(fill="x")

    def _field(self, master, label, default, hint=None):
        box = ctk.CTkFrame(master, fg_color="transparent")
        box.pack(fill="x", padx=20, pady=(8, 2))
        ctk.CTkLabel(box, text=label, font=(FONT_FAMILY, 12, "bold"),
                     text_color=COL_TXT, anchor="w").pack(fill="x", pady=(0, 4))
        e = ctk.CTkEntry(box, fg_color=COL_CARD, border_color=COL_BORDER,
                         border_width=1, text_color=COL_TXT, height=34)
        e.insert(0, default)
        e.pack(fill="x")
        if hint:
            ctk.CTkLabel(box, text=hint, font=(FONT_FAMILY, 10),
                         text_color=COL_TXT_DIM, anchor="w").pack(
                fill="x", pady=(3, 0))
        return e

    def _build_navigator(self):
        nav = ctk.CTkFrame(self, width=300, fg_color=COL_BG, corner_radius=0)
        nav.grid(row=0, column=1, sticky="nsew")
        nav.grid_rowconfigure(1, weight=1)
        nav.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(nav, text="ÉTAPES DU PIPELINE",
                     font=(FONT_FAMILY, 13, "bold"), text_color=COL_TXT_DIM,
                     anchor="w").grid(row=0, column=0, sticky="ew",
                                      padx=18, pady=(20, 6))
        lst = ctk.CTkScrollableFrame(nav, fg_color="transparent", width=280)
        lst.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 12))
        lst.grid_columnconfigure(0, weight=1)
        for i, (idx, title, sub, accent) in enumerate(PHASES):
            item = PhaseItem(lst, idx, title, sub, accent,
                             self._select_phase, self._run_phase)
            item.grid(row=i, column=0, sticky="ew", padx=4, pady=5)
            self.items[idx] = item

        ctk.CTkLabel(nav,
                     text="ℹ  Les étapes 2 à 6 nécessitent\nl'étape 1 au préalable.",
                     font=(FONT_FAMILY, 11), text_color=COL_TXT_DIM,
                     justify="left", anchor="w").grid(
            row=2, column=0, sticky="ew", padx=18, pady=(0, 16))

    def _build_output(self):
        wrap = ctk.CTkFrame(self, fg_color=COL_BG, corner_radius=0)
        wrap.grid(row=0, column=2, sticky="nsew")
        wrap.grid_rowconfigure(1, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        self.header = ctk.CTkFrame(wrap, fg_color=COL_PANEL, corner_radius=14,
                                   border_width=1, border_color=COL_BORDER)
        self.header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        self.header.grid_columnconfigure(1, weight=1)
        self.h_badge = ctk.CTkLabel(self.header, text="1", width=46, height=46,
                                    corner_radius=23, fg_color=COL_ACCENT,
                                    text_color="#ffffff",
                                    font=(FONT_FAMILY, 20, "bold"))
        self.h_badge.grid(row=0, column=0, rowspan=2, padx=(16, 12), pady=14)
        self.h_title = ctk.CTkLabel(self.header, text="", anchor="w",
                                    font=(FONT_FAMILY, 19, "bold"),
                                    text_color=COL_TXT)
        self.h_title.grid(row=0, column=1, sticky="ew", pady=(14, 0))
        self.h_status = ctk.CTkLabel(self.header, text="", anchor="w",
                                     font=(FONT_FAMILY, 12),
                                     text_color=COL_TXT_DIM)
        self.h_status.grid(row=1, column=1, sticky="ew", pady=(0, 14))
        self.h_run = ctk.CTkButton(self.header, text="▶  Exécuter cette étape",
                                   height=42, width=200,
                                   font=(FONT_FAMILY, 14, "bold"),
                                   fg_color=COL_ACCENT, text_color="#ffffff",
                                   hover_color="#0b7d72",
                                   command=lambda: self._run_phase(self.selected))
        self.h_run.grid(row=0, column=2, rowspan=2, padx=18)
        self.h_timer = ctk.CTkLabel(self.header, text="",
                                    font=(FONT_FAMILY, 18, "bold"),
                                    text_color=COL_RUN)
        self.h_timer.grid(row=0, column=3, rowspan=2, padx=(0, 16))

        self.prog = ctk.CTkProgressBar(wrap, progress_color=COL_ACCENT_2,
                                       height=6)
        self.prog.set(0)

        self.output = ctk.CTkFrame(wrap, fg_color="transparent")
        self.output.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.output.grid_rowconfigure(0, weight=1)
        self.output.grid_columnconfigure(0, weight=1)

    def _phase_meta(self, idx):
        for p in PHASES:
            if p[0] == idx:
                return p
        return PHASES[0]

    def _select_phase(self, idx):
        self.selected = idx
        _, title, sub, accent = self._phase_meta(idx)
        self.h_badge.configure(text=idx, fg_color=accent)
        self.h_title.configure(text=title)
        for k, it in self.items.items():
            it.set_selected(k == idx)
        self._update_header_status()
        self._render_output(idx)

    def _update_header_status(self):
        idx = self.selected
        _, _, sub, accent = self._phase_meta(idx)
        if self.running_idx == idx:
            self.h_status.configure(text="Exécution en cours…")
        elif self.done[idx]:
            self.h_status.configure(text="✓  Étape terminée — résultats ci-dessous")
        else:
            self.h_status.configure(text=sub)
        runnable = (idx == "1") or self.done["1"]
        self.h_run.configure(
            state="normal" if (runnable and self.running_idx is None) else "disabled",
            fg_color=accent if runnable else COL_PENDING,
            text="▶  Exécuter cette étape" if not self.done[idx]
            else "↻  Ré-exécuter cette étape")

    def _refresh_enabled(self):
        base = self.done["1"]
        for idx, it in self.items.items():
            it.set_enabled((idx == "1" or base) and self.running_idx is None)
        self.runall_btn.configure(
            state="normal" if self.running_idx is None else "disabled")
        self._update_header_status()

    def _gather_params(self):
        try:
            return {
                "m_per_class": int(self.m_entry.get()),
                "seed": int(self.seed_entry.get()),
                "ctgan_epochs": int(self.epochs_entry.get()),
                "ctgan_batch_size": int(self.batch_entry.get()),
                "output_dir": self.output_entry.get().strip() or "demo_output",
            }
        except ValueError:
            self.h_status.configure(
                text="⚠ Paramètres invalides (M, seed, epochs, batch = entiers).")
            return None

    def _run_phase(self, idx, _chain=None):
        if self.running_idx is not None:
            return
        if idx != "1" and not self.done["1"]:
            self._select_phase(idx)
            self.h_status.configure(
                text="⚠ Exécutez d'abord l'étape 1 (génération copule).")
            return
        params = self._gather_params()
        if params is None:
            return
        self.output_dir = Path(params["output_dir"])
        self._select_phase(idx)
        self.running_idx = idx
        self.items[idx].set_state("running")
        self.start_time = time.time()
        self.prog.set(0)
        if idx == "2":
            self.prog.grid(row=0, column=0, sticky="ew", padx=24, pady=(0, 2))
        self._refresh_enabled()
        self._update_header_status()
        self._tick()

        self._chain = _chain
        self.worker = threading.Thread(
            target=phase_worker, args=(self, idx, params, self.queue),
            daemon=True)
        self.worker.start()

    def _run_all(self):
        if self.running_idx is not None:
            return
        self._run_phase("1", _chain=["2", "3", "4", "5", "6"])

    def _tick(self):
        if self.running_idx and self.worker and self.worker.is_alive():
            el = int(time.time() - self.start_time)
            self.h_timer.configure(text=f"{el // 60:02d}:{el % 60:02d}")
            self.after(500, self._tick)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "status":
                    if self.running_idx == self.selected:
                        self.h_status.configure(text="⏳  " + str(payload)[:90])
                elif kind == "progress":
                    self.prog.set(payload)
                elif kind == "phase_result":
                    self._on_phase_done(payload)
                elif kind == "phase_error":
                    self._on_phase_error(*payload)
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _on_phase_done(self, idx):
        self.done[idx] = True
        self.items[idx].set_state("done")
        self.running_idx = None
        self.prog.grid_forget()
        self._summary_cache = None
        chain = getattr(self, "_chain", None)
        self._chain = None
        self._select_phase(idx)
        self._refresh_enabled()
        if chain:
            self.after(300, lambda: self._run_phase(chain[0], _chain=chain[1:]))

    def _on_phase_error(self, idx, tb):
        self.items[idx].set_state("error")
        self.running_idx = None
        self._chain = None
        self.prog.grid_forget()
        self._refresh_enabled()
        self._select_phase(idx)
        self._render_error(tb)

    def _summary(self):
        if self._summary_cache is None:
            p = self.output_dir / "reports" / "summary.json"
            try:
                self._summary_cache = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                self._summary_cache = {}
        return self._summary_cache

    def _df(self, which):
        if self.pipeline is not None:
            df = getattr(self.pipeline, f"df_{which}", None)
            if df is not None:
                return df
        path = self.output_dir / "datasets" / f"dataset_{which}.csv"
        if path.exists():
            try:
                import pandas as pd
                return pd.read_csv(path)
            except Exception:
                return None
        return None

    def _gen_stats(self, which):
        if self.pipeline is not None:
            gen = getattr(self.pipeline, f"{which}_generator", None)
            if gen is not None:
                return gen.stats.to_report()
        return self._summary().get("generation", {}).get(which)

    def _read_json(self, name):
        p = self.output_dir / "reports" / name
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _clear_output(self):
        for w in self.output.winfo_children():
            w.destroy()

    def _render_output(self, idx):
        self._clear_output()
        if self.running_idx == idx:
            self._render_running(idx)
            return
        if not self.done[idx]:
            self._render_placeholder(idx)
            return
        {"1": lambda: self._render_generation("copula"),
         "2": lambda: self._render_generation("ctgan"),
         "3": self._render_datasets,
         "4": self._render_analysis,
         "5": lambda: self._render_figures(["histograms", "boxplots",
                                            "scatter", "heatmap"]),
         "6": self._render_ml}.get(idx, lambda: None)()

    def _render_placeholder(self, idx):
        _, title, sub, accent = self._phase_meta(idx)
        f = ctk.CTkFrame(self.output, fg_color="transparent")
        f.grid(row=0, column=0)
        locked = (idx != "1" and not self.done["1"])
        ctk.CTkLabel(f, text="🔒" if locked else "▶",
                     font=(FONT_FAMILY, 52),
                     text_color=COL_PENDING if locked else accent).pack(pady=(0, 10))
        ctk.CTkLabel(f, text=title, font=(FONT_FAMILY, 20, "bold"),
                     text_color=COL_TXT).pack()
        msg = ("Exécutez d'abord l'étape 1 (génération copule)." if locked
               else "Cliquez sur « Exécuter cette étape » pour lancer "
                    "cette phase\net afficher ses résultats ici.")
        ctk.CTkLabel(f, text=msg, font=(FONT_FAMILY, 13),
                     text_color=COL_TXT_DIM, justify="center").pack(pady=8)

    def _render_running(self, idx):
        f = ctk.CTkFrame(self.output, fg_color="transparent")
        f.grid(row=0, column=0)
        ctk.CTkLabel(f, text="⏳", font=(FONT_FAMILY, 52),
                     text_color=COL_RUN).pack(pady=(0, 10))
        ctk.CTkLabel(f, text="Exécution en cours…",
                     font=(FONT_FAMILY, 18, "bold"), text_color=COL_TXT).pack()
        note = ("L'entraînement CTGAN peut durer plusieurs minutes."
                if idx == "2" else "Veuillez patienter.")
        ctk.CTkLabel(f, text=note, font=(FONT_FAMILY, 12),
                     text_color=COL_TXT_DIM).pack(pady=6)

    def _render_error(self, tb):
        self._clear_output()
        box = ctk.CTkFrame(self.output, fg_color=COL_PANEL, corner_radius=12,
                           border_width=1, border_color=COL_ERR)
        box.grid(row=0, column=0, sticky="nsew")
        box.grid_rowconfigure(1, weight=1)
        box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(box, text="✕  Une erreur est survenue", anchor="w",
                     font=(FONT_FAMILY, 15, "bold"), text_color=COL_ERR).grid(
            row=0, column=0, sticky="ew", padx=14, pady=(12, 4))
        t = ctk.CTkTextbox(box, fg_color=COL_CARD, text_color=COL_TXT,
                           font=("Consolas", 11))
        t.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        t.insert("end", tb)
        t.configure(state="disabled")

    def _render_generation(self, which):
        stats = self._gen_stats(which)
        df = self._df(which)
        scroll = ctk.CTkScrollableFrame(self.output, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        if stats:
            total = stats.get("total_accepted", len(df) if df is not None else 0)
            rate = stats.get("global_rejection_rate", 0.0)
            attempts = stats.get("total_attempts", 0)
            cards = [
                ("Patients générés", f"{total:,}".replace(",", " "), COL_ACCENT),
                ("Taux de rejet global", f"{rate:.1%}", COL_RUN),
                ("Tentatives totales", f"{attempts:,}".replace(",", " "),
                 COL_ACCENT_2),
                ("Classes", "6", COL_OK),
            ]
            self._cards_row(scroll, cards, 0)

            byc = stats.get("rejection_rate_by_class", {})
            if byc:
                card = self._panel(scroll, "Taux de rejet par classe", 1)
                for cls, r in byc.items():
                    self._rate_row(card, CLASS_FR.get(cls, cls), r)

        if df is not None:
            p = self._panel(scroll, f"Aperçu du dataset ({len(df):,} lignes × "
                            f"{df.shape[1]} colonnes)".replace(",", " "), 2)
            self._table(p, df, max_rows=100)
        elif not stats:
            ctk.CTkLabel(scroll, text="Aucune donnée disponible.",
                         text_color=COL_TXT_DIM).grid(row=0, column=0, pady=30)

    def _render_datasets(self):
        dd = self.output_dir / "datasets"
        csvs = sorted(dd.glob("*.csv")) if dd.exists() else []
        wrap = ctk.CTkFrame(self.output, fg_color="transparent")
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.grid_rowconfigure(2, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        pretty = {"dataset_copula": "Méthode 1 — Copule gaussienne",
                  "dataset_ctgan": "Méthode 2 — CTGAN"}
        self._ds_map = {}
        info = ctk.CTkFrame(wrap, fg_color="transparent")
        info.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        for p in csvs:
            label = pretty.get(p.stem, p.stem)
            self._ds_map[label] = p
            size = p.stat().st_size / 1024
            ctk.CTkLabel(info, text=f"✓  {p.name}   ({size:,.0f} Ko)".replace(",", " "),
                         font=(FONT_FAMILY, 12), text_color=COL_OK,
                         anchor="w").pack(anchor="w", pady=1)

        if not csvs:
            ctk.CTkLabel(wrap, text="Aucun fichier CSV. Exécutez l'étape 3.",
                         text_color=COL_TXT_DIM).grid(row=1, column=0, pady=30)
            return

        sel = ctk.CTkFrame(wrap, fg_color="transparent")
        sel.grid(row=1, column=0, sticky="ew", pady=8)
        ctk.CTkLabel(sel, text="Dataset :", font=(FONT_FAMILY, 13),
                     text_color=COL_TXT).pack(side="left", padx=(0, 8))
        keys = list(self._ds_map.keys())
        self._ds_menu = ctk.CTkOptionMenu(
            sel, values=keys, command=self._show_dataset, width=300,
            fg_color=COL_CARD, button_color=COL_ACCENT_2,
            button_hover_color="#4338ca", text_color=COL_TXT)
        self._ds_menu.pack(side="left")

        self._ds_table_holder = ctk.CTkFrame(wrap, fg_color=COL_PANEL,
                                             corner_radius=12, border_width=1,
                                             border_color=COL_BORDER)
        self._ds_table_holder.grid(row=2, column=0, sticky="nsew", pady=(4, 0))
        self._ds_table_holder.grid_rowconfigure(0, weight=1)
        self._ds_table_holder.grid_columnconfigure(0, weight=1)
        self._show_dataset(keys[0])

    def _show_dataset(self, label):
        for w in self._ds_table_holder.winfo_children():
            w.destroy()
        path = self._ds_map.get(label)
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
        except Exception as e:
            ctk.CTkLabel(self._ds_table_holder, text=f"Erreur : {e}",
                         text_color=COL_ERR).grid(row=0, column=0, pady=20)
            return
        self._table(self._ds_table_holder, df, max_rows=None, padded=True)

    def _render_analysis(self):
        ana = self._read_json("analysis_report.json")
        scroll = ctk.CTkScrollableFrame(self.output, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        if not ana:
            ctk.CTkLabel(scroll, text="Aucun rapport. Exécutez l'étape 4.",
                         text_color=COL_TXT_DIM).grid(row=0, column=0, pady=30)
            return
        r = 0
        for key, name in [("copula", "Méthode 1 — Copule gaussienne"),
                          ("ctgan", "Méthode 2 — CTGAN")]:
            block = ana.get(key)
            if not isinstance(block, dict):
                continue
            card = self._panel(scroll, name, r); r += 1
            gen = block.get("generation_stats", {})
            rej = gen.get("global_rejection_rate")
            if isinstance(rej, (int, float)):
                self._kv(card, "Taux de rejet global", f"{rej:.1%}")
            epi = block.get("epidemiological")
            if isinstance(epi, dict):
                for k, v in list(epi.items())[:8]:
                    self._kv(card, str(k), str(v))
            desc = block.get("descriptive")
            if isinstance(desc, dict):
                self._kv(card, "Variables décrites", str(len(desc)))
        if isinstance(ana.get("method_comparison"), dict):
            card = self._panel(scroll, "Comparaison Méthode 1 vs Méthode 2", r)
            r += 1
            mc = ana["method_comparison"]
            for k, v in list(mc.items())[:10]:
                if isinstance(v, (int, float, str)):
                    self._kv(card, str(k), str(v))

    def _render_figures(self, patterns):
        fd = self.output_dir / "figures"
        pngs = []
        if fd.exists():
            pngs = [p for p in sorted(fd.glob("*.png"))
                    if any(pat in p.stem for pat in patterns)]
        if not pngs:
            ctk.CTkLabel(self.output, text="Aucune figure. Exécutez cette étape.",
                         text_color=COL_TXT_DIM).grid(row=0, column=0, pady=30)
            return
        wrap = ctk.CTkFrame(self.output, fg_color="transparent")
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(1, weight=1)

        side = ctk.CTkScrollableFrame(wrap, width=230, fg_color=COL_PANEL,
                                      corner_radius=12, border_width=1,
                                      border_color=COL_BORDER,
                                      label_text="Figures", label_text_color=COL_TXT,
                                      label_fg_color=COL_CARD)
        side.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        disp = ctk.CTkFrame(wrap, fg_color=COL_PANEL, corner_radius=12,
                            border_width=1, border_color=COL_BORDER)
        disp.grid(row=0, column=1, sticky="nsew")
        disp.grid_rowconfigure(1, weight=1)
        disp.grid_columnconfigure(0, weight=1)
        self._fig_title = ctk.CTkLabel(disp, text="", font=(FONT_FAMILY, 14, "bold"),
                                       text_color=COL_TXT)
        self._fig_title.grid(row=0, column=0, pady=(12, 4))
        self._fig_lbl = ctk.CTkLabel(disp, text="")
        self._fig_lbl.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        self._fig_holder = disp
        disp.bind("<Configure>", lambda e: self._render_fig())

        for p in pngs:
            name = FIG_FR.get(p.stem, p.stem)
            ctk.CTkButton(side, text="🖼  " + name, anchor="w",
                          fg_color="transparent", hover_color=COL_CARD_HOVER,
                          text_color=COL_TXT, font=(FONT_FAMILY, 12),
                          command=lambda pp=p, nn=name: self._pick_fig(pp, nn)
                          ).pack(fill="x", padx=4, pady=2)
        self._pick_fig(pngs[0], FIG_FR.get(pngs[0].stem, pngs[0].stem))

    def _pick_fig(self, path, name):
        self._cur_fig = path
        self._fig_title.configure(text=name)
        self._render_fig()

    def _render_fig(self):
        if not self._cur_fig:
            return
        if not _HAS_PIL:
            self._fig_lbl.configure(text=str(self._cur_fig))
            return
        try:
            img = Image.open(self._cur_fig)
        except Exception as e:
            self._fig_lbl.configure(text=f"Erreur : {e}")
            return
        aw = max(self._fig_holder.winfo_width() - 50, 320)
        ah = max(self._fig_holder.winfo_height() - 70, 320)
        ratio = min(aw / img.width, ah / img.height)
        size = (max(int(img.width * ratio), 1), max(int(img.height * ratio), 1))
        cimg = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        self._fig_lbl.configure(image=cimg, text="")
        self._fig_ref = cimg

    def _render_ml(self):
        ml = self._read_json("ml_report.json")
        scroll = ctk.CTkScrollableFrame(self.output, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        r = 0
        if ml:
            n = ml.get("n_total", "—")
            self._cards_row(scroll, [
                ("Patients évalués", f"{n:,}".replace(",", " ")
                 if isinstance(n, int) else str(n), COL_ACCENT),
                ("Modèles", "3", COL_ACCENT_2),
                ("Protocole", "Split + CV 5-fold", COL_OK),
            ], r); r += 1

            single = ml.get("single_split", {})
            if single:
                card = self._panel(scroll, "Performances (split unique)", r); r += 1
                grid = ctk.CTkFrame(card, fg_color="transparent")
                grid.pack(fill="x", padx=12, pady=(0, 10))
                grid.grid_columnconfigure((0, 1, 2, 3), weight=1)
                for j, h in enumerate(["Modèle", "Accuracy", "F1 (macro)", "AUC"]):
                    ctk.CTkLabel(grid, text=h, font=(FONT_FAMILY, 12, "bold"),
                                 text_color=COL_ACCENT).grid(
                        row=0, column=j, sticky="ew", padx=8, pady=(6, 4))
                rr = 1
                for model, mt in single.items():
                    ctk.CTkLabel(grid, text=MODEL_FR.get(model, model), anchor="w",
                                 font=(FONT_FAMILY, 12), text_color=COL_TXT).grid(
                        row=rr, column=0, sticky="ew", padx=8, pady=3)
                    for j, key in enumerate(["accuracy", "f1_macro",
                                             "roc_auc_ovr_macro"], 1):
                        v = mt.get(key)
                        txt = f"{v:.3f}" if isinstance(v, (int, float)) else "—"
                        col = COL_OK if (isinstance(v, float) and v >= 0.9) else COL_TXT
                        ctk.CTkLabel(grid, text=txt, font=(FONT_FAMILY, 12, "bold"),
                                     text_color=col).grid(row=rr, column=j,
                                                          sticky="ew", padx=8, pady=3)
                    rr += 1

            cv = ml.get("cross_validation", {})
            if cv:
                scoring = next((st.get("scoring") for st in cv.values()
                                if isinstance(st, dict)), "f1_macro")
                card = self._panel(
                    scroll, f"Validation croisée 5-fold ({scoring})", r); r += 1
                for model, st in cv.items():
                    if isinstance(st, dict):
                        mean = st.get("mean")
                        std = st.get("std")
                        if isinstance(mean, (int, float)):
                            txt = f"{mean:.3f}"
                            if isinstance(std, (int, float)):
                                txt += f"  ± {std:.3f}"
                            self._kv(card, MODEL_FR.get(model, model), txt)

            cm = ml.get("cross_method", {})
            tstr = cm.get("train_on_main_test_on_other") if isinstance(cm, dict) else None
            if isinstance(tstr, dict):
                card = self._panel(
                    scroll, "Test croisé — entraîné sur Copule, testé sur CTGAN", r)
                r += 1
                for model, mt in tstr.items():
                    if isinstance(mt, dict) and isinstance(mt.get("accuracy"),
                                                           (int, float)):
                        self._kv(card, MODEL_FR.get(model, model),
                                 f"accuracy = {mt['accuracy']:.3f}")

        fd = self.output_dir / "figures"
        ml_figs = []
        if fd.exists():
            ml_figs = [p for p in sorted(fd.glob("*.png"))
                       if any(k in p.stem for k in ("confusion", "metrics", "cv"))]
        if ml_figs:
            card = self._panel(scroll, "Figures d'évaluation", r); r += 1
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=(0, 10))
            for p in ml_figs:
                name = FIG_FR.get(p.stem, p.stem)
                ctk.CTkButton(row, text="🖼  " + name, fg_color=COL_CARD,
                              border_width=1, border_color=COL_BORDER,
                              hover_color=COL_CARD_HOVER, text_color=COL_TXT,
                              font=(FONT_FAMILY, 12),
                              command=lambda pp=p, nn=name: self._popup_fig(pp, nn)
                              ).pack(side="left", padx=4)
        if r == 0 and not ml_figs:
            ctk.CTkLabel(scroll, text="Aucun résultat ML. Exécutez l'étape 6.",
                         text_color=COL_TXT_DIM).grid(row=0, column=0, pady=30)

    def _popup_fig(self, path, name):
        if not _HAS_PIL:
            return
        win = ctk.CTkToplevel(self)
        win.title(name)
        win.geometry("900x700")
        win.configure(fg_color=COL_BG)
        holder = ctk.CTkFrame(win, fg_color=COL_PANEL)
        holder.pack(fill="both", expand=True, padx=10, pady=10)
        lbl = ctk.CTkLabel(holder, text="")
        lbl.pack(fill="both", expand=True, padx=8, pady=8)

        def draw(_e=None):
            try:
                img = Image.open(path)
            except Exception:
                return
            aw = max(win.winfo_width() - 60, 300)
            ah = max(win.winfo_height() - 60, 300)
            ratio = min(aw / img.width, ah / img.height)
            size = (int(img.width * ratio), int(img.height * ratio))
            ci = ctk.CTkImage(light_image=img, dark_image=img, size=size)
            lbl.configure(image=ci, text="")
            lbl._ref = ci
        holder.bind("<Configure>", draw)
        win.after(200, draw)

    def _cards_row(self, parent, cards, row):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        for i, (lbl, val, col) in enumerate(cards):
            f.grid_columnconfigure(i, weight=1)
            c = ctk.CTkFrame(f, fg_color=COL_PANEL, corner_radius=12,
                             border_width=1, border_color=COL_BORDER)
            c.grid(row=0, column=i, sticky="ew", padx=5)
            ctk.CTkLabel(c, text=val, font=(FONT_FAMILY, 24, "bold"),
                         text_color=col, anchor="w").pack(anchor="w", padx=16,
                                                          pady=(12, 0))
            ctk.CTkLabel(c, text=lbl, font=(FONT_FAMILY, 11),
                         text_color=COL_TXT_DIM, anchor="w").pack(
                anchor="w", padx=16, pady=(0, 12))

    def _panel(self, parent, title, row):
        c = ctk.CTkFrame(parent, fg_color=COL_PANEL, corner_radius=12,
                         border_width=1, border_color=COL_BORDER)
        c.grid(row=row, column=0, sticky="ew", pady=8)
        c.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(c, text=title, anchor="w", font=(FONT_FAMILY, 15, "bold"),
                     text_color=COL_TXT).pack(anchor="w", padx=14, pady=(12, 6))
        return c

    def _kv(self, parent, key, value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=2)
        ctk.CTkLabel(row, text=key, anchor="w", font=(FONT_FAMILY, 12),
                     text_color=COL_TXT_DIM).pack(side="left")
        ctk.CTkLabel(row, text=value, anchor="e", font=(FONT_FAMILY, 12, "bold"),
                     text_color=COL_TXT).pack(side="right")

    def _rate_row(self, parent, name, rate):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=3)
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text=name, width=150, anchor="w",
                     font=(FONT_FAMILY, 12), text_color=COL_TXT).grid(
            row=0, column=0, sticky="w")
        bar = ctk.CTkProgressBar(row, height=12,
                                 progress_color=COL_RUN if rate > 0.5 else COL_ACCENT)
        bar.set(min(rate, 1.0))
        bar.grid(row=0, column=1, sticky="ew", padx=10)
        ctk.CTkLabel(row, text=f"{rate:.0%}", width=50, anchor="e",
                     font=(FONT_FAMILY, 12, "bold"), text_color=COL_TXT).grid(
            row=0, column=2, sticky="e")

    def _table(self, parent, df, max_rows=150, padded=False):
        import tkinter.font as tkfont

        holder = ctk.CTkFrame(parent, fg_color="transparent")
        if padded:
            holder.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        else:
            holder.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        tree = ttk.Treeview(holder, style="Clinic.Treeview", show="headings",
                            height=18)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(holder, orient="vertical", command=tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = ttk.Scrollbar(holder, orient="horizontal", command=tree.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        head_font = tkfont.Font(family=FONT_FAMILY, size=10, weight="bold")
        cell_font = tkfont.Font(family=FONT_FAMILY, size=10)

        cols = list(df.columns)
        tree["columns"] = cols

        # Échantillonne tout le dataset pour représenter toutes les classes.
        n = len(df)
        if max_rows is None or n <= max_rows:
            sample = df
        else:
            step = n / max_rows
            idx = sorted({min(int(i * step), n - 1) for i in range(max_rows)})
            sample = df.iloc[idx]

        display_rows = [
            [round(v, 2) if isinstance(v, float) else v for v in rrow]
            for _, rrow in sample.iterrows()
        ]

        for ci, c in enumerate(cols):
            w = head_font.measure(str(c)) + 30
            for dr in display_rows:
                w = max(w, cell_font.measure(str(dr[ci])) + 24)
            w = max(70, min(w, 320))
            tree.heading(c, text=c)
            tree.column(c, width=w, minwidth=w, anchor="center", stretch=False)

        for dr in display_rows:
            tree.insert("", "end", values=dr)

    def _on_load_existing(self):
        d = filedialog.askdirectory(
            title="Dossier de résultats (figures/, reports/, datasets/)")
        if not d:
            return
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, d)
        self.output_dir = Path(d)
        self.pipeline = None
        self._summary_cache = None
        ddir, fdir, rdir = (self.output_dir / "datasets",
                            self.output_dir / "figures",
                            self.output_dir / "reports")
        self.done["1"] = (ddir / "dataset_copula.csv").exists()
        self.done["2"] = (ddir / "dataset_ctgan.csv").exists()
        self.done["3"] = self.done["1"] or self.done["2"]
        self.done["4"] = (rdir / "analysis_report.json").exists()
        self.done["5"] = fdir.exists() and any(fdir.glob("histograms*.png"))
        self.done["6"] = (rdir / "ml_report.json").exists()
        for idx, it in self.items.items():
            it.set_state("done" if self.done[idx] else "pending")
        self._refresh_enabled()
        first_done = next((i for i in "123456" if self.done[i]), "1")
        self._select_phase(first_done)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Dossier de sortie")
        if d:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, d)

    def _open_folder(self):
        p = self.output_entry.get().strip()
        if p and Path(p).exists():
            try:
                os.startfile(Path(p).absolute())  # type: ignore[attr-defined]
            except Exception:
                pass


def main():
    DemoApp().mainloop()


if __name__ == "__main__":
    main()
