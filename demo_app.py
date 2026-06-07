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
    from PIL import Image, ImageTk
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
    ("V", "Validation",            "Rejets par règle • cohérence",     "#16a34a"),
    ("3", "Sauvegarde datasets",   "Export des fichiers CSV",          "#0891b2"),
    ("4", "Analyses statistiques", "Descriptives • corrélations",      "#7c3aed"),
    ("5", "Visualisations",        "Distributions • heatmaps",         "#db2777"),
    ("6", "Évaluation ML",         "3 modèles • validation croisée",   "#ea580c"),
]

VIEW_ONLY = {"V"}

CLASS_FR = {
    "sain": "Sain", "diabete": "Diabète", "dyslipidemie": "Dyslipidémie",
    "hypertension": "Hypertension", "obesite": "Obésité",
    "risque_cardiovasculaire": "Risque cardiovasc.",
}

MODEL_FR = {
    "logistic_regression": "Régression logistique",
    "random_forest": "Forêt aléatoire", "mlp": "Réseau de neurones (MLP)",
}

RULE_FR = {
    "bounds": "Bornes physiologiques absolues",
    "R1_quetelet": "R1 — IMC (formule de Quetelet)",
    "R2_friedewald": "R2 — Bilan lipidique (Friedewald)",
    "R3_pulse_pressure": "R3 — Pression pulsée (PAS − PAD)",
    "R4_glucose_hba1c": "R4 — Cohérence glycémie / HbA1c",
    "class_coherence": "Cohérence de classe diagnostique",
}

VAR_FR = {
    "age": "Âge (ans)", "height": "Taille (cm)", "weight": "Poids (kg)",
    "bmi": "IMC (kg/m²)", "heart_rate": "Fréq. cardiaque (bpm)",
    "sbp": "PAS (mmHg)", "dbp": "PAD (mmHg)", "resp_rate": "Fréq. resp. (/min)",
    "temp": "Température (°C)", "fasting_glucose": "Glycémie à jeun (mg/dL)",
    "hba1c": "HbA1c (%)", "total_chol": "Cholestérol total (mg/dL)",
    "ldl": "LDL (mg/dL)", "hdl": "HDL (mg/dL)", "triglycerides": "Triglycérides (mg/dL)",
}

CATVAR_FR = {
    "sex": "Sexe", "physical_activity": "Activité physique",
    "smoking": "Tabac", "alcohol": "Alcool", "diet_quality": "Qualité du régime",
}

MODALITY_FR = {
    "Female": "Femme", "Male": "Homme",
    "High": "Élevée", "Moderate": "Modérée", "Sedentary": "Sédentaire",
    "Current": "Fumeur actuel", "Former": "Ancien fumeur", "Never": "Jamais fumé",
    "Excessive": "Excessif", "None": "Aucun",
    "Average": "Moyenne", "Good": "Bonne", "Poor": "Mauvaise",
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


def _disable_console_quickedit():
    """Désactive le mode QuickEdit (Windows) : un clic/une sélection dans la
    console mettait en pause l'écriture stdout, ce qui gelait l'entraînement
    jusqu'à ce qu'on appuie sur Entrée."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080

        handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return
        new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
        kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        pass


class _StreamTee:
    _PCT = re.compile(r"(\d{1,3})%\|")
    _EPOCH = re.compile(r"(\d+)\s*/\s*(\d+)")
    _GEN = re.compile(r"Gen(?:erator)?\.?\s*\(?\s*(-?\d+(?:\.\d+)?)", re.I)
    _DISC = re.compile(r"Discrim(?:inator)?\.?\s*\(?\s*(-?\d+(?:\.\d+)?)", re.I)
    _ETA = re.compile(r"\[(\d+:\d+)<(\d+:\d+),\s*([\d.]+\s*(?:it/s|s/it))\]")
    _EPOCH_LINE = re.compile(
        r"Epoch\s+(\d+).*?Loss G:\s*(-?[\d.]+).*?Loss D:\s*(-?[\d.]+)", re.I)

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
            self._emit(line)

    def _emit(self, line):
        m = self._PCT.search(line)
        if m:
            self.q.put(("progress", min(int(m.group(1)) / 100.0, 1.0)))

        info = {}
        me = self._EPOCH.search(line)
        if me:
            info["epoch"], info["total"] = int(me.group(1)), int(me.group(2))
        mg = self._GEN.search(line)
        if mg:
            info["gen"] = mg.group(1)
        md = self._DISC.search(line)
        if md:
            info["disc"] = md.group(1)
        mt = self._ETA.search(line)
        if mt:
            info["eta"], info["rate"] = mt.group(2), mt.group(3)
        ml = self._EPOCH_LINE.search(line)
        if ml:
            info["epoch"] = int(ml.group(1))
            info["gen"], info["disc"] = ml.group(2), ml.group(3)
        if info:
            self.q.put(("ctgan", info))

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
            if idx == "6":
                pipe._step_run_ml_evaluation(
                    progress=lambda info: q.put(("ml", info)))
            else:
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


class _ImageViewer(ctk.CTkToplevel):
    """Fenêtre d'image avec zoom (molette/boutons) et déplacement (glisser)."""

    def __init__(self, master, path, title):
        super().__init__(master)
        self.title(title)
        self.geometry("1024x780")
        self.configure(fg_color=COL_BG)
        self._path = path
        self._img = Image.open(path)
        self._scale = 1.0
        self._fit_scale = 1.0
        self._min_scale = 0.05
        self._max_scale = 12.0
        self._photo = None

        bar = ctk.CTkFrame(self, fg_color=COL_PANEL, corner_radius=0)
        bar.pack(fill="x")

        def tb(txt, cmd, w=64, **kw):
            b = ctk.CTkButton(bar, text=txt, width=w, height=32, corner_radius=8,
                              font=(FONT_FAMILY, 13, "bold"), command=cmd, **kw)
            return b

        tb("➖", lambda: self._zoom(1 / 1.25)).pack(side="left", padx=(10, 4), pady=8)
        tb("➕", lambda: self._zoom(1.25)).pack(side="left", padx=4, pady=8)
        tb("Ajuster", self._fit, w=84, fg_color=COL_ACCENT,
           hover_color="#0b7d72").pack(side="left", padx=4)
        tb("100 %", self._actual, w=70).pack(side="left", padx=4)
        ctk.CTkLabel(
            bar, text="Molette : zoom    •    glisser : déplacer    •    "
            "double-clic : ajuster", font=(FONT_FAMILY, 11),
            text_color=COL_TXT_DIM).pack(side="left", padx=14)
        tb("🗂  Fichier", self._open_file, w=92, fg_color=COL_CARD,
           text_color=COL_TXT, hover_color=COL_CARD_HOVER).pack(side="right", padx=10)

        self._canvas = tk.Canvas(self, bg="#0f172a", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._img_id = self._canvas.create_image(0, 0, anchor="nw")

        self._canvas.bind("<MouseWheel>", self._on_wheel)
        self._canvas.bind("<Button-4>", lambda e: self._zoom(1.25))
        self._canvas.bind("<Button-5>", lambda e: self._zoom(1 / 1.25))
        self._canvas.bind("<ButtonPress-1>",
                          lambda e: self._canvas.scan_mark(e.x, e.y))
        self._canvas.bind("<B1-Motion>",
                          lambda e: self._canvas.scan_dragto(e.x, e.y, gain=1))
        self._canvas.bind("<Double-Button-1>", lambda e: self._fit())
        self.after(120, self._fit)

    def _resample(self):
        return getattr(getattr(Image, "Resampling", Image), "LANCZOS",
                       getattr(Image, "LANCZOS", 1))

    def _redraw(self):
        w = max(int(self._img.width * self._scale), 1)
        h = max(int(self._img.height * self._scale), 1)
        self._photo = ImageTk.PhotoImage(self._img.resize((w, h), self._resample()))
        cw = max(self._canvas.winfo_width(), 1)
        ch = max(self._canvas.winfo_height(), 1)
        ox = max((cw - w) // 2, 0)
        oy = max((ch - h) // 2, 0)
        self._canvas.itemconfigure(self._img_id, image=self._photo)
        self._canvas.coords(self._img_id, ox, oy)
        self._canvas.configure(scrollregion=(0, 0, max(w, cw), max(h, ch)))

    def _zoom(self, factor):
        xv, yv = self._canvas.xview(), self._canvas.yview()
        cx, cy = (xv[0] + xv[1]) / 2, (yv[0] + yv[1]) / 2
        self._scale = min(max(self._scale * factor, self._min_scale),
                          self._max_scale)
        self._redraw()
        xv, yv = self._canvas.xview(), self._canvas.yview()
        self._canvas.xview_moveto(max(0.0, cx - (xv[1] - xv[0]) / 2))
        self._canvas.yview_moveto(max(0.0, cy - (yv[1] - yv[0]) / 2))

    def _on_wheel(self, e):
        self._zoom(1.25 if e.delta > 0 else 1 / 1.25)

    def _fit(self):
        cw, ch = self._canvas.winfo_width(), self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.after(80, self._fit)
            return
        self._fit_scale = min(cw / self._img.width, ch / self._img.height)
        self._min_scale = self._fit_scale * 0.5
        self._max_scale = self._fit_scale * 16
        self._scale = self._fit_scale
        self._redraw()
        self._canvas.xview_moveto(0.0)
        self._canvas.yview_moveto(0.0)

    def _actual(self):
        self._scale = min(max(1.0, self._min_scale), self._max_scale)
        self._redraw()

    def _open_file(self):
        try:
            os.startfile(Path(self._path).absolute())  # type: ignore[attr-defined]
        except Exception:
            pass


class PhaseItem(ctk.CTkFrame):
    def __init__(self, master, idx, title, subtitle, accent,
                 on_select, on_run, runnable=True):
        super().__init__(master, fg_color=COL_PANEL, corner_radius=12,
                         border_width=1, border_color=COL_BORDER)
        self.idx = idx
        self.accent = accent
        self.on_select = on_select
        self.on_run = on_run
        self.runnable = runnable
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

        if runnable:
            self.run_btn = ctk.CTkButton(
                self, text="▶", width=40, height=40, corner_radius=10,
                font=(FONT_FAMILY, 15, "bold"), fg_color=accent,
                hover_color=COL_TXT, text_color="#ffffff",
                command=lambda: self.on_run(self.idx))
            self.run_btn.grid(row=0, column=2, rowspan=2, padx=(6, 12), pady=10)
        else:
            self.run_btn = None
            eye = ctk.CTkLabel(self, text="👁", width=40,
                               font=(FONT_FAMILY, 16), text_color=COL_TXT_DIM)
            eye.grid(row=0, column=2, rowspan=2, padx=(6, 12), pady=10)
            eye.bind("<Button-1>", lambda e: self.on_select(self.idx))

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
        if self.run_btn is None:
            return
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
        self._run_params: dict | None = None
        self._run_prog = None
        self._run_pct = None
        self._run_status = None
        self._run_epoch_lbl = None
        self._run_loss_lbl = None
        self._run_eta_lbl = None
        self._run_ml_stage = None
        self._run_ml_model = None
        self._run_ml_count = None

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
                             self._select_phase, self._run_phase,
                             runnable=idx not in VIEW_ONLY)
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
        for v in VIEW_ONLY:
            self.done[v] = self.done["1"]
        idx = self.selected
        _, _, sub, accent = self._phase_meta(idx)
        if idx in VIEW_ONLY:
            self.h_run.grid_remove()
            self.h_timer.configure(text="")
            self.h_status.configure(
                text="✓  Statistiques de validation ci-dessous"
                if self.done[idx] else sub)
            return
        self.h_run.grid()
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
        for v in VIEW_ONLY:
            self.done[v] = base
        for idx, it in self.items.items():
            if idx in VIEW_ONLY:
                it.set_state("done" if self.done[idx] else "pending")
                continue
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
        if idx in VIEW_ONLY:
            self._select_phase(idx)
            return
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
        self._run_params = params
        self.output_dir = Path(params["output_dir"])
        self.running_idx = idx
        self.items[idx].set_state("running")
        self.start_time = time.time()
        self._select_phase(idx)
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
                    if self._run_status is not None and self._run_status.winfo_exists():
                        self._run_status.configure(text=str(payload)[:120])
                elif kind == "progress":
                    if self._run_prog is not None and self._run_prog.winfo_exists():
                        self._run_prog.set(payload)
                    if self._run_pct is not None and self._run_pct.winfo_exists():
                        self._run_pct.configure(text=f"{int(payload * 100)} %")
                elif kind == "ctgan":
                    self._update_ctgan_info(payload)
                elif kind == "ml":
                    frac = payload.get("frac", 0.0)
                    if self._run_prog is not None and self._run_prog.winfo_exists():
                        self._run_prog.set(frac)
                    if self._run_pct is not None and self._run_pct.winfo_exists():
                        self._run_pct.configure(text=f"{int(frac * 100)} %")
                    self._update_ml_info(payload)
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
        self._run_prog = None
        self._run_pct = None
        self._run_status = None
        self._run_epoch_lbl = None
        self._run_loss_lbl = None
        self._run_eta_lbl = None
        self._run_ml_stage = None
        self._run_ml_model = None
        self._run_ml_count = None
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
         "V": self._render_validation,
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
        _, title, _, accent = self._phase_meta(idx)
        card = ctk.CTkFrame(self.output, fg_color=COL_PANEL, corner_radius=14,
                            border_width=1, border_color=COL_BORDER)
        card.grid(row=0, column=0, sticky="nsew")
        card.grid_rowconfigure(0, weight=1)
        card.grid_columnconfigure(0, weight=1)

        f = ctk.CTkFrame(card, fg_color="transparent", width=480)
        f.grid(row=0, column=0)
        f.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(f, text="⏳", font=(FONT_FAMILY, 48),
                     text_color=accent).grid(row=0, column=0, pady=(0, 6))
        ctk.CTkLabel(f, text=f"{title} — en cours…",
                     font=(FONT_FAMILY, 19, "bold"),
                     text_color=COL_TXT).grid(row=1, column=0)

        p = self._run_params or {}

        m = p.get("m_per_class")
        if idx == "2":
            rows = [
                ("Epochs d'entraînement", p.get("ctgan_epochs", "—")),
                ("Taille de batch", p.get("ctgan_batch_size", "—")),
                ("Patients par classe", m if m is not None else "—"),
                ("Total à générer", 6 * m if isinstance(m, int) else "—"),
                ("Graine (seed)", p.get("seed", "—")),
            ]
        elif idx == "6":
            rows = [
                ("Protocole", "Split 80/20 + CV 5-fold"),
                ("Modèles", "Rég. logistique • Forêt • MLP"),
                ("Patients évalués", 6 * m if isinstance(m, int) else "—"),
                ("Graine (seed)", p.get("seed", "—")),
            ]
        else:
            rows = []

        if rows:
            info = ctk.CTkFrame(f, fg_color=COL_CARD, corner_radius=10, width=440)
            info.grid(row=2, column=0, sticky="ew", pady=(16, 12))
            info.grid_columnconfigure(0, weight=1)
            for i, (k, v) in enumerate(rows):
                ctk.CTkLabel(info, text=k, font=(FONT_FAMILY, 12),
                             text_color=COL_TXT_DIM, anchor="w").grid(
                    row=i, column=0, sticky="w", padx=16, pady=5)
                ctk.CTkLabel(info, text=str(v), font=(FONT_FAMILY, 12, "bold"),
                             text_color=COL_TXT, anchor="e").grid(
                    row=i, column=1, sticky="e", padx=16, pady=5)

        if idx in ("2", "6"):
            barf = ctk.CTkFrame(f, fg_color="transparent", width=440)
            barf.grid(row=3, column=0, sticky="ew", pady=(0, 8))
            barf.grid_columnconfigure(0, weight=1)
            self._run_prog = ctk.CTkProgressBar(barf, progress_color=accent,
                                                height=16, width=380)
            self._run_prog.set(0)
            self._run_prog.grid(row=0, column=0, sticky="ew", padx=(0, 10))
            self._run_pct = ctk.CTkLabel(barf, text="0 %", width=48,
                                         font=(FONT_FAMILY, 13, "bold"),
                                         text_color=accent)
            self._run_pct.grid(row=0, column=1)

        if idx == "2":
            live = ctk.CTkFrame(f, fg_color=COL_CARD, corner_radius=10, width=440)
            live.grid(row=4, column=0, sticky="ew", pady=(0, 12))
            live.grid_columnconfigure((0, 1), weight=1)
            self._run_epoch_lbl = ctk.CTkLabel(
                live, text="Epoch  — / —", font=(FONT_FAMILY, 13, "bold"),
                text_color=COL_TXT)
            self._run_epoch_lbl.grid(row=0, column=0, columnspan=2,
                                     sticky="ew", pady=(10, 2))
            self._run_loss_lbl = ctk.CTkLabel(
                live, text="Pertes  —", font=(FONT_FAMILY, 11),
                text_color=COL_TXT_DIM)
            self._run_loss_lbl.grid(row=1, column=0, sticky="ew", pady=(0, 10))
            self._run_eta_lbl = ctk.CTkLabel(
                live, text="", font=(FONT_FAMILY, 11), text_color=COL_TXT_DIM)
            self._run_eta_lbl.grid(row=1, column=1, sticky="ew", pady=(0, 10))

        if idx == "6":
            live = ctk.CTkFrame(f, fg_color=COL_CARD, corner_radius=10, width=440)
            live.grid(row=4, column=0, sticky="ew", pady=(0, 12))
            live.grid_columnconfigure(0, weight=1)
            self._run_ml_stage = ctk.CTkLabel(
                live, text="Préparation…", font=(FONT_FAMILY, 11),
                text_color=COL_TXT_DIM)
            self._run_ml_stage.grid(row=0, column=0, sticky="ew", pady=(10, 0))
            self._run_ml_model = ctk.CTkLabel(
                live, text="🔄  En attente…", font=(FONT_FAMILY, 16, "bold"),
                text_color=accent)
            self._run_ml_model.grid(row=1, column=0, sticky="ew", pady=(0, 2))
            self._run_ml_count = ctk.CTkLabel(
                live, text="", font=(FONT_FAMILY, 11), text_color=COL_TXT_DIM)
            self._run_ml_count.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        note = ("Entraînement du réseau génératif puis génération validée — "
                "cela peut durer plusieurs minutes." if idx == "2"
                else "Entraînement et validation croisée des 3 modèles — "
                "le MLP peut prendre un moment." if idx == "6"
                else "Veuillez patienter, l'étape est en cours d'exécution.")
        ctk.CTkLabel(f, text=note, font=(FONT_FAMILY, 12),
                     text_color=COL_TXT_DIM, wraplength=440,
                     justify="center").grid(row=5, column=0, pady=(2, 8))

        self._run_status = ctk.CTkLabel(f, text="Initialisation…",
                                        font=(FONT_FAMILY, 12), text_color=accent,
                                        wraplength=440, justify="center")
        self._run_status.grid(row=6, column=0, pady=(0, 4))

    def _update_ctgan_info(self, info):
        el = self._run_epoch_lbl
        if el is not None and el.winfo_exists() and "epoch" in info:
            tot = info.get("total")
            el.configure(text=f"Epoch  {info['epoch']} / {tot}" if tot
                         else f"Epoch  {info['epoch']}")
        ll = self._run_loss_lbl
        if (ll is not None and ll.winfo_exists()
                and ("gen" in info or "disc" in info)):
            g, d = info.get("gen", "—"), info.get("disc", "—")
            ll.configure(text=f"Pertes  •  Générateur {g}  •  Discriminateur {d}")
        et = self._run_eta_lbl
        if et is not None and et.winfo_exists():
            parts = []
            if info.get("eta"):
                parts.append(f"reste ~{info['eta']}")
            if info.get("rate"):
                parts.append(str(info["rate"]))
            if parts:
                et.configure(text="  •  ".join(parts))

    def _update_ml_info(self, info):
        st = self._run_ml_stage
        if st is not None and st.winfo_exists() and info.get("stage"):
            st.configure(text=info["stage"])
        md = self._run_ml_model
        if md is not None and md.winfo_exists() and info.get("model"):
            md.configure(text=f"🔄  {info['model']}")
        ct = self._run_ml_count
        if ct is not None and ct.winfo_exists():
            d, t = info.get("done"), info.get("total")
            if d and t:
                ct.configure(text=f"Modèle {d} / {t}")

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

        loading = ctk.CTkLabel(self._ds_table_holder, text="⏳  Chargement…",
                               font=(FONT_FAMILY, 13), text_color=COL_TXT_DIM)
        loading.grid(row=0, column=0, pady=20)
        self.update_idletasks()

        try:
            import pandas as pd
            df = pd.read_csv(path)
        except Exception as e:
            loading.destroy()
            ctk.CTkLabel(self._ds_table_holder, text=f"Erreur : {e}",
                         text_color=COL_ERR).grid(row=0, column=0, pady=20)
            return

        loading.destroy()
        self._table(self._ds_table_holder, df, max_rows=None, padded=True)

    def _render_validation(self):
        methods = []
        for key, name in (("copula", "Méthode 1 — Copule gaussienne"),
                          ("ctgan", "Méthode 2 — CTGAN")):
            if self._gen_stats(key):
                methods.append((key, name))

        wrap = ctk.CTkFrame(self.output, fg_color="transparent")
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.grid_rowconfigure(1, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        if not methods:
            ctk.CTkLabel(
                wrap, text="Aucune statistique de validation disponible.\n"
                "Exécutez d'abord l'étape 1 (génération copule).",
                font=(FONT_FAMILY, 13), text_color=COL_TXT_DIM,
                justify="center").grid(row=0, column=0, pady=30)
            return

        self._val_map = {name: key for key, name in methods}
        sel = ctk.CTkFrame(wrap, fg_color="transparent")
        sel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(sel, text="Méthode :", font=(FONT_FAMILY, 13),
                     text_color=COL_TXT).pack(side="left", padx=(0, 8))
        keys = list(self._val_map.keys())
        self._val_menu = ctk.CTkOptionMenu(
            sel, values=keys, command=self._show_validation, width=300,
            fg_color=COL_CARD, button_color=COL_ACCENT_2,
            button_hover_color="#4338ca", text_color=COL_TXT)
        self._val_menu.pack(side="left")

        self._val_holder = ctk.CTkScrollableFrame(wrap, fg_color="transparent")
        self._val_holder.grid(row=1, column=0, sticky="nsew")
        self._val_holder.grid_columnconfigure(0, weight=1)
        self._show_validation(keys[0])

    def _show_validation(self, label):
        for w in self._val_holder.winfo_children():
            w.destroy()
        stats = self._gen_stats(self._val_map.get(label))
        if not stats:
            ctk.CTkLabel(self._val_holder, text="Aucune donnée.",
                         text_color=COL_TXT_DIM).grid(row=0, column=0, pady=20)
            return

        acc = stats.get("total_accepted", 0)
        att = stats.get("total_attempts", 0)
        rej = stats.get("total_rejected", 0)
        rate = stats.get("global_rejection_rate", 0.0)
        avg = stats.get("average_attempts_per_accepted")

        self._cards_row(self._val_holder, [
            ("Patients acceptés", f"{acc:,}".replace(",", " "), COL_OK),
            ("Tentatives totales", f"{att:,}".replace(",", " "), COL_ACCENT_2),
            ("Échantillons rejetés", f"{rej:,}".replace(",", " "), COL_RUN),
            ("Taux de rejet global", f"{rate:.1%}", COL_ERR),
        ], 0)

        r = 1
        if isinstance(avg, (int, float)):
            card = self._panel(self._val_holder,
                               "Efficacité de la génération", r); r += 1
            self._kv(card, "Tentatives moyennes par patient accepté",
                     f"{avg:.2f}")
            ctk.CTkLabel(
                card, text="Chaque patient est ré-échantillonné jusqu'à "
                "passer toute la cascade de validation.",
                font=(FONT_FAMILY, 11), text_color=COL_TXT_DIM,
                anchor="w", justify="left").pack(anchor="w", padx=16, pady=(2, 12))

        per_rule = stats.get("rejections_per_rule", {})
        if per_rule:
            card = self._panel(self._val_holder,
                               "Rejets par règle de validation", r); r += 1
            total = sum(per_rule.values()) or 1
            for rule, cnt in sorted(per_rule.items(), key=lambda kv: -kv[1]):
                self._rule_row(card, RULE_FR.get(rule, rule), cnt, cnt / total)
            ctk.CTkFrame(card, height=6, fg_color="transparent").pack()

        by_class = stats.get("rejection_rate_by_class", {})
        if by_class:
            card = self._panel(self._val_holder,
                               "Taux de rejet par classe", r); r += 1
            for cls, rt in by_class.items():
                self._rate_row(card, CLASS_FR.get(cls, cls), rt)
            ctk.CTkFrame(card, height=6, fg_color="transparent").pack()

    def _rule_row(self, parent, name, count, frac):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=3)
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text=name, width=230, anchor="w",
                     font=(FONT_FAMILY, 12), text_color=COL_TXT).grid(
            row=0, column=0, sticky="w")
        bar = ctk.CTkProgressBar(row, height=12, progress_color=COL_ACCENT_2)
        bar.set(min(frac, 1.0))
        bar.grid(row=0, column=1, sticky="ew", padx=10)
        ctk.CTkLabel(row, text=f"{count:,}".replace(",", " "), width=70,
                     anchor="e", font=(FONT_FAMILY, 12, "bold"),
                     text_color=COL_TXT).grid(row=0, column=2, sticky="e")

    def _render_analysis(self):
        ana = self._read_json("analysis_report.json")
        wrap = ctk.CTkFrame(self.output, fg_color="transparent")
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.grid_rowconfigure(1, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        if not ana:
            ctk.CTkLabel(
                wrap, text="Aucun rapport d'analyse.\nExécutez l'étape 4.",
                font=(FONT_FAMILY, 13), text_color=COL_TXT_DIM,
                justify="center").grid(row=0, column=0, pady=30)
            return

        self._ana = ana
        sections = []
        if isinstance(ana.get("copula"), dict):
            sections.append("Méthode 1 — Copule")
        if isinstance(ana.get("ctgan"), dict):
            sections.append("Méthode 2 — CTGAN")
        if isinstance(ana.get("method_comparison"), dict):
            sections.append("Comparaison")

        selrow = ctk.CTkFrame(wrap, fg_color="transparent")
        selrow.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._ana_seg = ctk.CTkSegmentedButton(
            selrow, values=sections, command=self._show_analysis,
            fg_color=COL_CARD, selected_color=COL_ACCENT,
            selected_hover_color="#0b7d72", unselected_color=COL_CARD,
            unselected_hover_color=COL_CARD_HOVER, text_color=COL_TXT,
            font=(FONT_FAMILY, 12, "bold"))
        self._ana_seg.pack(side="left")

        self._ana_holder = ctk.CTkScrollableFrame(wrap, fg_color="transparent")
        self._ana_holder.grid(row=1, column=0, sticky="nsew")
        self._ana_holder.grid_columnconfigure(0, weight=1)
        if sections:
            self._ana_seg.set(sections[0])
            self._show_analysis(sections[0])

    def _show_analysis(self, section):
        for w in self._ana_holder.winfo_children():
            w.destroy()
        if section.startswith("Méthode 1"):
            self._analysis_method(self._ana.get("copula", {}))
        elif section.startswith("Méthode 2"):
            self._analysis_method(self._ana.get("ctgan", {}))
        else:
            self._analysis_comparison(self._ana.get("method_comparison", {}))

    def _analysis_method(self, block):
        holder = self._ana_holder
        desc = block.get("descriptive", {}) or {}
        gen = block.get("generation_stats", {}) or {}
        epi = block.get("epidemiological", {}) or {}

        n_total = desc.get("n_total", 0)
        n_classes = len(desc.get("n_by_class", {}) or {})
        rate = gen.get("global_rejection_rate")
        avg = gen.get("average_attempts_per_accepted")
        self._cards_row(holder, [
            ("Patients", f"{n_total:,}".replace(",", " "), COL_ACCENT),
            ("Classes", str(n_classes), COL_OK),
            ("Taux de rejet", f"{rate:.1%}" if isinstance(rate, (int, float))
             else "—", COL_RUN),
            ("Tentatives moy.", f"{avg:.2f}" if isinstance(avg, (int, float))
             else "—", COL_ACCENT_2),
        ], 0)
        r = 1

        if epi:
            card = self._panel(holder, "Validation épidémiologique", r); r += 1
            npass, ntot = epi.get("n_patterns_passed"), epi.get("n_patterns_total")
            if isinstance(npass, int) and isinstance(ntot, int):
                self._badge_line(card, "Motifs cliniques validés",
                                 f"{npass} / {ntot}", npass == ntot)
            ao = epi.get("age_ordering")
            if isinstance(ao, list) and ao:
                n_ok = sum(1 for x in ao if isinstance(x, dict) and x.get("passed"))
                self._badge_line(card, "Ordres d'âge entre classes respectés",
                                 f"{n_ok} / {len(ao)}",
                                 bool(epi.get("age_ordering_all_passed",
                                              n_ok == len(ao))))
            pats = epi.get("patterns")
            if isinstance(pats, list) and pats:
                rows = []
                for p in pats:
                    if not isinstance(p, dict):
                        continue
                    thr = p.get("threshold")
                    rule = p.get("rule")
                    thr_txt = (f"≥ {thr}" if rule == "min_mean"
                               else f"≤ {thr}" if rule == "max_mean" else f"{thr}")
                    rows.append([
                        CLASS_FR.get(p.get("class"), p.get("class")),
                        VAR_FR.get(p.get("variable"), p.get("variable")),
                        str(p.get("observed_mean")), thr_txt,
                        "✓" if p.get("passed") else "✗",
                    ])
                self._grid_table(card, ["Classe", "Variable", "Observé",
                                        "Seuil", "OK"], rows)
            ctk.CTkFrame(card, height=4, fg_color="transparent").pack()

        cont = desc.get("continuous_global")
        if isinstance(cont, dict) and cont:
            card = self._panel(
                holder, "Statistiques descriptives — variables continues", r)
            r += 1
            rows = []
            for var, st in cont.items():
                if not isinstance(st, dict):
                    continue
                rows.append([
                    VAR_FR.get(var, var), str(st.get("mean", "—")),
                    str(st.get("std", "—")), str(st.get("median", "—")),
                    str(st.get("min", "—")), str(st.get("max", "—")),
                ])
            self._grid_table(card, ["Variable", "Moyenne", "Écart-type",
                                    "Médiane", "Min", "Max"], rows)
            ctk.CTkFrame(card, height=4, fg_color="transparent").pack()

        cat = desc.get("categorical_global")
        if isinstance(cat, dict) and cat:
            card = self._panel(holder, "Variables catégorielles (proportions)", r)
            r += 1
            for var, mods in cat.items():
                if not isinstance(mods, dict):
                    continue
                ctk.CTkLabel(card, text=CATVAR_FR.get(var, var), anchor="w",
                             font=(FONT_FAMILY, 12, "bold"),
                             text_color=COL_ACCENT_2).pack(
                    fill="x", padx=16, pady=(10, 2))
                for mod, prop in sorted(mods.items(), key=lambda kv: -kv[1]):
                    self._rate_row(card, MODALITY_FR.get(mod, mod), prop)
            ctk.CTkFrame(card, height=8, fg_color="transparent").pack()

    def _analysis_comparison(self, mc):
        holder = self._ana_holder
        n_c, n_g = mc.get("n_copula"), mc.get("n_ctgan")
        frob = mc.get("correlation_frobenius_distance")
        self._cards_row(holder, [
            ("Patients — Copule", f"{n_c:,}".replace(",", " ")
             if isinstance(n_c, int) else "—", COL_ACCENT),
            ("Patients — CTGAN", f"{n_g:,}".replace(",", " ")
             if isinstance(n_g, int) else "—", COL_ACCENT_2),
            ("Distance corrélations", f"{frob:.3f}"
             if isinstance(frob, (int, float)) else "—", COL_RUN),
        ], 0)
        r = 1

        div = mc.get("correlation_top_divergences")
        if isinstance(div, list) and div:
            card = self._panel(
                holder, "Plus fortes divergences de corrélation "
                "(Copule vs CTGAN)", r); r += 1
            rows = []
            for x in div:
                if not isinstance(x, dict):
                    continue
                pair = (f"{VAR_FR.get(x.get('var1'), x.get('var1'))}  ×  "
                        f"{VAR_FR.get(x.get('var2'), x.get('var2'))}")
                rows.append([pair, str(x.get("corr_a")), str(x.get("corr_b")),
                             str(x.get("delta"))])
            self._grid_table(card, ["Paire de variables", "Copule", "CTGAN",
                                    "Δ"], rows)
            ctk.CTkLabel(
                card, text="Δ = écart absolu entre les corrélations de Pearson "
                "des deux méthodes.", font=(FONT_FAMILY, 10),
                text_color=COL_TXT_DIM, anchor="w", justify="left").pack(
                fill="x", padx=16, pady=(0, 12))

        cm = mc.get("continuous_means")
        if isinstance(cm, dict) and cm:
            card = self._panel(
                holder, "Moyennes par classe — Copule vs CTGAN", r); r += 1
            self._cmp_means = cm
            self._cmp_class_map = {CLASS_FR.get(c, c): c for c in cm}
            selrow = ctk.CTkFrame(card, fg_color="transparent")
            selrow.pack(fill="x", padx=16, pady=(2, 6))
            ctk.CTkLabel(selrow, text="Classe :", font=(FONT_FAMILY, 12),
                         text_color=COL_TXT).pack(side="left", padx=(0, 8))
            pretty = list(self._cmp_class_map.keys())
            ctk.CTkOptionMenu(
                selrow, values=pretty, command=self._show_cmp_class, width=220,
                fg_color=COL_CARD, button_color=COL_ACCENT_2,
                button_hover_color="#4338ca", text_color=COL_TXT).pack(side="left")
            self._cmp_table_holder = ctk.CTkFrame(card, fg_color="transparent")
            self._cmp_table_holder.pack(fill="x")
            self._show_cmp_class(pretty[0])
            ctk.CTkFrame(card, height=4, fg_color="transparent").pack()

    def _show_cmp_class(self, label):
        for w in self._cmp_table_holder.winfo_children():
            w.destroy()
        data = self._cmp_means.get(self._cmp_class_map.get(label), {})
        rows = []
        for var, st in data.items():
            if not isinstance(st, dict):
                continue
            a, b = st.get("mean_copula"), st.get("mean_ctgan")
            delta = (f"{abs(a - b):.2f}" if isinstance(a, (int, float))
                     and isinstance(b, (int, float)) else "—")
            rows.append([VAR_FR.get(var, var), str(a), str(b), delta])
        self._grid_table(self._cmp_table_holder,
                         ["Variable", "Copule", "CTGAN", "Δ"], rows)

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

        top = ctk.CTkFrame(disp, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        top.grid_columnconfigure(0, weight=1)
        self._fig_title = ctk.CTkLabel(top, text="", anchor="w",
                                       font=(FONT_FAMILY, 14, "bold"),
                                       text_color=COL_TXT)
        self._fig_title.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(top, text="🔍  Agrandir / Zoom", width=160, height=32,
                      corner_radius=8, font=(FONT_FAMILY, 12, "bold"),
                      fg_color=COL_ACCENT_2, hover_color="#4338ca",
                      text_color="#ffffff",
                      command=self._open_viewer).grid(row=0, column=1, sticky="e")

        self._fig_lbl = ctk.CTkLabel(disp, text="", cursor="hand2")
        self._fig_lbl.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        self._fig_lbl.bind("<Double-Button-1>", lambda e: self._open_viewer())
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

    def _open_viewer(self):
        if self._cur_fig:
            self._popup_fig(self._cur_fig,
                            self._fig_title.cget("text") or Path(self._cur_fig).stem)

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
            try:
                os.startfile(Path(path).absolute())  # type: ignore[attr-defined]
            except Exception:
                pass
            return
        try:
            viewer = _ImageViewer(self, str(path), name)
            viewer.after(150, viewer.focus)
        except Exception:
            try:
                os.startfile(Path(path).absolute())  # type: ignore[attr-defined]
            except Exception:
                pass

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

    def _badge_line(self, parent, label, value, ok):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(row, text=label, anchor="w", font=(FONT_FAMILY, 12),
                     text_color=COL_TXT_DIM).pack(side="left")
        ctk.CTkLabel(row, text=("✓  " if ok else "✗  ") + value, anchor="e",
                     font=(FONT_FAMILY, 12, "bold"),
                     text_color=COL_OK if ok else COL_ERR).pack(side="right")

    def _grid_table(self, parent, headers, rows):
        t = ctk.CTkFrame(parent, fg_color=COL_BORDER, corner_radius=8)
        t.pack(fill="x", padx=14, pady=(2, 10))
        ncol = len(headers)
        for j in range(ncol):
            t.grid_columnconfigure(j, weight=3 if j == 0 else 1, uniform="col")
        for j, h in enumerate(headers):
            ctk.CTkLabel(t, text=h, font=(FONT_FAMILY, 11, "bold"),
                         text_color="#ffffff", fg_color=COL_ACCENT,
                         anchor="w" if j == 0 else "center").grid(
                row=0, column=j, sticky="nsew", padx=(0, 1), pady=(0, 1),
                ipady=5, ipadx=8)
        for i, rrow in enumerate(rows, 1):
            bg = COL_PANEL if i % 2 else COL_CARD
            for j, val in enumerate(rrow):
                ctk.CTkLabel(t, text=str(val), font=(FONT_FAMILY, 11),
                             text_color=COL_TXT, fg_color=bg,
                             anchor="w" if j == 0 else "center").grid(
                    row=i, column=j, sticky="nsew", padx=(0, 1), pady=(0, 1),
                    ipady=3, ipadx=8)
        return t

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
        holder.grid_rowconfigure(1, weight=1)
        holder.grid_columnconfigure(0, weight=1)

        cols = list(df.columns)

        n = len(df)
        if max_rows is None or n <= max_rows:
            sample = df
        else:
            step = n / max_rows
            idx = sorted({min(int(i * step), n - 1) for i in range(max_rows)})
            sample = df.iloc[idx]

        tree = ttk.Treeview(holder, style="Clinic.Treeview", show="headings",
                            height=18)
        tree.grid(row=1, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(holder, orient="vertical", command=tree.yview)
        vsb.grid(row=1, column=1, sticky="ns")
        hsb = ttk.Scrollbar(holder, orient="horizontal", command=tree.xview)
        hsb.grid(row=2, column=0, sticky="ew")
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        head_font = tkfont.Font(family=FONT_FAMILY, size=10, weight="bold")
        cell_font = tkfont.Font(family=FONT_FAMILY, size=10)

        tree["columns"] = cols

        display_rows = [
            [round(v, 2) if isinstance(v, float) else v for v in rrow]
            for rrow in sample.itertuples(index=False, name=None)
        ]

        width_probe = display_rows[:60]
        for ci, c in enumerate(cols):
            w = head_font.measure(str(c)) + 30
            for dr in width_probe:
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
    _disable_console_quickedit()
    DemoApp().mainloop()


if __name__ == "__main__":
    main()
