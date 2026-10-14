"""
Reconnaissance de locuteurs en temps réel (3 personnes max) - Français
 - Identification du locuteur : MFCC + GMM (fait maison, sklearn)
 - Parole / bruit / cri : descripteurs acoustiques (dsp.py)
 - Transcription : Vosk (modèle français hors-ligne)
"""
import json
import os
import pickle
import queue
import re
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from sklearn.mixture import GaussianMixture
from vosk import KaldiRecognizer, Model, SetLogLevel

import dsp

import customtkinter as ctk

SR = dsp.SR
BLOCK = 512                      # ~32 ms
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model-fr")
PROFILES = os.path.join(os.path.dirname(__file__), "profiles.pkl")
ENROLL_SECONDS = 12.0
MAX_SPEAKERS = 3
COLORS = ["#00E5FF", "#FF4081", "#B2FF59"]
SetLogLevel(-1)


# ============================================================ MOTEUR
class Engine(threading.Thread):
    def __init__(self, events):
        super().__init__(daemon=True)
        self.events = events
        self.audio_q = queue.Queue()
        self.seg_q = queue.Queue()
        self.running = True
        self.listening = True
        self.mode = "listen"            # "listen" | "enroll"
        self.profiles = {}              # nom -> {"gmm":..., "thr":...}
        self.enroll_feats, self.enroll_text, self.enroll_secs = [], [], 0.0
        self.model = None
        self._load_profiles()

    # ---------- profils
    def _load_profiles(self):
        if os.path.exists(PROFILES):
            with open(PROFILES, "rb") as f:
                self.profiles = pickle.load(f)

    def _save_profiles(self):
        with open(PROFILES, "wb") as f:
            pickle.dump(self.profiles, f)

    def reset_profiles(self):
        self.profiles = {}
        if os.path.exists(PROFILES):
            os.remove(PROFILES)

    # ---------- enrôlement
    def enroll_start(self):
        self.enroll_feats, self.enroll_text, self.enroll_secs = [], [], 0.0
        self.mode = "enroll"

    def enroll_cancel(self):
        self.mode = "listen"

    def enroll_commit(self, name):
        X = np.vstack(self.enroll_feats)
        gmm = GaussianMixture(8, covariance_type="diag", max_iter=150,
                              reg_covar=1e-3, random_state=0).fit(X)
        # seuil de rejet à partir des scores de petits morceaux de la voix
        chunks = [gmm.score(X[i:i + 100]) for i in range(0, len(X) - 100, 50)] or [gmm.score(X)]
        thr = float(np.mean(chunks) - 3 * np.std(chunks) - 1.0)
        self.profiles[name] = {"gmm": gmm, "thr": thr}
        self._save_profiles()
        self.mode = "listen"

    def identify(self, feats):
        if not self.profiles or len(feats) < 20:
            return None, 0.0
        scores = {n: p["gmm"].score(feats) for n, p in self.profiles.items()}
        best = max(scores, key=scores.get)
        others = [s for n, s in scores.items() if n != best]
        ref = max(others) if others else self.profiles[best]["thr"]
        conf = 1 / (1 + np.exp(-(scores[best] - ref) * 2))
        if scores[best] < self.profiles[best]["thr"]:
            return None, float(conf)
        return best, float(conf)

    # ---------- transcription
    def transcribe(self, audio):
        rec = KaldiRecognizer(self.model, SR)
        rec.AcceptWaveform(audio.tobytes())
        return json.loads(rec.FinalResult()).get("text", "").strip()

    @staticmethod
    def extract_name(text):
        m = re.search(r"(?:appelle|suis|nom est|c'est)\s+(\w+)", text)
        if m:
            return m.group(1).capitalize()
        w = text.split()
        return w[-1].capitalize() if w else ""

    # ---------- traitement d'un segment
    def process(self, audio):
        feats = dsp.mfcc(audio)

        if self.mode == "enroll":
            if len(feats):
                self.enroll_feats.append(feats)
                self.enroll_secs += len(audio) / SR
                self.events.put(("enroll_progress", min(self.enroll_secs / ENROLL_SECONDS, 1.0)))
            if self.enroll_secs >= ENROLL_SECONDS:
                self.mode = "wait"
                self.events.put(("enroll_done", None))
            return

        if not self.listening or self.mode != "listen":
            return
        text = self.transcribe(audio)
        info = dsp.analyze(audio)
        label = dsp.decide(info)
        if label == "bruit" and len(text.split()) < 2:
            text = ""                     # faux mots de Vosk sur du bruit
        if text:
            kind = "parole"
        elif label == "parole?":
            kind = "sans_mots"
        else:
            kind = label                  # bruit / cri
        who, conf = (None, 0.0) if kind == "bruit" else self.identify(feats)
        self.events.put(("result", dict(kind=kind, who=who, conf=conf, text=text, info=info)))

    def worker(self):
        while self.running:
            try:
                self.process(self.seg_q.get(timeout=0.2))
            except queue.Empty:
                pass
            except Exception as e:
                self.events.put(("status", f"Erreur : {e}"))

    # ---------- boucle audio + détection d'activité vocale
    def run(self):
        if not os.path.isdir(MODEL_DIR):
            self.events.put(("status", "Modèle Vosk introuvable : voir README (dossier model-fr)"))
            return
        self.events.put(("status", "Chargement du modèle français…"))
        self.model = Model(MODEL_DIR)
        threading.Thread(target=self.worker, daemon=True).start()
        self.events.put(("ready", None))

        cb = lambda indata, f, t, s: self.audio_q.put(indata[:, 0].copy())
        pre, seg = deque(maxlen=10), []
        in_speech, sil, hits, floor = False, 0, 0, 0.005
        with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=BLOCK, callback=cb):
            while self.running:
                try:
                    block = self.audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean((block / 32768.0) ** 2)))
                self.events.put(("level", rms))
                thr = max(floor * 3.5, 0.015)
                if not in_speech:
                    pre.append(block)
                    if rms > thr:
                        hits += 1
                    else:
                        hits = 0
                        floor = max(0.002, 0.97 * floor + 0.03 * rms)
                    if hits >= 3:
                        in_speech, seg, sil = True, list(pre), 0
                        self.events.put(("speaking", True))
                else:
                    seg.append(block)
                    sil = 0 if rms > thr * 0.7 else sil + 1
                    if sil >= 22 or len(seg) >= 470:       # 0,7 s de silence ou 15 s max
                        audio = np.concatenate(seg)
                        in_speech, hits = False, 0
                        pre.clear()
                        self.events.put(("speaking", False))
                        if len(audio) >= 0.3 * SR:
                            self.seg_q.put(audio)


# ============================================================ INTERFACE
class App(ctk.CTk):
    BG, CARD, TEXT = "#0B0F1A", "#141B2D", "#E8ECF4"

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("VoiceTrack - Reconnaissance de locuteurs")
        self.geometry("1000x640")
        self.configure(fg_color=self.BG)
        self.events = queue.Queue()
        self.engine = Engine(self.events)
        self.cards = {}
        self.enrolling = False
        self._build()
        for name in self.engine.profiles:
            self._add_card(name)
        self.engine.start()
        self.after(40, self._poll)

    def _build(self):
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=24, pady=(18, 6))
        ctk.CTkLabel(head, text="🎙  VoiceTrack", font=("Segoe UI", 28, "bold"),
                     text_color="#00E5FF").pack(side="left")
        self.status = ctk.CTkLabel(head, text="Démarrage…", text_color="#8A94A8")
        self.status.pack(side="right")

        self.level = ctk.CTkProgressBar(self, height=10, progress_color="#00E5FF", fg_color=self.CARD)
        self.level.set(0)
        self.level.pack(fill="x", padx=24, pady=(0, 12))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24)
        left = ctk.CTkFrame(body, fg_color="transparent", width=300)
        left.pack(side="left", fill="y", padx=(0, 16))
        self.cards_box = ctk.CTkFrame(left, fg_color="transparent")
        self.cards_box.pack(fill="x")

        self.btn_add = ctk.CTkButton(left, text="＋  Ajouter une personne", height=42,
                                     fg_color="#00E5FF", text_color="#00131A",
                                     hover_color="#5CF0FF", command=self._enroll)
        self.btn_add.pack(fill="x", pady=(14, 6))
        self.enroll_bar = ctk.CTkProgressBar(left, progress_color="#FF4081", fg_color=self.CARD)
        self.enroll_bar.set(0)
        self.enroll_lbl = ctk.CTkLabel(left, text="", text_color="#FFB3CB", wraplength=280, justify="left")
        self.switch = ctk.CTkSwitch(left, text="Écoute active", progress_color="#B2FF59",
                                    command=lambda: setattr(self.engine, "listening", bool(self.switch.get())))
        self.switch.select()
        self.switch.pack(anchor="w", pady=(12, 4))
        ctk.CTkButton(left, text="Effacer les profils", fg_color="#26304A", hover_color="#37436A",
                      command=self._reset).pack(fill="x", pady=4)

        right = ctk.CTkFrame(body, fg_color=self.CARD, corner_radius=16)
        right.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(right, text="Transcription en direct", font=("Segoe UI", 15, "bold"),
                     text_color=self.TEXT).pack(anchor="w", padx=16, pady=(12, 0))
        self.log = ctk.CTkTextbox(right, fg_color=self.CARD, text_color=self.TEXT,
                                  font=("Segoe UI", 14), wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=10)
        self.log.configure(state="disabled")
        self.log.tag_config("noise", foreground="#7C879E")
        self.log.tag_config("scream", foreground="#FFB74D")
        for i, c in enumerate(COLORS):
            self.log.tag_config(f"s{i}", foreground=c)
        self.log.tag_config("unk", foreground="#B0B8C8")

    # ---------- cartes
    def _add_card(self, name):
        i = len(self.cards)
        col = COLORS[i % 3]
        f = ctk.CTkFrame(self.cards_box, fg_color=self.CARD, corner_radius=14,
                         border_width=2, border_color=self.CARD)
        f.pack(fill="x", pady=5)
        ctk.CTkLabel(f, text="●", text_color=col, font=("Segoe UI", 26)).pack(side="left", padx=(14, 6), pady=12)
        box = ctk.CTkFrame(f, fg_color="transparent")
        box.pack(side="left")
        ctk.CTkLabel(box, text=name, font=("Segoe UI", 17, "bold"), text_color=self.TEXT).pack(anchor="w")
        st = ctk.CTkLabel(box, text="en attente", text_color="#8A94A8")
        st.pack(anchor="w")
        self.cards[name] = dict(frame=f, status=st, color=col, tag=f"s{i % 3}")
        self.btn_add.configure(state="normal" if len(self.cards) < MAX_SPEAKERS else "disabled")

    def _flash(self, name, text):
        c = self.cards[name]
        c["frame"].configure(border_color=c["color"])
        c["status"].configure(text=text, text_color=c["color"])
        self.after(1600, lambda: (c["frame"].configure(border_color=self.CARD),
                                  c["status"].configure(text="en attente", text_color="#8A94A8")))

    def _write(self, text, tag):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------- actions
    def _enroll(self):
        if self.enrolling or self.engine.model is None:
            return
        dlg = ctk.CTkInputDialog(text="Écrivez le nom de la personne :", title="Nouveau profil")
        name = (dlg.get_input() or "").strip().capitalize()
        if not name:
            return
        if name in self.cards:
            self.status.configure(text=f"{name} existe déjà", text_color="#FFB74D")
            return
        self.pending_name = name
        self.enrolling = True
        self.engine.enroll_start()
        self.btn_add.configure(text="Annuler", command=self._cancel)
        self.enroll_bar.set(0)
        self.enroll_bar.pack(fill="x", pady=(4, 2))
        self.enroll_lbl.configure(
            text=f"{name}, parlez maintenant pendant ~12 s (par exemple, lisez un texte).")
        self.enroll_lbl.pack(anchor="w")

    def _cancel(self):
        self.engine.enroll_cancel()
        self._end_enroll()

    def _end_enroll(self):
        self.enrolling = False
        self.enroll_bar.pack_forget()
        self.enroll_lbl.pack_forget()
        self.btn_add.configure(text="＋  Ajouter une personne", command=self._enroll,
                               state="normal" if len(self.cards) < MAX_SPEAKERS else "disabled")

    def _reset(self):
        self.engine.reset_profiles()
        for c in self.cards.values():
            c["frame"].destroy()
        self.cards = {}
        self.btn_add.configure(state="normal")

    # ---------- événements du moteur
    def _poll(self):
        try:
            while True:
                ev, val = self.events.get_nowait()
                if ev == "level":
                    self.level.set(min(1.0, val * 10))
                elif ev == "status":
                    self.status.configure(text=val)
                elif ev == "ready":
                    self.status.configure(text="● Prêt", text_color="#B2FF59")
                elif ev == "speaking":
                    self.status.configure(text="● Voix détectée…" if val else "● Prêt",
                                          text_color="#FF4081" if val else "#B2FF59")
                elif ev == "enroll_progress":
                    self.enroll_bar.set(val)
                elif ev == "enroll_done":
                    self._finish_enroll(val)
                elif ev == "result":
                    self._show(val)
        except queue.Empty:
            pass
        self.after(40, self._poll)

    def _finish_enroll(self, suggestion):
        try:
            self.engine.enroll_commit(self.pending_name)
            self._add_card(self.pending_name)
            self.status.configure(text=f"● Profil {self.pending_name} enregistré", text_color="#B2FF59")
        except Exception as e:
            self.engine.enroll_cancel()
            self.status.configure(text=f"Erreur : {e}", text_color="#FF4081")
        self._end_enroll()

    def _show(self, r):
        who, kind = r["who"], r["kind"]
        tag = self.cards[who]["tag"] if who in self.cards else "unk"
        name = who or "Inconnu"
        if kind == "parole":
            self._write(f"[{name}]  {r['text']}", tag)
        elif kind == "sans_mots":
            self._write(f"[{name}]  🗣 sons sans mots reconnus", tag)
        elif kind == "cri":
            self._write(f"[{name}]  📢 cri / son soutenu", "scream")
        else:
            self._write("🌫  bruit (pas de parole)", "noise")
        if who in self.cards:
            self._flash(who, {"parole": "parle", "cri": "crie"}.get(kind, "fait du bruit"))

    def destroy(self):
        self.engine.running = False
        super().destroy()


if __name__ == "__main__":
    App().mainloop()