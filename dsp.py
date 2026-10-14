"""Traitement du signal fait maison : MFCC + détection parole / bruit / cri."""
import numpy as np
from scipy.fft import dct, rfft, irfft

SR = 16000
N_FFT = 512
_FB = None


def _mel_filterbank(n_mels=26, n_fft=N_FFT, sr=SR):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


def _frames(x, flen, hop):
    if len(x) < flen:
        return np.zeros((0, flen))
    n = 1 + (len(x) - flen) // hop
    idx = np.arange(flen)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def to_float(x):
    x = np.asarray(x)
    return x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)


def mfcc(x, n_mfcc=20):
    """Retourne les coefficients MFCC (n_frames, n_mfcc) des trames de parole."""
    global _FB
    if _FB is None:
        _FB = _mel_filterbank()
    x = to_float(x)
    if len(x) < 400:
        return np.zeros((0, n_mfcc))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])  # pré-emphase
    fr = _frames(x, 400, 160) * np.hamming(400)  # 25 ms / pas 10 ms
    energy = np.log(np.sum(fr ** 2, axis=1) + 1e-10)
    power = np.abs(rfft(fr, N_FFT)) ** 2
    mel = np.log(power @ _FB.T + 1e-10)
    c = dct(mel, type=2, norm="ortho", axis=1)[:, 1:n_mfcc + 1]
    c = c[energy > np.percentile(energy, 30)]  # garde les trames les plus fortes
    return c - c.mean(axis=0)  # normalisation cepstrale (CMN)


def analyze(x):
    """Descripteurs acoustiques d'un segment audio."""
    x = to_float(x)
    dur = len(x) / SR
    # --- hauteur de voix par autocorrélation ---
    fr = _frames(x, 640, 320)
    if len(fr) == 0:
        return dict(dur=dur, voiced=0, pitch=0, flat=1, syl=0)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    active = rms > max(0.008, 0.2 * np.percentile(rms, 90))
    pitches = []
    for f in fr[active]:
        r = irfft(np.abs(rfft(f, 1280)) ** 2)[:320]
        if r[0] <= 0:
            continue
        r = r / r[0]
        lag = 20 + int(np.argmax(r[20:300]))  # 53 Hz .. 800 Hz
        if r[lag] > 0.4:
            pitches.append(SR / lag)
    n_act = max(int(active.sum()), 1)
    voiced = len(pitches) / n_act
    pitch = float(np.median(pitches)) if pitches else 0.0
    # --- platitude spectrale (bruit = proche de 1) ---
    sp = np.abs(rfft(_frames(x, 400, 400) * np.hamming(400), N_FFT)) ** 2 + 1e-10
    flat = float(np.median(np.exp(np.mean(np.log(sp), 1)) / np.mean(sp, 1)))
    # --- rythme syllabique (parole ~ 2-8 Hz) ---
    env = np.sqrt(np.mean(_frames(x, 160, 160) ** 2, axis=1))
    if len(env) > 5:
        env = np.convolve(env, np.ones(5) / 5, mode="same")
        th = 0.35 * env.max()
        up = np.sum((env[1:] > th) & (env[:-1] <= th))
        syl = up / max(dur, 0.1)
    else:
        syl = 0
    return dict(dur=dur, voiced=voiced, pitch=pitch, flat=flat, syl=float(syl))


def decide(info):
    """Étiquette acoustique : 'bruit', 'cri' ou 'parole?' (à confirmer par la transcription)."""
    if info["voiced"] < 0.15 or info["flat"] > 0.35:
        return "bruit"
    if info["pitch"] > 500:
        return "cri"
    if info["dur"] > 1.2 and info["syl"] < 0.8:
        return "cri"  # son soutenu sans rythme de syllabes
    return "parole?"
