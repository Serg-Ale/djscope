#!/usr/bin/env python3
"""djscope - painel de análise de áudio ao vivo no terminal (estilo MiniMeters + waveform de CDJ).

Capta o áudio que está saindo no sistema (monitor do sink padrão do PipeWire/Pulse) e mostra:
  * waveform rolante colorida por banda (graves=vermelho, médios=verde, agudos=azul) + beatgrid
  * espectro logarítmico com peak hold
  * stereo (goniômetro), correlação de fase
  * loudness M/S/I (LUFS, ITU BS.1770), picos L/R, PLR
  * BPM estimado e LED de batida
"""
import argparse, csv, json, math, os, re, select, shutil, subprocess, sys, termios, threading, time, tty
from collections import deque

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfilt

SR = 48000
HOP = 1600          # 33.3 ms -> uma coluna da waveform
EB = 400            # bloco do envelope de onsets (120 Hz)
ENV_RATE = SR / EB

RST = "\x1b[0m"
BG = np.array([12, 12, 18], dtype=np.float64)
RED = np.array([255, 70, 35], dtype=np.float64)
GRN = np.array([70, 235, 95], dtype=np.float64)
BLU = np.array([85, 150, 255], dtype=np.float64)
DIM = "\x1b[38;2;110;112;135m"
TXT = "\x1b[38;2;205;208;225m"


def fg(c): return f"\x1b[38;2;{int(c[0])};{int(c[1])};{int(c[2])}m"
def bgc(c): return f"\x1b[48;2;{int(c[0])};{int(c[1])};{int(c[2])}m"


def lufs(z): return -0.691 + 10 * math.log10(z) if z > 1e-12 else -70.0


def zone_color(f):
    if f < 0.72: return (60, 210, 110)
    if f < 0.90: return (235, 205, 60)
    return (250, 70, 60)


# ---------------------------------------------------------------- fontes de áudio
class Capture:
    def __init__(self, source):
        self.q, self.lock = deque(), threading.Lock()
        self.proc = subprocess.Popen(
            ["parec", "-d", source, "--format=float32le", f"--rate={SR}", "--channels=2", "--latency-msec=20"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        rem = b""
        while True:
            d = self.proc.stdout.read(3840)
            if not d: break
            d = rem + d
            n = len(d) // 8 * 8
            rem = d[n:]
            if n:
                arr = np.frombuffer(d[:n], dtype=np.float32).reshape(-1, 2).copy()
                with self.lock: self.q.append(arr)

    def get(self):
        with self.lock:
            items = list(self.q); self.q.clear()
        return np.concatenate(items) if items else np.zeros((0, 2), np.float32)

    def close(self):
        try: self.proc.terminate()
        except Exception: pass


class Demo:
    """Música sintética em Lá menor (8A), 128 bpm, com estrutura: acento no 1 do compasso e frases de 8 compassos."""
    def __init__(self, bpm=128, key_shift=0):
        self.n, self.t0, self.bpm = 0, time.time(), bpm
        self.shift = key_shift                      # semitons (transposição) p/ testar outras tonalidades
        self.rng = np.random.default_rng(1)

    def get(self):
        n = int((time.time() - self.t0) * SR) - self.n
        if n <= 0: return np.zeros((0, 2), np.float32)
        t = np.arange(self.n, self.n + n) / SR
        self.n += n
        return self.render(t)

    def render(self, t):
        n = len(t)
        beat = 60 / self.bpm
        bar_len, phrase_len = 4 * beat, 32 * beat
        k = 2 ** (self.shift / 12)
        ph = t % beat
        bar_i = (t // bar_len).astype(int)
        beat_in_bar = ((t // beat).astype(int)) % 4
        phr_i = (t // phrase_len).astype(int)
        phr_t = t % phrase_len
        accent = np.where(beat_in_bar == 0, 1.18, 1.0)                       # acento no 1 do compasso
        kick = np.sin(2 * np.pi * (45 * k * ph + 5 * (1 - np.exp(-30 * ph)))) * np.exp(-ph * 9) * accent
        ph2 = (t + beat / 2) % beat
        hat = np.diff(self.rng.standard_normal(n + 1))[:n] * np.exp(-ph2 * 70) * 0.35
        # baixo na tônica (A) ; pad com acorde de Lá menor (A C E) ; arpejo em escala menor natural
        bass = 0.25 * np.sin(2 * np.pi * 55 * k * t) * (1 - 0.85 * np.exp(-ph * 6))
        def chord(det):
            return 0.05 * sum(np.sin(2 * np.pi * f * k * det * t) for f in (220.0, 261.63, 329.63))
        padl, padr = chord(1.0), chord(1.003)
        bi = (t // beat).astype(int)
        phc = (t - bi * beat)
        step = (t // (beat / 4)).astype(int)
        notes = np.array([0, 3, 7, 10, 12, 10, 7, 3])[step % 8]                 # A C E G A G E C
        pa = (t % (beat / 4))
        arp = 0.10 * np.sin(2 * np.pi * 440 * k * 2 ** (notes / 12) * t) * np.exp(-pa * 14)
        arp = arp * (phr_i % 2 == 1)                                           # arpejo só nas frases ímpares
        clapL = np.diff(self.rng.standard_normal(n + 1))[:n] * np.exp(-phc * 28) * (bi % 2 == 1) * 0.16
        clapR = np.diff(self.rng.standard_normal(n + 1))[:n] * np.exp(-phc * 28) * (bi % 2 == 1) * 0.16
        airL = np.diff(self.rng.standard_normal(n + 1))[:n] * 0.004
        airR = np.diff(self.rng.standard_normal(n + 1))[:n] * 0.004
        lead = 0.06 * np.sin(2 * np.pi * 660 * k * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 5.5 * t))
        lead = lead * (phr_i % 4 >= 2)                                         # lead entra a cada 2 frases
        crash = np.diff(self.rng.standard_normal(n + 1))[:n] * np.exp(-phr_t * 3.0) * 0.30   # crash no início da frase
        pad_gain = np.where(phr_i % 2 == 0, 1.0, 0.45)
        hat = hat * np.where(phr_i % 2 == 0, 0.5, 1.0)
        l = 0.8 * kick + bass + 0.55 * padl * pad_gain + 0.95 * arp + 0.10 * hat + clapL + airL + 0.10 * lead + 0.5 * crash
        r = 0.8 * kick + bass + 0.55 * padr * pad_gain + 0.25 * arp + 0.60 * np.roll(hat, 3) + clapR + airR + 0.90 * lead + 0.5 * crash
        return (np.stack([l, r], 1) * 0.5).astype(np.float32)

    def close(self): pass


# ---------------------------------------------------------------- tonalidade (key) em Camelot
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
CAMELOT_MAJOR = {0: "8B", 7: "9B", 2: "10B", 9: "11B", 4: "12B", 11: "1B", 6: "2B", 1: "3B", 8: "4B", 3: "5B", 10: "6B", 5: "7B"}
CAMELOT_MINOR = {9: "8A", 4: "9A", 11: "10A", 6: "11A", 1: "12A", 8: "1A", 3: "2A", 10: "3A", 5: "4A", 0: "5A", 7: "6A", 2: "7A"}
_CH_WIN = np.hanning(8192)
_CH_NF = 32768
# Perfis de Temperley (2005): nos testes com música real foram bem mais estáveis que os de Krumhansl
TP_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TP_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
_PROFILES = np.array([np.roll(TP_MAJOR, t) for t in range(12)] + [np.roll(TP_MINOR, t) for t in range(12)])
_PROFILES = (_PROFILES - _PROFILES.mean(1, keepdims=True))
_PROFILES /= np.linalg.norm(_PROFILES, axis=1, keepdims=True)


def key_name(idx):
    """idx 0..11 = maior (tônica C..B), 12..23 = menor. Devolve (nome, camelot)."""
    if idx < 12: return f"{PC_NAMES[idx]} maj", CAMELOT_MAJOR[idx]
    return f"{PC_NAMES[idx - 12]} min", CAMELOT_MINOR[idx - 12]


def camelot_relation(a, b):
    """Relação harmônica entre dois códigos Camelot ('8A', '9A'...): '=', '±1', 'rel' (relativa), '±2' ou ''."""
    if not a or not b: return ""
    na, la, nb, lb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    d = min((na - nb) % 12, (nb - na) % 12)
    if la == lb: return "=" if d == 0 else "±1" if d == 1 else "±2" if d == 2 else ""
    return "rel" if d == 0 else ""


class KeyDetector:
    """Cromagrama a partir dos picos espectrais + correlação com perfis de Krumhansl (24 tonalidades)."""
    def __init__(self, decay=0.997):
        self.decay = decay                     # a 10 Hz: constante de tempo ~33 s
        self.reset()

    def reset(self):
        self.num = np.zeros(12)
        self.den = 0.0
        self.cur = None                      # tonalidade "estável" (com histerese)
        self.cand, self.cand_n = None, 0
        self.recent = deque(maxlen=20)       # últimos vencedores brutos (1 Hz) p/ medir estabilidade

    def soften(self, f=0.2):
        self.num *= f; self.den *= f
        self.cur, self.cand, self.cand_n = None, None, 0
        self.recent.clear()

    def update(self, ring):
        x = ring[-8192:].astype(np.float64).mean(axis=1) * _CH_WIN
        X = np.abs(np.fft.rfft(x, _CH_NF))
        df = SR / _CH_NF
        lo, hi = int(55 / df), int(2400 / df)
        seg = X[lo - 1:hi + 2]
        a, b, c = seg[:-2], seg[1:-1], seg[2:]
        mx = float(b.max())
        if mx <= 1e-6: return
        idx = np.nonzero((b > a) & (b >= c) & (b > 0.03 * mx))[0]      # picos até -30 dB do maior
        if not len(idx): return
        la, lb, lc = np.log(a[idx] + 1e-12), np.log(b[idx] + 1e-12), np.log(c[idx] + 1e-12)
        den = la - 2 * lb + lc
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (la - lc) / np.where(np.abs(den) > 1e-12, den, 1), 0.0)
        f = (lo + idx + np.clip(delta, -0.5, 0.5)) * df
        w = np.sqrt(b[idx]) / (1 + (f / 1500.0) ** 2)
        midi = 69 + 12 * np.log2(f / 440.0)
        fl = np.floor(midi)
        fr = midi - fl
        chroma = np.bincount((fl % 12).astype(int), w * (1 - fr), 12) + np.bincount(((fl + 1) % 12).astype(int), w * fr, 12)
        chroma /= chroma.sum() + 1e-12
        self.num = self.num * self.decay + chroma
        self.den = self.den * self.decay + 1.0

    def chroma(self):
        return self.num / self.den if self.den > 1e-9 else np.zeros(12)

    def _corr(self):
        ch = self.chroma()
        if ch.sum() < 1e-9 or self.den < 5: return None
        v = ch - ch.mean()
        n = np.linalg.norm(v)
        return None if n < 1e-9 else _PROFILES @ (v / n)

    def tick(self):
        """Chamar ~1x/s: atualiza a tonalidade estável. Só troca quando a nova candidata vence com folga clara
        ou se mantém na frente por 4 medições seguidas (evita oscilar entre duas tonalidades parecidas)."""
        corr = self._corr()
        if corr is None: return
        best = int(np.argmax(corr))
        self.recent.append(best)
        if self.cur is None:
            self.cur = best; return
        if best == self.cur:
            self.cand, self.cand_n = None, 0
            return
        if corr[best] - corr[self.cur] > 0.05:
            self.cur, self.cand, self.cand_n = best, None, 0
            return
        self.cand_n = self.cand_n + 1 if self.cand == best else 1
        self.cand = best
        if self.cand_n >= 4:
            self.cur, self.cand, self.cand_n = best, None, 0

    def estimate(self):
        """(idx, correlação, margem p/ o melhor concorrente, idx do concorrente, estabilidade 0..1) ou None."""
        corr = self._corr()
        if corr is None: return None
        if self.cur is None: self.cur = int(np.argmax(corr))
        cur = self.cur
        other = np.argsort(corr)[::-1]
        other = [int(i) for i in other if i != cur][0]
        stab = (sum(1 for r in self.recent if r == cur) / len(self.recent)) if len(self.recent) >= 5 else 0.5
        return cur, float(corr[cur]), float(corr[cur] - corr[other]), other, float(stab)


# ---------------------------------------------------------------- compasso e frase
_B12 = np.geomspace(40, 16000, 13)
_B12_FREQ = np.fft.rfftfreq(2048, 1 / SR)
_B12_IDX = np.searchsorted(_B12_FREQ, _B12)
_B12_IDX[1:] = np.maximum(_B12_IDX[1:], _B12_IDX[:-1] + 1)
_HOPWIN = np.hanning(HOP)


def spec12(mono):
    """Energia (log) em 12 bandas de 40 Hz a 16 kHz no hop atual."""
    p = np.abs(np.fft.rfft(mono * _HOPWIN, 2048)) ** 2
    cs = np.concatenate(([0.0], np.cumsum(p)))
    e = (cs[_B12_IDX[1:]] - cs[_B12_IDX[:-1]]) / (_B12_IDX[1:] - _B12_IDX[:-1])
    return np.log10(e * 1e-4 + 1e-12)


class PhraseTracker:
    """Conta batidas, descobre qual é o '1' do compasso e onde começam as frases de 8 compassos (heurística por novidade espectral)."""
    def __init__(self): self.reset()

    def reset(self):
        self.last_beat = None
        self.beat_idx = 0
        self.vecs = deque(maxlen=64)
        self.accv, self.accl, self.n = np.zeros(12), 0.0, 0
        self.bar_scores = np.zeros(4)
        self.phr_scores = np.zeros(8)
        self.bar_phase = 0
        self.phrase_phase = 0
        self.bar_conf = 0.0
        self.phr_conf = 0.0
        self.beats = 0
        self.bars = 0

    def feed_hop(self, env_now, P, a, vec, low):
        self.accv += vec; self.accl += low; self.n += 1
        if P is None or a is None: return
        if self.last_beat is None:
            self.last_beat = a + math.floor((env_now - a) / P) * P
            return
        while env_now >= self.last_beat + P:
            self.last_beat += P
            self.beat_idx += 1
            self._beat()

    def resync(self, a, P):
        """Corrige a fase da contagem de batidas em direção à grade (suave, sem pular batidas)."""
        if self.last_beat is None or a is None or P is None: return
        err = ((self.last_beat - a + P / 2) % P) - P / 2
        self.last_beat -= 0.3 * err

    def _beat(self):
        # o vetor acumulado descreve a batida que ACABOU (índice beat_idx-1), não a que começa agora
        v = self.accv / max(1, self.n)
        self.accv[:] = 0; self.accl = 0.0; self.n = 0
        self.vecs.append(v)
        self.beats += 1
        b = self.beat_idx - 1
        if len(self.vecs) >= 2:
            nov = float(np.linalg.norm(v - self.vecs[-2]))                  # "chegou algo novo nessa batida?"
            self.bar_scores *= 0.985
            self.bar_scores[b % 4] += nov
            if self.beats >= 12:
                best = int(np.argmax(self.bar_scores))
                if best != self.bar_phase and self.bar_scores[best] > 1.25 * self.bar_scores[self.bar_phase]:
                    self.bar_phase = best
                    self.phr_scores[:] = 0                                   # as frases dependiam da fase do compasso
                self.bar_conf = self._margin(self.bar_scores, self.bar_phase)
        # fim de um compasso (última batida dele) -> quanto ele difere do compasso anterior?
        if (b - self.bar_phase) % 4 == 3 and len(self.vecs) >= 8:
            arr = np.array(self.vecs)
            nb = float(np.linalg.norm(arr[-4:].mean(0) - arr[-8:-4].mean(0)))
            bar = (b - self.bar_phase) // 4
            self.bars += 1
            self.phr_scores *= 0.97
            self.phr_scores[bar % 8] += nb
            if self.bars >= 12 and self.phr_scores.sum() > 0:
                self.phrase_phase = int(np.argmax(self.phr_scores))
                self.phr_conf = self._margin(self.phr_scores, self.phrase_phase)

    @staticmethod
    def _margin(scores, best):
        """0..1: quão à frente o melhor candidato está do segundo colocado."""
        o = np.sort(scores)[::-1]
        return float((o[0] - o[1]) / o[0]) if o[0] > 0 else 0.0

    def pos(self):
        """(batida no compasso 0..3, compasso na frase 0..7, conf_compasso, conf_frase) ou None."""
        if self.last_beat is None or self.beats < 2: return None
        bib = (self.beat_idx - self.bar_phase) % 4
        bar = (self.beat_idx - self.bar_phase) // 4
        return bib, (bar - self.phrase_phase) % 8, self.bar_conf, self.phr_conf


# ---------------------------------------------------------------- faixa em reprodução (MPRIS) e registro do set
def _busctl(*args):
    r = subprocess.run(["busctl", "--user", "--json=short", *args], capture_output=True, text=True, timeout=2)
    if r.returncode != 0: raise RuntimeError(r.stderr.strip())
    return json.loads(r.stdout)


class NowPlaying(threading.Thread):
    """Lê título/artista do player via MPRIS (Spotify e outros que exponham org.mpris.MediaPlayer2) a cada ~1,5 s."""
    def __init__(self):
        super().__init__(daemon=True)
        self.info, self.alive = None, True
        self.start()

    def run(self):
        while self.alive:
            try: self.info = self.poll()
            except Exception: self.info = None
            for _ in range(15):
                if not self.alive: return
                time.sleep(0.1)

    @staticmethod
    def poll():
        names = [n["name"] for n in _busctl("list") if n["name"].startswith("org.mpris.MediaPlayer2.")]
        best = None
        for n in names:
            try:
                st = _busctl("get-property", n, "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player", "PlaybackStatus")["data"]
                md = _busctl("get-property", n, "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player", "Metadata")["data"]
            except Exception:
                continue
            g = lambda k, d=None: md.get(k, {}).get("data", d)
            info = dict(player=n.split(".")[-1], status=st, title=g("xesam:title"), artist=", ".join(g("xesam:artist", []) or []),
                        album=g("xesam:album"), length=(g("mpris:length", 0) or 0) / 1e6)
            if st == "Playing": return info
            best = best or info
        return best

    def close(self): self.alive = False


class TrackLog:
    """Detecta o começo/fim de cada faixa (título MPRIS, silêncio, salto de BPM ou marcação manual) e registra BPM, key, LUFS e true peak."""
    MIN_DUR = 8.0

    def __init__(self, eng, outdir=None):
        self.eng = eng
        self.tracks, self.cur = [], None
        self.t0 = time.time()
        self.outdir = outdir or os.path.expanduser("~/.local/share/djscope/sets")
        self.path = os.path.join(self.outdir, time.strftime("set-%Y%m%d-%H%M.csv"))
        self.src_title = None
        self.bpm_all, self.bpm_recent = [], deque(maxlen=12)
        self.last_sample, self.jump_since = 0.0, None

    # ---- ciclo de vida de uma faixa
    def _start(self, info, now, reason):
        title = info.get("title") if info and info.get("status") == "Playing" else None
        self.src_title = title
        self.cur = dict(start=now, title=title, artist=(info or {}).get("artist") if title else None, reason=reason)
        self.bpm_all.clear(); self.bpm_recent.clear(); self.jump_since = None

    def _finalize(self, now, reason_end):
        c = self.cur
        dur = now - c["start"]
        if dur < self.MIN_DUR: return None
        e = self.eng
        ki = e.key_info()
        bpm = float(np.median(self.bpm_all)) if len(self.bpm_all) >= 3 else (e.bpm if e.bpm else None)
        rec = dict(n=len(self.tracks) + 1, start_off=c["start"] - self.t0, dur=dur, title=c["title"], artist=c["artist"],
                   bpm=bpm, key=(ki["name"] if ki else None), camelot=(ki["camelot"] if ki else None),
                   key_level=(ki["level"] if ki else None), lufs_i=e.lufs_i, tp=e.tp_max if e.tp_max > -80 else None,
                   clips=e.clips, reason=c["reason"], wall=time.strftime("%H:%M:%S", time.localtime(c["start"])))
        self.tracks.append(rec)
        self.save()
        return rec

    def boundary(self, reason, info, now):
        if self.cur is not None: self._finalize(now, reason)
        self.eng.new_track()
        self._start(info, now, reason)

    def mark(self, info, now=None):
        """Marcação manual de 'nova faixa'."""
        self.boundary("manual", info, now if now is not None else time.time())

    # ---- chamado ~2x/s
    def tick(self, info, now=None):
        now = now if now is not None else time.time()
        e = self.eng
        playing_title = info.get("title") if info and info.get("status") == "Playing" else None
        if self.cur is None:
            if e.rms_db > -50: self._start(info, now, "início")
            return
        c = self.cur
        if playing_title and self.src_title and playing_title != self.src_title:
            self.boundary("título", info, now); return
        if c["title"] is None and playing_title:                       # o player só respondeu depois do áudio começar
            c["title"], c["artist"], self.src_title = playing_title, info.get("artist"), playing_title
        if e.resumed:
            e.resumed = False
            if now - c["start"] > self.MIN_DUR: self.boundary("silêncio", info, now); return
        # amostra de BPM (1 Hz) e salto sustentado de andamento
        if now - self.last_sample >= 1.0:
            self.last_sample = now
            if e.bpm and e.conf > 0.2:
                self.bpm_all.append(e.bpm); self.bpm_recent.append(e.bpm)
            if len(self.bpm_all) >= 40 and len(self.bpm_recent) >= 10:
                tr, rc = float(np.median(self.bpm_all)), float(np.median(self.bpm_recent))
                ratio = rc / tr
                halfdouble = abs(ratio - 2) < 0.08 or abs(ratio - 0.5) < 0.04
                if abs(ratio - 1) > 0.06 and not halfdouble:
                    self.jump_since = self.jump_since or now
                    if now - self.jump_since >= 15: self.boundary("BPM", info, now)
                else:
                    self.jump_since = None

    def snapshot(self, now=None):
        """Estado da faixa atual (p/ a GUI)."""
        if self.cur is None: return None
        now = now if now is not None else time.time()
        c = self.cur
        return dict(title=c["title"], artist=c["artist"], elapsed=now - c["start"], n=len(self.tracks) + 1,
                    bpm=(float(np.median(self.bpm_all)) if len(self.bpm_all) >= 3 else None), reason=c["reason"])

    def save(self):
        try:
            os.makedirs(self.outdir, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["n", "inicio", "offset_s", "duracao_s", "artista", "titulo", "bpm", "tonalidade", "camelot",
                            "confianca_key", "lufs_i", "true_peak_dbtp", "clips", "motivo"])
                for t in self.tracks:
                    w.writerow([t["n"], t["wall"], round(t["start_off"]), round(t["dur"]), t["artist"] or "", t["title"] or "",
                                f"{t['bpm']:.1f}" if t["bpm"] else "", t["key"] or "", t["camelot"] or "", t["key_level"] or "",
                                f"{t['lufs_i']:.1f}" if t["lufs_i"] is not None else "", f"{t['tp']:.1f}" if t["tp"] is not None else "",
                                t["clips"], t["reason"]])
            os.replace(tmp, self.path)
        except Exception:
            pass

    def finish(self, now=None):
        """Fecha a faixa em curso (ao sair do app) e salva."""
        if self.cur is not None:
            self._finalize(now if now is not None else time.time(), "fim")
            self.cur = None


class Engine:
    K1B = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    K1A = [1.0, -1.69065929318241, 0.73248077421585]
    K2B = [1.0, -2.0, 1.0]
    K2A = [1.0, -1.99004745483398, 0.99007225036621]

    def __init__(self):
        self.pend = np.zeros((0, 2), np.float32)
        self.ring = np.zeros((8192, 2), np.float32)
        self.zk1, self.zk2 = np.zeros((2, 2)), np.zeros((2, 2))
        self.sos_lo = butter(4, 200, "low", fs=SR, output="sos")
        self.sos_hi = butter(4, 2500, "high", fs=SR, output="sos")
        self.sos_mid = butter(2, [200, 2500], "band", fs=SR, output="sos")
        self.zmid = np.zeros((self.sos_mid.shape[0], 2))
        self.sos_k = butter(4, 150, "low", fs=SR, output="sos")
        self.zlo, self.zhi, self.zkick = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
        self.hist = deque(maxlen=4000)  # >= 120 s de histórico p/ o overview
        self.fine = deque(maxlen=4800)  # 4 colunas por hop (8,3 ms cada) = 40 s, p/ a waveform da GUI
        self.hop_count = 0
        self._fs = np.zeros(4)
        self._pk = 0.0
        self._lp_prev = np.zeros(2)
        self.z_acc, self.hop_mod = 0.0, 0
        self.blocks, self.win400 = deque(maxlen=60), []
        self.o = deque(maxlen=1500)
        self.ol = deque(maxlen=1500)          # só o onset do grave (kick): mais fiel à batida que o onset de banda larga
        self.env_count = 0
        self.pv_l = self.pv_f = 0.0
        self.pulse = 0.0
        self.reset_meters()
        self.bpm, self.conf = None, 0.0
        self.bpm_hist = deque(maxlen=7)
        self.grid_P, self.grid_a = None, None
        self.corr = 0.0
        self.win = np.hanning(8192)
        self._spec_cache = {}
        self.key = KeyDetector()
        self.phr = PhraseTracker()
        self._tail = np.zeros((64, 2))
        self.rms_db = -90.0
        self.quiet_hops, self.silence_seen, self.resumed = 0, False, False

    def reset_meters(self):
        self.pk = [-90.0, -90.0]
        self.hold = [-90.0, -90.0]
        self.hold_t = [0.0, 0.0]
        self.max_peak = -90.0
        self.tp, self.tp_max, self.clips, self._over = -90.0, -90.0, 0, False
        self.lufs_m = self.lufs_s = -70.0
        self.lufs_i = None
        self.win400 = []

    def new_track(self):
        """Começo de faixa nova: zera o que é por-faixa (LUFS integrado, picos, key, contagem de frase)."""
        self.reset_meters()
        self.key.soften(0.15)
        self.phr.reset()
        self.bpm_hist.clear()

    def feed(self, chunk):
        self.ring = np.concatenate((self.ring, chunk))[-8192:]
        self.pend = np.concatenate((self.pend, chunk))
        while len(self.pend) >= HOP:
            h, self.pend = self.pend[:HOP], self.pend[HOP:]
            self._hop(h)

    def _hop(self, h):
        now = time.time()
        h64 = h.astype(np.float64)
        mono = h64.mean(axis=1)
        for i in (0, 1):
            db = 20 * math.log10(max(float(np.abs(h64[:, i]).max()), 1e-5))
            self.pk[i] = max(db, self.pk[i] - 0.7)
            if db >= self.hold[i] or now > self.hold_t[i]:
                self.hold[i], self.hold_t[i] = db, now + 1.6
            self.max_peak = max(self.max_peak, db)
        # true peak (sobreamostragem 4x, com cauda do hop anterior p/ não criar artefato na borda)
        pad = np.concatenate([self._tail, h64])
        up = resample_poly(pad, 4, 1, axis=0)[4 * len(self._tail):]
        self._tail = h64[-64:]
        tpv = 20 * math.log10(max(float(np.abs(up).max()), 1e-5))
        self.tp = max(tpv, self.tp - 0.7)
        self.tp_max = max(self.tp_max, tpv)
        if tpv > 0.0:
            if not self._over: self.clips += 1
            self._over = True
        else:
            self._over = False
        # silêncio / retomada (p/ detectar troca de faixa)
        self.rms_db = 20 * math.log10(max(float(np.sqrt(np.mean(mono * mono))), 1e-6))
        if self.rms_db < -62:
            self.quiet_hops += 1
            if self.quiet_hops * HOP / SR >= 1.5: self.silence_seen = True
        else:
            if self.silence_seen and self.rms_db > -52: self.resumed = True
            if self.rms_db > -52: self.silence_seen = False
            self.quiet_hops = 0
        # bandas p/ cor da waveform
        lo, self.zlo = sosfilt(self.sos_lo, mono, zi=self.zlo)
        hi, self.zhi = sosfilt(self.sos_hi, mono, zi=self.zhi)
        mid, self.zmid = sosfilt(self.sos_mid, mono, zi=self.zmid)
        rb = lambda x: math.sqrt(float(np.mean(x * x)))
        self.hist.append((float(np.abs(mono).max()), rb(mono), rb(lo), rb(mid), rb(hi)))
        sq = lambda x: np.sqrt((x.reshape(4, -1) ** 2).mean(1))
        pk4, r4, l4, m4, h4 = np.abs(mono).reshape(4, -1).max(1), sq(mono), sq(lo), sq(mid), sq(hi)
        for i in range(4):
            cur = np.array((r4[i], l4[i], m4[i], h4[i]))
            self._fs = np.maximum(cur, self._fs * 0.80)
            self._pk = max(float(pk4[i]), self._pk * 0.86)
            self.fine.append((self._pk, *self._fs))
        self.hop_count += 1
        # loudness (K-weighting)
        k, self.zk1 = lfilter(self.K1B, self.K1A, h64, axis=0, zi=self.zk1)
        k, self.zk2 = lfilter(self.K2B, self.K2A, k, axis=0, zi=self.zk2)
        self.z_acc += float((k * k).sum())
        self.hop_mod += 1
        if self.hop_mod == 3:
            self.blocks.append(self.z_acc / (3 * HOP))
            self.z_acc, self.hop_mod = 0.0, 0
            if len(self.blocks) >= 4:
                b = list(self.blocks)
                self.lufs_m = lufs(sum(b[-4:]) / 4)
                self.lufs_s = lufs(sum(b[-30:]) / len(b[-30:]))
                self.win400.append(sum(b[-4:]) / 4)
                if self.hop_count % 30 == 0: self._integrated()
        # onsets p/ BPM
        kl, self.zkick = sosfilt(self.sos_k, mono, zi=self.zkick)
        # envelope do grave: potência numa janela de ~25 ms (3 blocos). A média de |x| em blocos curtos ondula ±15% com
        # baixos sustentados e isso vira "ataques" falsos; a janela de 3 blocos elimina a ondulação.
        pw = (kl * kl).reshape(4, EB).mean(1)
        seq = np.concatenate([self._lp_prev, pw])
        el = np.sqrt((seq[2:] + seq[1:-1] + seq[:-2]) / 3.0)
        self._lp_prev = seq[-2:]
        ef = np.abs(mono).reshape(4, EB).mean(1)
        new, newl = [], []
        for a, b in zip(el, ef):
            vl, vf = math.log1p(a * 40), math.log1p(b * 40)
            dl = max(0.0, vl - self.pv_l)
            new.append(dl + 0.5 * max(0.0, vf - self.pv_f))
            newl.append(dl)
            self.pv_l, self.pv_f = vl, vf
        self.o.extend(new)
        self.ol.extend(newl)
        self.env_count += 4
        self.phr.feed_hop(self.env_count, self.grid_P, self.grid_a, spec12(mono), float(el.sum()))
        if self.hop_count % 3 == 0 and self.rms_db > -55: self.key.update(self.ring)
        if self.hop_count % 30 == 0 and self.rms_db > -55: self.key.tick()
        base = float(np.mean(list(self.o)[-240:])) if len(self.o) > 20 else 1.0
        if max(new) > 2.2 * base + 0.04: self.pulse = 1.0
        self.pulse *= 0.84
        # correlação
        x = self.ring[-4096:].astype(np.float64)
        d = math.sqrt(float((x[:, 0] ** 2).sum() * (x[:, 1] ** 2).sum()))
        c = float((x[:, 0] * x[:, 1]).sum()) / d if d > 1e-9 else 0.0
        self.corr += 0.15 * (c - self.corr)

    def _integrated(self):
        if not self.win400: return
        a = np.array(self.win400)
        l = -0.691 + 10 * np.log10(np.maximum(a, 1e-12))
        g1 = a[l > -70]
        if not len(g1): return
        rel = lufs(float(g1.mean())) - 10
        g2 = a[(l > -70) & (l > rel)]
        if len(g2): self.lufs_i = lufs(float(g2.mean()))

    def estimate(self):
        if len(self.o) < 480: return
        o = np.array(self.o)[-960:]
        x = o - o.mean()
        n = len(x)
        r = np.correlate(x, x, "full")[n - 1:]
        if r[0] <= 1e-9: return
        r = r / r[0]
        ra = r * n / (n - np.arange(n))
        ax = np.arange(n)
        lags = np.arange(40.0, 104.0, 0.1)
        score = sum(np.interp(m * lags, ax, ra) for m in (1, 2, 3, 4)) / 4
        bpms = ENV_RATE * 60 / lags
        w = np.exp(-0.5 * (np.log2(bpms / 128) / 0.6) ** 2)
        i = int(np.argmax(score * w))
        lag = float(lags[i])
        ra = score
        self.conf = float(ra[i])
        if self.conf > 0.12:
            # banda larga refina melhor que o grave sozinho (testado): o grave sustentado cria ataques espúrios
            self.bpm_hist.append(self._refine(o, ENV_RATE * 60 / lag))
            self.bpm = float(np.median(self.bpm_hist))
            self._grid(o)
        elif self.conf < 0.05:
            self.bpm_hist.clear(); self.bpm = None; self.grid_P = None

    def _refine(self, sigs, bpm):
        """Refina o BPM por casamento de pulsos: testa tempos vizinhos (±8%) e fica com o que alinha melhor a grade
        de batidas aos ataques ao longo de toda a janela. `sigs`: sinais de ataque (banda larga e/ou grave), somados."""
        if isinstance(sigs, np.ndarray): sigs = [sigs]
        P0 = ENV_RATE * 60 / bpm
        cands = P0 * np.linspace(0.92, 1.08, 161)
        total = None
        for o in sigs:
            n = len(o)
            # limiar adaptativo: tira a média local (~0,1 s) -> subidas lentas (baixo, pads) somem, ataques bruscos ficam
            loc = np.convolve(o, np.ones(25) / 25.0, "same")
            sharp = np.maximum(0.0, o - 1.5 * loc)
            if sharp.sum() <= 1e-9: continue
            sm = np.convolve(sharp, np.array([1, 2, 3, 2, 1]) / 9.0, "same")
            ph = np.arange(0.0, math.ceil(cands.max()) + 1)
            K = int(n / cands.min())
            pos = ph[None, :, None] + np.arange(K)[None, None, :] * cands[:, None, None]
            valid = pos < n - 1
            vals = np.where(valid, sm[np.clip(np.rint(pos).astype(int), 0, n - 1)], 0.0)
            score = vals.sum(2) / np.maximum(valid.sum(2), 1)
            score = np.where(ph[None, :] < cands[:, None], score, 0.0)
            sc = score.max(1)
            sc = sc / (sc.mean() + 1e-12)                                    # normaliza p/ somar sinais de escalas diferentes
            total = sc if total is None else total + sc
        if total is None: return bpm
        sc = total
        i = int(np.argmax(sc))
        if 0 < i < len(sc) - 1:                                            # interpolação parabólica entre candidatos
            y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
            den = y0 - 2 * y1 + y2
            i_f = i + 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else float(i)
        else:
            i_f = float(i)
        P = P0 * (0.92 + 0.16 * i_f / 160.0)
        return ENV_RATE * 60 / P

    def _grid(self, o):
        P = ENV_RATE * 60 / self.bpm
        sm = np.convolve(o, [0.25, 0.5, 0.25], "same")
        e0 = self.env_count - len(o)
        best, bs = 0, -1.0
        for ph in range(int(P) + 1):
            idx = (ph + np.arange(0, len(o) / P) * P).round().astype(int)
            idx = idx[idx < len(o)]
            s = float(sm[idx].mean()) if len(idx) else 0
            if s > bs: bs, best = s, ph
        na = e0 + best - 2          # compensa a latência do detector (~17 ms)
        if self.grid_a is None or self.grid_P is None:
            self.grid_a = float(na)
        else:
            d = ((na - self.grid_a + P / 2) % P) - P / 2
            if abs(d) > 2.5: self.grid_a += d
        self.grid_P = P
        self.phr.resync(self.grid_a, self.grid_P)

    def beat_lattice(self):
        """(posição da última batida em unidades de env, período, índice absoluto) p/ desenhar a grade; None sem tempo."""
        if self.phr.last_beat is None or not self.grid_P: return None
        return self.phr.last_beat, self.grid_P, self.phr.beat_idx

    def key_info(self):
        """dict(name, camelot, conf, margin, chroma, idx) ou None."""
        est = self.key.estimate()
        if est is None: return None
        idx, corr, margin, idx2, stab = est
        name, cam = key_name(idx)
        name2, cam2 = key_name(idx2)
        # a confiança precisa valer também para a estabilidade: se a decisão ficou trocando, não é "alta"
        level = "alta" if (margin >= 0.08 and stab >= 0.85) else "média" if (margin >= 0.04 and stab >= 0.6) else "baixa"
        return dict(idx=idx, name=name, camelot=cam, conf=corr, margin=margin, stab=stab, level=level,
                    alt_name=name2, alt_camelot=cam2, chroma=self.key.chroma())

    def beat_pos(self):
        """posição fracionária (em batidas) em relação à âncora do grid; None se sem grid"""
        if not self.grid_P or self.grid_a is None: return None
        return (self.env_count - self.grid_a) / self.grid_P

    def spectrum(self, nb):
        c = self._spec_cache.get(nb)
        if c is None:
            freqs = np.fft.rfftfreq(8192, 1 / SR)
            edges = np.geomspace(30, 18000, nb + 1)
            lo = np.searchsorted(freqs, edges[:-1])
            hi = np.searchsorted(freqs, edges[1:])
            cen = np.sqrt(edges[:-1] * edges[1:])
            c = self._spec_cache[nb] = (freqs, lo, hi, cen)
        freqs, lo, hi, cen = c
        x = self.ring[-8192:].mean(axis=1) * self.win
        db = 20 * np.log10(np.abs(np.fft.rfft(x)) / (8192 / 4) + 1e-9)
        interp = np.interp(cen, freqs, db)
        out = np.array([db[a:b].max() if b > a else interp[i] for i, (a, b) in enumerate(zip(lo, hi))])
        out = out + 4.0 * np.log2(cen / 1000)
        return np.clip((out + 85) / 65, 0, 1)


# ---------------------------------------------------------------- tema
def C3(*a): return np.array(a, dtype=np.float64)

BG0 = C3(7, 8, 13)          # fundo geral
PANEL = C3(13, 15, 23)      # fundo dos painéis
TRACK = C3(25, 28, 42)      # trilho de medidores
BORDER = C3(42, 48, 72)
TTL = C3(150, 160, 200)
DIMC = C3(88, 95, 125)
TEXTC = C3(218, 222, 238)
ACC = C3(255, 184, 64)
ACC2 = C3(100, 200, 255)

WAVE_THEMES = {
    "3-band": dict(low=C3(35, 105, 255), mid=C3(255, 150, 35), high=C3(238, 242, 255)),
    "rgb": dict(low=C3(255, 55, 45), mid=C3(70, 235, 95), high=C3(80, 150, 255)),
}
SPEC_STOPS = [(0.0, (255, 60, 95)), (0.28, (255, 145, 50)), (0.52, (250, 225, 85)),
              (0.74, (70, 225, 150)), (1.0, (70, 150, 255))]
METER_STOPS = [(0.0, (45, 205, 125)), (0.6, (150, 225, 75)), (0.78, (255, 195, 50)), (1.0, (255, 70, 70))]


def grad(t, stops):
    t = np.asarray(t, dtype=np.float64)
    pos = [p for p, _ in stops]
    cols = np.array([c for _, c in stops], dtype=np.float64)
    return np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], -1)


def mix(a, b, t): return a * (1 - t) + b * t


# ---------------------------------------------------------------- canvas
class Canvas:
    def __init__(self, R, C):
        self.R, self.C = R, C
        self.ch = np.full((R, C), " ", dtype="<U1")
        self.fg = np.empty((R, C, 3), dtype=np.float64); self.fg[:] = TEXTC
        self.bg = np.empty((R, C, 3), dtype=np.float64); self.bg[:] = BG0

    def rect(self, r, c, h, w, bg):
        self.bg[r:r + h, c:c + w] = bg
        self.ch[r:r + h, c:c + w] = " "

    def text(self, r, c, t, fg, bg=None):
        if not (0 <= r < self.R): return
        for i, chh in enumerate(t):
            x = c + i
            if 0 <= x < self.C:
                self.ch[r, x] = chh
                self.fg[r, x] = fg
                if bg is not None: self.bg[r, x] = bg

    def box(self, r, c, h, w, title="", right="", accent=None):
        self.rect(r, c, h, w, PANEL)
        b = BORDER
        self.ch[r, c + 1:c + w - 1] = "─"; self.ch[r + h - 1, c + 1:c + w - 1] = "─"
        self.ch[r + 1:r + h - 1, c] = "│"; self.ch[r + 1:r + h - 1, c + w - 1] = "│"
        self.ch[r, c], self.ch[r, c + w - 1] = "╭", "╮"
        self.ch[r + h - 1, c], self.ch[r + h - 1, c + w - 1] = "╰", "╯"
        self.fg[r, c:c + w] = b; self.fg[r + h - 1, c:c + w] = b
        self.fg[r:r + h, c] = b; self.fg[r:r + h, c + w - 1] = b
        if title:
            self.text(r, c + 2, "◆", accent if accent is not None else ACC2)
            self.text(r, c + 4, title, TTL)
        if right:
            self.text(r, c + w - 2 - len(right), right, DIMC)

    def emit(self):
        """Gera as linhas ANSI. A detecção de mudança de cor é vetorizada; o Python só
        percorre os pontos onde fg/bg realmente mudam."""
        R, C = self.R, self.C
        fg8 = np.clip(self.fg + 0.5, 0, 255).astype(np.int16)
        bg8 = np.clip(self.bg + 0.5, 0, 255).astype(np.int16)
        nonsp = self.ch != " "
        cols = np.arange(C)[None, :]
        idx = np.maximum.accumulate(np.where(nonsp, cols, -1), axis=1)       # última célula não-vazia
        fg_eff = np.take_along_axis(fg8, np.clip(idx, 0, None)[..., None].repeat(3, -1), axis=1)
        prev_fg = np.concatenate([np.full((R, 1, 3), -1, np.int16), fg_eff[:, :-1]], axis=1)
        prev_bg = np.concatenate([np.full((R, 1, 3), -1, np.int16), bg8[:, :-1]], axis=1)
        need_fg = nonsp & (fg8 != prev_fg).any(-1)
        need_bg = (bg8 != prev_bg).any(-1)
        chl = self.ch.tolist()
        fgl, bgl = fg8.tolist(), bg8.tolist()
        nfl, nbl = need_fg.tolist(), need_bg.tolist()
        rows = []
        for r in range(R):
            cr, fr, br, nf, nb = chl[r], fgl[r], bgl[r], nfl[r], nbl[r]
            pts = [c for c in range(C) if nf[c] or nb[c]]
            out, last = [], 0
            for c in pts:
                if c > last: out.append("".join(cr[last:c]))
                f, b = fr[c], br[c]
                if nf[c] and nb[c]:
                    out.append(f"\x1b[38;2;{f[0]};{f[1]};{f[2]};48;2;{b[0]};{b[1]};{b[2]}m")
                elif nb[c]:
                    out.append(f"\x1b[48;2;{b[0]};{b[1]};{b[2]}m")
                else:
                    out.append(f"\x1b[38;2;{f[0]};{f[1]};{f[2]}m")
                last = c
            out.append("".join(cr[last:]))
            rows.append("".join(out))
        return rows


# ---------------------------------------------------------------- dígitos grandes (3 linhas)
BIG = {
    "0": ("┏━┓", "┃ ┃", "┗━┛"), "1": (" ╻ ", " ┃ ", " ╹ "), "2": ("╺━┓", "┏━┛", "┗━╸"),
    "3": ("╺━┓", " ━┫", "╺━┛"), "4": ("╻ ╻", "┗━┫", "  ╹"), "5": ("┏━╸", "┗━┓", "╺━┛"),
    "6": ("┏━╸", "┣━┓", "┗━┛"), "7": ("╺━┓", "  ┃", "  ╹"), "8": ("┏━┓", "┣━┫", "┗━┛"),
    "9": ("┏━┓", "┗━┫", "╺━┛"), "-": ("   ", "╺━╸", "   "), ".": (" ", " ", "▪"), " ": ("  ", "  ", "  "),
}


def big_width(s): return sum(len(BIG[ch][0]) + 1 for ch in s) - 1


def big_text(cv, r, c, s, fg):
    x = c
    for ch in s:
        g = BIG.get(ch, BIG[" "])
        for i in range(3): cv.text(r + i, x, g[i], fg)
        x += len(g[0]) + 1


# ---------------------------------------------------------------- waveform
def _blend(img, cov, color, alpha=1.0):
    a = (cov * alpha)[..., None]
    return img * (1 - a) + color * a


def draw_wave(cv, eng, r0, c0, h, w, z, theme, grid=False, ov_focus=None, agg="mean"):
    need = w * z
    data = list(eng.hist)[-need:]
    arr = np.zeros((need, 5))
    if data: arr[need - len(data):] = np.array(data)
    g = arr.reshape(w, z, 5)
    peak = g[:, :, 0].max(axis=1)
    red = (lambda a: a.max(axis=1)) if agg == "max" else (lambda a: a.mean(axis=1))
    rms = red(g[:, :, 1])
    bands = red(g[:, :, 2:])
    P = 2 * h
    half = P / 2
    d = np.abs(np.arange(P) + 0.5 - half)[:, None]
    valid = (peak > 1e-4)[None, :]
    cov = lambda ext: np.clip(ext[None, :] * half - d + 0.5, 0, 1) * valid
    a_pk = np.minimum(1.0, peak ** 0.9)
    a_rms = np.minimum(1.0, (rms * 2.8) ** 0.85)
    img = np.broadcast_to(PANEL, (P, w, 3)).copy()
    shade = (1.10 - 0.34 * (d / half))[..., None]
    T = WAVE_THEMES[theme]
    if theme == "3-band":
        img = _blend(img, cov(np.maximum(a_pk, 0.03)), T["low"] * 0.40 + PANEL * 0.2, 0.55)
        ext = [np.minimum(a_pk, (g_ * bands[:, k]) ** 0.62) for k, g_ in enumerate((4.4, 4.0, 9.0))]
        for k, key in enumerate(("low", "mid", "high")):
            cv_ = cov(np.maximum(ext[k], 0.03))[..., None]
            col = np.clip(T[key] * shade, 0, 255)
            img = img * (1 - cv_) + col * cv_
    else:
        wgt = (bands * np.array([1.0, 1.6, 4.0]) + 1e-7) ** 2.4
        wgt = wgt / wgt.sum(axis=1, keepdims=True)
        colw = wgt[:, :1] * T["low"] + wgt[:, 1:2] * T["mid"] + wgt[:, 2:3] * T["high"]
        colw = np.clip(colw / np.maximum(colw.max(axis=1, keepdims=True), 1) * 255, 0, 255)
        img = _blend(img, cov(np.maximum(a_pk, 0.03)), colw * 0.38, 0.9)
        cv_ = cov(np.maximum(a_rms, 0.03))[..., None]
        img = img * (1 - cv_) + np.clip(colw[None] * shade, 0, 255) * cv_
    # linha central sutil
    mid_rows = [P // 2 - 1, P // 2]
    for rr in mid_rows: img[rr] = mix(img[rr], C3(255, 255, 255), 0.05)
    # beatgrid
    if grid and eng.grid_P and eng.grid_a is not None:
        Pe, a = eng.grid_P, eng.grid_a
        k0 = math.ceil(((eng.hop_count - need) * 4 - a) / Pe)
        k1 = math.floor((eng.hop_count * 4 - a) / Pe)
        for k in range(k0, k1 + 1):
            x = int(((a + k * Pe) / 4.0 - (eng.hop_count - need)) / z)
            if 0 <= x < w:
                strong = (k % 4 == 0)
                img[:, x] = mix(img[:, x], C3(255, 255, 255), 0.30 if strong else 0.13)
                tk = 3 if strong else 1
                img[:tk, x] = mix(img[:tk, x], C3(255, 255, 255), 0.80 if strong else 0.45)
                img[-tk:, x] = mix(img[-tk:, x], C3(255, 255, 255), 0.80 if strong else 0.45)
    # overview: escurece fora da janela do waveform principal
    if ov_focus is not None:
        n = min(w, ov_focus)
        img[:, :w - n] = mix(img[:, :w - n], PANEL, 0.50)
        img[:, w - n] = mix(img[:, w - n], C3(255, 255, 255), 0.55)
    else:
        img[:, w - 1] = mix(img[:, w - 1], C3(255, 255, 255), 0.75)
        if w > 2: img[:, w - 2] = mix(img[:, w - 2], C3(255, 255, 255), 0.20)
    img = np.clip(img, 0, 255)
    cv.ch[r0:r0 + h, c0:c0 + w] = "▀"
    cv.fg[r0:r0 + h, c0:c0 + w] = img[0::2]
    cv.bg[r0:r0 + h, c0:c0 + w] = img[1::2]


# ---------------------------------------------------------------- espectro
PART = np.array(list(" ▁▂▃▄▅▆▇"))


def draw_spectrum(cv, v, eng, r0, c0, h, w):
    ph = h - 1
    nb = w
    vals = eng.spectrum(nb)
    vals = np.convolve(np.pad(vals, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    if v.spec is None or len(v.spec) != nb:
        v.spec, v.pkh = np.zeros(nb), np.zeros(nb)
    v.spec = np.where(vals > v.spec, v.spec + (vals - v.spec) * 0.75, np.maximum(vals, v.spec - 0.035))
    v.pkh = np.maximum(v.spec, v.pkh - 0.014)
    pkv = np.convolve(np.pad(v.pkh, 2, mode='edge'), [0.1, 0.2, 0.4, 0.2, 0.1], 'valid')
    hv = (v.spec * ph * 8).astype(int)
    full, rem = (hv // 8)[None, :], (hv % 8)[None, :]
    rb = np.arange(ph - 1, -1, -1)[:, None]
    isfull = rb < full
    ispart = (rb == full) & (rem > 0)
    chars = np.where(isfull, "█", np.where(ispart, PART[rem], " "))
    base = grad(np.linspace(0, 1, nb), SPEC_STOPS)                       # (nb,3)
    sh = (0.46 + 0.54 * ((rb + 0.5) / ph) ** 0.9)                        # (ph,1)
    cap = (ispart | ((rb == full - 1) & (rem == 0))) * 0.32 + 1.0
    fg = np.clip(base[None] * sh[..., None] * cap[..., None], 0, 255)
    bgm = np.broadcast_to(PANEL, (ph, nb, 3)).copy()
    # grade de fundo
    empty = chars == " "
    for fr in (0.25, 0.5, 0.75):
        rr = ph - 1 - int(fr * ph)
        m = empty[rr]
        chars[rr, m] = "╌"; fg[rr, m] = C3(48, 54, 80)
    for f in (100, 1000, 10000):
        x = int(math.log(f / 30) / math.log(18000 / 30) * nb)
        if 0 <= x < nb:
            m = empty[:, x] & (chars[:, x] == " ")
            chars[m, x] = "┆"; fg[m, x] = C3(48, 54, 80)
    # pico retido: brilho fantasma acima da barra (sem traços soltos)
    peak_rows = (pkv * ph * 8).astype(int)
    pfull = (peak_rows // 8)[None, :]
    ghost = (rb <= pfull) & (chars == " ")
    edge = (rb == pfull) & (chars == " ")
    bgm = np.where(ghost[..., None], PANEL + base[None] * 0.06, bgm)
    bgm = np.where(edge[..., None], PANEL + base[None] * 0.11, bgm)
    cv.bg[r0:r0 + ph, c0:c0 + nb] = bgm
    cv.ch[r0:r0 + ph, c0:c0 + nb] = chars
    cv.fg[r0:r0 + ph, c0:c0 + nb] = fg
    # rótulos de frequência
    last = -4
    for f, lab in ((50, "50"), (100, "100"), (200, "200"), (500, "500"), (1000, "1k"), (2000, "2k"),
                   (5000, "5k"), (10000, "10k"), (16000, "16k")):
        x = int(math.log(f / 30) / math.log(18000 / 30) * nb)
        if x >= last + 2 and x + len(lab) <= nb:
            cv.text(r0 + ph, c0 + x, lab, DIMC)
            last = x + len(lab)


# ---------------------------------------------------------------- stereo
BRAILLE = np.array([[1, 8], [2, 16], [4, 32], [64, 128]])
BRL = np.array([chr(0x2800 + i) for i in range(256)])


def draw_gonio(cv, v, eng, r0, c0, ch, cw):
    Wd, Hd = 2 * cw, 4 * ch
    if v.dens is None or v.dens.shape != (Hd, Wd): v.dens = np.zeros((Hd, Wd))
    x = eng.ring[-2048:].astype(np.float64)
    S = (x[:, 1] - x[:, 0]) * 0.7071
    M = (x[:, 0] + x[:, 1]) * 0.7071
    pk = max(0.25, float(max(np.abs(S).max(), np.abs(M).max())))
    tgt = 1.0 / pk
    v.gain = tgt if tgt < v.gain else v.gain * 0.93 + tgt * 0.07
    k = min(Wd, Hd) / 2 * 0.92 * min(v.gain, 4.0)
    px = np.clip((Wd / 2 + S * k).astype(int), 0, Wd - 1)
    py = np.clip((Hd / 2 - M * k).astype(int), 0, Hd - 1)
    v.dens *= 0.80
    np.add.at(v.dens, (py, px), 1.0)
    dots = (v.dens > 0.35).reshape(ch, 4, cw, 2)
    code = (dots * BRAILLE[None, :, None, :]).sum(axis=(1, 3))
    cellmax = v.dens.reshape(ch, 4, cw, 2).max(axis=(1, 3))
    inten = np.clip(cellmax / max(3.0, float(cellmax.max())), 0, 1) ** 0.7
    col = grad(inten, [(0, (55, 70, 200)), (0.35, (70, 150, 255)), (0.7, (90, 225, 255)), (1, (250, 252, 255))])
    # guias
    guide = C3(34, 39, 60)
    for r in range(ch):
        for c in range(cw):
            if code[r, c] == 0:
                onv, onh = c == cw // 2, r == ch // 2
                if onv and onh: cv.ch[r0 + r, c0 + c] = "┼"; cv.fg[r0 + r, c0 + c] = guide
                elif onv: cv.ch[r0 + r, c0 + c] = "│"; cv.fg[r0 + r, c0 + c] = guide
                elif onh: cv.ch[r0 + r, c0 + c] = "─"; cv.fg[r0 + r, c0 + c] = guide
    sl = (slice(r0, r0 + ch), slice(c0, c0 + cw))
    m = code > 0
    chars = BRL[code]
    cv.ch[sl] = np.where(m, chars, cv.ch[sl])
    cv.fg[sl] = np.where(m[..., None], col, cv.fg[sl])
    cv.text(r0, c0, "L", DIMC); cv.text(r0, c0 + cw - 1, "R", DIMC); cv.text(r0, c0 + cw // 2 + 1, "M", DIMC)


# ---------------------------------------------------------------- barras
def hbar(cv, r, c, w, frac, ticks=()):
    frac = min(1.0, max(0.0, frac))
    total = int(frac * w * 8)
    cv.bg[r, c:c + w] = TRACK
    colors = grad((np.arange(w) + 0.5) / w, METER_STOPS)
    for i in range(w):
        n = total - i * 8
        if n >= 8: chh = "█"
        elif n > 0: chh = " ▏▎▍▌▋▊▉"[n]
        else: chh = " "
        for tf in ticks:
            if int(tf * w) == i and chh == " ": chh = "╎"; cv.fg[r, c + i] = DIMC
        if chh != " " and chh != "╎":
            cv.fg[r, c + i] = colors[i]
        cv.ch[r, c + i] = chh


def phase_meter(cv, r, c, w, corr):
    cv.text(r, c, "PHASE", DIMC)
    x0, ww = c + 6, w - 6 - 6
    cv.ch[r, x0:x0 + ww] = "─"; cv.fg[r, x0:x0 + ww] = C3(48, 54, 80)
    mid = ww // 2
    pos = int(round((corr + 1) / 2 * (ww - 1)))
    colr = C3(70, 225, 130) if corr > 0.3 else C3(250, 205, 60) if corr > 0 else C3(255, 75, 70)
    a, b = sorted((mid, pos))
    cv.ch[r, x0 + a:x0 + b + 1] = "━"; cv.fg[r, x0 + a:x0 + b + 1] = colr
    cv.ch[r, x0 + mid] = "┃" if pos == mid else "┿"; cv.fg[r, x0 + mid] = colr if pos == mid else C3(150, 158, 190)
    cv.ch[r, x0 + pos] = "●"; cv.fg[r, x0 + pos] = colr
    cv.text(r, x0 + ww + 1, f"{corr:+.2f}", TEXTC)


def level_color(db, kind):
    if kind == "peak":
        return C3(255, 75, 70) if db > -1 else C3(255, 195, 55) if db > -6 else TEXTC
    return TEXTC


# ---------------------------------------------------------------- header
def draw_header(cv, eng, v, src, r0, C, big):
    if not big:
        cv.box(r0, 0, 3, C, "DJSCOPE", src)
        bpm = f"{eng.bpm:.1f}" if eng.bpm else "---"
        li = "---" if eng.lufs_i is None else f"{eng.lufs_i:.1f}"
        led = ACC * (0.35 + 0.65 * eng.pulse)
        cv.text(r0 + 1, 2, "●", led)
        txt = f"BPM {bpm}    LUFS M {eng.lufs_m:.1f}  I {li}    PEAK {eng.max_peak:.1f} dB"
        cv.text(r0 + 1, 4, txt, TEXTC)
        return 3
    cv.box(r0, 0, 6, C, "DJSCOPE", src)
    li = "---" if eng.lufs_i is None else f"{eng.lufs_i:.1f}"
    plr = "---" if eng.lufs_i is None else f"{eng.max_peak - eng.lufs_i:.1f}"
    bpm = f"{eng.bpm:.1f}" if eng.bpm else "---"
    tiles = [("BPM", bpm, "", ACC), ("LUFS-M", f"{eng.lufs_m:.1f}", "momentary", ACC2),
             ("LUFS-I", li, "integrated", ACC2), ("PEAK", f"{eng.max_peak:.1f}", "dBFS", TEXTC),
             ("PLR", plr, "dB", TEXTC)]
    iw = C - 2
    n = 5 if iw >= 100 else 4 if iw >= 80 else 3
    tiles = tiles[:n]
    tw = iw // n
    pulse = eng.pulse
    for i, (lab, val, sub, colr) in enumerate(tiles):
        x = 1 + i * tw
        if i:
            cv.ch[r0 + 1:r0 + 5, x] = "│"; cv.fg[r0 + 1:r0 + 5, x] = BORDER
        cv.text(r0 + 1, x + 2, lab, DIMC)
        if lab == "BPM":
            bp = eng.beat_pos()
            cur = int(math.floor(bp)) % 4 if bp is not None else -1
            for b in range(4):
                lit = (b == cur)
                dot = "●" if lit else "○"
                cc = mix(ACC * 0.5, C3(255, 255, 255), pulse * 0.8) if lit else C3(60, 66, 92)
                if b == 0 and lit: cc = mix(C3(255, 90, 70), C3(255, 255, 255), pulse * 0.7)
                cv.text(r0 + 1, x + 2 + 7 + b * 2, dot, cc)
            colr = mix(colr, C3(255, 255, 255), pulse * 0.55)
        elif sub:
            cv.text(r0 + 1, x + 2 + len(lab) + 1, sub, C3(62, 68, 96))
        if lab == "PEAK": colr = level_color(eng.max_peak, "peak")
        big_text(cv, r0 + 2, x + 2, val, colr)
    return 6


# ---------------------------------------------------------------- tela
class View:
    def __init__(self):
        self.zoom, self.grid, self.paused = 2, True, False
        self.theme_i = 0
        self.spec = self.pkh = None
        self.dens = None
        self.gain = 1.0
        self.rows = []

    @property
    def theme(self): return list(WAVE_THEMES)[self.theme_i % len(WAVE_THEMES)]


KEYS = (("q", "sair"), ("g", "beatgrid"), ("c", "cores"), ("+/-", "zoom"), ("espaço", "pausa"), ("r", "reset"))


def draw_footer(cv, r, C):
    x = 1
    for k, lab in KEYS:
        cv.text(r, x, f" {k} ", TEXTC, C3(30, 34, 52))
        x += len(k) + 2
        cv.text(r, x, f" {lab}", DIMC)
        x += len(lab) + 3


def render(v, eng, C, R, src):
    cv = Canvas(R, C)
    if C < 84 or R < 24:
        cv.text(1, 2, f"Terminal pequeno ({C}x{R}) — precisa de pelo menos 84x24.", TEXTC)
        return cv
    pc = 1
    X0, W = pc, C - 2 * pc
    big = R >= 34
    hh = 6 if big else 3
    # desenha header com padding
    hdr = Canvas(hh, W)
    draw_header(hdr, eng, v, src, 0, W, big)
    cv.ch[0:hh, X0:X0 + W] = hdr.ch; cv.fg[0:hh, X0:X0 + W] = hdr.fg; cv.bg[0:hh, X0:X0 + W] = hdr.bg
    body_r = hh
    body_h = R - hh - 1
    rw = 0
    if C >= 110: rw = max(32, min(46, W // 4))
    gap = 1 if rw else 0
    lw = W - rw - gap
    # ---- esquerda: waveform + espectro
    wbh = int(body_h * 0.52)
    sbh = body_h - wbh
    cv.box(body_r, X0, wbh, lw, "WAVEFORM", f"{lw * v.zoom * HOP / SR - 2 * 0:.0f}s · {v.theme} · grid {'on' if v.grid else 'off'}")
    ih, iw = wbh - 2, lw - 2
    ov = 3 if ih >= 15 else 0
    main_h = ih - ov - (1 if ov else 0)
    draw_wave(cv, eng, body_r + 1, X0 + 1, main_h, iw, v.zoom, v.theme, v.grid)
    if ov:
        z_ov = max(1, math.ceil(3600 / iw))
        draw_wave(cv, eng, body_r + 1 + main_h + 1, X0 + 1, ov, iw, z_ov, v.theme, False,
                  ov_focus=math.ceil(iw * v.zoom / z_ov))
    cv.box(body_r + wbh, X0, sbh, lw, "SPECTRUM", "30 Hz – 18 kHz")
    draw_spectrum(cv, v, eng, body_r + wbh + 1, X0 + 1, sbh - 2, lw - 2)
    # ---- direita: stereo + níveis
    if rw:
        rx = X0 + lw + gap
        lvl_h = 12 if body_h >= 30 else 9
        st_h = body_h - lvl_h
        cv.box(body_r, rx, st_h, rw, "STEREO", f"corr {eng.corr:+.2f}")
        iw2, ih2 = rw - 2, st_h - 2 - 2
        gc = min(iw2, ih2 * 2)
        gr = max(3, gc // 2)
        gr = min(gr, ih2)
        gc = min(gc, gr * 2)
        draw_gonio(cv, v, eng, body_r + 1 + (ih2 - gr) // 2, rx + 1 + (iw2 - gc) // 2, gr, gc)
        phase_meter(cv, body_r + st_h - 2, rx + 2, rw - 4, eng.corr)
        # níveis
        lr = body_r + st_h
        cv.box(lr, rx, lvl_h, rw, "LEVELS")
        bx = rx + 2 + 10
        bw = rw - 4 - 10 - 1
        rr = lr + 1
        li = eng.lufs_i
        cv.text(rr, rx + 2, "LUFS", DIMC)
        cv.text(rr, bx + int(0.65 * bw) - 1, "-14", C3(70, 76, 108))
        rr += 1
        fl = lambda x: (x + 40) / 40
        for lab, val in (("M", eng.lufs_m), ("S", eng.lufs_s), ("I", li)):
            cv.text(rr, rx + 2, lab, TEXTC)
            cv.text(rr, rx + 4, "  ---" if val is None else f"{val:5.1f}", TEXTC if val is not None else DIMC)
            hbar(cv, rr, bx, bw, 0 if val is None else fl(val), ticks=(fl(-14),))
            rr += 1
        rr += 1 if lvl_h >= 11 else 0
        cv.text(rr, rx + 2, "PEAK dBFS", DIMC)
        cv.text(rr, bx + int(0.9 * bw) - 1, "-6", C3(70, 76, 108))
        rr += 1
        for i, lab in enumerate("LR"):
            cv.text(rr, rx + 2, lab, TEXTC)
            hv = eng.hold[i]
            cv.text(rr, rx + 4, f"{hv:5.1f}", level_color(hv, "peak"))
            hbar(cv, rr, bx, bw, (eng.pk[i] + 60) / 60, ticks=((-6 + 60) / 60,))
            rr += 1
    draw_footer(cv, R - 1, C)
    return cv


def strip(s): return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", s)


# ---------------------------------------------------------------- main
def default_source():
    sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True).stdout.strip()
    return sink + ".monitor"


def short_src(s):
    return s if len(s) < 34 else s[:16] + "…" + s[-16:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="fonte do pactl (padrão: monitor do sink padrão)")
    ap.add_argument("--list", action="store_true", help="lista fontes e sai")
    ap.add_argument("--demo", action="store_true", help="áudio sintético de teste")
    ap.add_argument("--dump", type=float, help="roda N segundos sem tty e imprime o último frame")
    ap.add_argument("--size", default="150x44")
    a = ap.parse_args()
    if a.list:
        os.execvp("pactl", ["pactl", "list", "short", "sources"])
    src = "demo" if a.demo else short_src(a.source or default_source())
    cap = Demo() if a.demo else Capture(a.source or default_source())
    eng, v = Engine(), View()
    dt = 1 / 30

    if a.dump:
        C, R = map(int, a.size.split("x"))
        end, nxt, last_est, tmax, frames = time.time() + a.dump, time.time(), 0, 0, 0
        while time.time() < end:
            t0 = time.time()
            c = cap.get()
            if len(c): eng.feed(c)
            if t0 - last_est > 0.5: eng.estimate(); last_est = t0
            rows = render(v, eng, C, R, src).emit()
            tmax = max(tmax, time.time() - t0); frames += 1
            nxt += dt
            time.sleep(max(0, nxt - time.time()))
        print("\n".join(strip(l) for l in rows))
        print(f"frames={frames} worst_frame={tmax * 1000:.1f}ms bpm={eng.bpm}")
        cap.close()
        return

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    out = sys.stdout
    try:
        tty.setcbreak(fd)
        out.write("\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J")
        nxt, last_est = time.time(), 0
        while True:
            t0 = time.time()
            c = cap.get()
            if len(c): eng.feed(c)
            if t0 - last_est > 0.5: eng.estimate(); last_est = t0
            if not v.paused:
                C, R = shutil.get_terminal_size((120, 36))
                v.rows = render(v, eng, C, R, src).emit()
                out.write("\x1b[H" + "\x1b[0m\n".join(v.rows) + "\x1b[0m")
                out.flush()
            nxt += dt
            if nxt < time.time(): nxt = time.time() + dt
            r, _, _ = select.select([fd], [], [], max(0, nxt - time.time()))
            if r:
                k = os.read(fd, 32)
                if k in (b"q", b"\x03", b"Q"): break
                elif k == b"g": v.grid = not v.grid
                elif k == b"c": v.theme_i += 1
                elif k == b" ": v.paused = not v.paused
                elif k in (b"+", b"="): v.zoom = max(1, v.zoom - 1)
                elif k in (b"-", b"_"): v.zoom = min(8, v.zoom + 1)
                elif k == b"r": eng.reset_meters()
                elif k == b"\x1b[C" and False: pass
    finally:
        cap.close()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        out.write("\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l")
        out.flush()


if __name__ == "__main__":
    main()
