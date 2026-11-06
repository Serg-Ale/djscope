#!/usr/bin/env python3
"""djscope (GUI) - painel de análise de áudio ao vivo em Qt, no estilo MiniMeters + waveform de CDJ."""
import argparse, colorsys, math, os, sys, tempfile, time
from collections import deque

# BLAS em 1 thread: as matrizes aqui são pequenas e o multi-thread só gasta CPU (spin) sem acelerar
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
                           QRadialGradient)

import djscope as core

SR = core.SR
UI_FONT = "Noto Sans"
MONO_FONT = "JetBrainsMono Nerd Font"


def qc(c, a=255):
    return QColor(int(c[0]), int(c[1]), int(c[2]), int(a))


BG = (8, 9, 13)
CARD = (15, 17, 24)
BORDER = (30, 34, 48)
TXT = (224, 228, 242)
DIM = (112, 120, 152)
FAINT = (46, 52, 74)
ACC = (255, 176, 56)
ACC2 = (92, 188, 255)
GREEN = (60, 214, 126)
RED = (255, 84, 76)

SPEC_STOPS = [(0.0, (255, 60, 95)), (0.28, (255, 145, 50)), (0.52, (250, 225, 85)),
              (0.74, (70, 225, 150)), (1.0, (70, 150, 255))]
WAVE_THEMES = {
    "3-band": dict(low=(35, 105, 255), mid=(255, 150, 35), high=(238, 242, 255)),
    "rgb": dict(low=(255, 55, 45), mid=(70, 235, 95), high=(80, 150, 255)),
}


def grad_color(t, stops):
    t = float(min(1, max(0, t)))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if t <= p1:
            u = (t - p0) / max(1e-9, p1 - p0)
            return tuple(c0[i] + (c1[i] - c0[i]) * u for i in range(3))
    return stops[-1][1]


def font(px, family=UI_FONT, weight=QFont.Normal, spacing=0.0):
    f = QFont(family)
    f.setPixelSize(int(px))
    f.setWeight(weight)
    if spacing: f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return f


# ================================================================ análise auxiliar
WIN = np.hanning(8192).astype(np.float64)
NF = 32768
FREQS = np.fft.rfftfreq(NF, 1 / SR)
_edges_cache = {}


def spectrum_curve(ring, npts, slope=4.5, fmin=25.0, fmax=20000.0):
    """Espectro em dB (já com slope) em npts bandas log-espaçadas entre fmin e fmax."""
    x = ring[-8192:].astype(np.float64).mean(axis=1) * WIN
    X = np.abs(np.fft.rfft(x, NF)) / (8192 / 4)
    P = X * X
    key = (npts, fmin, fmax)
    c = _edges_cache.get(key)
    if c is None:
        edges = np.geomspace(fmin, fmax, npts + 1)
        lo = np.searchsorted(FREQS, edges[:-1])
        hi = np.maximum(lo + 1, np.searchsorted(FREQS, edges[1:]))
        c = _edges_cache[key] = (lo, hi, np.sqrt(edges[:-1] * edges[1:]))
    lo, hi, fc = c
    cs = np.concatenate(([0.0], np.cumsum(P)))
    mean_p = (cs[hi] - cs[lo]) / (hi - lo)
    db = 10 * np.log10(mean_p + 1e-18)
    return db + slope * np.log2(fc / 1000.0), fc


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_NAMES_PT = NOTES


def note_name(f):
    n = 12 * math.log2(f / 440.0) + 69
    k = int(round(n))
    return f"{NOTES[k % 12]}{k // 12 - 1}", (n - k) * 100


# ---- análise estéreo: campo (pan × frequência), largura por banda, bandas p/ goniômetro
from scipy.signal import sosfilt, butter as _butter
from scipy.ndimage import gaussian_filter

FN = 4096
FFTN = 16384                                   # zero-padding: bins 4x mais densos (interpola o campo nos graves)
FWIN = np.hanning(FN)
FFREQ = np.fft.rfftfreq(FFTN, 1 / SR)
NB_F, NPAN = 240, 241
_vmask = (FFREQ >= 25) & (FFREQ <= 20000)
_vband = np.clip((np.log(FFREQ[_vmask] / 25) / np.log(20000 / 25) * NB_F).astype(int), 0, NB_F - 1)
_vfreq = FFREQ[_vmask]
BAND_COL = [(255, 92, 72), (96, 232, 140), (96, 164, 255)]      # graves / médios / agudos


_interp_cache = {}


def cubic_taps(n_src, n_dst):
    """Reamostragem bicúbica (Keys, a=-0.5) em 4 taps por pixel: devolve (idx[4,n_dst], peso[4,n_dst])."""
    key = (n_src, n_dst)
    m = _interp_cache.get(key)
    if m is not None: return m
    x = (np.arange(n_dst) + 0.5) * n_src / n_dst - 0.5
    i0 = np.floor(x).astype(int)
    t = x - i0
    a = -0.5
    def w(d):
        d = np.abs(d)
        return np.where(d <= 1, (a + 2) * d ** 3 - (a + 3) * d ** 2 + 1, np.where(d < 2, a * d ** 3 - 5 * a * d ** 2 + 8 * a * d - 4 * a, 0.0))
    idx = np.stack([np.clip(i0 + off, 0, n_src - 1) for off in (-1, 0, 1, 2)])
    wt = np.stack([w(t - off) for off in (-1, 0, 1, 2)]).astype(np.float32)
    m = _interp_cache[key] = (idx, wt)
    return m


def cubic_resize(img, h, w):
    """img (H0,W0) -> (h,w) bicúbico, separável, sem BLAS."""
    iy, wy = cubic_taps(img.shape[0], h)
    tmp = sum(img[iy[k]] * wy[k][:, None] for k in range(4))              # (h, W0)
    ix, wx = cubic_taps(img.shape[1], w)
    return sum(tmp[:, ix[k]] * wx[k][None, :] for k in range(4))          # (h, w)


class StereoAnalyzer:
    def __init__(self):
        self.field = np.zeros((NB_F, NPAN), np.float32)
        self.width = np.zeros(3)
        self.bcorr = np.zeros(3)
        self.bass_width, self.bass_loss, self.mono_loss = 0.0, 0.0, 0.0     # grave <120 Hz: % lateral e perda ao somar em mono (dB)
        self.hist = deque(maxlen=260)                 # (t, largura[3], correlação[3])
        self.sos = [_butter(2, 250, "low", fs=SR, output="sos"), _butter(2, [250, 3000], "band", fs=SR, output="sos"),
                    _butter(2, 3000, "high", fs=SR, output="sos")]
        self.zi = [np.zeros((s.shape[0], 2, 2)) for s in self.sos]
        self.gbuf = np.zeros((3, 1536, 2), np.float32)

    def feed(self, chunk):
        if not len(chunk): return
        x = chunk.astype(np.float64)
        for b in range(3):
            y, self.zi[b] = sosfilt(self.sos[b], x, axis=0, zi=self.zi[b])
            self.gbuf[b] = np.concatenate([self.gbuf[b], y.astype(np.float32)])[-1536:]

    def analyze(self, ring):
        x = ring[-FN:].astype(np.float64) * FWIN[:, None]
        XL, XR = np.fft.rfft(x[:, 0], FFTN), np.fft.rfft(x[:, 1], FFTN)
        aL, aR = np.abs(XL)[_vmask], np.abs(XR)[_vmask]
        e = (aL * aL + aR * aR) / (FN / 4) ** 2 / (FFTN / FN)
        pan = (aR - aL) / (aR + aL + 1e-12)
        pi = np.clip(np.rint((pan + 1) / 2 * (NPAN - 1)).astype(int), 0, NPAN - 1)
        new = np.bincount(_vband * NPAN + pi, weights=e, minlength=NB_F * NPAN).reshape(NB_F, NPAN)
        k = np.array([0.07, 0.23, 0.40, 0.23, 0.07])
        sm = sum(w * np.roll(new, s, axis=1) for w, s in zip(k, (-2, -1, 0, 1, 2)))
        kv = np.array([1, 4, 6, 4, 1]) / 16.0
        sm = sum(w * np.roll(sm, s, axis=0) for w, s in zip(kv, (-2, -1, 0, 1, 2)))
        sm = sm.astype(np.float32)
        self.field = np.maximum(sm * 0.55, self.field * 0.80 + sm * 0.20)         # média móvel (estável) com ataque rápido
        # correlação por banda (a partir do sinal filtrado)
        for b in range(3):
            xb = self.gbuf[b][-1536:].astype(np.float64)
            d = math.sqrt(float((xb[:, 0] ** 2).sum() * (xb[:, 1] ** 2).sum()))
            c = float((xb[:, 0] * xb[:, 1]).sum()) / d if d > 1e-12 else 0.0
            self.bcorr[b] += 0.12 * (c - self.bcorr[b])
        # largura por banda: energia do lado / (meio + lado)
        Ms, Ss = np.abs(XL + XR)[_vmask] ** 2, np.abs(XL - XR)[_vmask] ** 2
        # compatibilidade mono: energia estéreo média vs energia da soma L+R (dB perdidos ao colapsar em mono)
        st_avg = (aL * aL + aR * aR) / 2
        mono_sum = Ms / 4
        bass = _vfreq < 120
        tb = Ms[bass].sum() + Ss[bass].sum()
        bw = float(Ss[bass].sum() / tb) if tb > 1e-12 else 0.0
        sb, mb = float(st_avg[bass].sum()), float(mono_sum[bass].sum())
        lb = min(30.0, 10 * math.log10(sb / max(mb, 1e-18))) if sb > 1e-12 else 0.0
        sa_, ma_ = float(st_avg.sum()), float(mono_sum.sum())
        la = min(30.0, 10 * math.log10(sa_ / max(ma_, 1e-18))) if sa_ > 1e-12 else 0.0
        self.bass_width += 0.10 * (bw - self.bass_width)
        self.bass_loss += 0.10 * (lb - self.bass_loss)
        self.mono_loss += 0.10 * (la - self.mono_loss)
        for i, (f0, f1) in enumerate(((25, 250), (250, 3000), (3000, 20000))):
            m = (_vfreq >= f0) & (_vfreq < f1)
            tot = Ms[m].sum() + Ss[m].sum()
            w = float(Ss[m].sum() / tot) if tot > 1e-12 else 0.0
            self.width[i] += 0.12 * (w - self.width[i])


# ================================================================ waveform em pixels
def wave_cols(peak, rms, bands, Hh, theme, gains=(4.4, 4.0, 7.0)):
    """Renderiza colunas (metade superior) -> uint8 (Hh, n, 3)."""
    n = len(peak)
    k5 = np.array([1, 2, 3, 2, 1], np.float32) / 9
    sm = lambda v: np.convolve(np.pad(v, 2, mode="edge"), k5, "valid")        # suavizado temporal curto (~±16 ms)
    peak = sm(peak)
    rms = sm(rms)
    bands = np.stack([sm(bands[:, k]) for k in range(3)], 1)
    d = (Hh - (np.arange(Hh, dtype=np.float32) + 0.5))[:, None]
    valid = (peak > 1e-4)[None, :]
    cov = lambda ext: np.clip(ext[None, :].astype(np.float32) * Hh - d + 0.5, 0, 1) * valid
    a_pk = np.minimum(1.0, peak ** 0.9)
    a_rms = np.minimum(1.0, (rms * 2.8) ** 0.85)
    card = np.array(CARD, np.float32)
    img = np.broadcast_to(card, (Hh, n, 3)).copy()
    shade = (1.12 - 0.40 * (d / Hh))[..., None]
    T = {k: np.array(v, np.float32) for k, v in WAVE_THEMES[theme].items()}
    if theme == "3-band":
        c = cov(np.maximum(a_pk, 0.02))[..., None] * 0.55
        img = img * (1 - c) + (T["low"] * 0.40 + card * 0.2) * c
        for k, (key, gain) in enumerate(zip(("low", "mid", "high"), gains)):
            ext = np.minimum(a_pk, (gain * bands[:, k]) ** 0.62)
            c = cov(np.maximum(ext, 0.02))[..., None]
            img = img * (1 - c) + np.clip(T[key] * shade, 0, 255) * c
    else:
        w = (bands * np.array([1.0, 1.6, 4.0]) + 1e-7) ** 2.4
        w = w / w.sum(axis=1, keepdims=True)
        col = w[:, :1] * T["low"] + w[:, 1:2] * T["mid"] + w[:, 2:3] * T["high"]
        col = np.clip(col / np.maximum(col.max(axis=1, keepdims=True), 1) * 255, 0, 255).astype(np.float32)
        c = cov(np.maximum(a_pk, 0.02))[..., None] * 0.9
        img = img * (1 - c) + col[None] * 0.38 * c
        c = cov(np.maximum(a_rms, 0.02))[..., None]
        img = img * (1 - c) + np.clip(col[None] * shade, 0, 255) * c
    return np.clip(img, 0, 255).astype(np.uint8)


def agg_cols(data, W, cpp, how="mean"):
    """data (N,5) -> peak, rms, bands(W,3) agrupando cpp amostras por coluna (alinhado ao fim)."""
    need = W * cpp
    arr = np.zeros((need, 5), np.float32)
    if len(data): arr[need - len(data):] = data[-need:]
    g = arr.reshape(W, cpp, 5)
    red = (lambda a: a.max(axis=1)) if how == "max" else (lambda a: a.mean(axis=1))
    return g[:, :, 0].max(axis=1), red(g[:, :, 1]), red(g[:, :, 2:])


class WaveBuf:
    """Imagem da waveform com rolagem incremental: só as colunas novas são calculadas a cada frame."""
    def __init__(self):
        self.key = None
        self.img = None
        self.count = 0

    def update(self, eng, W, Hh, cpp, theme):
        key = (W, Hh, cpp, theme)
        total = eng.hop_count * 4
        if key != self.key:
            self.key = key
            H = 2 * Hh
            self.img = np.zeros((H, W, 3), np.uint8)
            data = np.array(list(eng.fine)[-W * cpp:], np.float32).reshape(-1, 5)
            self._write(*agg_cols(data, W, cpp), 0, W, Hh, theme)
            self.count = total
            return
        ncols = (total - self.count) // cpp
        if ncols <= 0: return
        ncols = min(ncols, W)
        data = np.array(list(eng.fine)[-ncols * cpp:], np.float32).reshape(-1, 5)
        pk, rms, bands = agg_cols(data, ncols, cpp)
        self.img[:, :W - ncols] = self.img[:, ncols:].copy()
        self._write(pk, rms, bands, W - ncols, W, Hh, theme)
        self.count += ncols * cpp

    def _write(self, pk, rms, bands, x0, x1, Hh, theme):
        cols = wave_cols(pk, rms, bands, Hh, theme)
        self.img[:Hh, x0:x1] = cols
        self.img[Hh:, x0:x1] = cols[::-1]


def to_qimage(arr, dpr=1.0):
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    fmt = QImage.Format_RGB888 if arr.shape[2] == 3 else QImage.Format_RGBA8888
    q = QImage(arr.data, w, h, arr.strides[0], fmt)
    q._keep = arr
    q.setDevicePixelRatio(dpr)
    return q


# ================================================================ widgets
class Card(QtWidgets.QWidget):
    chips = ()

    def __init__(self, app, title, accent=ACC2):
        super().__init__()
        self.app, self.title, self.accent = app, title, accent
        self.setMinimumSize(140, 90)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._chip_rects = []
        self._max_rect = None
        self._pos_rect = None
        self._hot = False
        self._over_icon = False
        self._over_pos = False

    def right_text(self): return ""
    def title_extra(self, p, r, x): pass
    def chip_active(self): return -1
    def chip_select(self, i): pass
    def on_hover(self, pos): pass

    def is_compact(self): return self.width() < 520

    def mousePressEvent(self, e):
        pos = e.position()
        if self._max_rect is not None and self._max_rect.contains(pos):
            self.app.toggle_focus(self); return
        if self._pos_rect is not None and self._pos_rect.contains(pos):
            self.app.toggle_focus_pos(); return
        for rect, i in self._chip_rects:
            if rect.contains(pos):
                self.chip_select(i); return
        fc = self.app.focus_card()
        if fc is not None and fc is not self:               # clique numa miniatura = promove a principal
            self.app.set_focus_card(self); return
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if time.time() - self.app.focus_t < 0.45: return    # duplo clique que acabou de promover a miniatura
        self.app.toggle_focus(self)

    def mouseMoveEvent(self, e):
        pos = e.position()
        over_icon = self._max_rect is not None and self._max_rect.contains(pos)
        over_pos = self._pos_rect is not None and self._pos_rect.contains(pos)
        over = over_icon or over_pos or any(rect.contains(pos) for rect, _ in self._chip_rects) or \
            (self.app.focus_card() not in (None, self))
        self.setCursor(Qt.PointingHandCursor if over else Qt.ArrowCursor)
        if over_icon != self._over_icon or over_pos != self._over_pos:
            self._over_icon, self._over_pos = over_icon, over_pos; self.update()
        self.on_hover(pos)

    def enterEvent(self, e):
        self._hot = True; self.update()

    def leaveEvent(self, e):
        self._hot = False; self._over_icon = False; self._over_pos = False
        self.on_hover(None); self.update()

    def _draw_pos_icon(self, p, rect):
        """Mostra onde as miniaturas estão (lateral ou embaixo); clique alterna."""
        if self._over_pos:
            p.setPen(Qt.NoPen); p.setBrush(qc((34, 39, 56))); p.drawRoundedRect(rect, 6, 6)
        c = rect.center()
        col = qc(TXT if self._over_pos else DIM)
        box = QRectF(c.x() - 7.5, c.y() - 5.5, 15, 11)
        p.setPen(QPen(col, 1.4)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(box, 2.5, 2.5)
        p.setPen(Qt.NoPen); p.setBrush(QColor(col.red(), col.green(), col.blue(), 120))
        if self.app.focus_pos == "right":
            p.drawRect(QRectF(c.x() + 2.5, box.top() + 1.2, 4.3, box.height() - 2.4))
        else:
            p.drawRect(QRectF(box.left() + 1.2, c.y() + 1.8, box.width() - 2.4, 3.2))

    def _draw_max_icon(self, p, rect, restore):
        if self._over_icon:
            p.setPen(Qt.NoPen); p.setBrush(qc((34, 39, 56))); p.drawRoundedRect(rect, 6, 6)
        c = rect.center()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(qc(TXT if self._over_icon else DIM), 1.5, Qt.SolidLine, Qt.RoundCap))
        if not restore:
            s, L = 5.0, 3.2
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                x, y = c.x() + sx * s, c.y() + sy * s
                p.drawLine(QPointF(x, y), QPointF(x - sx * L, y)); p.drawLine(QPointF(x, y), QPointF(x, y - sy * L))
        else:
            p.drawRect(QRectF(c.x() - 6, c.y() - 2, 8, 8))
            p.drawPolyline([QPointF(c.x() - 2, c.y() - 2), QPointF(c.x() - 2, c.y() - 6), QPointF(c.x() + 6, c.y() - 6), QPointF(c.x() + 6, c.y() + 2), QPointF(c.x() + 2, c.y() + 2)])

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), qc(BG))
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        thumb = self.app.focus_card() not in (None, self)
        p.setPen(QPen(qc(self.accent, 150) if (thumb and self._hot) else qc(BORDER), 1.4 if (thumb and self._hot) else 1))
        p.setBrush(qc(CARD))
        p.drawRoundedRect(r, 11, 11)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(self.accent))
        p.drawEllipse(QPointF(r.left() + 18, r.top() + 21), 3, 3)
        p.setFont(font(11, weight=QFont.DemiBold, spacing=1.8))
        p.setPen(qc(DIM))
        p.drawText(QPointF(r.left() + 30, r.top() + 25), self.title)
        title_end = r.left() + 30 + QFontMetricsF(p.font()).horizontalAdvance(self.title)
        right = r.right() - 12
        compact = self.is_compact()
        self._chip_rects = []
        self._max_rect = QRectF(right - 22, r.top() + 10, 22, 21)
        self._draw_max_icon(p, self._max_rect, self.app.focus_card() is self)
        right -= 30
        self._pos_rect = None
        if self.app.focus_card() is self:
            self._pos_rect = QRectF(right - 22, r.top() + 10, 22, 21)
            self._draw_pos_icon(p, self._pos_rect)
            right -= 30
        chips_fit = False
        if self.chips:                                                  # os botões de modo só somem se NÃO couberem no título
            p.setFont(font(10, weight=QFont.DemiBold, spacing=1.3))
            fm0 = QFontMetricsF(p.font())
            need = sum(fm0.horizontalAdvance(c) + 18 for c in self.chips) + 6 * (len(self.chips) - 1)
            chips_fit = need <= (right - title_end - 28)
        if self.chips and chips_fit:
            p.setFont(font(10, weight=QFont.DemiBold, spacing=1.3))
            fm = QFontMetricsF(p.font())
            for i in reversed(range(len(self.chips))):
                lab = self.chips[i]
                w = fm.horizontalAdvance(lab) + 18
                rect = QRectF(right - w, r.top() + 10, w, 21)
                on = (i == self.chip_active())
                p.setPen(Qt.NoPen); p.setBrush(qc(self.accent, 46) if on else qc((26, 30, 44)))
                p.drawRoundedRect(rect, 10.5, 10.5)
                p.setPen(qc(self.accent) if on else qc(DIM))
                p.drawText(rect, Qt.AlignCenter, lab)
                self._chip_rects.append((rect, i))
                right -= w + 6
            right -= 6
        rt = self.right_text()
        if rt:
            p.setFont(font(11, MONO_FONT))
            fm = QFontMetricsF(p.font())
            if right - fm.horizontalAdvance(rt) - 16 > title_end:      # só desenha se couber sem encostar no título
                p.setPen(qc((78, 86, 118)))
                p.drawText(QPointF(right - fm.horizontalAdvance(rt), r.top() + 25), rt)
        self.title_extra(p, r, title_end)
        body = QRectF(r.left() + 16, r.top() + 38, r.width() - 32, r.height() - 52)
        if body.width() < 40 or body.height() < 30: return
        p.save()
        p.setClipRect(body.adjusted(-14, -4, 14, 10))
        self.paint_body(p, body)
        p.restore()


def bass_status(sa):
    """Estado da checagem de mono do grave (<120 Hz): (texto, cor)."""
    bw, bl = sa.bass_width, sa.bass_loss
    if bw < 0.10 and bl < 1.0: return "MONO OK", GREEN
    if bl < 3.0 and bw < 0.30: return "GRAVE ABRINDO", (250, 205, 60)
    if bl >= 3.0: return "FORA DE FASE", RED
    return "GRAVE LARGO", RED


PRESET_NAMES = ("MIX", "MASTER", "ANÁLISE")


class DecisionBar(QtWidgets.QWidget):
    """Faixa fixa no topo com o que decide a mixagem: BPM, tonalidade (e as compatíveis), frase, loudness,
    true peak e mono do grave. Também troca os presets (F1 MIX · F2 MASTER · F3 ANÁLISE)."""
    H = 78

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setFixedHeight(self.H)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._pills, self._hover = [], -1

    # ---- preset pills
    def mouseMoveEvent(self, e):
        h = next((i for r, i in self._pills if r.contains(e.position())), -1)
        if h != self._hover:
            self._hover = h; self.setCursor(Qt.PointingHandCursor if h >= 0 else Qt.ArrowCursor); self.update()

    def mousePressEvent(self, e):
        for r, i in self._pills:
            if r.contains(e.position()): self.app.set_preset(i); return

    def leaveEvent(self, e):
        self._hover = -1; self.update()

    # ---- blocos (cada um desenha a partir de x e devolve a largura usada)
    def _label(self, p, x, y, txt):
        p.setFont(font(9, weight=QFont.DemiBold, spacing=1.5)); p.setPen(qc(DIM)); p.drawText(QPointF(x, y), txt)

    def _bpm(self, p, x, r):
        e = self.app.eng
        sure = e.conf >= 0.2
        txt = (f"{e.bpm:.1f}" if sure else f"~{e.bpm:.0f}") if e.bpm else "---"
        col = tuple(ACC[i] + (255 - ACC[i]) * e.pulse * 0.65 for i in range(3)) if sure else (150, 138, 118)
        self._label(p, x, r.top() + 19, "BPM")
        p.setFont(font(36, MONO_FONT, QFont.Medium)); p.setPen(qc(col))
        p.drawText(QPointF(x, r.top() + 60), txt)
        w = QFontMetricsF(p.font()).horizontalAdvance(txt)
        pos = e.phr.pos(); cur = pos[0] if pos else -1
        for b in range(4):                                             # 4 LEDs: o 1 do compasso é vermelho
            cx, cy = x + w + 22 + b * 18, r.top() + 46
            if b == cur:
                bc = (255, 84, 66) if b == 0 else ACC
                gl = QRadialGradient(QPointF(cx, cy), 15)
                gl.setColorAt(0, qc(bc, 150 + 100 * e.pulse)); gl.setColorAt(1, qc(bc, 0))
                p.setPen(Qt.NoPen); p.setBrush(gl); p.drawEllipse(QPointF(cx, cy), 15, 15)
                p.setBrush(qc(tuple(bc[i] + (255 - bc[i]) * e.pulse * 0.7 for i in range(3))))
            else:
                p.setPen(Qt.NoPen); p.setBrush(qc((40, 45, 64)))
            p.drawEllipse(QPointF(cx, cy), 5.5, 5.5)
        return w + 22 + 4 * 18 + 6

    def _key(self, p, x, r, full=True):
        e = self.app.eng
        ky = e.key_info()
        self._label(p, x, r.top() + 19, "TONALIDADE")
        if not ky:
            p.setFont(font(13)); p.setPen(qc((78, 86, 118))); p.drawText(QPointF(x, r.top() + 50), "analisando…")
            return 150
        cc = camelot_rgb(ky["camelot"]); dim = ky["level"] == "baixa"
        br = QRectF(x, r.top() + 26, 66, 38)
        p.setPen(Qt.NoPen); p.setBrush(qc(cc, 34 if dim else 60)); p.drawRoundedRect(br, 9, 9)
        p.setPen(QPen(qc(cc, 110 if dim else 230), 1.4)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(br, 9, 9)
        p.setFont(font(23, MONO_FONT, QFont.Medium)); p.setPen(qc(cc, 150 if dim else 255))
        p.drawText(br, Qt.AlignCenter, ky["camelot"] + ("?" if dim else ""))
        p.setFont(font(13, MONO_FONT, QFont.Medium)); p.setPen(qc(TXT))
        p.drawText(QPointF(x + 78, r.top() + 42), ky["name"])
        lc = {"alta": (96, 232, 140), "média": (250, 205, 60), "baixa": (255, 112, 90)}[ky["level"]]
        p.setPen(Qt.NoPen); p.setBrush(qc(lc)); p.drawEllipse(QPointF(x + 82, r.top() + 54), 3.5, 3.5)
        p.setFont(font(10)); p.setPen(qc(DIM)); p.drawText(QPointF(x + 91, r.top() + 58), f"{ky['level']}")
        if not full: return 190
        nn, l2 = int(ky["camelot"][:-1]), ky["camelot"][-1]
        opp = "A" if l2 == "B" else "B"
        combos = (f"{(nn - 2) % 12 + 1}{l2}", f"{nn % 12 + 1}{l2}", f"{nn}{opp}", f"{(nn + 1) % 12 + 1}{l2}")
        x2 = x + 176
        self._label(p, x2, r.top() + 19, "COMBINA COM")
        for i, code in enumerate(combos):
            c2 = camelot_rgb(code)
            rr = QRectF(x2 + i * 46, r.top() + 30, 41, 24)
            p.setPen(Qt.NoPen); p.setBrush(qc(c2, 52)); p.drawRoundedRect(rr, 7, 7)
            p.setFont(font(12, MONO_FONT, QFont.Medium)); p.setPen(qc(c2)); p.drawText(rr, Qt.AlignCenter, code)
        p.setFont(font(8, weight=QFont.DemiBold, spacing=0.6)); p.setPen(qc((78, 86, 118)))
        for i, t in enumerate(("−1", "+1", "REL", "+2")):
            p.drawText(QRectF(x2 + i * 46, r.top() + 56, 41, 11), Qt.AlignHCenter, t)
        return 176 + 4 * 46

    def _phrase(self, p, x, r):
        e = self.app.eng
        pos = e.phr.pos()
        ok = bool(pos) and e.phr.phr_conf > 0.12 and e.phr.bars >= 12
        self._label(p, x, r.top() + 19, "FRASE" if ok else "FRASE  ~")
        bip = pos[1] if pos else -1
        for i in range(8):
            rr = QRectF(x + i * 20, r.top() + 28, 16, 22)
            lit = (i == bip)
            p.setPen(Qt.NoPen)
            if ok: p.setBrush(qc(ACC) if (lit and i == 0) else qc((255, 255, 255), 235) if lit else qc((70, 78, 108), 150) if i < bip else qc((38, 43, 62)))
            else: p.setBrush(qc((90, 98, 130), 110) if lit else qc((38, 43, 62)))
            p.drawRoundedRect(rr, 4, 4)
        left = 8 - bip if pos else 0
        p.setFont(font(22, MONO_FONT, QFont.Medium)); p.setPen(qc(ACC if ok and left <= 2 else (120, 128, 160)))
        p.drawText(QPointF(x + 172, r.top() + 49), (f"−{left}" if ok else "?"))
        if pos:
            p.setFont(font(9, MONO_FONT)); p.setPen(qc((78, 86, 118)))
            p.drawText(QPointF(x, r.top() + 66), f"compasso {bip + 1}/8  ·  batida {pos[0] + 1}")
        return 232

    def _lufs(self, p, x, r):
        e = self.app.eng
        self._label(p, x, r.top() + 19, "LUFS")
        for i, (lab, val) in enumerate((("MOMENT.", e.lufs_m), ("INTEGR.", e.lufs_i))):
            xx = x + i * 98
            p.setFont(font(8, weight=QFont.DemiBold, spacing=0.8)); p.setPen(qc((78, 86, 118))); p.drawText(QPointF(xx, r.top() + 36), lab)
            p.setFont(font(24, MONO_FONT, QFont.Medium)); p.setPen(qc(ACC2))
            p.drawText(QPointF(xx, r.top() + 62), "---" if val is None else f"{val:.1f}")
        return 196

    def _tp(self, p, x, r):
        e = self.app.eng
        tp = e.tp_max
        col = RED if tp > 0 else ACC if tp > -1 else GREEN
        self._label(p, x, r.top() + 19, "TRUE PEAK")
        p.setFont(font(24, MONO_FONT, QFont.Medium)); p.setPen(qc(col if tp > -80 else DIM))
        txt = f"{tp:+.1f}" if tp > -80 else "--"
        p.drawText(QPointF(x, r.top() + 54), txt)
        w = QFontMetricsF(p.font()).horizontalAdvance(txt)
        p.setFont(font(9, MONO_FONT)); p.setPen(qc((78, 86, 118))); p.drawText(QPointF(x + w + 5, r.top() + 54), "dBTP")
        if e.clips > 0:
            p.setFont(font(10, weight=QFont.DemiBold)); p.setPen(qc(RED)); p.drawText(QPointF(x, r.top() + 69), f"● {e.clips} clip{'s' if e.clips > 1 else ''}")
        return max(110, w + 50)

    def _bass(self, p, x, r):
        sa = self.app.sa
        txt, col = bass_status(sa)
        self._label(p, x, r.top() + 19, "GRAVE < 120 Hz")
        p.setFont(font(10, weight=QFont.Bold, spacing=1.2))
        tw = QFontMetricsF(p.font()).horizontalAdvance(txt) + 32
        pill = QRectF(x, r.top() + 28, tw, 24)
        p.setPen(QPen(qc(col, 150), 1)); p.setBrush(qc(col, 34)); p.drawRoundedRect(pill, 12, 12)
        p.setPen(Qt.NoPen); p.setBrush(qc(col)); p.drawEllipse(QPointF(pill.left() + 12, pill.center().y()), 3.8, 3.8)
        p.setPen(qc(col)); p.drawText(QRectF(pill.left() + 22, pill.top(), pill.width() - 22, pill.height()), Qt.AlignVCenter | Qt.AlignLeft, txt)
        p.setFont(font(9, MONO_FONT)); p.setPen(qc((90, 98, 130)))
        p.drawText(QPointF(x, r.top() + 68), f"lateral {int(sa.bass_width * 100)}%  ·  mono {-sa.bass_loss:+.1f} dB")
        return max(tw, 168)

    def paintEvent(self, ev):
        a = self.app
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        p.fillRect(self.rect(), qc(BG))
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(qc(BORDER), 1)); p.setBrush(qc(CARD)); p.drawRoundedRect(r, 11, 11)
        # pills de preset, à direita
        p.setFont(font(10, weight=QFont.DemiBold, spacing=1.2))
        fm = QFontMetricsF(p.font())
        names = [f"F{i + 1}  {n}" for i, n in enumerate(PRESET_NAMES)]
        widths = [fm.horizontalAdvance(n) + 26 for n in names]
        total = sum(widths) + 6 * (len(widths) - 1)
        self._pills = []
        px = r.right() - 14 - total
        show_pills = r.width() > 1100
        if show_pills:
            for i, (n, w) in enumerate(zip(names, widths)):
                rect = QRectF(px, r.center().y() - 14, w, 28)
                act = (i == a.preset)
                p.setPen(Qt.NoPen); p.setBrush(qc(ACC, 50) if act else qc((34, 39, 56)) if i == self._hover else qc((24, 28, 42)))
                p.drawRoundedRect(rect, 14, 14)
                p.setPen(qc(ACC) if act else qc(TXT) if i == self._hover else qc(DIM)); p.drawText(rect, Qt.AlignCenter, n)
                self._pills.append((rect, i)); px += w + 6
        avail = (px - 18 if show_pills else r.right() - 14) - (r.left() + 18)
        # o que cabe, por prioridade; a tonalidade perde a lista "combina com" antes de sumir
        want = [("bpm", 230), ("key", 394), ("phrase", 232), ("tp", 120), ("lufs", 196), ("bass", 176)]
        sep = 26
        chosen, used = [], 0
        for name, w in want:
            if name == "key" and used + w + sep > avail and used + 190 + sep <= avail:
                chosen.append(("key_small", 190)); used += 190 + sep
            elif used + w + sep <= avail or not chosen:
                chosen.append((name, w)); used += w + sep
        order = {"bpm": 0, "key": 1, "key_small": 1, "phrase": 2, "lufs": 3, "tp": 4, "bass": 5}
        chosen.sort(key=lambda t: order[t[0]])
        x = r.left() + 20
        for i, (name, w) in enumerate(chosen):
            if i:
                p.setPen(QPen(qc(BORDER), 1)); p.drawLine(QPointF(x - sep / 2, r.top() + 14), QPointF(x - sep / 2, r.bottom() - 14))
            if name == "bpm": used_w = self._bpm(p, x, r)
            elif name == "key": used_w = self._key(p, x, r, True)
            elif name == "key_small": used_w = self._key(p, x, r, False)
            elif name == "phrase": used_w = self._phrase(p, x, r)
            elif name == "lufs": used_w = self._lufs(p, x, r)
            elif name == "tp": used_w = self._tp(p, x, r)
            else: used_w = self._bass(p, x, r)
            x += max(used_w, w if name in ("bpm", "phrase") else used_w) + sep


class WaveCard(Card):
    OV_H = 52

    def __init__(self, app):
        super().__init__(app, "WAVEFORM", ACC)
        self.buf = WaveBuf()
        self.ov_img = None
        self.ov_key = None
        self.ov_t = 0.0

    chips = ("3-BAND", "RGB")

    def right_text(self):
        return f"{self.app.wave_secs():.0f}s  ·  grid {'on' if self.app.grid else 'off'}"

    def chip_active(self): return self.app.theme_i % 2
    def chip_select(self, i): self.app.theme_i = i

    def paint_body(self, p, body):
        a, e = self.app, self.app.eng
        dpr = self.devicePixelRatioF()
        gap = 12
        show_ov = body.height() > 150
        main_h = body.height() - (self.OV_H + gap if show_ov else 0)
        W = max(50, int(body.width() * dpr))
        Hh = max(8, int(main_h * dpr / 2))
        cpp = a.cpp
        self.buf.update(e, W, Hh, cpp, a.theme)
        H = 2 * Hh
        q = to_qimage(self.buf.img, dpr)
        top = body.top()
        p.drawImage(QPointF(body.left(), top), q)
        hl = H / dpr
        # centro
        p.setPen(QPen(QColor(255, 255, 255, 14), 1))
        p.drawLine(QPointF(body.left(), top + hl / 2), QPointF(body.right(), top + hl / 2))
        # beatgrid: batidas, início de compasso (1) e início de frase, a partir da contagem real de batidas
        lat = e.beat_lattice()
        if a.grid and lat:
            lb, Pe, bidx = lat
            ph = e.phr
            phr_ok = ph.phr_conf > 0.12 and ph.bars >= 12
            F = self.buf.count
            j0 = math.ceil((F - W * cpp - lb) / Pe)
            j1 = math.floor((F - lb) / Pe)
            for j in range(j0, j1 + 1):
                xp = W - (F - (lb + j * Pe)) / cpp
                if not (0 <= xp < W): continue
                xx = body.left() + xp / dpr
                idx = bidx + j
                bib = (idx - ph.bar_phase) % 4
                bar_in_phrase = (((idx - ph.bar_phase) // 4) - ph.phrase_phase) % 8
                is_bar = bib == 0
                is_phrase = is_bar and bar_in_phrase == 0 and phr_ok
                if is_phrase:
                    p.setPen(QPen(qc(ACC, 215), 2)); p.drawLine(QPointF(xx, top), QPointF(xx, top + hl))
                    p.setPen(Qt.NoPen); p.setBrush(qc(ACC, 235))
                    p.drawPolygon([QPointF(xx - 5, top), QPointF(xx + 5, top), QPointF(xx, top + 7)])
                    p.drawPolygon([QPointF(xx - 5, top + hl), QPointF(xx + 5, top + hl), QPointF(xx, top + hl - 7)])
                else:
                    p.setPen(QPen(QColor(255, 255, 255, 70 if is_bar else 24), 1))
                    p.drawLine(QPointF(xx, top), QPointF(xx, top + hl))
                    if is_bar:
                        p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255, 200))
                        p.drawPolygon([QPointF(xx - 4, top), QPointF(xx + 4, top), QPointF(xx, top + 6)])
                        p.drawPolygon([QPointF(xx - 4, top + hl), QPointF(xx + 4, top + hl), QPointF(xx, top + hl - 6)])
                if is_bar and phr_ok and hl > 70 and Pe / cpp / dpr * 4 > 34:
                    p.setFont(font(10, MONO_FONT, QFont.Medium if is_phrase else QFont.Normal))
                    p.setPen(qc(ACC) if is_phrase else QColor(255, 255, 255, 120))
                    p.drawText(QPointF(xx + 5, top + 17), str(bar_in_phrase + 1))
        # playhead
        gx = body.right()
        gr = QLinearGradient(gx - 26, 0, gx, 0)
        gr.setColorAt(0, QColor(255, 255, 255, 0)); gr.setColorAt(1, QColor(255, 255, 255, 38))
        p.fillRect(QRectF(gx - 26, top, 26, hl), gr)
        p.setPen(QPen(QColor(255, 255, 255, 230), 2))
        p.drawLine(QPointF(gx - 1, top), QPointF(gx - 1, top + hl))
        # overview
        if show_ov:
            self._overview(p, body, top + hl + gap, dpr, W, cpp)

    def _overview(self, p, body, oy, dpr, mainW, cpp):
        a, e = self.app, self.app.eng
        Wo = max(50, int(body.width() * dpr))
        Hh = max(4, int(self.OV_H * dpr / 2))
        now = time.time()
        L = len(e.hist)
        span = int(min(3600, max(900, math.ceil(L / 900) * 900)))        # 30 s .. 2 min, em passos de 30 s (30 hops/s)
        key = (Wo, Hh, a.theme, span)
        if key != self.ov_key or now - self.ov_t > 0.12:
            self.ov_key, self.ov_t = key, now
            data = np.array(list(e.hist)[-span:], np.float32).reshape(-1, 5)
            full = np.zeros((span, 5), np.float32)
            if len(data): full[span - len(data):] = data
            edges = np.unique(np.linspace(0, span, Wo + 1).astype(int)[:-1])
            cnt = np.diff(np.append(edges, span))[:, None]
            pk = np.maximum.reduceat(full[:, 0], edges)
            mean = np.add.reduceat(full[:, 1:], edges, axis=0) / cnt
            rms, bands = mean[:, 0], mean[:, 1:]
            if len(edges) < Wo:
                pad = Wo - len(edges); pk = np.pad(pk, (pad, 0)); rms = np.pad(rms, (pad, 0)); bands = np.pad(bands, ((pad, 0), (0, 0)))
            cols = wave_cols(pk, rms, bands, Hh, a.theme, gains=(4.6, 4.0, 2.6))
            self.ov_img = np.concatenate([cols, cols[::-1]], axis=0)
        q = to_qimage(self.ov_img, dpr)
        p.drawImage(QPointF(body.left(), oy), q)
        oh = self.ov_img.shape[0] / dpr
        secs = mainW * cpp / 120.0
        span_s = span / 30.0
        spanpx = Wo / dpr * min(1.0, secs / span_s)
        fx = body.right() - spanpx
        p.setPen(Qt.NoPen); p.setBrush(QColor(8, 9, 13, 150))
        p.drawRect(QRectF(body.left(), oy, fx - body.left(), oh))
        p.setPen(QPen(QColor(255, 255, 255, 170), 1.2)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(fx, oy + 0.5, spanpx - 0.5, oh - 1), 3, 3)
        p.setFont(font(10, MONO_FONT)); p.setPen(qc((78, 86, 118)))
        p.drawText(QPointF(body.left() + 4, oy + 11), f"-{int(span_s // 60)}:{int(span_s % 60):02d}")


class SpectrumCard(Card):
    DB_TOP, DB_BOT = -26.0, -90.0
    chips = ("CURVE", "BARS", "3D", "KEY")

    def __init__(self, app):
        super().__init__(app, "SPECTRUM", (255, 112, 80))
        self._hover = None
        self._skey = self._fkey = self._lkey = None

    def right_text(self): return f"slope {self.app.slope:.1f} dB/oct"
    def chip_active(self): return self.app.spec_mode
    def chip_select(self, i): self.app.spec_mode = i
    def on_hover(self, pos): self._hover = pos

    # ---- utilidades
    def _plot(self, body):
        return QRectF(body.left() + 34, body.top() + 4, body.width() - 38, body.height() - 26)

    def _ypos(self, db, plot): return plot.bottom() - (db - self.DB_BOT) / (self.DB_TOP - self.DB_BOT) * plot.height()

    def _ys(self, v, plot):
        v = np.clip(v, self.DB_BOT, self.DB_TOP + 4)
        return plot.bottom() - (v - self.DB_BOT) / (self.DB_TOP - self.DB_BOT) * plot.height()

    @staticmethod
    def _poly(xs, ys):
        path = QPainterPath()
        path.addPolygon(QtGui.QPolygonF([QPointF(x, y) for x, y in zip(xs.tolist(), ys.tolist())]))
        return path

    def _static(self, body, dpr):
        """Grade + rótulos (cacheados em pixmap)."""
        key = (int(body.width()), int(body.height()), dpr)
        if self._skey == key: return self._spm
        pm = QtGui.QPixmap(int(body.width() * dpr) + 1, int(body.height() * dpr) + 1)
        pm.setDevicePixelRatio(dpr); pm.fill(Qt.transparent)
        p = QPainter(pm); p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        plot = self._plot(QRectF(0, 0, body.width(), body.height()))
        b = QRectF(0, 0, body.width(), body.height())
        fx = lambda f: plot.left() + math.log(f / 25) / math.log(20000 / 25) * plot.width()
        p.setFont(font(10, MONO_FONT))
        for db in range(-30, -90, -10):
            y = self._ypos(db, plot)
            p.setPen(QPen(QColor(255, 255, 255, 14), 1)); p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(qc((78, 86, 118))); p.drawText(QRectF(b.left(), y - 8, 30, 16), Qt.AlignRight | Qt.AlignVCenter, f"{db}")
        for f, lab, major in ((30, "30", 0), (50, "50", 0), (100, "100", 1), (200, "200", 0), (500, "500", 0),
                              (1000, "1k", 1), (2000, "2k", 0), (5000, "5k", 0), (10000, "10k", 1), (20000, "20k", 0)):
            x = fx(f)
            p.setPen(QPen(QColor(255, 255, 255, 22 if major else 10), 1)); p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            p.setPen(qc(DIM if major else (78, 86, 118)))
            p.drawText(QRectF(x - 20, plot.bottom() + 4, 40, 16), Qt.AlignHCenter | Qt.AlignTop, lab)
        p.end()
        self._skey, self._spm = key, pm
        return pm

    def _fill_texture(self, plot, dpr):
        key = (int(plot.width()), int(plot.height()), dpr)
        if self._fkey == key: return self._ftex
        w, h = max(2, int(plot.width() * dpr)), max(2, int(plot.height() * dpr))
        xs = np.linspace(0, 1, w)
        cols = np.stack([np.interp(xs, [s_[0] for s_ in SPEC_STOPS], [s_[1][k] for s_ in SPEC_STOPS]) for k in range(3)], -1)
        ys = np.linspace(0, 1, h)
        alpha = (0.62 * (1 - ys) ** 1.1 + 0.05)[:, None]
        rgb = cols[None] * alpha[..., None]
        arr = np.concatenate([rgb, np.broadcast_to(alpha[..., None] * 255, (h, w, 1))], axis=-1).astype(np.uint8)
        q = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGBA8888_Premultiplied)
        pm = QtGui.QPixmap.fromImage(q.copy()); pm.setDevicePixelRatio(dpr)
        self._fkey, self._ftex = key, pm
        return pm

    def _led_mask(self, plot, dpr):
        """Linhas escuras horizontais + sombreamento: dá o aspecto de LED segmentado nas barras."""
        key = (int(plot.width()), int(plot.height()), dpr)
        if self._lkey == key: return self._lpm
        pm = QtGui.QPixmap(int(plot.width() * dpr) + 1, int(plot.height() * dpr) + 1)
        pm.setDevicePixelRatio(dpr); pm.fill(Qt.transparent)
        q = QPainter(pm)
        q.setPen(QPen(qc(CARD), 1.6))
        y = plot.height()
        while y > 0:
            q.drawLine(QPointF(0, y), QPointF(plot.width(), y)); y -= 4.5
        q.end()
        self._lkey, self._lpm = key, pm
        return pm

    # ---- hover: frequência, nota, dB
    def _hover_label(self, p, plot, xs, vals):
        h = self._hover
        if h is None: return
        x = h.x() - self.geometry().left() if False else h.x()
        if not (plot.left() <= x <= plot.right()): return
        f = 25 * (20000 / 25) ** ((x - plot.left()) / plot.width())
        db = float(np.interp(x, xs, vals))
        nm, cents = note_name(f)
        txt = f"{f:,.0f} Hz   {nm} {cents:+.0f}¢   {db:.1f} dB".replace(",", ".")
        p.setPen(QPen(QColor(255, 255, 255, 70), 1, Qt.DashLine))
        p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        y = float(self._ys(np.array([db]), plot)[0])
        p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255, 235)); p.drawEllipse(QPointF(x, y), 3.5, 3.5)
        p.setFont(font(11, MONO_FONT))
        fm = QFontMetricsF(p.font())
        w = fm.horizontalAdvance(txt) + 18
        bx = min(max(x + 12, plot.left()), plot.right() - w)
        rect = QRectF(bx, plot.top() + 6, w, 24)
        p.setBrush(QColor(8, 9, 13, 230)); p.setPen(QPen(qc(BORDER), 1)); p.drawRoundedRect(rect, 6, 6)
        p.setPen(qc(TXT)); p.drawText(rect, Qt.AlignCenter, txt)

    def paint_body(self, p, body):
        a = self.app
        plot = self._plot(body)
        dpr = self.devicePixelRatioF()
        mode = a.spec_mode
        if mode in (0, 1):
            p.drawPixmap(QPointF(body.left(), body.top()), self._static(body, dpr))
        if mode == 0 and a.spec is not None: self._paint_curve(p, plot, dpr)
        elif mode == 1 and a.bars is not None: self._paint_bars(p, plot, dpr)
        elif mode == 2: self._paint_3d(p, body, plot)
        elif mode == 3: self._paint_key(p, body)

    # ---- modo 0: curva
    def _paint_curve(self, p, plot, dpr):
        a = self.app
        n = len(a.spec)
        xs = np.linspace(plot.left(), plot.right(), n)
        path = self._poly(xs, self._ys(a.spec, plot))
        fill = QPainterPath(path)
        fill.lineTo(plot.right(), plot.bottom()); fill.lineTo(plot.left(), plot.bottom()); fill.closeSubpath()
        brush = QtGui.QBrush(self._fill_texture(plot, dpr))
        brush.setTransform(QtGui.QTransform.fromTranslate(plot.left(), plot.top()))
        p.setPen(Qt.NoPen)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillPath(fill, brush)
        p.setRenderHint(QPainter.Antialiasing, True)
        line = QLinearGradient(plot.left(), 0, plot.right(), 0)
        glow = QLinearGradient(plot.left(), 0, plot.right(), 0)
        for t, c in SPEC_STOPS: line.setColorAt(t, qc(c)); glow.setColorAt(t, qc(c, 38))
        p.setBrush(Qt.NoBrush)
        p.save()
        for dy, br in ((-2.4, glow), (2.4, glow), (-1.6, glow), (1.6, glow), (-0.8, line), (0.8, line), (0.0, line)):
            p.setPen(QPen(QtGui.QBrush(br), 0))
            p.translate(0, dy); p.drawPath(path); p.translate(0, -dy)
        p.restore()
        p.setPen(QPen(QColor(235, 240, 255, 95), 0)); p.drawPath(self._poly(xs, self._ys(a.spec_pk, plot)))
        self._hover_label(p, plot, xs, a.spec)

    # ---- modo 1: barras LED (1/6 de oitava)
    def _paint_bars(self, p, plot, dpr):
        a = self.app
        n = len(a.bars)
        pitch = plot.width() / n
        bw = pitch * 0.72
        ys = self._ys(a.bars, plot)
        yp = self._ys(a.bars_pk, plot)
        p.setPen(Qt.NoPen)
        p.setRenderHint(QPainter.Antialiasing, False)
        for i in range(n):
            x = plot.left() + i * pitch + (pitch - bw) / 2
            col = grad_color(i / max(1, n - 1), SPEC_STOPS)
            g = QLinearGradient(0, plot.bottom(), 0, plot.top())
            g.setColorAt(0, qc(col, 110)); g.setColorAt(0.55, qc(col, 235)); g.setColorAt(1, qc(col))
            p.setBrush(g)
            p.drawRect(QRectF(x, ys[i], bw, plot.bottom() - ys[i]))
        p.drawPixmap(QPointF(plot.left(), plot.top()), self._led_mask(plot, dpr))
        p.setBrush(QColor(238, 242, 255, 215))
        for i in range(n):
            x = plot.left() + i * pitch + (pitch - bw) / 2
            p.drawRect(QRectF(x, yp[i] - 3, bw, 2))
        p.setRenderHint(QPainter.Antialiasing, True)
        xs = plot.left() + (np.arange(n) + 0.5) * pitch
        self._hover_label(p, plot, xs, a.bars)

    # ---- modo 2: waterfall 3D em perspectiva (profundidade contínua no tempo)
    def _paint_3d(self, p, body, plot):
        a = self.app
        if a.spec3_live is None: return
        T, now = a.T3D, a.now()
        items = [(max(0.0, (now - t) / T), arr) for t, arr in a.spec3d if (now - t) < T]
        items.append((0.0, a.spec3_live))                   # frente = espectro ao vivo
        items.sort(key=lambda it: -it[0])                   # fundo -> frente
        n = len(a.spec3_live)
        W = body.width() - 4
        H = body.height() - 22
        X_SH, Y_SH = W * 0.12, H * 0.50
        wd = W - X_SH
        amp = H * 0.74
        x0, ybase = body.left() + 2, body.top() + H
        t_idx = np.linspace(0, 1, n)
        cardc = qc(CARD)
        p.setBrush(Qt.NoBrush)
        last = len(items) - 1
        for j, (u, arr) in enumerate(items):
            k = min(1.0, u) ** 0.82                          # easing: fatias novas se afastam mais rápido
            ox = x0 + X_SH * k
            by = ybase - Y_SH * k
            lv = np.clip((arr - self.DB_BOT) / (self.DB_TOP - self.DB_BOT), 0, 1.05)
            xs = ox + t_idx * wd
            ys = by - lv * amp * (1 - 0.35 * k)
            pts = QtGui.QPolygonF([QPointF(x, y) for x, y in zip(xs.tolist(), ys.tolist())])
            poly = QtGui.QPolygonF(pts)
            poly.append(QPointF(xs[-1], by + 3)); poly.append(QPointF(xs[0], by + 3))
            fade = min(1.0, (1.0 - k) * 7.0)                 # some suavemente no fundo (sem "pop")
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(CARD[0], CARD[1], CARD[2], 255))
            p.drawPolygon(poly)
            if j == last:
                gr = QLinearGradient(0, by - amp, 0, by)
                gr.setColorAt(0, QColor(255, 255, 255, 46)); gr.setColorAt(1, QColor(CARD[0], CARD[1], CARD[2], 0))
                p.setBrush(gr); p.drawPolygon(poly)
            alpha = int((55 + 200 * (1 - k) ** 1.3) * fade)
            gl = QLinearGradient(ox, 0, ox + wd, 0)
            for tt, c in SPEC_STOPS: gl.setColorAt(tt, qc(c, alpha))
            p.setBrush(Qt.NoBrush)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(QPen(QtGui.QBrush(gl), 0))
            p.drawPolyline(pts)
            if k < 0.10:
                p.translate(0, 0.8); p.drawPolyline(pts); p.translate(0, -0.8)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setFont(font(10, MONO_FONT))
        for f, lab in ((50, "50"), (100, "100"), (500, "500"), (1000, "1k"), (5000, "5k"), (10000, "10k")):
            x = x0 + math.log(f / 25) / math.log(20000 / 25) * wd
            p.setPen(qc((78, 86, 118)))
            p.drawText(QRectF(x - 20, ybase + 5, 40, 14), Qt.AlignHCenter, lab)
        p.setPen(qc((78, 86, 118)))
        p.drawText(QRectF(body.right() - 60, body.top() - 2, 58, 14), Qt.AlignRight, f"depth −{T:.0f} s")

    # ---- modo 3: tonalidade (cromagrama + roda Camelot)
    SCALE_MAJ = (0, 2, 4, 5, 7, 9, 11)
    SCALE_MIN = (0, 2, 3, 5, 7, 8, 10)

    @staticmethod
    def _sector(cx, cy, r0, r1, qa, sweep=30.0):
        """Setor de anel (ângulos no padrão Qt: graus, anti-horário, 0 = 3h)."""
        path = QPainterPath()
        path.arcMoveTo(cx - r1, cy - r1, 2 * r1, 2 * r1, qa)
        path.arcTo(cx - r1, cy - r1, 2 * r1, 2 * r1, qa, sweep)
        path.arcTo(cx - r0, cy - r0, 2 * r0, 2 * r0, qa + sweep, -sweep)
        path.closeSubpath()
        return path

    def _paint_key(self, p, body):
        a, e = self.app, self.app.eng
        ky = e.key_info()
        W, H = body.width(), body.height()
        wide = W >= 640 and H >= 190
        zl = QRectF(body.left(), body.top(), W * (0.30 if wide else 0.42), H)
        zc = QRectF(zl.right() + 10, body.top() + 6, W * (0.38 if wide else 0.58) - 10, H - 6)
        zr = QRectF(zc.right() + 14, body.top(), body.right() - zc.right() - 14, H) if wide else None
        # ---------------- esquerda: nome da tonalidade + confiança
        p.setFont(font(10, weight=QFont.DemiBold, spacing=1.6)); p.setPen(qc(DIM))
        p.drawText(QPointF(zl.left() + 2, zl.top() + 12), "TONALIDADE")
        if not ky:
            p.setFont(font(15)); p.setPen(qc((78, 86, 118)))
            p.drawText(QRectF(zl.left(), zl.top() + 30, zl.width(), 60), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                       "Analisando…\ntoque algo com notas")
            col = (96, 112, 150)
        else:
            col = camelot_rgb(ky["camelot"])
            nm = ky["name"]
            fs = int(max(26, min(54, zl.width() / 4.4, H / 3.2)))
            p.setFont(font(fs, MONO_FONT, QFont.Medium)); p.setPen(qc(TXT))
            p.drawText(QPointF(zl.left() + 2, zl.top() + 18 + fs), nm)
            by = zl.top() + 18 + fs + 14
            bw, bh = fs * 1.9, fs * 0.92
            br = QRectF(zl.left() + 2, by, bw, bh)
            p.setPen(Qt.NoPen); p.setBrush(qc(col, 52)); p.drawRoundedRect(br, 10, 10)
            p.setPen(QPen(qc(col, 200), 1.5)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(br, 10, 10)
            p.setFont(font(int(fs * 0.62), MONO_FONT, QFont.Medium)); p.setPen(qc(col))
            p.drawText(br, Qt.AlignCenter, ky["camelot"])
            ly = by + bh + 22
            lv = ky["level"]
            lc = {"alta": (96, 232, 140), "média": (250, 205, 60), "baixa": (255, 112, 90)}[lv]
            p.setPen(Qt.NoPen); p.setBrush(qc(lc)); p.drawEllipse(QPointF(zl.left() + 7, ly - 4), 4, 4)
            p.setFont(font(12, weight=QFont.DemiBold)); p.setPen(qc(TXT))
            p.drawText(QPointF(zl.left() + 18, ly), f"confiança {lv}")
            p.setFont(font(10, MONO_FONT)); p.setPen(qc((78, 86, 118)))
            p.drawText(QPointF(zl.left() + 2, ly + 17), f"corr {ky['conf']:.2f}  ·  margem {ky['margin']:.2f}")
            p.drawText(QPointF(zl.left() + 2, ly + 31), f"estável {int(ky['stab'] * 100)}%")
            yy2 = ly + 52
            if lv != "alta" and zl.height() > yy2 - zl.top():
                p.setFont(font(11)); p.setPen(qc(DIM))
                p.drawText(QPointF(zl.left() + 2, yy2), "ou  ")
                p.setFont(font(11, MONO_FONT, QFont.Medium)); p.setPen(qc(camelot_rgb(ky["alt_camelot"])))
                p.drawText(QPointF(zl.left() + 26, yy2), f"{ky['alt_name']}  {ky['alt_camelot']}")
                yy2 += 30
            # mixagem harmônica: as 4 vizinhas clássicas da roda
            if zl.height() > yy2 - zl.top() + 40:
                nn, l2 = int(ky["camelot"][:-1]), ky["camelot"][-1]
                opp = "A" if l2 == "B" else "B"
                combos = ((f"{(nn - 2) % 12 + 1}{l2}", "−1"), (f"{nn % 12 + 1}{l2}", "+1"), (f"{nn}{opp}", "relativa"), (f"{(nn + 1) % 12 + 1}{l2}", "+2 energia"))
                p.setFont(font(9, weight=QFont.DemiBold, spacing=1.4)); p.setPen(qc((78, 86, 118)))
                p.drawText(QPointF(zl.left() + 2, yy2), "COMBINA COM")
                for i_, (code, why) in enumerate(combos):
                    cy_ = yy2 + 12 + i_ * 30
                    if cy_ + 26 > zl.bottom(): break                      # só as linhas que cabem
                    cc_ = camelot_rgb(code)
                    rr_ = QRectF(zl.left() + 2, cy_, 52, 23)
                    p.setPen(Qt.NoPen); p.setBrush(qc(cc_, 52)); p.drawRoundedRect(rr_, 7, 7)
                    p.setFont(font(13, MONO_FONT, QFont.Medium)); p.setPen(qc(cc_)); p.drawText(rr_, Qt.AlignCenter, code)
                    p.setFont(font(11)); p.setPen(qc(DIM)); p.drawText(QPointF(rr_.right() + 10, cy_ + 16), why)
        # ---------------- centro: cromagrama
        chroma = ky["chroma"] if ky else e.key.chroma()
        mx = float(chroma.max()) if chroma.max() > 0 else 1.0
        tonic = (ky["idx"] % 12) if ky else None
        scale = set(((tonic + s) % 12) for s in (self.SCALE_MAJ if ky["idx"] < 12 else self.SCALE_MIN)) if ky else set()
        n = 12
        gap = 6.0
        bw = (zc.width() - gap * (n - 1)) / n
        base_y = zc.bottom() - 22
        top_y = zc.top() + 16
        p.setFont(font(10, weight=QFont.DemiBold, spacing=1.6)); p.setPen(qc(DIM))
        p.drawText(QPointF(zc.left(), zc.top() + 4), "CROMAGRAMA")
        for i in range(n):
            x = zc.left() + i * (bw + gap)
            hfrac = float(chroma[i] / mx) ** 0.8
            h = max(3.0, hfrac * (base_y - top_y))
            r = QRectF(x, base_y - h, bw, h)
            inscale = (i in scale)
            c = col if inscale else (62, 70, 98)
            gr = QLinearGradient(0, base_y, 0, base_y - h)
            gr.setColorAt(0, qc(c, 70 if inscale else 60)); gr.setColorAt(1, qc(c, 235 if inscale else 150))
            p.setPen(Qt.NoPen); p.setBrush(gr); p.drawRoundedRect(r, 4, 4)
            if i == tonic:
                p.setPen(QPen(QColor(255, 255, 255, 230), 1.6)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(r, 4, 4)
            p.setFont(font(11, weight=QFont.Bold if i == tonic else QFont.Normal, spacing=0.5))
            p.setPen(qc(TXT if i == tonic else DIM if inscale else (78, 86, 118)))
            p.drawText(QRectF(x - 4, base_y + 3, bw + 8, 16), Qt.AlignHCenter | Qt.AlignTop, PC_NAMES_PT[i])
        # ---------------- direita: roda Camelot
        if zr is not None and zr.width() > 120:
            cx, cy = zr.center().x(), zr.center().y() + 4
            R1 = max(40.0, min(zr.width() / 2, zr.height() / 2 - 6))
            Rm = R1 * 0.64
            R0 = R1 * 0.30
            cur = ky["camelot"] if ky else None
            alt = ky["alt_camelot"] if ky and ky["level"] != "alta" else None
            comp = set()
            if cur:
                nn, ll = int(cur[:-1]), cur[-1]
                comp = {f"{(nn - 2) % 12 + 1}{ll}", f"{nn % 12 + 1}{ll}", f"{nn}{'A' if ll == 'B' else 'B'}"}
            p.setBrush(Qt.NoBrush)
            for n_ in range(1, 13):
                qc_ = 90 - (n_ % 12) * 30
                for ll, ra, rb in (("B", Rm, R1), ("A", R0, Rm)):
                    code = f"{n_}{ll}"
                    cc = camelot_rgb(code)
                    path = self._sector(cx, cy, ra + 1.2, rb - 1.2, qc_ - 15 + 1.0, 28.0)
                    if code == cur: al, pen = 255, QPen(QColor(255, 255, 255, 240), 2)
                    elif code in comp: al, pen = 150, QPen(qc(cc, 200), 1)
                    else: al, pen = 42, Qt.NoPen
                    p.setPen(pen); p.setBrush(qc(cc, al)); p.drawPath(path)
                    if code == alt:
                        p.setBrush(Qt.NoBrush); p.setPen(QPen(QColor(255, 255, 255, 200), 1.4, Qt.DashLine)); p.drawPath(path)
                    ang = math.radians(qc_)
                    rr = (ra + rb) / 2
                    p.setFont(font(10 if rb - ra < 40 else 11, MONO_FONT, QFont.Medium if code in comp or code == cur else QFont.Normal))
                    p.setPen(QColor(10, 12, 18) if code == cur else qc(cc, 255 if code in comp else 120))
                    p.drawText(QRectF(cx + rr * math.cos(ang) - 16, cy - rr * math.sin(ang) - 8, 32, 16), Qt.AlignCenter, code)
            p.setFont(font(9, weight=QFont.DemiBold, spacing=1.3)); p.setPen(qc((78, 86, 118)))
            p.drawText(QRectF(cx - 40, cy - 7, 80, 14), Qt.AlignCenter, "CAMELOT")


def _magma_lut():
    stops = [(0.0, (10, 11, 20)), (0.18, (40, 20, 90)), (0.38, (130, 30, 130)), (0.58, (230, 70, 90)),
             (0.8, (255, 170, 60)), (1.0, (255, 245, 200))]
    lut = np.zeros((256, 3), np.uint8)
    for i in range(256): lut[i] = grad_color(i / 255, stops)
    return lut


MAGMA = _magma_lut()


class SpectrogramCard(Card):
    def __init__(self, app):
        super().__init__(app, "SPECTROGRAM", (170, 90, 255))

    def right_text(self): return "30 s"

    def paint_body(self, p, body):
        a = self.app
        plot = QRectF(body.left() + 34, body.top(), body.width() - 34, body.height() - 16)
        img = MAGMA[a.sgram[::-1]]                       # (ny, nt, 3), baixa freq embaixo
        q = to_qimage(img)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(plot, q)
        p.setFont(font(10, MONO_FONT)); p.setPen(qc((78, 86, 118)))
        for f, lab in ((100, "100"), (1000, "1k"), (10000, "10k")):
            y = plot.bottom() - math.log(f / 25) / math.log(20000 / 25) * plot.height()
            p.drawText(QRectF(body.left(), y - 8, 30, 16), Qt.AlignRight | Qt.AlignVCenter, lab)
            p.setPen(QPen(QColor(255, 255, 255, 22), 1)); p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(qc((78, 86, 118)))
        p.setFont(font(10, MONO_FONT)); p.setPen(qc((78, 86, 118)))
        for fr, lab in ((0.0, "-30 s"), (0.5, "-15 s"), (1.0, "agora")):
            x = plot.left() + fr * plot.width()
            al = Qt.AlignLeft if fr == 0 else Qt.AlignRight if fr == 1 else Qt.AlignHCenter
            r_ = QRectF(x - (0 if fr == 0 else 60 if fr == 1 else 30), plot.bottom() + 2, 60, 14)
            p.drawText(r_, al, lab)


class StereoCard(Card):
    N = 192
    chips = ("GONIO", "POLAR", "FIELD")

    def __init__(self, app):
        super().__init__(app, "STEREO", (90, 220, 255))
        self.dens = np.zeros((3, self.N, self.N), np.float32)
        self.gain = 1.0
        self.polar = np.zeros(90, np.float32)
        self.polar_pk = np.zeros(90, np.float32)
        self.polar_b = np.zeros((3, 90), np.float32)
        self.mdens = None
        self.bgain = [1.0, 1.0, 1.0]
        self._hcache = {}
        self._gq, self._gq_n = None, 0
        self._mq = {}

    def right_text(self): return ""
    def chip_active(self): return self.app.stereo_mode
    def chip_select(self, i): self.app.stereo_mode = i

    # ---------------------------------------------------------------- layout
    BAND_NAMES = (("LOW", "< 250 Hz"), ("MID", "250 Hz – 3 kHz"), ("HIGH", "> 3 kHz"))

    def paint_body(self, p, body):
        a = self.app
        mode = a.stereo_mode
        strip = 0
        hgt = body.height()
        if mode == 0:                                                     # as larguras por banda já estão no painel
            strip = 58 if hgt > 190 else 34 if hgt > 150 else 0           # grave-mono + correlação
        else:
            strip = 88 if hgt > 235 else 58 if hgt > 190 else 30 if hgt > 150 else 0
        vis = QRectF(body.left(), body.top(), body.width(), body.height() - strip - (6 if strip else 0))
        if mode == 0: self._gonio(p, vis)
        elif mode == 1: self._polar(p, vis)
        else: self._field(p, vis)
        if strip:
            self._strip(p, QRectF(body.left(), body.bottom() - strip, body.width(), strip), widths=(mode != 0 and strip >= 80))

    # ---------------------------------------------------------------- utilidades de plot
    @staticmethod
    def _gonio_points(xb, N, k, U=3):
        """Traço contínuo (U sub-amostras entre pontos) com splat bilinear: sem granulado, em resolução de pixel."""
        S = (xb[:, 1] - xb[:, 0]) * 0.7071
        M = (xb[:, 0] + xb[:, 1]) * 0.7071
        px = N / 2 + S * k
        py = N / 2 - M * k
        t = np.arange(U, dtype=np.float32) / U
        px = (px[:-1, None] + (px[1:] - px[:-1])[:, None] * t).ravel()
        py = (py[:-1, None] + (py[1:] - py[:-1])[:, None] * t).ravel()
        x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
        fx = (px - x0).astype(np.float32); fy = (py - y0).astype(np.float32)
        ok = (x0 >= 0) & (x0 < N - 1) & (y0 >= 0) & (y0 < N - 1)
        x0, y0, fx, fy = x0[ok], y0[ok], fx[ok], fy[ok]
        out = np.zeros(N * N, np.float32)
        for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            out += np.bincount((y0 + dy) * N + (x0 + dx), weights=wgt, minlength=N * N).astype(np.float32)
        return out.reshape(N, N) / U

    @staticmethod
    def _peak_sm(xb):
        """Raio que contém 95% dos pontos (robusto a picos isolados)."""
        S = (xb[:, 1] - xb[:, 0]) * 0.7071
        M = (xb[:, 0] + xb[:, 1]) * 0.7071
        return float(np.percentile(np.hypot(S, M), 95))

    def _guides(self, p, box):
        c, R = box.center(), box.width() / 2
        p.setBrush(Qt.NoBrush)
        for rr, al in ((1.0, 44), (0.66, 22), (0.33, 16)):
            p.setPen(QPen(QColor(255, 255, 255, al), 1)); p.drawEllipse(c, R * rr, R * rr)
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        p.drawLine(QPointF(c.x(), box.top()), QPointF(c.x(), box.bottom()))
        p.drawLine(QPointF(box.left(), c.y()), QPointF(box.right(), c.y()))
        p.setPen(QPen(QColor(255, 255, 255, 16), 1))
        d = R * 0.7071
        p.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
        p.drawLine(QPointF(c.x() - d, c.y() + d), QPointF(c.x() + d, c.y() - d))

    # ---------------------------------------------------------------- modo 0: goniômetro + painel por banda
    def _gonio(self, p, vis):
        a, e, sa = self.app, self.app.eng, self.app.sa
        panel = None
        if vis.height() >= 220 and vis.width() >= 560:
            side = min(vis.height(), vis.width() * 0.54)
            main = QRectF(vis.left(), vis.top() + (vis.height() - side) / 2, side, side)
            panel = QRectF(main.right() + 22, vis.top(), vis.right() - main.right() - 22, vis.height())
        else:
            side = max(30.0, min(vis.width(), vis.height()))
            main = QRectF(vis.center().x() - side / 2, vis.top() + (vis.height() - side) / 2, side, side)
        self._guides(p, main)
        dpr = self.devicePixelRatioF()
        N = int(np.clip(main.width() * dpr, 160, 420))
        if self.dens.shape[1] != N: self.dens = np.zeros((3, N, N), np.float32)
        sc = N / 192.0
        x = e.ring[-1536:].astype(np.float32)
        tgt = 0.78 / max(0.06, self._peak_sm(x))
        self.gain = tgt if tgt < self.gain else self.gain * 0.96 + tgt * 0.04
        k = N / 2 * min(self.gain, 9.0)
        self.dens *= 0.88
        for b in range(3):
            self.dens[b] += self._gonio_points(sa.gbuf[b][-1536:], N, k) * 1.3 / sc
        self._gf = getattr(self, "_gf", 0) + 1
        if self._gq is None or self._gq_n != N or self._gf % 2 == 0:       # a imagem muda devagar: recalcula a cada 2 frames
            v = gaussian_filter(self.dens, (0, 0.8, 0.8))
            inten = 1 - np.exp(-v * 2.4)
            cols = np.array(BAND_COL, np.float32) / 255
            rgb = np.clip(np.tensordot(inten, cols, axes=([0], [0])) * (255 * 1.1), 0, 255).astype(np.uint8)   # (N,N,3)
            self._gq, self._gq_n = to_qimage(rgb), N
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.drawImage(main, self._gq)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setFont(font(11, weight=QFont.DemiBold)); p.setPen(qc((78, 86, 118)))
        c = main.center()
        p.drawText(QPointF(main.left() + 6, main.top() + 16), "L"); p.drawText(QPointF(main.right() - 14, main.top() + 16), "R")
        p.drawText(QPointF(c.x() + 7, main.top() + 14), "M"); p.drawText(QPointF(c.x() + 7, main.bottom() - 6), "−M")
        if panel is None:
            p.setFont(font(9, weight=QFont.DemiBold, spacing=1.2))
            lx = vis.left() + 2
            ly = vis.bottom() - 6                                     # embaixo: não colide com o L / M / R do círculo
            for (lab, _), col in zip(self.BAND_NAMES, BAND_COL):
                p.setBrush(qc(col)); p.setPen(Qt.NoPen); p.drawEllipse(QPointF(lx + 3, ly - 3), 3, 3)
                p.setPen(qc(DIM)); p.drawText(QPointF(lx + 11, ly), lab); lx += 52
        else:
            self._bands_panel(p, panel)

    def _bands_panel(self, p, panel):
        """Três bandas (grave/médio/agudo). Largo e baixo -> 3 colunas; alto e estreito -> 3 linhas.
        Cada banda: mini goniômetro + largura + correlação + histórico de 20 s."""
        a, sa = self.app, self.app.sa
        dpr = self.devicePixelRatioF()
        gap = 10.0
        cols = panel.width() / max(1.0, panel.height()) >= 1.45
        if cols:
            cw = (panel.width() - 2 * gap) / 3
            cells = [QRectF(panel.left() + i * (cw + gap), panel.top(), cw, panel.height()) for i in range(3)]
            side = max(40.0, min(cw - 28, panel.height() * 0.46))
        else:
            rh = (panel.height() - 2 * gap) / 3
            cells = [QRectF(panel.left(), panel.top() + i * (rh + gap), panel.width(), rh) for i in range(3)]
            side = max(40.0, min(rh - 18, panel.width() * 0.46))
        M = int(np.clip(side * dpr, 64, 260))
        if self.mdens is None or self.mdens.shape[1] != M: self.mdens = np.zeros((3, M, M), np.float32)
        self.mdens *= 0.86
        for b in range(3):
            xb = sa.gbuf[b][-1024:]
            tgt = 0.75 / max(0.004, self._peak_sm(xb))
            self.bgain[b] = tgt if tgt < self.bgain[b] else self.bgain[b] * 0.96 + tgt * 0.04
            self.mdens[b] += self._gonio_points(xb, M, M / 2 * min(self.bgain[b], 90.0)) * 1.1 / (M / 92.0)
        self._mf = getattr(self, "_mf", 0) + 1
        refresh = (self._mf % 2 == 0) or len(self._mq) < 3 or self._mq.get("M") != M
        mv = gaussian_filter(self.mdens, (0, 0.6, 0.6)) if refresh else None
        if refresh: self._mq["M"] = M
        for b, cell in enumerate(cells):
            col = BAND_COL[b]
            p.setPen(QPen(qc(BORDER), 1)); p.setBrush(qc((19, 22, 32)))
            p.drawRoundedRect(cell, 10, 10)
            w = float(min(1.0, sa.width[b] * 2.0))
            cb = float(sa.bcorr[b])
            ccol = (96, 232, 140) if cb > 0.3 else (250, 205, 60) if cb > 0 else (255, 84, 76)
            if cols:
                x0, tw = cell.left() + 14, cell.width() - 28
                y = cell.top() + 26
                self._band_header(p, x0, y, tw, b)
                box = QRectF(cell.center().x() - side / 2, cell.top() + 40, side, side)
                self._mini_gonio(p, box, mv[b] if refresh else None, col, b)
                yb = box.bottom() + 26
                self._mini_bar(p, x0, yb, tw, "WIDTH", w, col, f"{int(w * 100)}%", centered=False)
                self._mini_bar(p, x0, yb + 24, tw, "CORR", (cb + 1) / 2, ccol, f"{cb:+.2f}", centered=True)
                ch = QRectF(x0, yb + 42, tw, cell.bottom() - 12 - (yb + 42))
            else:
                box = QRectF(cell.left() + 10, cell.top() + (cell.height() - side) / 2, side, side)
                self._mini_gonio(p, box, mv[b] if refresh else None, col, b)
                tx = box.right() + 16
                tw = cell.right() - tx - 12
                y = cell.top() + 24
                self._band_header(p, tx, y, tw, b)
                self._mini_bar(p, tx, y + 22, tw, "WIDTH", w, col, f"{int(w * 100)}%", centered=False)
                self._mini_bar(p, tx, y + 46, tw, "CORR", (cb + 1) / 2, ccol, f"{cb:+.2f}", centered=True)
                ch = QRectF(tx, y + 62, tw, cell.bottom() - 12 - (y + 62))
            if ch.height() >= 46 and ch.width() >= 120:
                self._hist_chart(p, ch, b)

    def _band_header(self, p, x, y, w, b):
        col = BAND_COL[b]
        p.setPen(Qt.NoPen); p.setBrush(qc(col)); p.drawEllipse(QPointF(x + 4, y - 4), 4, 4)
        p.setFont(font(13, weight=QFont.Bold, spacing=2.0)); p.setPen(qc(TXT))
        p.drawText(QPointF(x + 16, y), self.BAND_NAMES[b][0])
        p.setFont(font(10, MONO_FONT)); p.setPen(qc((78, 86, 118)))
        fmt = QFontMetricsF(p.font()).horizontalAdvance(self.BAND_NAMES[b][1])
        p.drawText(QPointF(x + w - fmt, y), self.BAND_NAMES[b][1])

    def _mini_gonio(self, p, box, dens, col, b):
        p.setPen(Qt.NoPen); p.setBrush(qc((11, 13, 19))); p.drawRoundedRect(box, 9, 9)
        cc, R = box.center(), box.width() / 2
        p.setBrush(Qt.NoBrush)
        for rr, al in ((0.92, 28), (0.62, 16)):
            p.setPen(QPen(QColor(255, 255, 255, al), 1)); p.drawEllipse(cc, R * rr, R * rr)
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        p.drawLine(QPointF(cc.x(), box.top() + 4), QPointF(cc.x(), box.bottom() - 4))
        p.drawLine(QPointF(box.left() + 4, cc.y()), QPointF(box.right() - 4, cc.y()))
        if dens is not None:
            it = 1 - np.exp(-dens * 2.4)
            c3 = np.array(col, np.float32)
            img = np.clip(it[..., None] * c3 * 1.1 + (it ** 2.2)[..., None] * 90, 0, 255).astype(np.uint8)
            self._mq[b] = to_qimage(img)
        qi = self._mq.get(b)
        if qi is not None:
            p.save(); path = QPainterPath(); path.addRoundedRect(box, 9, 9); p.setClipPath(path)
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            p.drawImage(box, qi)
            p.restore()

    def _hist_chart(self, p, r, b):
        """Largura (área) e correlação (linha) dessa banda nos últimos 20 s. Cacheado: os dados mudam a 10 Hz."""
        col = BAND_COL[b]
        p.setFont(font(9, weight=QFont.DemiBold, spacing=1.1))
        p.setPen(qc(col, 230)); p.drawText(QPointF(r.left() + 2, r.top() + 11), "WIDTH")
        p.setPen(QColor(255, 255, 255, 215)); p.drawText(QPointF(r.left() + 48, r.top() + 11), "CORR")
        p.setFont(font(9, MONO_FONT)); p.setPen(qc((70, 78, 108)))
        p.drawText(QRectF(r.right() - 60, r.top(), 60, 13), Qt.AlignRight | Qt.AlignVCenter, "20 s")
        plot = QRectF(r.left(), r.top() + 17, r.width(), r.height() - 17)
        dpr = self.devicePixelRatioF()
        h = self.app.sa.hist
        key = (int(plot.width()), int(plot.height()), dpr, h[-1][0] if h else 0)
        cache = self._hcache.get(b)
        if cache is None or cache[0] != key:
            pm = QtGui.QPixmap(int(plot.width() * dpr) + 1, int(plot.height() * dpr) + 1)
            pm.setDevicePixelRatio(dpr); pm.fill(Qt.transparent)
            q = QPainter(pm); q.setRenderHints(QPainter.Antialiasing)
            self._draw_hist(q, QRectF(0, 0, plot.width(), plot.height()), b)
            q.end()
            cache = self._hcache[b] = (key, pm)
        p.drawPixmap(QPointF(plot.left(), plot.top()), cache[1])

    def _draw_hist(self, p, r, b):
        h = self.app.sa.hist
        col = BAND_COL[b]
        p.setPen(Qt.NoPen); p.setBrush(qc((13, 15, 22))); p.drawRoundedRect(r, 6, 6)
        p.save(); path = QPainterPath(); path.addRoundedRect(r, 6, 6); p.setClipPath(path)
        for fr in (0.25, 0.5, 0.75):
            p.setPen(QPen(QColor(255, 255, 255, 12 if fr != 0.5 else 22), 1))
            y = r.bottom() - fr * r.height()
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        if len(h) > 3:
            T = 20.0
            t = np.array([x[0] for x in h]); wv = np.array([x[1][b] for x in h]); cv = np.array([x[2][b] for x in h])
            xs = r.right() - (t[-1] - t) / T * r.width()
            m = xs >= r.left()
            xs, wv, cv = xs[m], wv[m], cv[m]
            if len(xs) > 2:
                yw = r.bottom() - np.minimum(1.0, wv * 2.0) * (r.height() - 4) - 2
                yc = r.bottom() - (np.clip(cv, -1, 1) + 1) / 2 * (r.height() - 4) - 2
                area = QtGui.QPolygonF([QPointF(xs[0], r.bottom())] + [QPointF(x, y) for x, y in zip(xs.tolist(), yw.tolist())] + [QPointF(xs[-1], r.bottom())])
                gr = QLinearGradient(0, r.top(), 0, r.bottom())
                gr.setColorAt(0, qc(col, 120)); gr.setColorAt(1, qc(col, 10))
                p.setPen(Qt.NoPen); p.setBrush(gr); p.drawPolygon(area)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(qc(col, 235), 1.5)); p.drawPolyline(QtGui.QPolygonF([QPointF(x, y) for x, y in zip(xs.tolist(), yw.tolist())]))
                p.setPen(QPen(QColor(255, 255, 255, 215), 1.2)); p.drawPolyline(QtGui.QPolygonF([QPointF(x, y) for x, y in zip(xs.tolist(), yc.tolist())]))
        p.restore()

    def _mini_bar(self, p, x, y, w, label, frac, col, text, centered):
        p.setFont(font(9, weight=QFont.DemiBold, spacing=1.2)); p.setPen(qc(DIM))
        p.drawText(QPointF(x, y + 4), label)
        bx, bw = x + 48, max(20.0, w - 48 - 40)
        tr = QRectF(bx, y - 2, bw, 7)
        p.setPen(Qt.NoPen); p.setBrush(qc((26, 30, 44))); p.drawRoundedRect(tr, 3.5, 3.5)
        frac = min(1.0, max(0.0, frac))
        if centered:
            mid = tr.center().x(); xx = tr.left() + frac * tr.width()
            r = QRectF(min(mid, xx), tr.top(), max(3.0, abs(xx - mid)), tr.height())
            p.setBrush(qc(col, 235)); p.drawRoundedRect(r, 3.5, 3.5)
            p.setBrush(QColor(255, 255, 255, 230)); p.drawEllipse(QPointF(xx, tr.center().y()), 4.5, 4.5)
            p.setPen(QPen(QColor(255, 255, 255, 60), 1)); p.drawLine(QPointF(mid, tr.top() - 2), QPointF(mid, tr.bottom() + 2))
        else:
            p.setBrush(qc(col, 235)); p.drawRoundedRect(QRectF(tr.left(), tr.top(), max(5.0, tr.width() * frac), tr.height()), 3.5, 3.5)
        p.setFont(font(10, MONO_FONT)); p.setPen(qc(TXT))
        p.drawText(QRectF(tr.right() + 6, y - 8, 36, 16), Qt.AlignLeft | Qt.AlignVCenter, text)

    # ---------------------------------------------------------------- modo 1: polar level por banda
    def _polar_lv(self, S, M, prev):
        sgn = np.where(M < 0, -1.0, 1.0)
        S, M = S * sgn, M * sgn
        r = np.sqrt(S * S + M * M)
        th = np.arctan2(S, M)
        nb = len(prev)
        idx = np.clip(((th + np.pi / 2) / np.pi * nb).astype(int), 0, nb - 1)
        mx = np.zeros(nb, np.float32)
        np.maximum.at(mx, idx, r.astype(np.float32))
        lv = np.clip((20 * np.log10(mx + 1e-6) + 54) / 54, 0, 1)
        k = np.array([1, 2, 3, 4, 3, 2, 1], np.float32) / 16
        lv = np.maximum(np.convolve(np.pad(lv, 3, mode="edge"), k, "valid"), lv * 0.55)      # suaviza sem perder picos
        return np.where(lv > prev, lv, prev * 0.90)

    def _polar(self, p, vis):
        a, e, sa = self.app, self.app.eng, self.app.sa
        W, H = vis.width(), vis.height()
        R = max(30.0, min(W / 2 - 14, H - 26))
        c = QPointF(vis.center().x(), vis.top() + 16 + R)
        x = e.ring[-1536:].astype(np.float32)
        self.polar = self._polar_lv((x[:, 1] - x[:, 0]) * 0.7071, (x[:, 0] + x[:, 1]) * 0.7071, self.polar)
        self.polar_pk = np.maximum(self.polar, self.polar_pk * 0.985)
        for b in range(3):
            xb = sa.gbuf[b][-1280:]
            self.polar_b[b] = self._polar_lv((xb[:, 1] - xb[:, 0]) * 0.7071, (xb[:, 0] + xb[:, 1]) * 0.7071, self.polar_b[b])
        nb = len(self.polar)
        p.setBrush(Qt.NoBrush)
        for fr, lab in ((1.0, "0 dB"), (0.78, "-12"), (0.56, "-24"), (0.33, "-36")):
            p.setPen(QPen(QColor(255, 255, 255, 36 if fr == 1.0 else 18), 1))
            p.drawArc(QRectF(c.x() - R * fr, c.y() - R * fr, 2 * R * fr, 2 * R * fr), 0, 180 * 16)
            p.setFont(font(9, MONO_FONT)); p.setPen(qc((70, 78, 108)))
            p.drawText(QPointF(c.x() + 4, c.y() - R * fr + 12), lab)
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        p.drawLine(QPointF(c.x() - R, c.y()), QPointF(c.x() + R, c.y()))
        for ang in (-90, -60, -30, 0, 30, 60, 90):
            t = math.radians(ang)
            p.drawLine(c, QPointF(c.x() + R * math.sin(t), c.y() - R * math.cos(t)))
        def poly(arr):
            pts = [c]
            for i in range(nb):
                t = (i + 0.5) / nb * math.pi - math.pi / 2
                pts.append(QPointF(c.x() + R * arr[i] * math.sin(t), c.y() - R * arr[i] * math.cos(t)))
            return QtGui.QPolygonF(pts)
        p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255, 12)); p.drawPolygon(poly(self.polar_pk))
        for b in (2, 1, 0):                                             # agudos ao fundo, graves na frente
            col = BAND_COL[b]
            p.setPen(Qt.NoPen); p.setBrush(qc(col, 78)); p.drawPolygon(poly(self.polar_b[b]))
            p.setBrush(Qt.NoBrush); p.setPen(QPen(qc(col, 235), 1.5)); p.drawPolygon(poly(self.polar_b[b]))
        p.setBrush(Qt.NoBrush); p.setPen(QPen(QColor(255, 255, 255, 190), 1.2)); p.drawPolygon(poly(self.polar))
        p.setFont(font(11, weight=QFont.DemiBold)); p.setPen(qc((78, 86, 118)))
        p.drawText(QPointF(c.x() - R, c.y() + 15), "L"); p.drawText(QPointF(c.x() + R - 9, c.y() + 15), "R")
        p.drawText(QPointF(c.x() - 4, c.y() - R - 4), "M")
        p.setFont(font(9, weight=QFont.DemiBold, spacing=1.2))
        lx = vis.left() + 2
        for (lab, rng), col in zip(self.BAND_NAMES, BAND_COL):
            p.setBrush(qc(col)); p.setPen(Qt.NoPen); p.drawEllipse(QPointF(lx + 3, vis.top() + 8), 3.5, 3.5)
            p.setPen(qc(TXT)); p.drawText(QPointF(lx + 12, vis.top() + 12), lab); lx += 58
        p.setPen(qc(DIM)); p.drawText(QPointF(lx + 4, vis.top() + 12), "—  total")

    # ---------------------------------------------------------------- modo 2: campo (pan × frequência)
    def _field(self, p, vis):
        sa = self.app.sa
        left = 30.0
        box = QRectF(vis.left() + left, vis.top() + 2, vis.width() - left - 2, vis.height() - 18)
        dpr = self.devicePixelRatioF()
        w, h = max(40, int(box.width() * dpr)), max(40, int(box.height() * dpr))
        # a imagem do campo muda devagar (decaimento): recalcula a cada 2 frames e reaproveita entre eles
        self._fn = getattr(self, "_fn", 0) + 1
        if getattr(self, "_fq", None) is None or self._fq_key != (h, w) or self._fn % 2 == 0:
            db = 10 * np.log10(sa.field + 1e-12)
            I = (np.clip((db + 84) / 46, 0, 1) ** 1.25)[::-1].astype(np.float32)    # linha 0 = agudos (topo)
            up = np.clip(cubic_resize(I, h, w), 0, 1)                                # bicúbico até a resolução da tela
            if getattr(self, "_fcol_key", None) != (h,):
                t = 1 - (np.arange(h) + 0.5) / h
                self._fcol = np.array([grad_color(tt, SPEC_STOPS) for tt in t], np.float32)
                self._fcol_key = (h,)
            rgb = self._fcol[:, None, :] * (0.10 + 0.90 * up[..., None]) * 1.15
            arr = np.empty((h, w, 4), np.uint8)
            arr[..., :3] = np.minimum(rgb, 255)
            arr[..., 3] = up ** 0.75 * 255
            self._fq, self._fq_key = to_qimage(arr, dpr), (h, w)
        q = self._fq
        p.setPen(Qt.NoPen); p.setBrush(qc((11, 13, 19))); p.drawRoundedRect(box, 6, 6)
        p.save(); path = QPainterPath(); path.addRoundedRect(box, 6, 6); p.setClipPath(path)
        p.drawImage(box.topLeft(), q)
        # grade
        p.setPen(QPen(QColor(255, 255, 255, 30), 1))
        p.drawLine(QPointF(box.center().x(), box.top()), QPointF(box.center().x(), box.bottom()))
        for fr in (-0.5, 0.5):
            p.setPen(QPen(QColor(255, 255, 255, 12), 1))
            xx = box.center().x() + fr * box.width() / 2
            p.drawLine(QPointF(xx, box.top()), QPointF(xx, box.bottom()))
        for f in (100, 1000, 10000):
            y = box.bottom() - math.log(f / 25) / math.log(20000 / 25) * box.height()
            p.setPen(QPen(QColor(255, 255, 255, 22), 1)); p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        p.restore()
        p.setFont(font(10, MONO_FONT)); p.setPen(qc((78, 86, 118)))
        for f, lab in ((100, "100"), (1000, "1k"), (10000, "10k")):
            y = box.bottom() - math.log(f / 25) / math.log(20000 / 25) * box.height()
            p.drawText(QRectF(vis.left(), y - 8, left - 4, 16), Qt.AlignRight | Qt.AlignVCenter, lab)
        p.setFont(font(11, weight=QFont.DemiBold))
        p.drawText(QRectF(box.left(), box.bottom() + 2, 30, 14), Qt.AlignLeft, "L")
        p.drawText(QRectF(box.right() - 30, box.bottom() + 2, 30, 14), Qt.AlignRight, "R")
        p.drawText(QRectF(box.center().x() - 20, box.bottom() + 2, 40, 14), Qt.AlignHCenter, "C")

    # ---------------------------------------------------------------- faixa inferior: correlação + largura por banda
    def _bass_row(self, p, r):
        """Checagem de mono do grave (<120 Hz): sistemas de clube costumam somar o grave em mono."""
        sa = self.app.sa
        bw, bl, ml = sa.bass_width, sa.bass_loss, sa.mono_loss
        txt, col = bass_status(sa)
        p.setFont(font(9, weight=QFont.DemiBold, spacing=1.2)); p.setPen(qc(DIM))
        p.drawText(QPointF(r.left() + 6, r.top() + 15), "GRAVE < 120 Hz")
        lab_w = QFontMetricsF(p.font()).horizontalAdvance("GRAVE < 120 Hz") + 14
        p.setFont(font(10, weight=QFont.Bold, spacing=1.2))
        tw = QFontMetricsF(p.font()).horizontalAdvance(txt) + 30
        pill = QRectF(r.left() + 6 + lab_w, r.top() + 2, tw, 20)
        p.setPen(QPen(qc(col, 150), 1)); p.setBrush(qc(col, 34)); p.drawRoundedRect(pill, 10, 10)
        p.setPen(Qt.NoPen); p.setBrush(qc(col)); p.drawEllipse(QPointF(pill.left() + 11, pill.center().y()), 3.5, 3.5)
        p.setPen(qc(col)); p.drawText(QRectF(pill.left() + 20, pill.top(), pill.width() - 20, pill.height()), Qt.AlignVCenter | Qt.AlignLeft, txt)
        p.setFont(font(10, MONO_FONT))
        det = f"lateral {int(bw * 100)}%  ·  soma mono {-bl:+.1f} dB"
        det_w = QFontMetricsF(p.font()).horizontalAdvance(det)
        x = pill.right() + 12
        mc = GREEN if ml < 1.5 else (250, 205, 60) if ml < 3 else RED
        tot = f"SOMA MONO TOTAL {-ml:+.1f} dB"
        p.setFont(font(9, weight=QFont.DemiBold, spacing=1.1))
        tw2 = QFontMetricsF(p.font()).horizontalAdvance(tot)
        room = r.right() - 6 - x
        if room >= det_w + tw2 + 24:                                   # cabe tudo
            p.setFont(font(10, MONO_FONT)); p.setPen(qc((120, 128, 160))); p.drawText(QRectF(x, r.top(), det_w + 4, 24), Qt.AlignVCenter | Qt.AlignLeft, det)
            p.setFont(font(9, weight=QFont.DemiBold, spacing=1.1)); p.setPen(qc(mc)); p.drawText(QPointF(r.right() - 6 - tw2, r.top() + 15), tot)
        elif room >= tw2 + 12:                                         # só o total (o mais importante)
            p.setPen(qc(mc)); p.drawText(QPointF(r.right() - 6 - tw2, r.top() + 15), tot)

    def _strip(self, p, r, widths=False):
        e, sa = self.app.eng, self.app.sa
        y = r.top()
        if r.height() >= 50:
            self._bass_row(p, QRectF(r.left(), y, r.width(), 24)); y += 28
        mx0, mx1 = r.left() + 6, r.right() - 6
        my = y + 8
        track = QRectF(mx0, my - 3, mx1 - mx0, 6)
        gr = QLinearGradient(mx0, 0, mx1, 0)
        gr.setColorAt(0, qc(RED, 150)); gr.setColorAt(0.5, qc((250, 205, 60), 90)); gr.setColorAt(1, qc(GREEN, 170))
        p.setPen(Qt.NoPen); p.setBrush(gr); p.drawRoundedRect(track, 3, 3)
        cx = mx0 + (e.corr + 1) / 2 * (mx1 - mx0)
        p.setBrush(QColor(255, 255, 255)); p.drawEllipse(QPointF(cx, my), 6, 6)
        p.setFont(font(9, MONO_FONT)); p.setPen(qc((78, 86, 118)))
        p.drawText(QRectF(mx0, my + 7, 60, 12), Qt.AlignLeft, "-1"); p.drawText(QRectF(mx1 - 60, my + 7, 60, 12), Qt.AlignRight, "+1")
        p.drawText(QRectF(r.center().x() - 50, my + 7, 100, 12), Qt.AlignHCenter, f"corr {e.corr:+.2f}")
        if not widths: return
        y0 = my + 30
        wcell = (r.width() - 12) / 3
        for i, (lab, col) in enumerate(zip(("LOW", "MID", "HIGH"), BAND_COL)):
            x = r.left() + 6 + i * wcell
            p.setFont(font(9, weight=QFont.DemiBold, spacing=1.2)); p.setPen(qc(DIM))
            p.drawText(QPointF(x, y0 + 4), lab)
            bw = wcell - 82
            tr = QRectF(x + 34, y0 - 3, bw, 7)
            p.setPen(Qt.NoPen); p.setBrush(qc((24, 27, 40))); p.drawRoundedRect(tr, 3.5, 3.5)
            w = float(min(1.0, sa.width[i] * 2.0))                       # 0..1 (S/(M+S) = 0.5 equivale a "totalmente largo")
            p.setBrush(qc(col, 230)); p.drawRoundedRect(QRectF(tr.left(), tr.top(), max(5.0, tr.width() * w), tr.height()), 3.5, 3.5)
            p.setFont(font(10, MONO_FONT)); p.setPen(qc(TXT))
            p.drawText(QRectF(tr.right() + 4, y0 - 8, 28, 16), Qt.AlignLeft | Qt.AlignVCenter, f"{int(w * 100)}%")


class LoudCard(Card):
    RANGE = (-40.0, -6.0)

    def __init__(self, app):
        super().__init__(app, "LOUDNESS", (92, 188, 255))
        self._pm, self._key = None, None

    def right_text(self): return "60 s"

    def paint_body(self, p, body):
        dpr = self.devicePixelRatioF()
        key = (int(body.width()), int(body.height()), self.app.last_lh, dpr)
        if key != self._key:
            self._key = key
            pm = QtGui.QPixmap(int(body.width() * dpr) + 1, int(body.height() * dpr) + 1)
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.transparent)
            q = QPainter(pm)
            q.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
            self._draw(q, QRectF(0, 0, body.width(), body.height()))
            q.end()
            self._pm = pm
        p.drawPixmap(QPointF(body.left(), body.top()), self._pm)

    def _draw(self, p, body):
        a, e = self.app, self.app.eng
        lo, hi = self.RANGE
        plot = QRectF(body.left() + 34, body.top() + 4, body.width() - 38, body.height() - 8)
        y = lambda db: plot.bottom() - (min(hi, max(lo, db)) - lo) / (hi - lo) * plot.height()
        p.setFont(font(10, MONO_FONT))
        for db in (-10, -20, -30):
            yy = y(db)
            p.setPen(QPen(QColor(255, 255, 255, 14), 1)); p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setPen(qc((78, 86, 118))); p.drawText(QRectF(body.left(), yy - 8, 30, 16), Qt.AlignRight | Qt.AlignVCenter, f"{db}")
        # alvo streaming
        yy = y(-14)
        pen = QPen(qc(GREEN, 110), 1, Qt.DashLine); p.setPen(pen); p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
        p.setFont(font(10, MONO_FONT))
        p.setPen(Qt.NoPen); p.setBrush(QColor(CARD[0], CARD[1], CARD[2], 235)); p.drawRoundedRect(QRectF(plot.right() - 82, yy - 8, 80, 16), 4, 4)
        p.setPen(qc(GREEN, 220)); p.drawText(QRectF(plot.right() - 82, yy - 8, 80, 16), Qt.AlignCenter, "-14 stream")
        h = [x for x in a.lufs_hist if x[1] > -65]            # ignora o silêncio antes da música (evita o risco vertical)
        if len(h) > 2:
            t_end = h[-1][0]
            pts_m = [(plot.right() - (t_end - t) / 60.0 * plot.width(), y(m)) for t, m, s in h]
            pts_s = [(plot.right() - (t_end - t) / 60.0 * plot.width(), y(s)) for t, m, s in h]
            pts_m = [pt for pt in pts_m if pt[0] >= plot.left()][::2]; pts_s = [pt for pt in pts_s if pt[0] >= plot.left()][::2]
            if len(pts_m) > 2:
                path = QPainterPath(QPointF(*pts_m[0]))
                for pt in pts_m[1:]: path.lineTo(*pt)
                fill = QPainterPath(path); fill.lineTo(pts_m[-1][0], plot.bottom()); fill.lineTo(pts_m[0][0], plot.bottom()); fill.closeSubpath()
                g = QLinearGradient(0, plot.top(), 0, plot.bottom())
                g.setColorAt(0, qc(ACC2, 90)); g.setColorAt(1, qc(ACC2, 0))
                p.setPen(Qt.NoPen); p.setBrush(g); p.drawPath(fill)
                p.setBrush(Qt.NoBrush)
                pen = QPen(qc(ACC2), 1.6); pen.setJoinStyle(Qt.RoundJoin); p.setPen(pen); p.drawPath(path)
                sp = QPainterPath(QPointF(*pts_s[0]))
                for pt in pts_s[1:]: sp.lineTo(*pt)
                pen = QPen(QColor(255, 255, 255, 190), 1.4); pen.setJoinStyle(Qt.RoundJoin); p.setPen(pen); p.drawPath(sp)
        # legenda
        p.setFont(font(10, weight=QFont.DemiBold, spacing=1.2))
        lx = plot.left() + 8
        for lab, c in (("MOMENTARY", ACC2), ("SHORT-TERM", (235, 238, 250))):
            p.setPen(qc(c)); p.drawText(QPointF(lx, plot.top() + 12), lab)
            lx += QFontMetricsF(p.font()).horizontalAdvance(lab) + 20
        if e.lufs_i is not None:
            yy = y(e.lufs_i)
            p.setPen(QPen(qc(ACC, 190), 1, Qt.DotLine)); p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setPen(qc(ACC)); p.setFont(font(10, weight=QFont.DemiBold, spacing=1.2)); p.drawText(QPointF(lx, plot.top() + 12), f"INTEGRATED {e.lufs_i:.1f}")


class PeakCard(Card):
    def __init__(self, app):
        super().__init__(app, "LEVELS", (255, 120, 90))
        self.setMinimumWidth(150)

    def title_extra(self, p, r, x):
        e = self.app.eng
        tp = e.tp_max
        col = RED if tp > 0 else ACC if tp > -1 else GREEN                 # EBU R128: limite -1 dBTP
        txt = f"TP {tp:+.1f}" if tp > -80 else "TP --"
        p.setFont(font(11, MONO_FONT, QFont.Medium)); p.setPen(qc(col if tp > -80 else DIM))
        p.drawText(QPointF(x + 16, r.top() + 25), txt)
        if e.clips > 0:
            adv = QFontMetricsF(p.font()).horizontalAdvance(txt)
            p.setPen(qc(RED)); p.drawText(QPointF(x + 16 + adv + 10, r.top() + 25), f"● {e.clips}")

    def paint_body(self, p, body):
        e = self.app.eng
        top, bot = body.top() + 22, body.bottom() - 18
        ypk = lambda db: bot - (1 - (min(0.0, max(-60.0, db)) / -60.0) ** 0.6) * (bot - top)
        yl = lambda db: bot - (min(0.0, max(-40.0, db)) + 40.0) / 40.0 * (bot - top)
        # área: [escala pico] [L] [R]   [LUFS-M] [escala LUFS]
        sw = 26.0
        bw = max(14.0, min(34.0, (body.width() - 2 * sw - 40) / 3))
        gap = max(6.0, (body.width() - 2 * sw - 3 * bw) / 4)
        xL = body.left() + sw + gap * 0.6
        xR = xL + bw + gap * 0.5
        xM = xR + bw + gap * 1.3
        # grades + escala de pico (esquerda)
        p.setFont(font(10, MONO_FONT)); last_y = -99.0
        for db in (0, -6, -12, -24, -36, -48, -60):
            yy = ypk(db)
            p.setPen(QPen(QColor(255, 255, 255, 12), 1)); p.drawLine(QPointF(xL, yy), QPointF(xM + bw, yy))
            if abs(yy - last_y) >= 13:
                p.setPen(qc((78, 86, 118))); p.drawText(QRectF(body.left() - 6, yy - 8, sw, 16), Qt.AlignRight | Qt.AlignVCenter, f"{db}")
                last_y = yy
        # escala LUFS (direita)
        for db in (-10, -20, -30):
            yy = yl(db)
            p.setPen(qc((78, 86, 118))); p.drawText(QRectF(xM + bw + 4, yy - 8, sw + 8, 16), Qt.AlignLeft | Qt.AlignVCenter, f"{db}")
        gp = QLinearGradient(0, bot, 0, top)
        for db, c in ((-60, (40, 190, 120)), (-24, (70, 215, 120)), (-12, (200, 225, 70)), (-6, (255, 190, 50)), (-1, (255, 84, 76)), (0, (255, 70, 70))):
            gp.setColorAt(1 - (db / -60.0) ** 0.6, qc(c))
        gl = QLinearGradient(0, bot, 0, top)
        for db, c in ((-40, (60, 150, 255)), (-24, (70, 190, 255)), (-14, (110, 225, 200)), (-9, (255, 190, 50)), (-4, (255, 84, 76))):
            gl.setColorAt((db + 40) / 40.0, qc(c))
        def bar(x, level, hold, yfun, grad, label, valtxt, valcol):
            tr = QRectF(x, top, bw, bot - top)
            p.setPen(Qt.NoPen); p.setBrush(qc((22, 25, 36))); p.drawRoundedRect(tr, 4, 4)
            yy = yfun(level)
            p.save(); p.setClipRect(QRectF(x, yy, bw, bot - yy)); p.setBrush(grad); p.drawRoundedRect(tr, 4, 4); p.restore()
            p.setPen(QPen(qc(CARD), 1.2)); ys = top
            while ys < bot:
                p.drawLine(QPointF(x, ys), QPointF(x + bw, ys)); ys += 4
            if hold is not None:
                hy = yfun(hold)
                p.setPen(QPen(qc(RED if hold > -1 and yfun is ypk else TXT), 2)); p.drawLine(QPointF(x, hy), QPointF(x + bw, hy))
            p.setFont(font(11, MONO_FONT, QFont.Medium)); p.setPen(qc(valcol))
            p.drawText(QRectF(x - 10, body.top() - 2, bw + 20, 16), Qt.AlignHCenter | Qt.AlignVCenter, valtxt)
            p.setFont(font(10, weight=QFont.DemiBold, spacing=1.2)); p.setPen(qc(DIM))
            p.drawText(QRectF(x - 10, bot + 2, bw + 20, 14), Qt.AlignHCenter | Qt.AlignTop, label)
        for i, (x, lab) in enumerate(((xL, "L"), (xR, "R"))):
            hv = e.hold[i]
            bar(x, e.pk[i], e.hold[i], ypk, gp, lab, f"{hv:.1f}", RED if hv > -1 else ACC if hv > -6 else TXT)
        bar(xM, e.lufs_m, e.lufs_i, yl, gl, "M", f"{e.lufs_m:.1f}", ACC2)
        # referência -14 LUFS (streaming) cruzando a barra do LUFS
        y14 = yl(-14)
        p.setPen(QPen(qc(GREEN, 200), 1.4)); p.drawLine(QPointF(xM - 4, y14), QPointF(xM + bw + 4, y14))


def fmt_t(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def camelot_rgb(code):
    """Cor da roda Camelot (1..12 em matiz; B = mais claro, A = mais escuro)."""
    n = int(code[:-1]); minor = code[-1] == "A"
    h = ((n - 1) * 30 + 165) % 360
    r, g_, b = colorsys.hsv_to_rgb(h / 360, 0.62 if minor else 0.50, 0.88 if minor else 0.98)
    return int(r * 255), int(g_ * 255), int(b * 255)


REL_STYLE = {"=": ((96, 232, 140), "="), "±1": ((96, 232, 140), "±1"), "rel": ((96, 232, 140), "rel"),
             "±2": ((250, 205, 60), "±2"), "": ((255, 90, 80), "✗")}


class TracklistCard(Card):
    def __init__(self, app):
        super().__init__(app, "SET", (255, 200, 80))
        self.setMinimumWidth(230)
        self.note, self.note_t = "", 0.0

    def right_text(self):
        a = self.app
        return f"faixa {len(a.tracklog.tracks) + 1}  ·  set {fmt_t(a.wall_now() - a.tracklog.t0)}"

    def flash(self, text):
        self.note, self.note_t = text, time.time()

    def _badge(self, p, x, y, code, level="alta", w=40):
        col = camelot_rgb(code)
        dim = level == "baixa"
        r = QRectF(x, y - 11, w, 17)
        p.setPen(Qt.NoPen); p.setBrush(qc(col, 30 if dim else 52)); p.drawRoundedRect(r, 5, 5)
        p.setFont(font(11, MONO_FONT, QFont.Medium)); p.setPen(qc(col, 150 if dim else 255))
        p.drawText(r, Qt.AlignCenter, code + ("?" if dim else ""))

    # ---- linha do tempo da faixa: loudness desde o começo, playhead e "zona de saída" (últimos 32 s)
    def _timeline(self, p, r, snap, length, elapsed):
        total = length if length > 0 else max(90.0, elapsed + 20.0)
        p.setPen(Qt.NoPen); p.setBrush(qc((20, 23, 34))); p.drawRoundedRect(r, 7, 7)
        p.save()
        path = QPainterPath(); path.addRoundedRect(r, 7, 7); p.setClipPath(path)
        if length > 0:
            zx = r.left() + max(0.0, (length - 32.0) / total) * r.width()
            p.setBrush(qc(ACC, 30)); p.drawRect(QRectF(zx, r.top(), r.right() - zx, r.height()))
            if r.right() - zx > 52:
                p.setFont(font(8, weight=QFont.DemiBold, spacing=1.2)); p.setPen(qc(ACC, 200))
                p.drawText(QRectF(zx + 5, r.top() + 2, r.right() - zx - 8, 12), Qt.AlignLeft, "SAÍDA")
        prof = snap["prof"] if snap else []
        pos0 = snap["pos0"] if snap else 0.0
        px = r.left() + min(1.0, elapsed / total) * r.width()
        if len(prof) > 1:
            lo, hi = -34.0, -6.0
            X = lambda i: r.left() + min(1.0, (pos0 + i) / total) * r.width()
            Y = lambda v: r.bottom() - 3 - min(1.0, max(0.0, (v - lo) / (hi - lo))) * (r.height() - 9)
            pts = [QPointF(X(i), Y(v)) for i, v in enumerate(prof)]
            poly = QtGui.QPolygonF([QPointF(pts[0].x(), r.bottom())] + pts + [QPointF(pts[-1].x(), r.bottom())])
            gr = QLinearGradient(0, r.top(), 0, r.bottom())
            gr.setColorAt(0, qc(ACC2, 140)); gr.setColorAt(1, qc(ACC2, 14))
            p.setPen(Qt.NoPen); p.setBrush(gr); p.drawPolygon(poly)
            p.setBrush(Qt.NoBrush); p.setPen(QPen(qc(ACC2, 230), 1.3)); p.drawPolyline(QtGui.QPolygonF(pts))
        # marcas de minuto
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        m = 60.0
        while m < total:
            xx = r.left() + m / total * r.width()
            p.drawLine(QPointF(xx, r.bottom() - 5), QPointF(xx, r.bottom()))
            m += 60.0
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 235), 2)); p.drawLine(QPointF(px, r.top() + 1), QPointF(px, r.bottom() - 1))

    def paint_body(self, p, body):
        a, e, tl = self.app, self.app.eng, self.app.tracklog
        info = a.nowplaying.info if a.nowplaying else None
        snap = tl.snapshot(a.wall_now())
        playing = bool(info and info.get("status") == "Playing")
        fmx = lambda txt, wpx: QFontMetricsF(p.font()).elidedText(txt, Qt.ElideRight, wpx)
        x0, W = body.left(), body.width()
        y = body.top() + 2
        grand = body.height() > 520                                    # card grande (modo foco): tudo escala
        title = (snap and snap["title"]) or (playing and info.get("title")) or None
        artist = (snap and snap["artist"]) or (playing and info.get("artist")) or ""
        length = float(info.get("length") or 0.0) if playing else 0.0
        pos = info.get("position") if playing else None
        elapsed = float(pos) if (pos is not None and length > 0) else ((snap["elapsed"] + snap["pos0"]) if snap else 0.0)
        remain = max(0.0, length - elapsed) if length > 0 else None
        # ---- A: título, artista, tempo restante
        rw = 0.0
        if remain is not None:
            rc = qc(RED) if remain <= 20 else qc(ACC) if remain <= 60 else qc(TXT)
            p.setFont(font(8, weight=QFont.DemiBold, spacing=1.4)); p.setPen(qc(DIM)); lab_w = QFontMetricsF(p.font()).horizontalAdvance("RESTA")
            txt = f"−{fmt_t(remain)}"
            p.setFont(font(22, MONO_FONT, QFont.Medium)); rw = QFontMetricsF(p.font()).horizontalAdvance(txt)
            p.setPen(rc); p.drawText(QPointF(x0 + W - rw, y + 24), txt)
            p.setFont(font(8, weight=QFont.DemiBold, spacing=1.4)); p.setPen(qc(DIM)); p.drawText(QPointF(x0 + W - rw, y + 6), "RESTA")
        p.setFont(font(22 if grand else 15, weight=QFont.DemiBold)); p.setPen(qc(TXT if title else DIM))
        p.drawText(QPointF(x0, y + (24 if grand else 17)), fmx(title or ("Tocando…" if snap else "Aguardando áudio…"), W - rw - 14))
        p.setFont(font(14 if grand else 11)); p.setPen(qc(DIM))
        tm = f"{fmt_t(elapsed)} / {fmt_t(length)}" if length > 0 else fmt_t(elapsed)
        p.drawText(QPointF(x0, y + (48 if grand else 36)), fmx((artist + "   ·   " if artist else "") + tm, W - rw - 14))
        # ---- B: linha do tempo
        th = 96 if grand else 34 if body.height() > 230 else 24
        ty = y + (62 if grand else 46)
        self._timeline(p, QRectF(x0, ty, W, th), snap, length, elapsed)
        # ---- C: estatísticas ao vivo da faixa
        ky = e.key_info()
        bpm = (snap and snap["bpm"]) or e.bpm
        sure = e.conf >= 0.2
        yy = ty + th + (44 if grand else 28)
        p.setFont(font(30 if grand else 20, MONO_FONT, QFont.Medium)); p.setPen(qc(ACC if sure else (150, 138, 118)))
        txt = (f"{bpm:.1f}" if sure else f"~{bpm:.0f}") if bpm else "---"
        p.drawText(QPointF(x0, yy), txt)
        adv = QFontMetricsF(p.font()).horizontalAdvance(txt)
        p.setFont(font(9, weight=QFont.DemiBold, spacing=1.3)); p.setPen(qc(DIM)); p.drawText(QPointF(x0 + adv + 5, yy), "BPM")
        xk = x0 + adv + (60 if grand else 52)
        if ky: self._badge(p, xk, yy - 3, ky["camelot"], ky["level"], 44)
        if W > 330:
            li = f"{e.lufs_i:.1f}" if e.lufs_i is not None else "---"
            p.setFont(font(22 if grand else 15, MONO_FONT, QFont.Medium)); p.setPen(qc(ACC2)); p.drawText(QPointF(xk + 64, yy), li)
            p.setFont(font(9, weight=QFont.DemiBold, spacing=1.3)); p.setPen(qc(DIM)); p.drawText(QPointF(xk + 64 + (74 if grand else 52), yy), "LUFS-I")
        y_next = yy + (24 if grand else 14)
        bottom = body.bottom() - 16
        has_tracks = len(tl.tracks) > 0
        # ---- D: próxima faixa (compatíveis + janela de BPM)
        ndrawn = 0
        if ky or bpm:
            big = (not has_tracks or grand) and (bottom - y_next >= 78)
            hgt = 74 if big else 26
            if bottom - y_next >= hgt:
                self._next(p, x0, y_next, W, ky, bpm if sure else None, big)
                ndrawn = hgt + 8
        # ---- E: histórico do set
        ly = y_next + ndrawn
        rows_h = bottom - ly
        if rows_h >= 40 and has_tracks:
            self._list(p, body, ly, bottom)
        elif not has_tracks and rows_h >= 24:
            p.setFont(font(10)); p.setPen(qc((70, 78, 108)))
            p.drawText(QRectF(x0, ly + 4, W, 30), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                       "O histórico do set aparece aqui quando a próxima faixa começar (por título, silêncio ou salto de BPM).")
        # ---- rodapé
        p.setFont(font(9, MONO_FONT)); p.setPen(qc((70, 78, 108)))
        foot = self.note if (self.note and time.time() - self.note_t < 4) else "N nova faixa · E salvar CSV"
        p.drawText(QRectF(x0, body.bottom() - 13, W, 13), Qt.AlignLeft | Qt.AlignVCenter, fmx(foot, W))

    def _next(self, p, x0, y0, W, ky, bpm, big):
        """O que procurar para a próxima faixa: tonalidades compatíveis e janela de BPM (±3 % / ±6 % de pitch)."""
        codes = []
        if ky:
            nn, l2 = int(ky["camelot"][:-1]), ky["camelot"][-1]
            opp = "A" if l2 == "B" else "B"
            codes = [(f"{(nn - 2) % 12 + 1}{l2}", "−1"), (f"{nn % 12 + 1}{l2}", "+1"), (f"{nn}{opp}", "REL"), (f"{(nn + 1) % 12 + 1}{l2}", "+2")]
        p.setFont(font(9, weight=QFont.DemiBold, spacing=1.4)); p.setPen(qc((78, 86, 118)))
        p.drawText(QPointF(x0, y0 + 10), "PRÓXIMA FAIXA")
        bx = x0
        by = y0 + (18 if big else 0)
        if not big: bx = x0 + 104
        for code, why in codes:
            col = camelot_rgb(code)
            rr = QRectF(bx, by if big else y0 - 3, 40, 22 if big else 20)
            p.setPen(Qt.NoPen); p.setBrush(qc(col, 52)); p.drawRoundedRect(rr, 6, 6)
            p.setFont(font(12 if big else 11, MONO_FONT, QFont.Medium)); p.setPen(qc(col)); p.drawText(rr, Qt.AlignCenter, code)
            if big:
                p.setFont(font(8, weight=QFont.DemiBold, spacing=0.8)); p.setPen(qc((78, 86, 118)))
                p.drawText(QRectF(bx, rr.bottom() + 2, 40, 11), Qt.AlignHCenter, why)
            bx += 46
        if bpm:
            f3, f6 = (bpm * 0.97, bpm * 1.03), (bpm * 0.94, bpm * 1.06)
            if big and W > 380:
                tx = x0 + 4 * 46 + 14
                p.setFont(font(9, weight=QFont.DemiBold, spacing=1.2)); p.setPen(qc((78, 86, 118)))
                p.drawText(QPointF(tx, y0 + 28), "BPM  ±3%"); p.drawText(QPointF(tx, y0 + 52), "BPM  ±6%")
                p.setFont(font(13, MONO_FONT, QFont.Medium)); p.setPen(qc(TXT))
                p.drawText(QPointF(tx + 70, y0 + 28), f"{f3[0]:.0f}–{f3[1]:.0f}")
                p.setPen(qc(DIM)); p.drawText(QPointF(tx + 70, y0 + 52), f"{f6[0]:.0f}–{f6[1]:.0f}")
            elif big:
                p.setFont(font(11, MONO_FONT)); p.setPen(qc(TXT))
                p.drawText(QPointF(x0, y0 + 74), f"BPM {f3[0]:.0f}–{f3[1]:.0f} (±3%)")
            else:
                p.setFont(font(11, MONO_FONT)); p.setPen(qc(TXT))
                p.drawText(QPointF(bx + 6, y0 + 15), f"{f3[0]:.0f}–{f3[1]:.0f} bpm")

    def _list(self, p, body, ly, bottom):
        """Faixas anteriores do set: início, título, BPM, key, mix com a anterior e LUFS-I."""
        tl = self.app.tracklog
        fmx = lambda txt, wpx: QFontMetricsF(p.font()).elidedText(txt, Qt.ElideRight, wpx)
        x0, W = body.left(), body.width()
        p.setPen(QPen(qc(BORDER), 1)); p.drawLine(QPointF(x0, ly), QPointF(x0 + W, ly))
        cols_rel, cols_lufs = W > 300, W > 380
        right = x0 + W
        c_lufs = right - 46 if cols_lufs else right
        c_rel = (c_lufs - 38) if cols_rel else c_lufs
        c_key = c_rel - 48
        c_bpm = c_key - 46
        c_title = x0 + 68
        p.setFont(font(9, weight=QFont.DemiBold, spacing=1.2)); p.setPen(qc((78, 86, 118)))
        hy = ly + 14
        p.drawText(QPointF(x0, hy), "#"); p.drawText(QPointF(x0 + 22, hy), "INÍCIO")
        p.drawText(QPointF(c_title, hy), "FAIXA")
        p.drawText(QPointF(c_bpm, hy), "BPM"); p.drawText(QPointF(c_key + 4, hy), "KEY")
        if cols_rel: p.drawText(QPointF(c_rel + 4, hy), "MIX")
        if cols_lufs: p.drawText(QPointF(c_lufs + 2, hy), "LUFS")
        rh = 28 if (bottom - ly) > 420 else 22
        top = hy + 8
        nrows = int((bottom - top) // rh)
        recs = list(reversed(tl.tracks))[:max(0, nrows)]
        for i, t in enumerate(recs):
            ry = top + i * rh + (18 if rh > 22 else 15)
            prev = tl.tracks[t["n"] - 2] if t["n"] >= 2 else None
            if i % 2 == 0:
                p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255, 6)); p.drawRoundedRect(QRectF(x0 - 4, top + i * rh, W + 8, rh), 4, 4)
            p.setFont(font(10, MONO_FONT)); p.setPen(qc(DIM))
            p.drawText(QPointF(x0, ry), f"{t['n']}"); p.drawText(QPointF(x0 + 22, ry), fmt_t(t["start_off"]))
            p.setFont(font(11)); p.setPen(qc(TXT))
            name = (t["title"] or "—") + (f" · {t['artist']}" if t["artist"] and W > 420 else "")
            p.drawText(QPointF(c_title, ry), fmx(name, max(30, c_bpm - c_title - 6)))
            p.setFont(font(11, MONO_FONT)); p.setPen(qc(TXT))
            p.drawText(QPointF(c_bpm, ry), f"{t['bpm']:.0f}" if t["bpm"] else "--")
            if t["camelot"]: self._badge(p, c_key, ry - 1, t["camelot"], t["key_level"] or "alta", 38)
            if cols_rel and prev and prev["camelot"] and t["camelot"]:
                col, sym = REL_STYLE[core.camelot_relation(prev["camelot"], t["camelot"])]
                p.setFont(font(11, weight=QFont.DemiBold)); p.setPen(qc(col)); p.drawText(QPointF(c_rel + 6, ry), sym)
            if cols_lufs and t["lufs_i"] is not None:
                p.setFont(font(10, MONO_FONT)); p.setPen(qc(ACC2)); p.drawText(QPointF(c_lufs + 2, ry), f"{t['lufs_i']:.1f}")


# ================================================================ app
class Main(QtWidgets.QWidget):
    def __init__(self, cap, src_name, persist=False):
        super().__init__()
        self.cap, self.src_name, self.persist = cap, src_name, persist
        self.eng = core.Engine()
        self.nowplaying = core.NowPlaying() if persist else None            # título da faixa via MPRIS (só no app real)
        sets_dir = os.environ.get("DJSCOPE_SETS_DIR")                    # (testes usam isso p/ não escrever na pasta real)
        self.tracklog = core.TrackLog(self.eng, outdir=(sets_dir or None) if persist else tempfile.mkdtemp(prefix="djscope-set-"))
        self.last_tl = 0.0
        self.grid, self.theme_i, self.cpp, self.paused = True, 0, 1, False
        self.slope = 4.5
        self.spec = self.spec_pk = None
        self.spec_hold = None
        self.sgram = np.zeros((200, 900), np.uint8)
        self.stereo_mode, self.spec_mode = 0, 0
        self.sa = StereoAnalyzer()
        self.bars = self.bars_pk = None
        self.T3D = 3.4                                  # segundos de histórico no waterfall
        self.spec3d = deque(maxlen=60)                  # (t, curva)
        self.spec3_live = None
        self.last_3d = 0.0
        self.last_bh = 0.0
        self.offline = False
        self.clock = 0.0
        self.last_sg = 0.0
        self.last_lh = 0.0
        self.last_est = 0.0
        self.lufs_hist = deque(maxlen=700)
        self.setWindowTitle("DJScope")
        self.setMinimumSize(980, 620)
        self.setStyleSheet(f"background:rgb{BG};")
        self.header = DecisionBar(self)
        self.wave, self.spectrum = WaveCard(self), SpectrumCard(self)
        self.sg, self.stereo, self.loud, self.peak = SpectrogramCard(self), StereoCard(self), LoudCard(self), PeakCard(self)
        self.tracks_card = TracklistCard(self)
        self.glayout = QtWidgets.QGridLayout(self)            # (não chamar de 'grid': esse nome é o beatgrid on/off)
        self.glayout.setContentsMargins(14, 14, 14, 14); self.glayout.setSpacing(12)
        self.cards = [self.wave, self.spectrum, self.stereo, self.sg, self.loud, self.peak, self.tracks_card]
        self.focus, self.focus_t = None, 0.0
        self.focus_pos = "right"                      # miniaturas: "right" (lateral) ou "bottom" (tira embaixo)
        self.preset = 0                               # 0 MIX · 1 MASTER · 2 ANÁLISE (F1/F2/F3)
        self.set_preset(0)                            # aplica layout e modos do preset (o estado salvo, se existir, vem depois)
        self.widgets = [self.header] + self.cards
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(lambda: self.process(self.cap.get()))
        self.timer.start(16)
        if persist: self._load_state()

    # ---------------------------------------------------------------- modo foco
    def focus_card(self):
        return None if self.focus is None else self.cards[self.focus]

    # cada preset: modos dos cards + linhas [(altura relativa, [(card, colunas de 60), ...]), ...]; o que não está nele fica oculto
    PRESETS = (
        dict(spec=3, stereo=0, rows=[(34, [("wave", 60)]), (40, [("spectrum", 24), ("tracks", 18), ("stereo", 18)]),
                                      (26, [("sg", 22), ("loud", 24), ("peak", 14)])]),
        dict(spec=0, stereo=0, rows=[(24, [("wave", 60)]), (46, [("spectrum", 30), ("stereo", 30)]),
                                      (30, [("loud", 24), ("peak", 12), ("sg", 24)])]),
        dict(spec=2, stereo=2, rows=[(16, [("wave", 60)]), (50, [("spectrum", 34), ("stereo", 26)]),
                                      (34, [("sg", 26), ("loud", 20), ("peak", 14)])]),
    )

    def set_preset(self, i, apply_modes=True):
        self.preset = i % len(self.PRESETS)
        if apply_modes:
            self.spec_mode, self.stereo_mode = self.PRESETS[self.preset]["spec"], self.PRESETS[self.preset]["stereo"]
        self.focus = None
        self.relayout()
        self.header.update()

    def relayout(self):
        g = self.glayout
        for w in [self.header] + self.cards: g.removeWidget(w)
        for r in range(8): g.setRowStretch(r, 0)
        NC = 60                                           # 60 colunas: divide bem por 12 (normal), 4/1 (lateral) e 5 (tira)
        for c in range(NC): g.setColumnStretch(c, 1)
        g.addWidget(self.header, 0, 0, 1, NC)
        byname = dict(wave=self.wave, spectrum=self.spectrum, stereo=self.stereo, sg=self.sg, loud=self.loud,
                      peak=self.peak, tracks=self.tracks_card)
        if self.focus is None:
            shown = set()
            for ri, (stretch, items) in enumerate(self.PRESETS[self.preset]["rows"]):
                col = 0
                for name, span in items:
                    g.addWidget(byname[name], 1 + ri, col, 1, span); col += span; shown.add(name)
                g.setRowStretch(1 + ri, stretch)
            for name, w in byname.items(): w.setVisible(name in shown)         # fora do preset = oculto (volta no modo foco)
        else:
            for w in self.cards: w.setVisible(True)
            main = self.cards[self.focus]
            others = [c for i, c in enumerate(self.cards) if i != self.focus]
            if self.focus_pos == "right":
                g.addWidget(main, 1, 0, len(others), 45)                  # principal: 3/4 da largura
                for i, c in enumerate(others):                            # miniaturas empilhadas à direita
                    g.addWidget(c, 1 + i, 45, 1, 15)
                    g.setRowStretch(1 + i, 1)
            else:
                g.addWidget(main, 1, 0, 1, NC)                            # principal: largura toda
                g.setRowStretch(1, 72)
                w = NC // len(others)
                for i, c in enumerate(others):                            # miniaturas lado a lado embaixo
                    g.addWidget(c, 2, i * w, 1, w)
                g.setRowStretch(2, 28)
        for c in self.cards: c.update()

    def toggle_focus_pos(self):
        self.focus_pos = "bottom" if self.focus_pos == "right" else "right"
        self.relayout()

    def set_focus(self, i):
        self.focus = i
        self.focus_t = time.time()
        self.relayout()

    def set_focus_card(self, card): self.set_focus(self.cards.index(card))

    def toggle_focus(self, card):
        i = self.cards.index(card)
        self.set_focus(None if self.focus == i else i)

    def wall_now(self):
        return (1000.0 + self.clock) if self.offline else time.time()

    def now(self):
        """Relógio da animação: tempo real no app; relógio de amostras nos testes offscreen."""
        return self.clock if self.offline else time.perf_counter()

    @property
    def theme(self): return list(WAVE_THEMES)[self.theme_i % len(WAVE_THEMES)]

    def wave_secs(self):
        w = self.wave.width() - 32
        return w * self.cpp / 120.0 if w > 0 else 0

    def process(self, chunk):
        if self.paused: return
        e = self.eng
        if len(chunk):
            e.feed(chunk)
            self.clock += len(chunk) / SR
        if self.clock - self.last_est > 0.5:
            self.last_est = self.clock
            e.estimate()
            self.setWindowTitle(f"DJScope · {e.bpm:.1f} BPM" if e.bpm else "DJScope")
        # espectro suavizado (dB)
        npts = int(max(120, min(300, self.spectrum.width() / 3.0)))
        db, _ = spectrum_curve(e.ring, npts, self.slope)
        if self.spec is None or len(self.spec) != npts:
            self.spec = db.copy(); self.spec_pk = db.copy()
        else:
            self.spec = np.where(db > self.spec, self.spec + (db - self.spec) * 0.55, np.maximum(db, self.spec - 1.1))
            self.spec_pk = np.maximum(self.spec, self.spec_pk - 0.18)
        if self.clock - self.last_sg >= 1 / 30:
            self.last_sg = self.clock
            col, _ = spectrum_curve(e.ring, self.sgram.shape[0], self.slope)
            v = np.clip((col + 90) / 66, 0, 1)
            self.sgram[:, :-1] = self.sgram[:, 1:]
            self.sgram[:, -1] = (v * 255).astype(np.uint8)
        if self.clock - self.last_lh >= 0.1:
            self.last_lh = self.clock
            self.lufs_hist.append((self.clock, e.lufs_m, e.lufs_s))
        if self.clock - self.last_tl >= 0.5:
            self.last_tl = self.clock
            self.tracklog.tick(self.nowplaying.info if self.nowplaying else None, now=self.wall_now())
        # modo barras (1/6 de oitava)
        if self.spec_mode == 1:
            nb = int(max(24, min(90, self.spectrum.width() / 15)))
            bdb, _ = spectrum_curve(e.ring, nb, self.slope, 31.5, 16000)
            if self.bars is None or len(self.bars) != nb:
                self.bars, self.bars_pk = bdb.copy(), bdb.copy()
            else:
                self.bars = np.where(bdb > self.bars, self.bars + (bdb - self.bars) * 0.75, np.maximum(bdb, self.bars - 1.5))
                self.bars_pk = np.maximum(self.bars, self.bars_pk - 0.20)
        # waterfall 3D: fatia ao vivo (todo frame) + histórico (12 Hz) com timestamp
        n = len(self.spec)
        xi = np.linspace(0, n - 1, 120)
        live = np.interp(xi, np.arange(n), self.spec)
        k5 = np.array([1, 2, 3, 2, 1]) / 9.0
        self.spec3_live = np.convolve(np.pad(live, 2, mode="edge"), k5, "valid")
        tnow = self.now()
        if tnow - self.last_3d >= 1 / 12:
            self.last_3d = tnow
            self.spec3d.append((tnow, self.spec3_live.copy()))
        # estéreo
        self.sa.feed(chunk)
        self.sa.analyze(e.ring)
        if self.clock - self.last_bh >= 0.1:
            self.last_bh = self.clock
            self.sa.hist.append((self.clock, self.sa.width.copy(), self.sa.bcorr.copy()))
        for w in self.widgets: w.update()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_Q: self.close()
        elif k == Qt.Key_Escape:
            if self.focus is not None: self.set_focus(None)
            elif self.windowState() & Qt.WindowFullScreen: self.setWindowState(self.windowState() & ~Qt.WindowFullScreen)
        elif Qt.Key_1 <= k <= Qt.Key_7:
            i = k - Qt.Key_1
            self.set_focus(None if self.focus == i else i)
        elif k == Qt.Key_0: self.set_focus(None)
        elif k == Qt.Key_L: self.toggle_focus_pos()
        elif k == Qt.Key_Tab: self.set_focus(0 if self.focus is None else (self.focus + 1) % len(self.cards))
        elif k == Qt.Key_G: self.grid = not self.grid
        elif k == Qt.Key_C: self.theme_i += 1
        elif k == Qt.Key_Space: self.paused = not self.paused
        elif k in (Qt.Key_Plus, Qt.Key_Equal): self.cpp = max(1, self.cpp - 1)
        elif k in (Qt.Key_Minus, Qt.Key_Underscore): self.cpp = min(4, self.cpp + 1)
        elif k == Qt.Key_R: self.eng.reset_meters()
        elif k == Qt.Key_H: self.header.setVisible(not self.header.isVisible())
        elif k in (Qt.Key_F1, Qt.Key_F2, Qt.Key_F3): self.set_preset(k - Qt.Key_F1)
        elif k == Qt.Key_S: self.stereo_mode = (self.stereo_mode + 1) % 3
        elif k == Qt.Key_D: self.spec_mode = (self.spec_mode + 1) % 4
        elif k == Qt.Key_N:
            self.tracklog.mark(self.nowplaying.info if self.nowplaying else None, self.wall_now())
            self.tracks_card.flash("nova faixa marcada")
        elif k == Qt.Key_E:
            self.tracklog.save()
            self.tracks_card.flash("CSV: " + os.path.basename(self.tracklog.path))
        elif k in (Qt.Key_F, Qt.Key_F11): self.setWindowState(self.windowState() ^ Qt.WindowFullScreen)

    STATE = os.path.expanduser("~/.config/djscope/state.json")

    def _load_state(self):
        try:
            import json
            with open(self.STATE) as f_:
                s = json.load(f_)
            self.stereo_mode = int(s.get("stereo_mode", 0)) % 3
            self.spec_mode = int(s.get("spec_mode", 0)) % 4
            self.theme_i = int(s.get("theme_i", 0))
            self.cpp = min(4, max(1, int(s.get("cpp", 1))))
            self.grid = bool(s.get("grid", True))
            self.header.setVisible(bool(s.get("decision", True)))
            self.preset = int(s.get("preset", 0)) % len(self.PRESETS)
            self.focus_pos = s.get("focus_pos", "right") if s.get("focus_pos") in ("right", "bottom") else "right"
            f = s.get("focus")
            self.focus = f if isinstance(f, int) and 0 <= f < len(self.cards) else None
            self.relayout()
            if s.get("w") and s.get("h"): self.resize(int(s["w"]), int(s["h"]))
        except Exception:
            pass

    def _save_state(self):
        try:
            import json
            os.makedirs(os.path.dirname(self.STATE), exist_ok=True)
            data = dict(stereo_mode=self.stereo_mode, spec_mode=self.spec_mode, theme_i=self.theme_i, cpp=self.cpp,
                        grid=self.grid, decision=self.header.isVisible(), preset=self.preset, focus=self.focus, focus_pos=self.focus_pos,
                        w=self.width(), h=self.height())
            tmp = self.STATE + ".tmp"
            with open(tmp, "w") as f_:
                json.dump(data, f_)
            os.replace(tmp, self.STATE)                 # escrita atômica: nunca deixa o arquivo pela metade
        except Exception:
            pass

    def focusNextPrevChild(self, nxt): return False          # Tab chega no keyPressEvent

    def closeEvent(self, ev):
        if self.persist:
            self._save_state()
            self.tracklog.finish(self.wall_now())
        if self.nowplaying: self.nowplaying.close()
        self.cap.close()
        super().closeEvent(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--shot", help="renderiza offscreen com áudio demo e salva PNG")
    ap.add_argument("--secs", type=float, default=130)
    ap.add_argument("--size", default="1600x960")
    ap.add_argument("--theme", type=int, default=0)
    ap.add_argument("--stereo", type=int, default=-1)
    ap.add_argument("--focus", type=int, default=-1)
    ap.add_argument("--preset", type=int, default=-1)
    ap.add_argument("--focus-pos", default="right")
    ap.add_argument("--spec", type=int, default=-1)
    a = ap.parse_args()
    if a.shot: os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("djscope"); app.setDesktopFileName("djscope")
    W, H = map(int, a.size.split("x"))
    if a.shot:
        dm = core.Demo(128); dm.t0 = time.time() - a.secs
        cap = dm
        win = Main(core.Demo(), "demo")
        win.timer.stop()
        win.offline = True
        win.tracklog.t0 = win.wall_now()
        if a.preset >= 0: win.set_preset(a.preset)
        win.focus_pos = a.focus_pos
        if a.focus >= 0: win.focus = a.focus; win.relayout()
        win.resize(W, H); win.theme_i = a.theme
        if a.stereo >= 0: win.stereo_mode = a.stereo
        if a.spec >= 0: win.spec_mode = a.spec
        win.show(); app.processEvents()
        x = dm.get()
        for i in range(0, len(x), 800):
            win.process(x[i:i + 800])
        for _ in range(3): app.processEvents()
        win.grab().save(a.shot)
        t0 = time.process_time()
        for _ in range(20):
            win.process(np.zeros((0, 2), np.float32)); win.repaint()
        print("ok", a.shot, f"cpu/frame {(time.process_time() - t0) / 20 * 1000:.1f} ms")
        return
    src = "demo" if a.demo else (a.source or core.default_source())
    cap = core.Demo() if a.demo else core.Capture(src)
    win = Main(cap, core.short_src(src), persist=True)
    win.resize(W, H)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
