#!/usr/bin/env python3
"""Gera os screenshots do README (offscreen, com áudio sintético: nada de áudio real nas imagens).

    .venv/bin/python dev/make_screenshots.py
"""
import os, sys, tempfile
os.environ["QT_QPA_PLATFORM"] = "offscreen"                 # força offscreen mesmo que o ambiente use wayland
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["DJSCOPE_SETS_DIR"] = tempfile.mkdtemp(prefix="djscope-shots-")   # não escreve na sua pasta de sets
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import djscope_gui as g                      # importa antes do numpy: fixa o BLAS em 1 thread
import numpy as np
from PySide6 import QtWidgets
import djscope as core

OUT = os.path.join(ROOT, "docs", "screenshots")
os.makedirs(OUT, exist_ok=True)
app = QtWidgets.QApplication([])
win = g.Main(core.Demo(), "demo")
win.timer.stop(); win.offline = True; win.tracklog.t0 = win.wall_now()
win.resize(1900, 1034); win.show(); app.processEvents()

# um "set" de 4 faixas sintéticas (andamento e tonalidade diferentes); a troca vem do título, como no MPRIS
TRACKS = [("Sunrise Loop", "Demo Artist", 128, 0), ("Midnight Run", "Demo Artist", 126, 2),
          ("Deep Cut", "Demo Artist", 124, 7), ("Closer", "Demo Artist", 130, -3)]
for title, artist, bpm, shift in TRACKS:
    x = core.Demo(bpm, key_shift=shift).render(np.arange(62 * 48000) / 48000)
    pos = [0.0]

    class NP:                                   # "player" falso: título, duração e posição como no MPRIS
        info = property(lambda self: {"status": "Playing", "title": title, "artist": artist, "length": 192.0, "position": pos[0]})
        def close(self): pass
    win.nowplaying = NP()
    for j in range(0, len(x), 800):
        pos[0] = j / 48000
        win.process(x[j:j + 800])
    print("processado:", title, flush=True)


def settle(n=3):
    win.glayout.activate(); app.processEvents()
    for _ in range(n): win.process(np.zeros((0, 2), np.float32))


def shot(name, widget=None):
    settle()
    (widget or win).grab().save(os.path.join(OUT, name))
    print("salvo", name, flush=True)


# visão geral = preset MIX (espectro em KEY); a barra de decisão aparece no topo
win.set_preset(0)
shot("overview.png")
shot("decision_bar.png", win.header)
# cards individuais
shot("card_waveform.png", win.wave)
shot("card_spectrum_curve.png", win.spectrum)
win.spec_mode = 1; shot("card_spectrum_bars.png", win.spectrum)
win.spec_mode = 2; shot("card_spectrum_3d.png", win.spectrum)
win.spec_mode = 3; shot("card_key.png", win.spectrum)
win.spec_mode = 0
shot("card_set.png", win.tracks_card)
shot("card_levels.png", win.peak)
for m, name in ((0, "gonio"), (1, "polar"), (2, "field")):
    win.stereo_mode = m; shot(f"card_stereo_{name}.png", win.stereo)
# presets
win.set_preset(1); shot("preset_master.png")
win.set_preset(2); shot("preset_analise.png")
win.set_preset(0)
# modo foco
win.stereo_mode = 0; win.focus, win.focus_pos = 2, "bottom"; win.relayout(); shot("focus_stereo_bottom.png")
win.spec_mode = 3; win.focus, win.focus_pos = 1, "right"; win.relayout(); shot("focus_key_right.png")
win.spec_mode = 2; win.focus, win.focus_pos = 1, "bottom"; win.relayout(); shot("focus_3d_bottom.png")
win.stereo_mode = 2; win.focus, win.focus_pos = 2, "right"; win.relayout(); shot("focus_field_right.png")
print("ok")
