#!/usr/bin/env python3
"""Verificações da interface (offscreen): presets, barra de decisão, modo foco, atalhos e persistência.

    .venv/bin/python tests/check_ui.py

Sai com código 1 se algo falhar. Não escreve nada na sua pasta de sets nem no seu estado salvo.
"""
import json, os, sys, tempfile, time

os.environ["QT_QPA_PLATFORM"] = "offscreen"                      # força offscreen mesmo que o ambiente use wayland
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["DJSCOPE_SETS_DIR"] = tempfile.mkdtemp(prefix="djscope-check-sets-")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import djscope_gui as g              # importa antes do numpy: fixa o BLAS em 1 thread
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
import djscope as core

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FALHA ") + msg, flush=True)
    if not cond: failures.append(msg)


g.Main.STATE = os.path.join(tempfile.mkdtemp(prefix="djscope-check-state-"), "state.json")
app = QtWidgets.QApplication([])
win = g.Main(core.Demo(), "demo", persist=True)
win.timer.stop(); win.offline = True; win.resize(1900, 1034); win.show(); app.processEvents()
x = core.Demo(128).render(np.arange(40 * 48000) / 48000)
for i in range(0, len(x), 800): win.process(x[i:i + 800])


def settle():
    win.glayout.activate(); app.processEvents()
    for c in win.cards: c.grab()
    win.header.grab()


def click(widget, rect):
    QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, QPoint(int(rect.center().x()), int(rect.center().y())))
    app.processEvents()


settle()
print("== presets")
check(win.preset == 0 and win.spec_mode == 3 and win.stereo_mode == 0, "abre no preset MIX (espectro em KEY)")
check(win.tracks_card.isVisible() and win.wave.width() > win.width() * 0.9, "MIX: SET visível e waveform em largura total")
QTest.keyClick(win, Qt.Key_F2); settle()
check(win.preset == 1 and win.spec_mode == 0 and not win.tracks_card.isVisible(), "F2 -> MASTER (espectro em CURVE, SET oculto)")
QTest.keyClick(win, Qt.Key_F3); settle()
check(win.preset == 2 and win.spec_mode == 2 and win.stereo_mode == 2, "F3 -> ANÁLISE (espectro 3D, stereo FIELD)")
check(win.wave.height() < win.spectrum.height() / 2, "ANÁLISE: waveform vira uma faixa fina")
QTest.keyClick(win, Qt.Key_F1); settle()
check(win.preset == 0 and win.tracks_card.isVisible(), "F1 -> MIX de volta (SET visível de novo)")
settle()
pill = [r for r, i in win.header._pills if i == 1][0]
click(win.header, pill)
check(win.preset == 1, "clicar na pílula MASTER da barra de decisão troca o preset")
QTest.keyClick(win, Qt.Key_F1); settle()

print("== barra de decisão")
check(win.header.isVisible() and win.header.height() == g.DecisionBar.H, "visível por padrão, no topo")
QTest.keyClick(win, Qt.Key_H); settle()
check(not win.header.isVisible(), "H oculta a barra")
QTest.keyClick(win, Qt.Key_H); settle()
check(win.header.isVisible(), "H mostra a barra de novo")
check(win.header.grab().width() > 100, "a barra renderiza")

print("== botões de modo cabem quando há espaço (regressão: sumiam em cards médios)")
win.resize(1700, 1000); settle()
check(len(win.stereo._chip_rects) == 3, f"stereo a {win.stereo.width()} px mostra os 3 botões (GONIO/POLAR/FIELD)")
check(len(win.spectrum._chip_rects) == 4, f"espectro a {win.spectrum.width()} px mostra os 4 botões")
click(win.stereo, [r for r, i in win.stereo._chip_rects if i == 2][0]); check(win.stereo_mode == 2, "clique no botão FIELD muda o modo")
click(win.spectrum, [r for r, i in win.spectrum._chip_rects if i == 1][0]); check(win.spec_mode == 1, "clique no botão BARS muda o modo")
win.resize(1900, 1034); settle()

print("== modo foco")
QTest.mouseDClick(win.spectrum, Qt.LeftButton, Qt.NoModifier, QPoint(300, 200)); app.processEvents(); settle()
check(win.focus == 1 and all(c.isVisible() for c in win.cards), "duplo clique foca e mostra todos os cards (mesmo os ocultos pelo preset)")
check(win.header.isVisible(), "a barra de decisão continua visível em foco")
time.sleep(0.5)
QTest.mouseClick(win.stereo, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100)); app.processEvents()
check(win.focus == 2, "clique numa miniatura promove o card")
QTest.keyClick(win, Qt.Key_L); settle()
check(win.focus_pos == "bottom" and win.stereo.width() > win.width() * 0.9 and win.wave.geometry().top() > win.stereo.geometry().bottom(),
      "L move as miniaturas para uma tira embaixo do card principal (que ocupa a largura toda)")
QTest.keyClick(win, Qt.Key_Escape); settle()
check(win.focus is None and win.preset == 0 and win.wave.width() > win.width() * 0.9 and win.tracks_card.isVisible(), "Esc volta ao layout do preset MIX")
for f in range(7):
    for sm in range(3):
        for pm in range(4):
            win.focus = f; win.relayout(); win.stereo_mode, win.spec_mode = sm, pm
            win.process(x[:800]); [c.grab() for c in win.cards]
check(True, "7 cards em foco x 12 combinações de modo renderizam sem erro")
QTest.keyClick(win, Qt.Key_G); QTest.keyClick(win, Qt.Key_3); QTest.keyClick(win, Qt.Key_G); QTest.keyClick(win, Qt.Key_0)
check(isinstance(win.grid, bool) and win.focus is None, "G (beatgrid) não quebra o layout")

print("== persistência")
win.set_preset(2); win.stereo_mode = 1; win.spec_mode = 3; win.focus, win.focus_pos = 3, "bottom"; win.theme_i = 1
win.close()
s = json.load(open(g.Main.STATE))
check(s["preset"] == 2 and s["focus"] == 3 and s["focus_pos"] == "bottom", "estado salvo (preset, foco, posição das miniaturas)")
w2 = g.Main(core.Demo(), "demo", persist=True); w2.timer.stop()
check((w2.preset, w2.focus, w2.focus_pos, w2.stereo_mode, w2.spec_mode, w2.theme_i) == (2, 3, "bottom", 1, 3, 1), "reabre com o mesmo estado")

print()
if failures:
    print(f"{len(failures)} verificação(ões) falharam"); sys.exit(1)
print("todas as verificações passaram")
