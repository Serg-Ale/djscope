#!/usr/bin/env python3
"""Verificações do motor de análise com áudio sintético de verdade-terreno conhecida.

    .venv/bin/python tests/check_engine.py

Cobre: tonalidade (Camelot), contagem de compasso/frase, BPM, true peak, mono do grave e a lógica da tracklist.
Sai com código 1 se algo falhar. Leva ~1–2 min.
"""
import collections, os, sys, tempfile, time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import djscope_gui as g          # importa antes do numpy (fixa o BLAS em 1 thread) e fornece StereoAnalyzer
import numpy as np
import djscope as d

SR = 48000
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FALHA ") + msg, flush=True)
    if not cond: failures.append(msg)


def engine_on(x, estimate_every=30):
    e = d.Engine()
    for i in range(0, len(x), 800):
        e.feed(x[i:i + 800])
        if (i // 800) % estimate_every == 0: e.estimate()
    return e


print("== tonalidade (Demo está em Lá menor = 8A; key_shift transpõe em semitons)")
for shift in (0, 2, 3, 5, 7, -2):
    x = d.Demo(128, key_shift=shift).render(np.arange(45 * SR) / SR)
    e = engine_on(x)
    ki = e.key_info()
    want = d.key_name(12 + (9 + shift) % 12)
    check(ki is not None and ki["idx"] == 12 + (9 + shift) % 12,
          f"transposição {shift:+d}: detectou {ki['name']} {ki['camelot']}, esperado {want[0]} {want[1]}")

print("== BPM")
for bpm in (100, 120, 124, 128, 140):
    x = d.Demo(bpm).render(np.arange(38 * SR) / SR)
    e = d.Engine(); got = []
    for i in range(0, len(x), 800):
        e.feed(x[i:i + 800])
        if (i // 800) % 30 == 0:
            e.estimate()
            if e.bpm and i / SR > 18: got.append(e.bpm)
    m = float(np.median(got))
    check(abs(m - bpm) < 0.6, f"{bpm} bpm real -> {m:.2f}")

print("== compasso e frase (o 1 do compasso e o início da frase de 8 compassos)")
for bpm in (120, 128, 140):
    beat = 60 / bpm
    P_true = beat * 120                                    # unidades de env por batida
    x = d.Demo(bpm).render(np.arange(130 * SR) / SR)
    e = d.Engine(); events = []
    orig = d.PhraseTracker._beat
    def rec(self, orig=orig, events=events):
        orig(self)
        events.append((self.last_beat, self.beat_idx, self.bar_phase, self.phrase_phase))
    d.PhraseTracker._beat = rec
    try:
        for i in range(0, len(x), 800):
            e.feed(x[i:i + 800])
            if (i // 800) % 30 == 0: e.estimate()
    finally:
        d.PhraseTracker._beat = orig
    late = [ev for ev in events if ev[0] / 120 > 70]
    bar_off, phr_off = collections.Counter(), collections.Counter()
    for lb, idx, bp, pp in late:
        tb = round(lb / P_true)
        bar_off[(tb % 4 - (idx - bp) % 4) % 4] += 1
        phr_off[(((tb // 4) % 8) - (((idx - bp) // 4) - pp) % 8) % 8] += 1
    check(len(late) > 40 and set(bar_off) == {0}, f"{bpm} bpm: '1' do compasso certo em {bar_off.get(0, 0)}/{len(late)} batidas")
    check(len(late) > 40 and set(phr_off) == {0}, f"{bpm} bpm: início da frase certo em {phr_off.get(0, 0)}/{len(late)} batidas")

print("== true peak (seno em fs/4 com fase de 45°: amostras = ±0,707, pico verdadeiro = 1,0)")
t = np.arange(SR)
x = np.sin(2 * np.pi * (SR / 4) * t / SR + np.pi / 4)
st = np.stack([x, x], 1).astype(np.float32)
e = d.Engine(); e.feed(st)
check(abs(e.max_peak + 3.01) < 0.3, f"pico de amostra {e.max_peak:.2f} dBFS (esperado -3,01)")
check(abs(e.tp_max) < 0.4, f"true peak {e.tp_max:.2f} dBTP (esperado ~0)")
e = d.Engine(); e.feed((1.2 * st).astype(np.float32))
check(e.tp_max > 1.0 and e.clips >= 1, f"acima de 0 dBTP detectado: {e.tp_max:.2f} dBTP, clips={e.clips}")

print("== mono do grave (<120 Hz)")
tt = np.arange(SR * 2) / SR
for phase, want_w, want_loss in ((0, (0, 5), (0, 0.5)), (90, (40, 60), (2.5, 3.5)), (180, (95, 100.1), (20, 30.1))):
    sa = g.StereoAnalyzer()
    ring = np.stack([0.5 * np.sin(2 * np.pi * 60 * tt), 0.5 * np.sin(2 * np.pi * 60 * tt + np.deg2rad(phase))], 1).astype(np.float32)
    for _ in range(80): sa.analyze(ring[-8192:])
    w, loss = sa.bass_width * 100, sa.bass_loss
    check(want_w[0] <= w <= want_w[1] and want_loss[0] <= loss <= want_loss[1],
          f"grave com fase {phase:3d}°: lateral {w:5.1f}% / perda no mono {loss:4.1f} dB")

print("== tracklist (relógio virtual)")
e = d.Engine()
tl = d.TrackLog(e, outdir=tempfile.mkdtemp(prefix="djscope-check-")); tl.t0 = 1000.0
e.rms_db = -20
A = {"status": "Playing", "title": "Faixa A", "artist": "X"}
B = {"status": "Playing", "title": "Faixa B", "artist": "Y"}
e.bpm, e.conf = 128.0, 0.5
for i in range(60): tl.tick(A, now=1000.0 + i)
tl.tick(B, now=1061.0)
check(len(tl.tracks) == 1 and tl.cur["title"] == "Faixa B" and tl.cur["reason"] == "título", "troca de título cria uma faixa nova")
for i in range(40): tl.tick(B, now=1061.0 + i)
e.resumed = True; tl.tick(B, now=1110.0)
check(len(tl.tracks) == 2 and tl.cur["reason"] == "silêncio", "silêncio seguido de retomada cria uma faixa nova")
tl.src_title = None
for i in range(50): e.bpm, e.conf = 128.0, 0.5; tl.tick(None, now=1111.0 + i)
n0 = len(tl.tracks)
for i in range(30): e.bpm, e.conf = 140.0, 0.5; tl.tick(None, now=1170.0 + i)
check(len(tl.tracks) == n0 + 1 and tl.cur["reason"] == "BPM", "salto sustentado de BPM (128 -> 140) cria uma faixa nova")
tl2 = d.TrackLog(e, outdir=tempfile.mkdtemp(prefix="djscope-check-")); e.rms_db = -20; tl2.tick(None, now=0.0)
for i in range(50): e.bpm, e.conf = 128.0, 0.5; tl2.tick(None, now=1.0 + i)
for i in range(30): e.bpm, e.conf = 64.0, 0.5; tl2.tick(None, now=60.0 + i)
check(len(tl2.tracks) == 0, "meio-tempo (128 -> 64) NÃO cria faixa nova")
tl.finish(now=1400.0)
check(os.path.exists(tl.path) and open(tl.path, encoding="utf-8").read().count("\n") >= 4, "CSV do set foi salvo")

print("== linha do tempo da faixa (perfil de loudness + posição do player)")
e = d.Engine(); tl = d.TrackLog(e, outdir=tempfile.mkdtemp(prefix="djscope-check-")); tl.t0 = 0.0; e.rms_db = -20
info = {"status": "Playing", "title": "P", "artist": "Q", "length": 200.0, "position": 37.0}
tl.tick(info, now=0.0)
for i in range(1, 31):
    e.lufs_m = -14.0 - 0.1 * i
    tl.tick(info, now=float(i))
snap = tl.snapshot(now=30.0)
check(abs(snap["pos0"] - 37.0) < 1e-6, "posição inicial vem do player (alinha a linha do tempo com o que já tinha tocado)")
check(25 <= len(snap["prof"]) <= 31 and snap["prof"][0] > snap["prof"][-1], f"perfil de LUFS-M amostrado a 1 Hz ({len(snap['prof'])} pontos)")
tl.boundary("manual", info, 31.0)
check(len(tl.snapshot(now=32.0)["prof"]) <= 1, "o perfil reinicia na faixa nova")

print()
if failures:
    print(f"{len(failures)} verificação(ões) falharam"); sys.exit(1)
print("todas as verificações passaram")
