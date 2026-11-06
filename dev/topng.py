#!/usr/bin/env python3
"""Renderiza um frame do djscope (modo demo) para PNG, célula a célula. Uso: topng.py out.png [COLSxROWS] [segundos]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import djscope as d

out = sys.argv[1]
C, R = map(int, (sys.argv[2] if len(sys.argv) > 2 else "150x44").split("x"))
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 24
# alimenta o engine com áudio sintético "pré-gravado" (não depende de tempo real)
dm = d.Demo(128); dm.t0 = time.time() - secs
x = dm.get(); e = d.Engine(); v = d.View()
for i in range(0, len(x), 480):
    e.feed(x[i:i + 480])
    if (i // 480) % 40 == 0: e.estimate()
for _ in range(3): e.estimate()
v.theme_i = int(sys.argv[4]) if len(sys.argv) > 4 else 0
for _ in range(25): cv = d.render(v, e, C, R, "demo")
CW, CH = 10, 20
font = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", 16)
img = Image.new("RGB", (C * CW, R * CH))
px = ImageDraw.Draw(img)
fg = np.clip(cv.fg + .5, 0, 255).astype(int); bg = np.clip(cv.bg + .5, 0, 255).astype(int)
EIGHTHS = " ▁▂▃▄▅▆▇█"
for r in range(R):
    for c in range(C):
        x0, y0 = c * CW, r * CH
        px.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1], fill=tuple(bg[r, c]))
        ch = cv.ch[r, c]
        if ch == " ": continue
        f = tuple(fg[r, c])
        if ch == "▀": px.rectangle([x0, y0, x0 + CW - 1, y0 + CH // 2 - 1], fill=f)
        elif ch in EIGHTHS[1:]:
            h = EIGHTHS.index(ch) * CH // 8
            px.rectangle([x0, y0 + CH - h, x0 + CW - 1, y0 + CH - 1], fill=f)
        elif ch == "▔": px.rectangle([x0, y0, x0 + CW - 1, y0 + 2], fill=f)
        else: px.text((x0, y0 + 1), ch, font=font, fill=f)
img.save(out)
print("ok", img.size)
