#!/usr/bin/env bash
# Instala o DJScope no usuário atual: venv local + comandos `djscope` / `djscope-tui` + entrada no menu de apps.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

for cmd in python3 parec pactl; do
  command -v "$cmd" >/dev/null 2>&1 || echo "aviso: '$cmd' não encontrado (necessário p/ captar o áudio do sistema: PipeWire/PulseAudio)" >&2
done
command -v busctl >/dev/null 2>&1 || echo "aviso: 'busctl' não encontrado — o título da faixa (MPRIS) não será lido (opcional)" >&2

python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

BIN="$HOME/.local/bin"; APPS="$HOME/.local/share/applications"
mkdir -p "$BIN" "$APPS"
printf '#!/bin/sh\nexec "%s/.venv/bin/python" "%s/djscope_gui.py" "$@"\n' "$DIR" "$DIR" > "$BIN/djscope"
printf '#!/bin/sh\nexec "%s/.venv/bin/python" "%s/djscope.py" "$@"\n' "$DIR" "$DIR" > "$BIN/djscope-tui"
chmod +x "$BIN/djscope" "$BIN/djscope-tui"
cat > "$APPS/djscope.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=DJScope
Comment=Análise de áudio ao vivo para DJs (waveform, espectro, key, LUFS, stereo)
Exec=$BIN/djscope
Icon=audio-card
Terminal=false
Categories=AudioVideo;Audio;
StartupWMClass=djscope
DESKTOP

echo "pronto. rode:  djscope        (ou: djscope --demo  para testar sem música)"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "obs: adicione $BIN ao seu PATH";; esac
