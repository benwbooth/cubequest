#!/bin/bash

# Download Phoenix MT-32 SoundFont
OUTPUT_FILE="music.sf2"
URL="https://musical-artifacts.com/artifacts/1481/Phoenix_MT-32.sf2"

echo "Downloading Phoenix MT-32 SoundFont..."

if command -v curl &> /dev/null; then
    curl -fL -o "$OUTPUT_FILE" "$URL"
else
    wget -O "$OUTPUT_FILE" "$URL"
fi

if [ -f "$OUTPUT_FILE" ] && head -c 4 "$OUTPUT_FILE" | grep -q "RIFF"; then
    echo ""
    echo "Downloaded: $OUTPUT_FILE ($(du -h "$OUTPUT_FILE" | cut -f1))"
    echo "Run the game with: cargo run --release"
else
    echo "Download failed"
    exit 1
fi
