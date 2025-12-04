#!/bin/bash

# Roland SC-55 SoundFont (General MIDI compatible)
SOUNDFONT="SC55_zzdenis_v0.5.sf2"
URL="https://drive.google.com/file/d/1wdYpwoCka8r7ZuIzPn1CHy13_4aP9oSq/view"

if [ -f "$SOUNDFONT" ]; then
    echo "SoundFont already exists: $SOUNDFONT"
    echo "Run the game with: cargo run --release"
    exit 0
fi

echo "Opening download page for SC-55 SoundFont..."
echo ""
echo "Please download the file and save it to:"
echo "  $(pwd)/$SOUNDFONT"
echo ""

# Open URL in default browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$URL"
elif command -v xdg-open &> /dev/null; then
    xdg-open "$URL"
else
    echo "Could not open browser. Please visit:"
    echo "  $URL"
fi

echo "After downloading, run the game with: cargo run --release"
