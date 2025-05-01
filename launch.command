#!/bin/bash

cd "$(dirname "$0")"

# Activate virtual environment
source faceid-env/bin/activate || {
  echo "Failed to activate virtual environment."
  read -p "Press Enter to close..."
  exit 1
}

# Start Streamlit backend silently
streamlit run app.py --server.headless true &

# Wait for server to boot
sleep 5

# Open the Nativefier app without opening its folder
open -n "/Users/pouriaasadi/Real-Time Face ID/Real-Time Face ID.app"

# Keep terminal open
read -p "Press Enter to exit this terminal..."




