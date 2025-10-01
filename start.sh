#!/bin/bash
# Force execution using the Python 3.11 binary installed via packages.txt
# This ensures the environment variables and paths are set correctly for the older torch version.
python3.11 -m streamlit run app.py