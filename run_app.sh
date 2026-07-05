#!/usr/bin/env bash
# Launch the PD-Agent Streamlit demo with an environment that has torch +
# transformers installed. The default `streamlit` on PATH (base conda) lacks
# torch, so we call the interpreter that has the pipeline dependencies.
set -e
cd "$(dirname "$0")"

# Env known to have torch + transformers + networkx + matplotlib + streamlit.
PYENV="/opt/conda/envs/tracecal_rl/bin"

if [ ! -x "$PYENV/streamlit" ]; then
    echo "ERROR: $PYENV/streamlit not found."
    echo "Install into a torch env, e.g.:"
    echo "  uv pip install --python <env>/bin/python streamlit"
    exit 1
fi

exec "$PYENV/streamlit" run app.py "$@"
