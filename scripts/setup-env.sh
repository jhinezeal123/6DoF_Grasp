#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="$root/.venv"

if [[ -n "${CONDA_EXE:-}" ]]; then
  conda_bin="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
  conda_bin="$(command -v conda)"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
  conda_bin="$HOME/miniforge3/bin/conda"
else
  echo "Conda was not found. Set CONDA_EXE to its executable path." >&2
  exit 1
fi

if [[ -e "$env_dir" && ! -d "$env_dir/conda-meta" ]]; then
  echo "$env_dir exists but is not a Conda environment." >&2
  exit 1
fi

if [[ -d "$env_dir/conda-meta" ]]; then
  "$conda_bin" env update --prefix "$env_dir" --file "$root/environment.yml" --yes
else
  "$conda_bin" env create --prefix "$env_dir" --file "$root/environment.yml" --yes
fi

"$env_dir/bin/python" -m pip install --no-deps --no-build-isolation -e "$root"
"$env_dir/bin/python" -m pip check
printf '6DoF environment ready: %s\n' "$env_dir"
