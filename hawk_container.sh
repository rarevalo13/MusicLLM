#!/bin/bash
# Run a command inside the Ubuntu 22.04 Singularity image on Hawk.
# Hawk's RHEL 7.9 (glibc 2.17) is too old for the PyTorch 2.7.1 wheels, so
# anything that imports torch/stable_audio_3 must run in this container.
# The stable-audio-3 venv goes first on PATH; GPU driver libraries are bound in
# only when they exist (GPU nodes), so the same script works on login nodes.
#
# Usage (from the repo root on Hawk):
#   ./hawk_container.sh hf auth whoami
#   ./hawk_container.sh python -c "import torch; print(torch.cuda.is_available())"
#   SA3_IMAGE=/path/to/other/image ./hawk_container.sh python ...
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${SA3_IMAGE:-$REPO/containers/ubuntu-22.04}"
VENV="$REPO/stable-audio-3/.venv"
[ -d "$IMAGE" ] || { echo "container image not found: $IMAGE" >&2; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "Python environment not found: $VENV" >&2; exit 1; }
opts=(-B /nfs)
# A job's scratch dir ($TMPDIR) is only visible inside if it's under a path that is bound anyway
case "${TMPDIR:-/tmp}" in
    /tmp | /tmp/* | /nfs | /nfs/* | "$HOME" | "$HOME"/*) ;;
    *) [ -d "$TMPDIR" ] && opts+=(-B "$TMPDIR") ;;
esac
if [ -e /usr/lib64/libcuda.so.1 ]; then
    # --nv alone doesn't make the driver visible in this image: bind the driver libraries explicitly
    opts+=(--nv)
    bind_lib() {  # bind_lib <host soname> <names inside the container...>
        local src; src="$(readlink -f "/usr/lib64/$1")"; shift
        [ -e "$src" ] || return 0
        for name in "$@"; do opts+=(-B "$src:/usr/lib/x86_64-linux-gnu/$name"); done
    }
    bind_lib libcuda.so.1 libcuda.so.1 libcuda.so
    bind_lib libnvidia-ml.so.1 libnvidia-ml.so.1 libnvidia-ml.so
    bind_lib libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so.1
    bind_lib libnvidia-nvvm.so.4 libnvidia-nvvm.so.4
fi
exec singularity -s exec "${opts[@]}" "$IMAGE" \
    env PATH="$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt "$@"
