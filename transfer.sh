#!/usr/bin/env bash
# Copy single blueprints between this machine and a training server.
# Run it on your own machine, from your poker_table_v5 folder.
#
#   bash transfer.sh push USER@HOST:/var/tmp/USER/poker_table_v5 6x100
#   bash transfer.sh pull USER@HOST:/var/tmp/USER/poker_table_v5 6x100
#   bash transfer.sh setup USER@HOST:/var/tmp/USER/poker_table_v5
#
# setup  copies cfr_config.txt and abstraction/ to the server (needed once,
#        and again whenever you change either).
# push   copies cfr_config.txt and the blueprint's checkpoint to the server.
# pull   copies the blueprint's checkpoint and exported strategy back here.
# Several blueprints may be named at once: "push HOST:DIR 6x100 6x60 3x10-25-25".
# A blueprint name is PLAYERSxSTACK as in "train_cfr.py --only"; the files are
# checkpoints/pPLAYERS_sSTACK.ckpt and blueprints/pPLAYERS_sSTACK.strat.
#
# Files are only replaced by newer ones (rsync --update), so a pull cannot
# overwrite a checkpoint here that is newer than the server's copy. Do not
# train the same blueprint on both machines at once.
set -euo pipefail
cd "$(dirname "$0")"

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -ge 2 ] || usage
cmd="$1"; remote="${2%/}"; shift 2

if command -v rsync >/dev/null; then
  copy() { rsync -a --update --partial --progress "$@"; }
else
  echo "(rsync not found; using scp, which always overwrites)"
  copy() { scp -r "$@"; }
fi

host="${remote%%:*}"; rdir="${remote#*:}"
remote_mkdir() {
  if [ "$host" != "$remote" ]; then ssh "$host" "mkdir -p '$rdir/checkpoints' '$rdir/blueprints'"
  else mkdir -p "$rdir/checkpoints" "$rdir/blueprints"; fi
}

file_name() {   # 6x100 -> p6_s100
  local b="$1"
  [[ "$b" =~ ^([2-6])x([0-9.]+(-[0-9.]+)*)$ ]] || { echo "bad blueprint name '$b' (expected e.g. 6x100 or 3x10-25-25)" >&2; exit 1; }
  echo "p${BASH_REMATCH[1]}_s${BASH_REMATCH[2]}"
}

case "$cmd" in
  setup)
    [ -f cfr_config.txt ] && [ -f abstraction/meta.txt ] || { echo "Run this from the folder with cfr_config.txt and abstraction/"; exit 1; }
    remote_mkdir
    copy cfr_config.txt "$remote/"
    copy abstraction "$remote/"
    ;;
  push)
    [ $# -ge 1 ] || usage
    remote_mkdir
    copy cfr_config.txt "$remote/"
    for b in "$@"; do
      f="$(file_name "$b")"
      if [ -f "checkpoints/$f.ckpt" ]; then copy "checkpoints/$f.ckpt" "$remote/checkpoints/"
      else echo "$b: no checkpoints/$f.ckpt here; the server will start it from scratch"; fi
    done
    echo "Now on the server:  python train_cfr.py --only $*"
    ;;
  pull)
    [ $# -ge 1 ] || usage
    mkdir -p checkpoints blueprints
    for b in "$@"; do
      f="$(file_name "$b")"
      copy "$remote/checkpoints/$f.ckpt" checkpoints/
      copy "$remote/blueprints/$f.strat" blueprints/
    done
    ;;
  *) usage ;;
esac
echo "Done."
