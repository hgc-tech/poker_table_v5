# v5 poker bots: run, train and play

This folder holds only the code. You supply `cfr_config.txt`, `abstraction/`, `checkpoints/` and `blueprints/`.

## 1. Install

Copy the zip to the machine and unzip it on the local disk (not a network home folder):

```
scp ~/Downloads/poker_table_v5_min.zip USER@HOST:/var/tmp/USER/
ssh USER@HOST
cd /var/tmp/USER && unzip -n poker_table_v5_min.zip && bash poker_table_v5/setup.sh
```

`setup.sh` checks for g++, python3-venv and python3-dev, creates a private Python environment in `.venv`, installs numpy, rich, CPU-only torch, cython, eval7 and pytest into it, and compiles the C++ engine. It changes nothing outside the folder. If it reports something missing, install it (Ubuntu: `sudo apt install build-essential python3-venv python3-dev`) and run it again.

## 2. Add your files

Put these inside `poker_table_v5/`, next to `train_cfr.py`:

```
cfr_config.txt
abstraction/
checkpoints/
blueprints/
```

The folder names must match `abstraction_dir`, `checkpoint_dir` and `blueprint_dir` in `cfr_config.txt` (relative paths are relative to that file).

## 3. Every new terminal

```
cd /var/tmp/USER/poker_table_v5 && source .venv/bin/activate
```

This makes `python` use the private environment. Without it you get "No module named torch" or "externally-managed-environment".

## 4. Train

Run inside tmux so training survives a closed ssh session:

```
tmux new -s train
cd /var/tmp/USER/poker_table_v5 && source .venv/bin/activate
python train_cfr.py --memory     # RAM per blueprint and the bucket level each uses
python train_cfr.py              # train all blueprints, round robin, until Ctrl-C
```

Detach with Ctrl-b then d; reattach with `tmux attach -t train`.

- Stop: press Ctrl-C once and wait; the current blueprint saves and exports.
- Resume: run the same command; every blueprint continues from `checkpoints/`.
- Progress: `python train_cfr.py --status`.
- Variants: `--hours 72`, `--only 6x20 3x10-25-25`, `--threads 12`, `--slice 60`.
- Rebuild the abstraction (only if you change bucket counts): `python train_cfr.py --abstraction`.

Unequal-stack blueprints come from `profiles@N` lines in `cfr_config.txt`, one stack per position, button first, for example `profiles@3 = 10-25-25 25-10-25 25-25-10`. Adding profiles does not affect existing checkpoints. Changing blueprint sizes, payouts or bucket counts does: delete `checkpoints/` and `blueprints/` and retrain.

## 4b. Train one blueprint at a time on a server

Keep your full set of files on your own machine and send the server one blueprint at a time. Run `transfer.sh` on your own machine, from your `poker_table_v5` folder (copy `transfer.sh` there if needed). `SERVER` stands for `USER@HOST:/var/tmp/USER/poker_table_v5`.

```
bash transfer.sh setup SERVER          # once: cfr_config.txt and abstraction/ (0.6 GB)
bash transfer.sh push SERVER 6x100     # cfr_config.txt and checkpoints/p6_s100.ckpt
```

On the server, in tmux:

```
cd /var/tmp/USER/poker_table_v5 && source .venv/bin/activate
python train_cfr.py --only 6x100 --hours 48
```

Training with `--only` needs only that blueprint's checkpoint; the other blueprints' files may be absent. With no checkpoint it starts from scratch. Back on your own machine:

```
bash transfer.sh pull SERVER 6x100     # checkpoints/p6_s100.ckpt and blueprints/p6_s100.strat
```

Several names may be given at once (`push SERVER 6x100 6x60`). Copies use rsync and replace a file only with a newer one, so a pull never overwrites a newer checkpoint on your machine. Keep `cfr_config.txt` and `abstraction/` identical on both machines (run `setup` again after changing either), and do not train the same blueprint on both machines at once. If the server lacks `abstraction/`, `train_cfr.py` starts building one, which takes hours; stop it with Ctrl-C and run `setup`.

## 5. Play

```
python play.py                    # you vs five v5 bots
python play.py --search-ms 1500   # longer search per decision
python play.py --diagnostic       # all cards face up, and each bot's plan
python play.py --no-search        # blueprints only (fastest)
```

## 6. Measure

```
python train_cfr.py --lbr                                   # exploitability of the heads-up blueprints
python snapshot.py day1                                     # snapshot the current blueprints
python compare.py --opponent snapshot --against snapshots/day1 --tournaments 300 --workers 4
python compare.py --opponent v4 --v4 /path/to/poker_table --tournaments 300 --workers 4
python compare.py --opponent check_call --tournaments 100   # fixed sparring bots
```

An average seat earns 16.7% of the prize pool; v5 is stronger when its share is above that and the interval excludes it. The v3 neural opponents (`--opponent neural`, `play.py --bots neural`) need the old `weights/` folder, which this package leaves out.

## 7. Check the install

```
python -m pytest -q tests
```

The strategy tests use your blueprints and fail if they are undertrained; the other tests check the code.
