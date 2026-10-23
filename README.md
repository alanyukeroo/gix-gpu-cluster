# GIX GPU Cluster

Train your models on **NVIDIA RTX 5090** GPUs from your own laptop.
Send a job, close your laptop, come back to the results.

```
your laptop ──gix──► GIX cluster queue ──► free GPU ──► results in your cluster folder
```

- No server login, no Linux admin skills needed. Everything happens with the `gix` command on your laptop.
- If all GPUs are busy, your job **waits in line and starts by itself**.
- Works on **macOS and Linux**. Windows is supported but still in beta: tell your TA if something doesn't work.

**Contents:**
[1. Before you start](#1-before-you-start) ·
[2. Install gix](#2-install-gix-one-time) ·
[3. Connect your account](#3-connect-your-account-one-time) ·
[4. Your first job](#4-your-first-job-5-minutes) ·
[5. Your own project](#5-run-your-own-project) ·
[Cheat sheet](#cheat-sheet) ·
[Good to know](#good-to-know) ·
[Troubleshooting](#troubleshooting)

---

## 1. Before you start

You need:

| | How to check |
|---|---|
| A **UW NetID** | You use it for UW email |
| The **UW network**: on campus, or **Husky OnNet** VPN when off campus | |
| **Python 3.9 or newer** | `python3 --version` (Windows: `py --version`) |
| **git** | `git --version` |

Missing Python or git?

| | Install |
|---|---|
| macOS | `xcode-select --install` (gives you git), Python from [python.org](https://www.python.org/downloads/) |
| Windows | `winget install Python.Python.3.12 Git.Git`, then open a new terminal |
| Linux | `sudo apt install python3-pip git` |

> **Terminal** = the Terminal app on macOS, **PowerShell** on Windows.

## 2. Install gix (one time)

macOS / Linux:

```sh
python3 -m pip install "git+https://github.com/alanyukeroo/gix-gpu-cluster#subdirectory=client"
```

Windows:

```powershell
py -m pip install "git+https://github.com/alanyukeroo/gix-gpu-cluster#subdirectory=client"
```

Check it:

```sh
gix --version
```

`gix: command not found`? Close the terminal, open a new one and try again.
Still not found (common on Windows)? Use `py -m gix_cli` instead of `gix`, e.g. `py -m gix_cli setup`
(macOS/Linux: `python3 -m gix_cli setup`).

## 3. Connect your account (one time)

```sh
gix setup
```

1. Type your **NetID** (e.g. `jdoe`).
2. Type your **NetID password** when asked. This is the only time.

You should see:

```
Added gix-login and gix-gpu to /Users/jdoe/.ssh/config
Copying your key to the cluster. Enter your NetID password:
jdoe@10.155.32.7's password:

NetID:      jdoe
NAS folder: /mnt/qnap/jdoe

Done. Try:  gix status
```

`NAS folder` is **your folder on the cluster**. Your projects, data and results live there.

---

## 4. Your first job (5 minutes)

Train an image classifier on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
(60,000 small photos of planes, cars, cats, …) to check that everything works.

**Step 1. Get the example** (downloads this repo to your laptop):

```sh
git clone https://github.com/alanyukeroo/gix-gpu-cluster.git
cd gix-gpu-cluster/examples
```

**Step 2. Upload it** to your cluster folder:

```sh
gix upload cifar10
```

**Step 3. Start the job:**

```sh
gix submit cifar10/train.sbatch
```

```
Job 149 submitted (cifar10, shard:1)
Queue position: 1 of 1. It starts by itself when a GPU is free.
Output: /mnt/qnap/jdoe/cifar10/slurm-149.out
```

Remember the job number (here **149**).

**Step 4. Watch it run** (use your job number):

```sh
gix logs 149 -f
```

```
GPU: NVIDIA GeForce RTX 5090
downloading CIFAR-10 train set...
epoch  1/30  loss 1.467  test accuracy 57.5%  (1s)
epoch  2/30  loss 1.283  test accuracy 65.0%  (1s)
...
epoch 30/30  loss 0.674  test accuracy 91.4%  (1s)
done: final test accuracy 91.4%, model saved to outputs/model.pt
== job 149 has ended
```

The first run takes a few minutes longer: the cluster downloads the PyTorch image and the dataset once.
You can press `Ctrl+C` any time. That only stops watching; the job keeps running.

**Step 5. Get the results** to your laptop:

```sh
gix download cifar10/outputs
```

You now have `outputs/model.pt` (the trained model) and `outputs/metrics.json` (accuracy per epoch). 🎉

---

## 5. Run your own project

### a. Prepare the folder on your laptop

```
my-project/
├── train.py             ← your code
├── requirements.txt     ← extra Python packages (one per line), can be empty
├── data/                ← your dataset (optional)
└── train.sbatch         ← copy of examples/train.sbatch
```

### b. Edit `train.sbatch`

Only change the lines marked `CHANGE`:

```bash
#SBATCH --job-name=my-project       # name shown in `gix status`
#SBATCH --gres=gpu:1                # GPU size, see below
#SBATCH --time=08:00:00             # max run time hh:mm:ss, the job stops after this

IMAGE=docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN='pip install -q -r requirements.txt && python train.py --data data/ --out outputs/'
```

| Line | What it means |
|---|---|
| `--gres=gpu:1` | A **whole** RTX 5090 (32 GB). Use this for real training |
| `--gres=shard:1` | A **quarter** of a GPU, shared with others. Fine for small jobs and tests, and usually starts sooner |
| `--time` | Ask for a bit more than you need. Shorter jobs often start sooner |
| `IMAGE` | The software environment (container). The default has PyTorch + CUDA 12.8. Any Docker image with CUDA ≥ 12.8 works, e.g. from [Docker Hub](https://hub.docker.com/) or [NVIDIA NGC](https://catalog.ngc.nvidia.com/) |
| `RUN` | The command(s) to run inside the image, from your project folder |

### c. Upload, run, get results

```sh
gix upload my-project
gix submit my-project/train.sbatch
gix logs <job> -f
gix download my-project/outputs
```

Changed your code? Run `gix upload my-project` again. It overwrites the files on the cluster.

---

## Cheat sheet

| Command | What it does |
|---|---|
| `gix setup` | One-time setup (NetID + password) |
| `gix upload <folder>` | Copy a folder or file from your laptop to your cluster folder |
| `gix submit <folder>/train.sbatch` | Start a job (it waits in line if GPUs are busy) |
| `gix status` | Your jobs: waiting, running, recently finished |
| `gix logs <job>` | Last lines of a job's output |
| `gix logs <job> -f` | Follow the output live until the job ends |
| `gix cancel <job>` | Stop a job (`gix cancel all` stops all of yours) |
| `gix download <folder>/outputs` | Copy results from the cluster to your laptop |
| `gix whoami` | Your NetID and cluster folder |
| `gix --help` | All commands |

Paths like `cifar10/train.sbatch` are always **relative to your cluster folder**.

---

## Good to know

- **Waiting in line:** `gix status` shows your position. The job starts by itself; you don't have to stay online.
- **Sharing a GPU (`shard:1`):** up to 4 people share one GPU and its 32 GB memory. If your job runs out of
  GPU memory, use `gpu:1`.
- **Time limit:** your job stops at the `--time` you asked for. Save checkpoints in `outputs/` during
  long runs, so you can continue from them.
- **Saving results:** write everything you want to keep to your project folder (e.g. `outputs/`).
  Anything else inside the container is gone when the job ends.
- **Internet:** jobs can download packages, models and datasets.
- **Secrets:** don't put passwords or API keys in your cluster folder.
- **Be fair:** cancel jobs you don't need (`gix cancel <job>`), so others can use the GPU.

---

## Optional: interactive work with VS Code

For writing and debugging code directly on a GPU (instead of sending jobs). `gix setup` already prepared this.

1. Install the VS Code extensions **Remote - SSH** and **Dev Containers**.
2. `Cmd/Ctrl+Shift+P` → **Remote-SSH: Connect to Host…** → **gix-gpu** (choose **Linux** if asked).
3. **File → Open Folder…** → a project with a `.devcontainer` folder
   (example: [`examples/vscode-devcontainer`](examples/vscode-devcontainer)) → **Reopen in Container**.

You get a quarter GPU for up to 4 hours; it is released 30 minutes after you close VS Code.
Files you create there are on the GPU node, not in your cluster folder, so keep your code in git.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `gix: command not found` | Open a new terminal. Still missing: use `python3 -m gix_cli` (Windows: `py -m gix_cli`) instead of `gix` |
| `gix setup` hangs or "Connection timed out" | You must be on the UW network or connected to **Husky OnNet** |
| `Permission denied` during `gix setup` | Wrong NetID or password. Run `gix setup` again |
| `gix: run gix setup first` | Run `gix setup` |
| `file not found` on `gix submit` | Upload the folder first (`gix upload <folder>`), and use the path inside your cluster folder |
| Job waits for a long time | All GPUs are busy. It starts by itself. `shard:1` and a shorter `--time` usually start sooner |
| `CUDA out of memory` | Use `--gres=gpu:1`, or a smaller batch size |
| Job stopped early | Check `gix logs <job>`. If it ran out of time, ask for a longer `--time` |
| Errors about your code / packages | `gix logs <job>` shows the error. Fix it on your laptop, `gix upload`, `gix submit` again |

Still stuck? Send your TA the output of `gix status` and `gix logs <job>`.
