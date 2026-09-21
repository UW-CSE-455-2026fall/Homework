# Welcome to CSE 576, 2026 Spring

In this repository, you will find instructions on how to build your own image processing/computer vision library from (mostly) scratch. The work is divided out into different homework assignments, found in the `src/` directory.

To get started, make sure you have `git` and Python installed. Then run:

```
git clone https://github.com/UW-CSE-576-2026SP/Homework.git
cd Homework
```

and check to see that everything runs correctly. We recommend using Linux or MacOS for the homework for a smoother setup.

## Due Dates
** HW1 is due on April 13 (11:59 pm).**

** HW2 is due on April 24 (11:59 pm).**

** HW3 is due on May 12 (11:59 pm).**

** HW4 is due on May 19 (11:59 pm).**

** HW5 is due on May 27 (11:59 pm).**

## Get started on HW1

Open up the README for homework 1 in src/hw1/README.md, or view it [here](src/hw1/README.md). Good luck and have fun!

## Setup

Do this once, in order. You build the environment a single time; after that you just
**activate** it at the start of every session.

### 1. Install Miniconda (or Anaconda)

We use [conda](https://docs.conda.io/) to manage Python and every dependency, so you do
**not** need a separate system Python.

- **macOS / Linux:** install the command-line installer and answer **yes** when it asks
  to initialize conda at startup (fewer headaches later).
- **Windows:** install Anaconda and run all commands below in the **Anaconda Prompt**
  (it uses Bash-like syntax). Windows works, but macOS/Linux is smoother.
- **Low on disk (<10 GB, e.g. attu):** use Miniconda instead of full Anaconda.

Download:
- [Anaconda](https://www.anaconda.com/download)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (smaller)

After installing, **open a new terminal** and confirm conda is on your PATH:

```
conda --version
```

### 2. Clone the repository

```
git clone https://github.com/UW-CSE-576-2026SP/Homework.git
cd Homework
```

### 3. Create the environment

From the repo directory (the one containing `environment.yaml`):

```
conda env create -f environment.yaml
```

This creates an environment named **`cse576`** with Python 3.11, NumPy, Pillow,
Matplotlib, pandas, tqdm, and PyTorch/torchvision. PyTorch is only used in HW5, but it is
the largest download — expect to pull **~5 GB**, so make sure you have the space and a
stable connection. The solve can take several minutes.

### 4. Activate the environment

```
conda activate cse576
```

> **Important — do this in every new terminal.** You must run `conda activate cse576`
> before running any homework code or tests. If you skip it, you'll be using your *base*
> Python (a different, often older set of packages), which can make tests fail for no real
> reason. When the environment is active your prompt shows `(cse576)`.

### 5. Verify it works

```
python -m src.main test hw1
```

The HW1 tests should run and pass. (Extra-credit tests may fail until you implement those
functions — that's expected.) If you get import errors, re-check step 4: your prompt
should show `(cse576)`.

### Updating later

If `environment.yaml` changes during the quarter, update your environment with:

```
conda env update -f environment.yaml --prune
```


