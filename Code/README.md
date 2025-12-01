# Domain Adaptation with DANN: MNIST → MNIST-M

This repository implements a **Domain-Adversarial Neural Network (DANN)** to adapt from the **MNIST** source domain to the **MNIST-M** target domain.

The training pipeline has **two phases**:

1. **Source-Only Baseline** – train on MNIST only, evaluate on both MNIST and MNIST-M to measure the domain gap.
2. **DANN** – adversarially align features using a Gradient Reversal Layer (GRL) and a domain classifier to improve target performance.

---

## 1. Project Structure

    .
    ├── DANN.py                   # Main training script (Source-Only baseline + DANN)
    ├── log.txt                   # Example training log from one run (optional)
    ├── checkpoints/              # Folder where best model weights will be saved (auto-created)
    │   └── best_dann_final.pth   # Best DANN model (saved after Phase 2)
    ├── tsne_paper_before.png     # t-SNE plot before adaptation (generated)
    └── tsne_paper_after.png      # t-SNE plot after adaptation (generated)

At the beginning, you only need **DANN.py** (and the MNIST-M data).  
All other files and folders will be created automatically after running the script.

---

## 2. Environment & Dependencies

Tested with:

- **Python** ≥ 3.8  
- **PyTorch** + **torchvision** (CUDA GPU recommended but not required)

Main Python packages:

- torch, torchvision  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn (for t-SNE)

Install dependencies:

    # (Optional) create a virtual environment
    python -m venv venv

    # Linux / macOS
    source venv/bin/activate
    # Windows (PowerShell)
    # .\venv\Scripts\Activate.ps1

    # Install packages
    pip install torch torchvision torchaudio    # choose version matching your CUDA
    pip install numpy matplotlib seaborn scikit-learn

If you already have PyTorch installed, you can just install the remaining packages:

    pip install matplotlib seaborn scikit-learn

---

## 3. Dataset Preparation

The code uses:

- **Source domain**: MNIST (from `torchvision.datasets.MNIST`)
- **Target domain**: MNIST-M (loaded via `torchvision.datasets.ImageFolder`)

### 3.1 MNIST (Source)

No manual download is needed.  
`DANN.py` will automatically download MNIST into a local `./data` folder.

- Training: 60,000 images  
- Test: 10,000 images  

### 3.2 MNIST-M (Target)

You must download MNIST-M manually (the standard version used in DANN papers).

Expected folder structure:

    /path/to/mnist_m/
    ├── training/
    │   ├── 0/
    │   ├── 1/
    │   ├── ...
    │   └── 9/
    └── testing/
        ├── 0/
        ├── 1/
        ├── ...
        └── 9/

Each subfolder 0–9 contains RGB images of that digit.

When running the script, pass `/path/to/mnist_m` to the `--mnistm-root` argument.

---

## 4. How to Run the Code

Everything is controlled by **one file**: `DANN.py`.

### 4.1 Basic Command

    python DANN.py \
      --mnistm-root /path/to/mnist_m \
      --epochs 50 \
      --batch-size 512 \
      --lr 1e-3 \
      --seed 42 \
      --num-workers 8 \
      --gamma 10.0 \
      --dom-weight 1.0 \
      --tgt-split 0.8 0.1 0.1 \
      --patience 7 \
      --warmup-epochs 10

This will:

1. Train the **Source-Only** baseline with early stopping on MNIST validation loss.  
2. Evaluate the Source-Only model on MNIST and MNIST-M test sets.  
3. Train **DANN** with:
   - Gradient Reversal Layer (GRL) and lambda schedule,
   - Classification loss on source samples,
   - Domain loss on source + target samples,
   - Early stopping after a warm-up period.
4. Generate t-SNE plots before and after adaptation:
   - `tsne_paper_before.png`
   - `tsne_paper_after.png`
5. Save the best DANN model to `checkpoints/best_dann_final.pth`.

To save the console output into `log.txt`:

    python DANN.py --mnistm-root /path/to/mnist_m > log.txt 2>&1

---

## 5. Command-Line Arguments

All arguments are defined in `main()` inside `DANN.py`.

- `--mnistm-root` (str, **required**)  
  Path to the MNIST-M folder (the folder that contains `training/` and `testing/`).

- `--epochs` (int, default 50)  
  Number of epochs for **each phase** (Source-Only and DANN).

- `--batch-size` (int, default 512)  
  Mini-batch size for all dataloaders.

- `--lr` (float, default 1e-3)  
  Learning rate for the AdamW optimizer.

- `--seed` (int, default 42)  
  Random seed for reproducibility.

- `--num-workers` (int, default 8)  
  Number of worker processes for DataLoader.

- `--gamma` (float, default 10.0)  
  Gamma parameter used in the DANN lambda schedule.  
  Higher values make lambda grow faster from 0 → 1.

- `--dom-weight` (float, default 1.0)  
  Weight for the domain loss when combining with classification loss.

- `--tgt-split` (3 floats, default 0.8 0.1 0.1)  
  Ratios for splitting the target dataset into Train / Val / Test.  
  Example: `0.8 0.1 0.1` means 80% train, 10% validation, 10% test.

- `--patience` (int, default 7)  
  Patience for early stopping (number of epochs without improvement).

- `--warmup-epochs` (int, default 10)  
  Number of DANN epochs to run **before** early stopping becomes active.  
  This stabilizes training at the beginning.

If you omit an argument, the script uses the default value defined in `DANN.py`.

---

## 6. Training Flow (What the Code Does)

### 6.1 Phase 1 – Source-Only Baseline

- Uses the same backbone as DANN but **turns off the domain loss**.  
- Optimizes only the classification loss on MNIST.  
- Uses EarlyStopping on MNIST validation loss.  
- At the end, reports:
  - Test accuracy on MNIST (source),
  - Test accuracy on MNIST-M (target baseline, domain gap).

The best Source-Only model is kept in memory and used as a baseline reference.

### 6.2 Phase 2 – DANN

- Re-initializes a new DANN model with:
  - Feature extractor,
  - Label classifier,
  - Domain classifier,
  - Gradient Reversal Layer.

For each training iteration:

1. Take a batch of **labeled source** images and **unlabeled target** images.  
2. Pass both through the feature extractor.  
3. Compute classification loss on source predictions vs. source labels.  
4. Pass features through GRL + domain classifier to predict domain labels (source or target).  
5. Compute domain loss on source + target.  
6. Total loss = `classification_loss + dom_weight * domain_loss`.  
7. Update parameters with back-propagation.

The lambda for GRL is scheduled as training progresses (classic DANN schedule):

- At the beginning, lambda ≈ 0 → almost no adversarial signal.  
- Near the end, lambda → 1 → strong adversarial alignment.

Early stopping in DANN still monitors **source validation loss**.  
This is conservative and prevents over-fitting the domain alignment at the cost of source performance.

After training DANN, the script:

- Evaluates the final model on:
  - MNIST (source test set),
  - MNIST-M (target test set),
- Prints the improvement on the target domain compared to the Source-Only model,
- Computes domain classifier accuracy (close to 50% means good confusion),
- Generates t-SNE visualizations to show how features move from “separated” to “aligned”.

---

## 7. Reproducibility

The script calls `set_seed(seed)` at the start, which sets:

- Python `random` seed,  
- NumPy seed,  
- Torch CPU and CUDA seeds,  
- `torch.backends.cudnn.deterministic = True`,  
- `torch.backends.cudnn.benchmark = False`.

To reproduce a run, keep the same:

- `--seed`  
- `--batch-size`  
- `--tgt-split`  
- Dataset path and contents  

Some small variation can still appear across machines and GPUs, but results should be close.
