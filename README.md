# Predikce výnosu plodin v Indii

Semestrální projekt pro předmět **Úvod do strojového učení – MML1**.

Projekt řeší regresní predikci výnosu zemědělských plodin v Indii podle plodiny, lokality, sezóny, roku a meteorologických údajů.

## Hlavní výstup

Kompletní porovnání modelů, výpočty, grafy a diskuze jsou v notebooku:

```text
notebooks/notebook_hw3.ipynb
```

Notebook obsahuje:

- natrénování a vyhodnocení všech porovnávaných modelů,
- baseline modely,
- MAE, RMSE, R² a MedianAE,
- tabulky a grafy výsledků,
- analýzu přeučení,
- porovnání nejlepšího modelu s baseline,
- diskuzi výsledků a závěr.

## Data

Cílovou proměnnou je korigovaný výnos:

```text
target_yield = Production_corrected / Area_corrected
```

Použitý chronologický split:

```text
data/processed/train_1997_2010.parquet
data/processed/validation_2011_2012.parquet
data/processed/test_2013_2014.parquet
```

- trénovací období: 1997–2010,
- validační období: 2011–2012,
- testovací období: 2013–2014.

## Spuštění

Vytvoření virtuálního prostředí a instalace závislostí:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Otevření notebooku:

```powershell
jupyter notebook notebooks/notebook_hw3.ipynb
```

## Struktura projektu

```text
notebooks/notebook_hw3.ipynb   hlavní odevzdávaný notebook
src/                           příprava dat, modelů a evaluace
data/processed/                train, validation a test data
data/reference/                uložené konfigurace a metadata
reports/                       výsledkové tabulky a grafy
requirements.txt               Python závislosti
```

## HW4: PyTorch MLP and ablation study

HW4 adds two advanced workshop-based techniques:

1. A configurable PyTorch MLP regressor for nonlinear interactions among crop,
   geography, season, area, year, and weather features.
2. An ablation study comparing small `[64]`, medium `[128, 64]`, and deep
   `[256, 128, 64]` networks, plus ReLU versus Tanh for the medium network.

These techniques are appropriate because crop yield depends on nonlinear
interactions between weather, crop type, place, and growing season. The ablation
separates the effect of network capacity from the effect of activation choice,
instead of relying on one arbitrary neural-network setup.

The experiment reuses the chronological processed splits, fits scaling and
one-hot encoding on training data only, and reports MAE, RMSE, and R2 against
Dummy and Ridge baselines. Run it with:

```powershell
jupyter notebook notebooks/notebook_hw4.ipynb
```

For a non-interactive execution from the repository root:

```powershell
jupyter nbconvert --to notebook --execute notebooks/notebook_hw4.ipynb --inplace --ExecutePreprocessor.timeout=-1
```

The main comparison is saved to `results/hw4_mlp_ablation_results.csv`; error
plots and the worst-predictions table are saved in the same directory. PyTorch
and the remaining notebook dependencies are listed in `requirements.txt`.

