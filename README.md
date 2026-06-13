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
