#!/usr/bin/env python3
"""
Final clean bootstrap for MML1 HW2 repository.

Použitie:
1. Daj tento súbor do priečinka, kde máš:
   - crop_with_coords_final.csv
   - weather_data_final.csv
   Voliteľne:
   - Indian_crop_production_yield_dataset.csv
2. Spusti:
   python create_hw2_repo_final_clean.py
3. Vytvorí sa priečinok MML1-project-HW2 s hotovou štruktúrou na upload na GitHub.

Tento skript robí finálnu verziu podľa aktuálneho stavu:
- crop data + prepočítané weather features
- weather-complete subset
- chronologický split 1997-2010 / 2011-2012 / 2013-2014
- notebooky, README, reporty a src skripty
"""

from __future__ import annotations

import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd
    import numpy as np
except ImportError as exc:
    print("Chýba pandas/numpy. Nainštaluj: pip install pandas numpy")
    raise exc

# =========================
# Nastavenia finálnej verzie
# =========================
ROOT = Path("MML1-project-HW2")
DATA = ROOT / "data"
RAW = DATA / "raw"
SRC = ROOT / "src"
REPORTS = ROOT / "reports"

CROP_WITH_COORDS = Path("crop_with_coords_final.csv")
CROP_RAW_OPTIONAL = Path("Indian_crop_production_yield_dataset.csv")
WEATHER_FILE = Path("weather_data_final.csv")

JOIN_KEYS = ["State_Name", "District_Name", "Crop_Year", "Season", "Crop"]
CROP_COLUMNS = ["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production", "yield"]
COORD_COLUMNS = ["State_Name", "District_Name", "latitude", "longitude"]

WEATHER_COLUMNS = [
    "rain_sum_mm",
    "rainy_days_ge1mm",
    "dry_days_lt1mm",
    "longest_dry_spell_days",
    "longest_wet_spell_days",
    "heavy_rain_days_ge20mm",
    "temp_mean_c",
    "temp_max_mean_c",
    "temp_min_mean_c",
    "hot_days_tmax_ge35c",
    "humidity_mean_pct",
    "solar_mean",
    "planting_dry_flag",
    "harvest_too_wet_flag",
]

# Podľa aktuálneho rozhodnutia: finálny weather-complete dataset používame 1997-2014.
TRAIN_START, TRAIN_END = 1997, 2010
VAL_START, VAL_END = 2011, 2012
TEST_START, TEST_END = 2013, 2014
FINAL_START, FINAL_END = TRAIN_START, TEST_END


def robust_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "State_Name": "string",
            "District_Name": "string",
            "Season": "string",
            "Crop": "string",
        },
    )


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["State_Name", "District_Name", "Season", "Crop"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
    return df


def backup_existing_repo() -> None:
    if ROOT.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = Path(f"MML1-project-HW2_backup_{timestamp}")
        print(f"Existujúci priečinok {ROOT} presúvam do {backup}")
        shutil.move(str(ROOT), str(backup))


def make_dirs() -> None:
    for p in [ROOT, DATA, RAW, SRC, REPORTS]:
        p.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    missing = []
    if not CROP_WITH_COORDS.exists() and not CROP_RAW_OPTIONAL.exists():
        missing.append("crop_with_coords_final.csv alebo Indian_crop_production_yield_dataset.csv")
    if not WEATHER_FILE.exists():
        missing.append("weather_data_final.csv")
    if missing:
        raise FileNotFoundError("Chýbajú vstupné súbory: " + ", ".join(missing))


def load_crop() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return crop-only dataframe and optional coordinate dataframe."""
    coords = None

    if CROP_WITH_COORDS.exists():
        crop_coords = robust_read_csv(CROP_WITH_COORDS)
        crop_coords = clean_text_columns(crop_coords)
        missing = set(CROP_COLUMNS) - set(crop_coords.columns)
        if missing:
            raise ValueError(f"crop_with_coords_final.csv nemá potrebné stĺpce: {missing}")

        crop = crop_coords[CROP_COLUMNS].copy()
        if set(COORD_COLUMNS).issubset(crop_coords.columns):
            coords = crop_coords[COORD_COLUMNS].drop_duplicates().copy()
            coords["latitude"] = pd.to_numeric(coords["latitude"], errors="coerce")
            coords["longitude"] = pd.to_numeric(coords["longitude"], errors="coerce")
            coords = coords.dropna(subset=["latitude", "longitude"])

        shutil.copy2(CROP_WITH_COORDS, RAW / "crop_with_coords_final.csv")
        crop.to_csv(RAW / "Indian_crop_production_yield_dataset.csv", index=False)
        if coords is not None:
            coords.to_csv(RAW / "crop_regions_cleaned.csv", index=False)
    else:
        crop = robust_read_csv(CROP_RAW_OPTIONAL)
        crop = clean_text_columns(crop)
        missing = set(CROP_COLUMNS) - set(crop.columns)
        if missing:
            raise ValueError(f"Indian_crop_production_yield_dataset.csv nemá potrebné stĺpce: {missing}")
        crop = crop[CROP_COLUMNS].copy()
        shutil.copy2(CROP_RAW_OPTIONAL, RAW / "Indian_crop_production_yield_dataset.csv")

    return crop, coords


def load_weather() -> pd.DataFrame:
    weather = robust_read_csv(WEATHER_FILE)
    weather = clean_text_columns(weather)
    required = set(JOIN_KEYS + WEATHER_COLUMNS)
    missing = required - set(weather.columns)
    if missing:
        raise ValueError(f"weather_data_final.csv nemá potrebné stĺpce: {missing}")
    weather = weather[JOIN_KEYS + WEATHER_COLUMNS].copy()
    shutil.copy2(WEATHER_FILE, DATA / "weather_data_final.csv")
    return weather


def add_lag_yield(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_cols = ["State_Name", "District_Name", "Season", "Crop"]
    df = df.sort_values(group_cols + ["Crop_Year"])
    df["lag_yield"] = df.groupby(group_cols, dropna=False)["yield"].shift(1)
    return df


def prepare_data() -> Dict[str, Any]:
    crop, coords = load_crop()
    weather = load_weather()

    # Numeric conversions
    for col in ["Crop_Year", "Area", "Production", "yield"]:
        crop[col] = pd.to_numeric(crop[col], errors="coerce")
    crop = crop.dropna(subset=["Crop_Year", "yield"]).copy()
    crop["Crop_Year"] = crop["Crop_Year"].astype(int)
    crop = crop.drop_duplicates()

    for col in ["Crop_Year"] + WEATHER_COLUMNS:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather["Crop_Year"] = weather["Crop_Year"].astype("Int64")
    weather = weather.dropna(subset=["Crop_Year"]).copy()
    weather["Crop_Year"] = weather["Crop_Year"].astype(int)
    weather = weather.drop_duplicates(subset=JOIN_KEYS)

    joined = crop.merge(weather, on=JOIN_KEYS, how="left", validate="many_to_one")
    joined.to_csv(DATA / "crop_weather_joined.csv", index=False)

    weather_complete = joined.dropna(subset=["rain_sum_mm"]).copy()
    weather_complete = weather_complete[
        weather_complete["Crop_Year"].between(FINAL_START, FINAL_END)
    ].copy()
    weather_complete = weather_complete.dropna(subset=["yield"]).copy()
    weather_complete = add_lag_yield(weather_complete)

    # Remove latitude/longitude if somehow present in modelling data
    for col in ["latitude", "longitude"]:
        if col in weather_complete.columns:
            weather_complete = weather_complete.drop(columns=[col])

    weather_complete.to_csv(DATA / "final_dataset_weather_complete.csv", index=False)

    train = weather_complete[weather_complete["Crop_Year"].between(TRAIN_START, TRAIN_END)].copy()
    validation = weather_complete[weather_complete["Crop_Year"].between(VAL_START, VAL_END)].copy()
    test = weather_complete[weather_complete["Crop_Year"].between(TEST_START, TEST_END)].copy()

    train.to_csv(DATA / "train.csv", index=False)
    validation.to_csv(DATA / "validation.csv", index=False)
    test.to_csv(DATA / "test.csv", index=False)

    summary = {
        "crop_shape": crop.shape,
        "weather_shape": weather.shape,
        "joined_shape": joined.shape,
        "weather_complete_shape": weather_complete.shape,
        "train_shape": train.shape,
        "validation_shape": validation.shape,
        "test_shape": test.shape,
        "weather_missing_ratio_full_join": float(joined["rain_sum_mm"].isna().mean()),
        "years_weather_complete": (int(weather_complete["Crop_Year"].min()), int(weather_complete["Crop_Year"].max())) if len(weather_complete) else None,
        "coords_shape": coords.shape if coords is not None else None,
        "year_counts": weather_complete["Crop_Year"].value_counts().sort_index().to_dict(),
    }
    return summary


def write_requirements() -> None:
    (ROOT / "requirements.txt").write_text(
        "\n".join([
            "pandas",
            "numpy",
            "scikit-learn",
            "matplotlib",
            "requests",
            "jupyter",
            "nbconvert",
        ]) + "\n",
        encoding="utf-8",
    )


def write_readme(summary: Dict[str, Any]) -> None:
    txt = f"""
# Predikce výnosu plodin v Indii – MML1 HW2

Tento repozitář obsahuje řešení domácího úkolu 2 pro projekt **Predikce produkce/výnosu plodin v Indii**.
Cílem není maximalizovat skóre pokročilého modelu, ale vytvořit korektní datovou pipeline: popis dat, prevence leakage, časový split, uložení train/validation/test sad a jednoduchý benchmark.

## Struktura repozitáře

```text
MML1-project-HW2/
├─ README.md
├─ dataprocessing.ipynb
├─ dataprocessing.html
├─ benchmark.ipynb
├─ benchmark.html
├─ requirements.txt
├─ data/
│  ├─ raw/
│  │  ├─ Indian_crop_production_yield_dataset.csv
│  │  ├─ crop_with_coords_final.csv
│  │  └─ crop_regions_cleaned.csv
│  ├─ weather_data_final.csv
│  ├─ crop_weather_joined.csv
│  ├─ final_dataset_weather_complete.csv
│  ├─ train.csv
│  ├─ validation.csv
│  └─ test.csv
├─ reports/
└─ src/
   ├─ finalize_hw2.py
   └─ join_crop_weather.py
```

## Základní crop data

Původní crop tabulka má sloupce:

```text
State_Name, District_Name, Crop_Year, Season, Crop, Area, Production, yield
```

Jednotka pozorování je:

```text
State_Name + District_Name + Crop_Year + Season + Crop
```

## Weather data

Weather features byly vytvořeny z denních dat NASA POWER podle souřadnic indických okresů. Weather tabulka se připojuje přes stejné klíče:

```text
State_Name, District_Name, Crop_Year, Season, Crop
```

Použité weather features:

```text
rain_sum_mm, rainy_days_ge1mm, dry_days_lt1mm,
longest_dry_spell_days, longest_wet_spell_days,
heavy_rain_days_ge20mm,
temp_mean_c, temp_max_mean_c, temp_min_mean_c,
hot_days_tmax_ge35c, humidity_mean_pct, solar_mean,
planting_dry_flag, harvest_too_wet_flag
```

`crop_weather_joined.csv` obsahuje plný join crop + weather. Finální benchmark používá pouze řádky s úspěšně připojeným počasím, uložené v `final_dataset_weather_complete.csv`.

## Target a leakage audit

Hlavní cílová proměnná je `yield`. Sloupec `Production` není používán jako modelová feature, protože je přímo provázaný s výnosem a plochou a mohl by způsobit target leakage. Denní weather data nejsou připojena přímo; nejdříve jsou agregována na úroveň crop pozorování.

## Split

Finální weather-complete dataset je omezen na roky **{FINAL_START}–{FINAL_END}**. Použitý split je chronologický:

- train: {TRAIN_START}–{TRAIN_END}
- validation: {VAL_START}–{VAL_END}
- test: {TEST_START}–{TEST_END}

Test set je vytvořen a uložen, ale nepoužívá se pro výběr modelu ani pro porovnávání benchmarků.

## Velikost dat po zpracování

- crop data: {summary['crop_shape']}
- weather data: {summary['weather_shape']}
- crop_weather_joined: {summary['joined_shape']}
- final_dataset_weather_complete: {summary['weather_complete_shape']}
- train: {summary['train_shape']}
- validation: {summary['validation_shape']}
- test: {summary['test_shape']}
- podíl řádků bez weather hodnot v plném joinu: {summary['weather_missing_ratio_full_join']:.3f}

## Jak spustit

```bash
pip install -r requirements.txt
python src/finalize_hw2.py
```

Notebooky:

```text
dataprocessing.ipynb
benchmark.ipynb
```
"""
    (ROOT / "README.md").write_text(textwrap.dedent(txt).strip() + "\n", encoding="utf-8")


def nb_cell(cell_type: str, source: str) -> Dict[str, Any]:
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}
    if cell_type == "code":
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}
    raise ValueError(cell_type)


def write_notebook(path: Path, cells: List[Dict[str, Any]]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataprocessing_notebook() -> None:
    cells = [
        nb_cell("markdown", """# Data processing – Predikce výnosu plodin v Indii

Tento notebook připravuje finální data pro HW2: crop data, weather join, leakage audit, chronologický split a uložení train/validation/test sad."""),
        nb_cell("markdown", f"""## 1. Task framing

Cílem projektu je predikovat `yield` zemědělských plodin v Indii na úrovni:

`State_Name + District_Name + Crop_Year + Season + Crop`.

Hlavní target je `yield`. Sloupec `Production` není používán jako modelová feature, protože je přímo navázaný na výnos a pěstovanou plochu."""),
        nb_cell("code", """import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path('data')
RAW = DATA / 'raw'
JOIN_KEYS = ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop']
WEATHER_COLS = [
    'rain_sum_mm', 'rainy_days_ge1mm', 'dry_days_lt1mm',
    'longest_dry_spell_days', 'longest_wet_spell_days',
    'heavy_rain_days_ge20mm', 'temp_mean_c', 'temp_max_mean_c',
    'temp_min_mean_c', 'hot_days_tmax_ge35c', 'humidity_mean_pct',
    'solar_mean', 'planting_dry_flag', 'harvest_too_wet_flag'
]

def clean_text_columns(df):
    df = df.copy()
    for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
        if col in df.columns:
            df[col] = df[col].astype('string').str.strip().str.replace(r'\\s+', ' ', regex=True)
    return df
"""),
        nb_cell("markdown", """## 2. Načtení dat"""),
        nb_cell("code", """crop = pd.read_csv(RAW / 'Indian_crop_production_yield_dataset.csv', low_memory=False)
weather = pd.read_csv(DATA / 'weather_data_final.csv', low_memory=False)

crop = clean_text_columns(crop)
weather = clean_text_columns(weather)

print('crop:', crop.shape)
print('weather:', weather.shape)
display(crop.head())
display(weather.head())
"""),
        nb_cell("markdown", """## 3. Popis dat"""),
        nb_cell("code", """summary = pd.DataFrame({
    'missing_count': crop.isna().sum(),
    'missing_ratio': crop.isna().mean(),
    'n_unique': crop.nunique(dropna=True),
})
display(summary)
print('Crop years:', crop['Crop_Year'].min(), crop['Crop_Year'].max())
print('States:', crop['State_Name'].nunique())
print('District combinations:', crop[['State_Name','District_Name']].drop_duplicates().shape[0])
print('Crops:', crop['Crop'].nunique())
print('Duplicate rows:', crop.duplicated().sum())
display(crop['yield'].describe())
"""),
        nb_cell("markdown", """## 4. Weather join

Weather data jsou agregována na stejnou jednotku pozorování jako crop dataset. Denní weather data se nepřipojují přímo, aby nevznikla změna jednotky pozorování.

Připojení probíhá přes:

`State_Name, District_Name, Crop_Year, Season, Crop`."""),
        nb_cell("code", """weather = weather.drop_duplicates(subset=JOIN_KEYS)
joined = crop.merge(weather, on=JOIN_KEYS, how='left', validate='many_to_one')
joined.to_csv(DATA / 'crop_weather_joined.csv', index=False)

print('joined:', joined.shape)
print('rows preserved:', joined.shape[0] == crop.shape[0])
print('missing weather ratio:', joined['rain_sum_mm'].isna().mean())
"""),
        nb_cell("markdown", """## 5. Leakage audit

- Target je `yield`.
- `Production` se nepoužívá jako feature, protože by vedl k target leakage.
- Weather features jsou agregované před joinem.
- Split je chronologický podle `Crop_Year`, ne náhodný.
- Preprocessing modelů v benchmarku je fitovaný pouze na train sadě.
- Test set se pouze vytvoří a uloží; nepoužívá se pro výběr modelu."""),
        nb_cell("markdown", """## 6. Finální weather-complete dataset a lag feature"""),
        nb_cell("code", f"""df = joined.dropna(subset=['rain_sum_mm']).copy()
df = df[df['Crop_Year'].between({FINAL_START}, {FINAL_END})].copy()
df = df.dropna(subset=['yield']).copy()

for col in ['Crop_Year', 'Area', 'Production', 'yield'] + WEATHER_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['Crop_Year'] = df['Crop_Year'].astype(int)
df = df.sort_values(['State_Name', 'District_Name', 'Season', 'Crop', 'Crop_Year'])
df['lag_yield'] = df.groupby(['State_Name', 'District_Name', 'Season', 'Crop'])['yield'].shift(1)

df.to_csv(DATA / 'final_dataset_weather_complete.csv', index=False)
print('final weather-complete:', df.shape)
print('years:', df['Crop_Year'].min(), df['Crop_Year'].max())
"""),
        nb_cell("markdown", f"""## 7. Chronologický split

Použitý split:

- train: {TRAIN_START}–{TRAIN_END}
- validation: {VAL_START}–{VAL_END}
- test: {TEST_START}–{TEST_END}"""),
        nb_cell("code", f"""train = df[df['Crop_Year'].between({TRAIN_START}, {TRAIN_END})].copy()
validation = df[df['Crop_Year'].between({VAL_START}, {VAL_END})].copy()
test = df[df['Crop_Year'].between({TEST_START}, {TEST_END})].copy()

train.to_csv(DATA / 'train.csv', index=False)
validation.to_csv(DATA / 'validation.csv', index=False)
test.to_csv(DATA / 'test.csv', index=False)

print('train:', train.shape, train['Crop_Year'].min(), train['Crop_Year'].max())
print('validation:', validation.shape, validation['Crop_Year'].min(), validation['Crop_Year'].max())
print('test:', test.shape, test['Crop_Year'].min(), test['Crop_Year'].max())
"""),
        nb_cell("markdown", """## 8. Shrnutí

Výsledkem je finální weather-complete dataset a tři uložené datasety `train.csv`, `validation.csv` a `test.csv`. Test set zůstává stranou pro pozdější finální vyhodnocení."""),
    ]
    write_notebook(ROOT / "dataprocessing.ipynb", cells)


def write_benchmark_notebook() -> None:
    cells = [
        nb_cell("markdown", """# Benchmark – Predikce výnosu plodin v Indii

Tento notebook vytváří jednoduchý benchmark. Modely se porovnávají pouze na validační sadě. Test set se nepoužívá."""),
        nb_cell("code", """import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

DATA = Path('data')
train = pd.read_csv(DATA / 'train.csv', low_memory=False)
validation = pd.read_csv(DATA / 'validation.csv', low_memory=False)

print('train:', train.shape)
print('validation:', validation.shape)
display(train.head())
"""),
        nb_cell("markdown", """## 1. Metriky

Používám MAE, RMSE a R². Hlavní metrika pro interpretaci je MAE."""),
        nb_cell("code", """def evaluate(y_true, y_pred, name):
    return {
        'model': name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': r2_score(y_true, y_pred),
    }

results = []
y_train = train['yield']
y_val = validation['yield']
"""),
        nb_cell("markdown", """## 2. Naivní baseline"""),
        nb_cell("code", """results.append(evaluate(y_val, np.repeat(y_train.mean(), len(validation)), 'Naive baseline: train mean'))
results.append(evaluate(y_val, np.repeat(y_train.median(), len(validation)), 'Naive baseline: train median'))
"""),
        nb_cell("markdown", """## 3. Historický baseline přes lag_yield"""),
        nb_cell("code", """lag_pred = validation['lag_yield'].fillna(y_train.median())
results.append(evaluate(y_val, lag_pred, 'Historical baseline: lag_yield'))
"""),
        nb_cell("markdown", """## 4. Jednoduché ML benchmarky

`Production` záměrně není mezi features, protože by způsobovala leakage."""),
        nb_cell("code", """weather_cols = [
    'rain_sum_mm', 'rainy_days_ge1mm', 'dry_days_lt1mm',
    'longest_dry_spell_days', 'longest_wet_spell_days',
    'heavy_rain_days_ge20mm', 'temp_mean_c', 'temp_max_mean_c',
    'temp_min_mean_c', 'hot_days_tmax_ge35c', 'humidity_mean_pct',
    'solar_mean', 'planting_dry_flag', 'harvest_too_wet_flag'
]

numeric_features = ['Crop_Year', 'Area', 'lag_yield'] + [c for c in weather_cols if c in train.columns]
categorical_features = ['State_Name', 'District_Name', 'Season', 'Crop']
feature_cols = numeric_features + categorical_features

X_train = train[feature_cols].copy()
X_val = validation[feature_cols].copy()

try:
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
except TypeError:
    onehot = OneHotEncoder(handle_unknown='ignore', sparse=True)

preprocess = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False)),
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', onehot),
        ]), categorical_features),
    ]
)

models = {
    'Ridge regression': Ridge(alpha=1.0, random_state=42),
    'Decision tree regressor': DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42),
}

for name, model in models.items():
    pipe = Pipeline([('preprocess', preprocess), ('model', model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    results.append(evaluate(y_val, pred, name))
"""),
        nb_cell("markdown", """## 5. Vyhodnocení na validační sadě"""),
        nb_cell("code", """results_df = pd.DataFrame(results).sort_values('MAE')
display(results_df)
results_df.to_csv(DATA / 'benchmark_validation_results.csv', index=False)
"""),
        nb_cell("markdown", """## 6. Shrnutí

Benchmark obsahuje naivní baseline, historický baseline a dva jednoduché ML modely. Test set nebyl použit."""),
    ]
    write_notebook(ROOT / "benchmark.ipynb", cells)


def write_html_fallbacks() -> None:
    dp = f"""<!doctype html><html><head><meta charset='utf-8'><title>dataprocessing</title></head><body>
<h1>Data processing – Predikce výnosu plodin v Indii</h1>
<p>HTML fallback. Primary evaluated file is dataprocessing.ipynb.</p>
<p>Split: train {TRAIN_START}-{TRAIN_END}, validation {VAL_START}-{VAL_END}, test {TEST_START}-{TEST_END}.</p>
</body></html>"""
    bm = """<!doctype html><html><head><meta charset='utf-8'><title>benchmark</title></head><body>
<h1>Benchmark – Predikce výnosu plodin v Indii</h1>
<p>HTML fallback. Primary evaluated file is benchmark.ipynb.</p>
<p>Benchmark uses validation set only. Test set is not used.</p>
</body></html>"""
    (ROOT / "dataprocessing.html").write_text(dp, encoding="utf-8")
    (ROOT / "benchmark.html").write_text(bm, encoding="utf-8")


def write_scripts() -> None:
    finalize = Path(__file__).read_text(encoding="utf-8")
    (SRC / "finalize_hw2.py").write_text(finalize, encoding="utf-8")

    join_script = """
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA = BASE_DIR / 'data'
RAW = DATA / 'raw'
JOIN_KEYS = ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop']

def clean_text_columns(df):
    df = df.copy()
    for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
        if col in df.columns:
            df[col] = df[col].astype('string').str.strip().str.replace(r'\\s+', ' ', regex=True)
    return df

crop = pd.read_csv(RAW / 'Indian_crop_production_yield_dataset.csv', low_memory=False)
weather = pd.read_csv(DATA / 'weather_data_final.csv', low_memory=False)
crop = clean_text_columns(crop)
weather = clean_text_columns(weather)
weather = weather.drop_duplicates(subset=JOIN_KEYS)
joined = crop.merge(weather, on=JOIN_KEYS, how='left', validate='many_to_one')
joined.to_csv(DATA / 'crop_weather_joined.csv', index=False)
print('Crop:', crop.shape)
print('Weather:', weather.shape)
print('Joined:', joined.shape)
print('Missing rain_sum_mm ratio:', joined['rain_sum_mm'].isna().mean())
""".strip() + "\n"
    (SRC / "join_crop_weather.py").write_text(join_script, encoding="utf-8")

    weather_note = """
# Weather data generation note
#
# The file data/weather_data_final.csv was generated from NASA POWER daily data
# using district-level latitude/longitude coordinates. It is included in the repository
# as a processed input for HW2. The final modelling datasets are regenerated by
# src/finalize_hw2.py.
""".strip() + "\n"
    (SRC / "build_weather_data.py").write_text(weather_note, encoding="utf-8")


def write_reports(summary: Dict[str, Any]) -> None:
    summary_txt = f"""
FINAL DATASET SUMMARY
=====================

Crop shape: {summary['crop_shape']}
Weather shape: {summary['weather_shape']}
Joined shape: {summary['joined_shape']}
Weather-complete final dataset: {summary['weather_complete_shape']}
Train: {summary['train_shape']} ({TRAIN_START}-{TRAIN_END})
Validation: {summary['validation_shape']} ({VAL_START}-{VAL_END})
Test: {summary['test_shape']} ({TEST_START}-{TEST_END})
Missing weather ratio in full joined data: {summary['weather_missing_ratio_full_join']:.4f}
Years in final weather-complete dataset: {summary['years_weather_complete']}

Year counts:
{summary['year_counts']}
"""
    (REPORTS / "final_dataset_summary.txt").write_text(textwrap.dedent(summary_txt).strip() + "\n", encoding="utf-8")

    notes = f"""
## Notes to mention in dataprocessing.ipynb

Weather features were generated from NASA POWER daily weather data and joined to the crop dataset using State_Name, District_Name, Crop_Year, Season and Crop.

Because not all crop observations had matching generated weather rows, the final benchmark dataset uses the weather-complete subset. The full joined dataset is kept as `data/crop_weather_joined.csv`.

The final weather-complete dataset covers years {FINAL_START}-{FINAL_END}. A chronological split is used:

- train: {TRAIN_START}-{TRAIN_END}
- validation: {VAL_START}-{VAL_END}
- test: {TEST_START}-{TEST_END}

Target is `yield`. `Production` is not used as a feature due to target leakage risk.
"""
    (REPORTS / "notes_to_paste.md").write_text(textwrap.dedent(notes).strip() + "\n", encoding="utf-8")


def main() -> None:
    print("Vytváram čistý MML1-project-HW2 odznova...")
    validate_inputs()
    backup_existing_repo()
    make_dirs()
    summary = prepare_data()
    write_requirements()
    write_readme(summary)
    write_dataprocessing_notebook()
    write_benchmark_notebook()
    write_html_fallbacks()
    write_scripts()
    write_reports(summary)

    print("\nHOTOVO")
    print(f"Repo vytvorené v: {ROOT.resolve()}")
    print("Skontroluj hlavne:")
    print("- reports/final_dataset_summary.txt")
    print("- README.md")
    print("- dataprocessing.ipynb")
    print("- benchmark.ipynb")
    print("Potom nahraj priečinok MML1-project-HW2 na GitHub.")


if __name__ == "__main__":
    main()
