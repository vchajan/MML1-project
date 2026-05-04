# Predikce výnosu plodin v Indii – HW2

Autor: Peter Briedoň

Tento repozitář obsahuje řešení domácího úkolu 2 pro projekt **Predikce produkce plodin v Indii**. Cílem úkolu je připravit korektní datovou pipeline: popsat data, vytvořit train/validation/test split, ošetřit leakage, připravit preprocessing a vytvořit jednoduchý benchmark.

## 1. Cíl projektu

Cílem projektu je predikovat výnos (`yield`) zemědělských plodin v Indii na úrovni:

```text
State_Name + District_Name + Crop_Year + Season + Crop
```

Projekt nepoužívá jako hlavní target celkovou produkci (`Production`), protože produkce je výrazně ovlivněna obdělávanou plochou (`Area`). Hlavní modelovací úloha je proto formulována jako regresní predikce výnosu (`yield`).

## 2. Struktura repozitáře

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
│  ├─ final_dataset_summary.txt
│  └─ notes_to_paste.md
└─ src/
   ├─ build_weather_data.py
   ├─ join_crop_weather.py
   └─ finalize_hw2.py
```

## 3. Použitá data

Základní crop dataset obsahuje sloupce:

```text
State_Name, District_Name, Crop_Year, Season, Crop, Area, Production, yield
```

K crop datasetu byla doplněna meteorologická data z NASA POWER. Počasí bylo staženo podle souřadnic okresů a následně agregováno na úroveň pozorování odpovídající crop datasetu.

Weather features se připojují pomocí klíčů:

```text
State_Name, District_Name, Crop_Year, Season, Crop
```

Finální weather sloupce jsou:

```text
rain_sum_mm
rainy_days_ge1mm
dry_days_lt1mm
longest_dry_spell_days
longest_wet_spell_days
heavy_rain_days_ge20mm
temp_mean_c
temp_max_mean_c
temp_min_mean_c
hot_days_tmax_ge35c
humidity_mean_pct
solar_mean
planting_dry_flag
harvest_too_wet_flag
```

Soubor `data/crop_weather_joined.csv` obsahuje crop data spojená s weather features. Soubor `data/final_dataset_weather_complete.csv` obsahuje pouze řádky, kde se podařilo počasí úspěšně připojit.

## 4. Finální modelovací dataset

Protože souřadnice byly dostupné pouze pro část okresů, finální benchmark používá pouze pozorování s kompletními weather features. Řádky bez připojeného počasí zůstávají ve spojeném datasetu `crop_weather_joined.csv`, ale nejsou použity ve finálním benchmarku.

Finální datasety pro modelování jsou:

```text
data/train.csv
data/validation.csv
data/test.csv
```

## 5. Split dat

Použitý split je chronologický podle `Crop_Year`:

```text
train:       1997–2010
validation: 2011–2012
test:       2013–2014
```

Chronologický split byl zvolen proto, aby validace lépe odpovídala budoucímu použití modelu. Náhodný split by mohl vést k příliš optimistickému odhadu výkonu, protože by promíchal starší a novější roky.

Test set je pouze vytvořen a uložen. Nepoužívá se pro výběr modelu, ladění parametrů ani porovnávání benchmarků.

## 6. Leakage audit

Hlavní body prevence leakage:

- Target proměnná je `yield`.
- `Production` se nepoužívá jako modelová feature, protože je přímo svázána s výnosem a plochou.
- Weather features jsou agregované před joinem, takže denní weather data nemění jednotku pozorování.
- Split je chronologický podle `Crop_Year`, ne náhodný.
- Preprocessing modelů v benchmarku je fitovaný pouze na train sadě.
- Validation sada slouží pro porovnání benchmarků.
- Test sada zůstává stranou pro pozdější finální vyhodnocení.

## 7. Notebooky

Hlavní notebooky:

```text
dataprocessing.ipynb
benchmark.ipynb
```

Exportované zálohy:

```text
dataprocessing.html
benchmark.html
```

Notebook `dataprocessing.ipynb` obsahuje:

- připomenutí task framingu,
- popis dat,
- weather join,
- leakage audit,
- vytvoření train/validation/test sad,
- preprocessing rozhodnutí,
- krátké shrnutí.

Notebook `benchmark.ipynb` obsahuje:

- naivní baseline,
- jednoduchý historický baseline,
- jednoduchý ML benchmark,
- vyhodnocení pouze na validační sadě.

## 8. Jak spustit projekt

Nejprve je vhodné nainstalovat závislosti:

```bash
pip install -r requirements.txt
```

Poté je možné spustit notebooky:

```text
dataprocessing.ipynb
benchmark.ipynb
```

Pokud je potřeba znovu spojit crop a weather data:

```bash
python src/join_crop_weather.py
```

Pokud je potřeba znovu vytvořit finální split:

```bash
python src/finalize_hw2.py
```

## 9. Benchmark

Benchmark je záměrně jednoduchý. Cílem HW2 není maximalizovat skóre, ale vytvořit metodicky správný základ pro další modelování.

Použité benchmarky:

- naivní baseline,
- historický baseline přes předchozí známý výnos,
- jednoduchý ML benchmark.

Modely jsou porovnávány pouze na validační sadě. Test set není použit v benchmark notebooku.
