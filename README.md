# Predikce výnosu plodin v Indii

Tento repozitář je pracovní větev pro čistou přestavbu crop, geography, weather a soil pipeline. Původní odevzdaná HW2 verze je zachovaná v Git tagu `hw2-original`.

## Aktuální stav

- Projekt se přerábí nad crop daty pro období 1997-2014.
- Jednotka pozorování zůstává `State_Name + District_Name + Crop_Year + Season + Crop`.
- Všech 486 680 crop řádků má přiřazené `start_date`, `end_date`, `district_id`, `weather_point_id` a `weather_window_id`.
- Crop calendar má 2 069 pravidel a pokrytí aplikace kalendáře je 100 %.
- Pro 701 weather bodů jsou kompletně stažená denní NASA POWER data za období 1997-03-01 až 2015-10-14.
- Weather features jsou agregované podle crop-specific `start_date` a `end_date`.
- Agregace vytvořila 150 832 weather-window feature řádků; všechna okna jsou validní a minimum coverage je 1.000000.
- Výsledný crop-weather dataset má 486 680 řádků, 65 sloupců, 727 okresů, 700 použitých weather bodů a 150 832 weather windows.
- Velké Parquet výstupy a NASA cache zůstávají lokální a jsou ignorované Gitem.
- Modelování a čištění konfliktního targetu ještě nebyly vykonané.
- Půdní vlastnosti budou později získané ze SoilGrids.

## Geografické přiřazení

Okresní body jsou vytvořené z Census 2001 a Census 2011 polygonů přes representative point. Pro každý crop řádek se používá census verze podle `Crop_Year`.

Geografické přiřazení zůstává označené podle confidence:

- `confirmed`
- `working_strong`
- `working_fallback`
- `historical_fallback`

Fuzzy fallbacky jsou ponechané jako auditovatelná omezení další práce.

## Modelovací rozhodnutí

- Úloha je regresní predikce targetu `yield`.
- `Production` nebude použité jako modelová feature.
- Split bude chronologický:
  - train: 1997-2010
  - validation: 2011-2012
  - test: 2013-2014
- Test set nebude použitý před finální evaluací.

## Důležité soubory

- `data/raw/Indian_crop_production_yield_dataset.csv` - hlavní crop dataset uložený v repozitáři.
- `data/reference/crop_calendar_rules_1997_2014_v1.csv` - pravidla crop kalendáře.
- `data/reference/required_districts_1997_2014.csv` - reference soubor pro potřebné okresy.
- `data/reference/district_boundary_assignments_working.csv` - pracovní přiřazení okresů k boundary vrstvám.
- `data/reference/district_point_versions.csv` - reprezentativní body pro census verze.
- `data/reference/district_points_by_crop_year.csv` - body použité podle crop roku.
- `data/reference/weather_points_unique.csv` - unikátní NASA POWER body.
- `data/reference/nasa_power_request_manifest.json` - manifest NASA POWER downloadu.
- `data/interim/crop_with_calendar_dates_1997_2014.csv` - velký lokální mezivýsledek ignorovaný Gitem.
- `data/interim/weather_daily/` - lokální NASA POWER cache ignorovaná Gitem.
- `data/interim/weather_features_by_window_1997_2014.parquet` - lokální agregované weather features ignorované Gitem.
- `data/interim/crop_weather_dataset_1997_2014.parquet` - lokální spojený crop-weather dataset ignorovaný Gitem.
- `reports/weather_window_aggregation_summary.md` - shrnutí agregace počasí.
- `reports/crop_weather_dataset_summary.md` - shrnutí finálního crop-weather datasetu.

## Spuštění bez nového stahování

```powershell
python src\run_geography_weather_pipeline.py --skip-download
```

Samostatné kroky:

```powershell
python src\aggregate_crop_weather_windows.py
python src\build_crop_weather_dataset.py
```

Testy:

```powershell
python -m pytest -q -p no:cacheprovider
```

## Struktura

```text
data/
  raw/
  reference/
  interim/
    weather_daily/
  processed/
reports/
notebooks/
src/
maps/
models/
```
