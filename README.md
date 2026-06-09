# Predikce výnosu plodin v Indii

Tento repozitář je pracovní větev pro čistou přestavbu crop, geography, weather a soil pipeline. Původní odevzdaná HW2 verze je zachovaná v Git tagu `hw2-original`.

## Aktuální stav

- Projekt se přerábí nad crop daty pro období 1997-2014.
- Jednotka pozorování pro kanonický dataset je `canonical_state_name + canonical_district_name + Crop_Year + Season_canonical + Crop_canonical`.
- Raw crop dataset obsahoval dvě zdrojové verze: `legacy_source` a `expanded_source_x100`.
- Expanded verze měla systematickou měřítkovou chybu `×100`; při reconciliaci byla normalizovaná faktorem `0.01`.
- Při konfliktním překryvu má prioritu legacy zdroj. Hodnoty se neprůměrují a nesčítají.
- Kanonický crop-weather dataset má 270 300 řádků.
- Základní modelový dataset má 267 150 řádků.
- Coconut a agregované crop kategorie zůstávají v úplném kanonickém datasetu zdokumentované, ale nejsou v základním modelovém datasetu.
- Weather pipeline nebyla při reconciliaci přepočítaná; byly zachované již vytvořené weather features.
- Chronologický train/validation/test split je hotový. Outlier treatment a modelování ještě nebyly vykonané.
- Půdní vlastnosti budou později získané ze SoilGrids.

## Hotové mezikroky

- Crop calendar má 2 069 pravidel a pokrytí aplikace kalendáře je 100 %.
- Všech 486 680 crop-weather řádků má `start_date`, `end_date`, `district_id`, `weather_point_id` a `weather_window_id`.
- Pro 701 weather bodů jsou kompletně stažená denní NASA POWER data za období 1997-03-01 až 2015-10-14.
- Agregace vytvořila 150 832 weather-window feature řádků; všechna okna jsou validní a minimum coverage je 1.000000.
- Spojený crop-weather dataset před reconciliací měl 486 680 řádků, 65 sloupců a 0 missing weather feature values.

## Source Reconciliation

Rozdělení zdrojů podle `source_row_id`:

- `source_row_id <= 236378`: `legacy_source`
- `source_row_id >= 236379`: `expanded_source_x100`

Měřítko:

- `legacy_source`: `production_scale_factor = 1.0`
- `expanded_source_x100`: `production_scale_factor = 0.01`

Definitivní target:

```text
target_yield = Production_corrected / Area_corrected
```

Raw sloupec `yield` není definitivní target. Slouží pouze jako původní zdrojová hodnota a diagnostika.

Výsledek reconciliace:

- legacy-only keys: 19 437
- expanded-only keys: 34 483
- overlapping keys: 216 380
- corroborated overlaps: 213 267
- conflicting overlaps: 3 113

## Základní Modelová Vhodnost

Z úplného kanonického datasetu jsou pro základní modelování vynechané pouze známé nekompatibilní jednotky a agregované kategorie:

- Coconut: 2 260 řádků
- Total foodgrain: 188 řádků
- Pulses total: 255 řádků
- Oilseeds total: 447 řádků

Nevyřazují se další target outliery. Outlier treatment se musí později fitovat pouze na train období.

## Chronologický Modelovací Dataset

Modelovací dataset je vytvořený z lokálního `data/interim/crop_weather_model_base_1997_2014.parquet` bez změny vstupních řádků:

- model dataset: 267 150 řádků
- train 1997-2010: 202 166 řádků
- validation 2011-2012: 32 388 řádků
- test 2013-2014: 32 596 řádků
- target `target_yield`: 0 missing hodnot
- weather feature missing hodnoty: 0

Schválené feature sety jsou uložené v `data/reference/model_feature_manifest.json`:

- `core_without_lag`: 31 features
- `core_with_lag`: 33 features
- 4 kategorické features, 4 numerické core features a 23 weather features

Lag feature `lag_yield_1y` je vytvořený explicitním self-joinem na stejný district, crop a season s `Crop_Year = Y - 1`. Nejedná se o `groupby().shift()` nad seřazenou tabulkou.

- řádky s dostupným lagem: 213 187
- řádky bez dostupného lagu: 53 963

Žádný imputer, encoder, scaler, outlier threshold ani model se při stavbě datasetu nefituje. Tyto kroky musí být později fitované pouze na train splitu. Test split 2013-2014 nebyl použitý pro model selection ani preprocessing rozhodnutí.

## Geografické Přiřazení

Okresní body jsou vytvořené z Census 2001 a Census 2011 polygonů přes representative point. Pro každý crop řádek se používá census verze podle `Crop_Year`.

Geografické přiřazení zůstává označené podle confidence:

- `confirmed`
- `working_strong`
- `working_fallback`
- `historical_fallback`

Fuzzy fallbacky jsou ponechané jako auditovatelná omezení další práce.

## Modelovací Rozhodnutí

- Úloha je regresní predikce `target_yield`.
- `Production` nebude použité jako modelová feature.
- Split je chronologický:
  - train: 1997-2010
  - validation: 2011-2012
  - test: 2013-2014
- Test set nebyl použitý před finální evaluací ani pro model selection.
- Baseline modely zatím nebyly vytvořené.

## Důležité Soubory

- `data/raw/Indian_crop_production_yield_dataset.csv` - původní raw crop dataset.
- `data/reference/crop_source_reconciliation_rules.json` - pravidla source reconciliace.
- `data/reference/crop_calendar_rules_1997_2014_v1.csv` - pravidla crop kalendáře.
- `data/reference/district_boundary_assignments_working.csv` - pracovní přiřazení okresů k boundary vrstvám.
- `data/reference/district_point_versions.csv` - reprezentativní body pro census verze.
- `data/reference/district_points_by_crop_year.csv` - body použité podle crop roku.
- `data/reference/weather_points_unique.csv` - unikátní NASA POWER body.
- `data/reference/nasa_power_request_manifest.json` - manifest NASA POWER downloadu.
- `data/interim/weather_daily/` - lokální NASA POWER cache ignorovaná Gitem.
- `data/interim/weather_features_by_window_1997_2014.parquet` - lokální agregované weather features ignorované Gitem.
- `data/interim/crop_weather_dataset_1997_2014.parquet` - lokální spojený crop-weather dataset ignorovaný Gitem.
- `data/interim/crop_weather_canonical_1997_2014.parquet` - lokální úplný kanonický dataset ignorovaný Gitem.
- `data/interim/crop_weather_model_base_1997_2014.parquet` - lokální základní modelový dataset ignorovaný Gitem.
- `data/processed/model_dataset_1997_2014.parquet` - lokální modelovací dataset ignorovaný Gitem.
- `data/processed/train_1997_2010.parquet` - lokální train split ignorovaný Gitem.
- `data/processed/validation_2011_2012.parquet` - lokální validation split ignorovaný Gitem.
- `data/processed/test_2013_2014.parquet` - lokální test split ignorovaný Gitem.
- `data/reference/model_feature_manifest.json` - schválené feature sety a modeling zásady.
- `reports/crop_source_reconciliation_summary.md` - shrnutí source reconciliace.
- `reports/crop_source_conflicts.csv` - konflikty mezi zdroji, kde byl vybraný legacy zdroj.
- `reports/crop_basic_model_exclusions.csv` - řádky vynechané ze základního modelového datasetu.
- `reports/crop_canonical_dataset_validation.csv` - validační kontroly kanonického datasetu.
- `reports/modeling_dataset_summary.md` - shrnutí modeling datasetu a splitu.
- `reports/chronological_split_validation.csv` - validační kontroly splitu.
- `reports/modeling_feature_schema.csv` - schema modelovacích sloupců a missing counts.
- `reports/modeling_unseen_categories.csv` - kategorie ve validation/test, které nejsou v train.
- `reports/modeling_lag_summary.csv` - dostupnost lag feature podle splitu.

## Spuštění

Reconciliace zdrojů:

```powershell
python src\build_canonical_crop_weather_dataset.py
```

Weather pipeline bez nového stahování:

```powershell
python src\run_geography_weather_pipeline.py --skip-download
```

Modelovací dataset a chronologický split:

```powershell
python src\build_modeling_dataset.py
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
