# Predikce výnosu plodin v Indii

Tento repozitář je pracovní větev pro čistou přestavbu crop, geography, weather a soil pipeline. Původní odevzdaná HW2 verze je zachovaná v Git tagu `hw2-original`.

## Aktuální stav

- Projekt se přerábí od začátku nad crop daty pro období 1997-2014.
- Jednotka pozorování zůstává `State_Name + District_Name + Crop_Year + Season + Crop`.
- Každý sezonní crop řádek v připravovaném calendar výstupu má explicitní weather okno `start_date` až `end_date`.
- Crop calendar má 2 069 pravidel.
- Všech 486 680 řádků má přiřazené `start_date` a `end_date`; pokrytí aplikace kalendáře je 100 %.
- Dalším krokem je vytvoření district crosswalku a ověřených bodů okresů.
- Denní počasí bude později získané z NASA POWER.
- Půdní vlastnosti budou později získané ze SoilGrids.
- Stará geografická a weather pipeline se už nepoužívá.
- Audit názvů okresů byl vytvořen; souřadnice zatím nebyly přiřazeny.

Nová geography, weather ani soil pipeline zatím nejsou dokončené.

## Modelovací rozhodnutí

- Úloha je regresní predikce targetu `yield`.
- `Production` nebude použité jako modelová feature.
- Split bude chronologický:
  - train: 1997-2010
  - validation: 2011-2012
  - test: 2013-2014
- Test set nebude použitý před finální evaluací.

## Projektové soubory

- `data/raw/Indian_crop_production_yield_dataset.csv` - hlavní crop dataset uložený v repozitáři.
- `data/reference/crop_calendar_rules_1997_2014_v1.csv` - pravidla crop kalendáře.
- `data/reference/required_districts_1997_2014.csv` - reference soubor pro potřebné okresy.
- `data/reference/district_crosswalk_template_1997_2014.csv` - šablona district crosswalku.
- `data/interim/crop_with_calendar_dates_1997_2014.csv` - velký lokální mezivýsledek ignorovaný Gitem.
- `reports/crop_calendar_application_validation.csv` - validační výstup aplikace kalendáře.
- `reports/crop_calendar_application_summary.txt` - textové shrnutí aplikace kalendáře.
- `reports/district_requirements_summary.txt` - shrnutí požadavků na okresy.

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
