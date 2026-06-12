# Predikce výnosu plodin v Indii – MML1

Autor: Peter Briedoň

Tento repozitář obsahuje semestrální projekt pro předmět **Úvod do strojového učení – MML1**. Projekt řeší regresní predikci výnosu zemědělských plodin v Indii a navazuje na tři úkoly:

1. task framing,
2. příprava dat, leakage audit, split a benchmark,
3. komplexní evaluace, porovnání a diskuze modelů.

Původní odevzdaná verze HW2 je zachována v Git tagu `hw2-original`. Finální HW3 verze je zachována v odovzdávacím Git tagu a je dostupná na hlavní větvi repozitáře.

## Aktuální stav

Finální HW3 workflow je dokončeno:

- konfigurace modelů byly zmrazeny před zpřístupněním testovací sady,
- model selection proběhl na train období 1997–2010 a validation období 2011–2012,
- finální modely byly natrénovány na období 1997–2012,
- jednorázová finální evaluace proběhla na test období 2013–2014,
- po zobrazení testovacích výsledků neproběhl žádný další tuning,
- finální notebook `ukol3_evaluace.ipynb` byl spuštěn od začátku do konce a obsahuje uložené výstupy,
- HTML export je uložen jako `ukol3_evaluace.html`,
- kompletní testovací sada projektu prošla: **163 testů**.

Autoritativní audit finální evaluace je uložen v:

```text
data/reference/hw3_final_evaluation_record.json
```

## Cíl projektu

Cílem je predikovat spojitou cílovou proměnnou:

```text
target_yield = Production_corrected / Area_corrected
```

Jednotka pozorování:

```text
canonical_state_name
+ canonical_district_name
+ Crop_Year
+ Season_canonical
+ Crop_canonical
```

Projekt predikuje zemědělský výnos. Nepredikuje ekonomický zisk a výsledky nelze interpretovat kauzálně.

## Data a jejich opravy

Raw crop dataset obsahoval dvě zdrojové verze:

```text
legacy_source
expanded_source_x100
```

Druhá verze měla systematickou chybu měřítka produkce `×100`. Při source reconciliation byla proto normalizována faktorem `0.01`.

Definitivní target je odvozen z opravené produkce a opravené plochy. Původní raw sloupec `yield` zůstává pouze jako diagnostická zdrojová hodnota.

Hlavní velikosti dat:

```text
kanonický crop-weather dataset: 270 300 řádků
model-base dataset:             267 150 řádků
modelovací dataset:             267 148 řádků
train 1997–2010:                202 164 řádků
validation 2011–2012:            32 388 řádků
test 2013–2014:                  32 596 řádků
finální fit 1997–2012:          234 552 řádků
```

Dva konkrétní interně poškozené záznamy byly vyloučeny pouze z modelovací vrstvy před vytvořením lag feature. Neprobíhá obecné mazání target outlierů, clipping ani winsorizace podle testovacích výsledků.

## Weather a geografie

Denní meteorologická data pocházejí z NASA POWER. Byla získána podle reprezentativních bodů okresů a agregována na crop-specific časová okna.

Modelovací manifest obsahuje:

- 4 kategorické features,
- 4 numerické core features,
- 23 weather features,
- volitelně 2 lag features.

Hlavní finální HW3 porovnání používá společný feature set:

```text
core_without_lag
```

Tím mají full-data modely stejnou informační základnu. Historické lag experimenty zůstávají v repozitáři jako doplňková vývojová větev, ale nejsou hlavním autoritativním pořadím Úlohy 3.

## Leakage audit

Mezi zakázané modelové vstupy patří zejména:

```text
Production
Production_corrected
yield
yield_source_corrected
target_yield
```

a technické sloupce popisující source reconciliation, identifikátory a validitu weather oken.

Další ochrany:

- split je chronologický, nikoli náhodný,
- imputery, encodery a scalery se fitují pouze uvnitř modelových pipeline,
- test 2013–2014 nebyl použit pro feature selection, preprocessing rozhodnutí, tuning ani výběr modelu,
- model list, features, preprocessing a hyperparametry byly uloženy před obnovením testovacího souboru.

## Chronologický experimentální protokol

```text
train:       1997–2010
validation:  2011–2012
final fit:   1997–2012
test:        2013–2014
```

Před finálním testem byl vytvořen zmrazený plán:

```text
data/reference/hw3_frozen_evaluation_plan.json
```

Finální audit:

```text
data/reference/hw3_final_evaluation_record.json
```

Audit potvrzuje:

```text
configuration_frozen_before_test = true
test_used_for_model_selection = false
test_used_for_hyperparameter_tuning = false
post_test_tuning_performed = false
```

## Modely v hlavním full-data porovnání

Finální full-data evaluace zahrnuje 10 konfigurací:

1. DummyRegressor – mean,
2. DummyRegressor – median,
3. Linear Regression,
4. Ridge,
5. Lasso,
6. Elastic Net,
7. Decision Tree,
8. Random Forest,
9. Gradient Boosting,
10. LinearSVR.

Neuronové sítě nejsou podle zadání Úlohy 3 zahrnuty.

## Resource-limited experiment

KNN a kernelový RBF-SVR byly vyhodnoceny odděleně na deterministickém vzorku, protože jejich náklady špatně škálují na plném vysoko-dimenzionálním one-hot datasetu.

Samostatný resource-limited experiment obsahuje:

- DummyRegressor mean,
- DummyRegressor median,
- Decision Tree,
- KNN,
- RBF-SVR.

Jeho výsledky se nesmějí přímo míchat do hlavního full-data pořadí.

## Finální výsledky

Validace vybrala:

```text
Random Forest
run_id: random_forest_200_leaf_20
```

Validační metriky:

```text
MAE:  1.618119
RMSE: 5.712142
R²:   0.794407
```

Finální testovací metriky:

```text
MAE:      1.611674
RMSE:     5.277702
R²:       0.839797
MedianAE: 0.424710
```

Random Forest zlepšil MAE:

```text
oproti mean baseline:   72.86 %
oproti median baseline: 64.34 %
```

Decision Tree dosáhl téměř stejného MAE:

```text
Decision Tree MAE: 1.612646
```

Random Forest je však preferovaný finální model, protože má podstatně nižší RMSE a vyšší R², tedy lépe omezuje velké chyby.

## Interpretace overfittingu

Random Forest měl:

```text
validation MAE: 1.618119
test MAE:       1.611674
```

Mezi validation a test obdobím tedy nedošlo k výraznému propadu. Rozdíly mezi obdobími však nelze automaticky označit za overfitting, protože chronologický split zachycuje také možný temporal dataset shift.

Reziduální analýza ukazuje, že nejvyšší kvartil targetu je výrazně obtížnější a model v něm častěji podhodnocuje vysoké výnosy.

## Hlavní soubory Úlohy 3

```text
ukol3_evaluace.ipynb
ukol3_evaluace.html
create_hw3_artifacts.py

src/run_hw3_pretest_selection.py
src/run_hw3_final_test.py

tests/test_hw3_pretest_selection.py
tests/test_hw3_final_test.py

data/reference/hw3_frozen_evaluation_plan.json
data/reference/hw3_final_evaluation_record.json

reports/hw3_pretest_full_validation_results.csv
reports/hw3_pretest_resource_validation_results.csv
reports/hw3_pretest_selection_summary.md

reports/hw3_final_full_test_results.csv
reports/hw3_final_resource_test_results.csv
reports/hw3_final_evaluation_summary.md
reports/hw3_validation_test_comparison.csv
reports/hw3_residual_summary_by_target_quartile.csv
reports/hw3_worst_predictions.csv
```

Grafy finální evaluace jsou uloženy v `reports/` pod prefixem `hw3_`.

## Notebook Úlohy 3

Notebook:

```text
ukol3_evaluace.ipynb
```

obsahuje:

- task framing,
- popis targetu, dat a chronologického splitu,
- leakage audit,
- baseline,
- lineární a regularizované modely,
- stromové a ensemble modely,
- SVM,
- oddělený KNN a RBF-SVR experiment,
- validační výběr modelu,
- finální MAE, RMSE, R² a MedianAE,
- porovnávací grafy,
- diskuzi overfittingu,
- reziduální analýzu,
- nejhorší predikce,
- porovnání s baseline,
- kritické zhodnocení přínosu složitějších modelů,
- limity interpretace.

Notebook načítá již vytvořené, auditované reporty. Neprovádí další post-test tuning.

## Instalace

```bash
python -m venv .venv
```

Aktivace ve Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instalace závislostí:

```bash
pip install -r requirements.txt
```

## Reprodukce HW3 workflow

### 1. Pre-test validace a zmrazení konfigurací

Tuto fázi již není vhodné opakovat za účelem změny konfigurací podle známých testovacích výsledků. Příkazy dokumentují původní workflow:

```bash
python src/run_hw3_pretest_selection.py --phase full-validation --resume
python src/run_hw3_pretest_selection.py --phase resource-validation --resume
python src/run_hw3_pretest_selection.py --phase freeze
```

### 2. Jednorázová finální testovací evaluace

```bash
python src/run_hw3_final_test.py --phase full-test --resume
python src/run_hw3_final_test.py --phase resource-test --resume
python src/run_hw3_final_test.py --phase finalize
```

### 3. Vytvoření finálního notebooku a HTML

```bash
python create_hw3_artifacts.py --execute
```

### 4. Testy

```bash
python -m pytest -q -p no:cacheprovider
```

Ověřený stav finální verze:

```text
163 passed
```

## Historické a doplňkové experimenty

Repozitář obsahuje i předchozí validační, time-aware, lag, residual a log-target experimenty. Ty dokumentují vývoj projektu a další aplikační scénáře.

Pro finální Úlohu 3 jsou autoritativní zejména:

```text
data/reference/hw3_frozen_evaluation_plan.json
data/reference/hw3_final_evaluation_record.json
reports/hw3_final_*
ukol3_evaluace.ipynb
```

Starší soubory `frozen_model_configuration.json` a
`frozen_tuned_model_configuration.json` nejsou hlavním finálním pořadím Úlohy 3.

## Omezení

- Model predikuje výnos, nikoli ekonomický profit.
- Výsledky nejsou kauzálním důkazem vlivu počasí.
- Dataset neobsahuje kompletní náklady, ceny, odrůdy, zavlažování ani lokální agronomické zásahy.
- Výkon na období 2013–2014 nezaručuje stejný výkon v současnosti.
- Nejvyšší výnosové hodnoty jsou obtížnější a častěji podhodnocované.
- Resource-limited experiment není přímo srovnatelný s full-data pořadím.
- Další tuning podle finální testovací sady by porušil její roli nezávislého auditu.

