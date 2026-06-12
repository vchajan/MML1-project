# Predikce výnosu plodin v Indii

Semestrální projekt pro předmět **Úvod do strojového učení – MML1**.

Projekt řeší regresní úlohu: predikci výnosu zemědělských plodin v Indii na základě informací o lokalitě, plodině, sezóně, roku a počasí.

## Cíl Úkolu 3

Cílem třetího úkolu je komplexně porovnat dosud probírané modely, vyhodnotit je pomocí vhodných regresních metrik a kriticky diskutovat jejich výsledky.

Neuronové sítě nejsou podle zadání do srovnání zahrnuty.

Hlavním výstupem je notebook:

```text
ukol3_evaluace.ipynb
```

Notebook je uložen se všemi výstupy, tabulkami a grafy. K dispozici je také HTML export:

```text
ukol3_evaluace.html
```

## Data a cílová proměnná

Jedním pozorováním je kombinace:

```text
stát + okres + rok + sezóna + plodina
```

Cílovou proměnnou je:

```text
target_yield = Production_corrected / Area_corrected
```

Původní sloupec `yield` není použit jako vstupní proměnná modelu.

Modelovací dataset obsahuje 267 148 řádků.

## Rozdělení dat

Data jsou rozdělena chronologicky:

```text
train:       1997–2010   202 164 řádků
validation:  2011–2012    32 388 řádků
test:        2013–2014    32 596 řádků
```

Chronologické rozdělení bylo zvoleno proto, aby evaluace odpovídala reálnému scénáři predikce budoucích období z historických dat.

Po ukončení výběru modelu byly train a validation spojeny. Finální modely byly natrénovány na období 1997–2012 a jednorázově vyhodnoceny na testovacím období 2013–2014.

## Vstupní proměnné

Finální srovnání používá společný feature set `core_without_lag`.

Obsahuje:

- stát,
- okres,
- plodinu,
- sezónu,
- rok,
- opravenou plochu,
- zeměpisnou šířku a délku,
- meteorologické charakteristiky odvozené z dat NASA POWER.

Historické lag proměnné nejsou součástí hlavního srovnání Úkolu 3, aby všechny modely používaly stejnou informační základnu.

## Ochrana proti data leakage

Mezi zakázané vstupní proměnné patří zejména:

```text
Production
Production_corrected
yield
yield_source_corrected
target_yield
```

Preprocessing je součástí modelových pipeline. Imputace, kódování kategorií a škálování se fitují pouze na trénovacích datech.

Testovací sada nebyla použita pro:

- výběr modelu,
- výběr vstupních proměnných,
- tuning hyperparametrů,
- rozhodování o preprocessingu.

Před otevřením testovací sady byl uložen zmrazený plán modelů:

```text
data/reference/hw3_frozen_evaluation_plan.json
```

Finální audit evaluace je uložen v:

```text
data/reference/hw3_final_evaluation_record.json
```

## Porovnané modely

### Hlavní full-data experiment

1. DummyRegressor – mean
2. DummyRegressor – median
3. Linear Regression
4. Ridge
5. Lasso
6. Elastic Net
7. Decision Tree
8. Random Forest
9. Gradient Boosting
10. LinearSVR

### Resource-limited experiment

KNN a RBF-SVR byly kvůli výpočetní náročnosti vyhodnoceny odděleně na deterministickém vzorku.

Tento experiment obsahuje:

- DummyRegressor – mean,
- DummyRegressor – median,
- Decision Tree,
- KNN,
- RBF-SVR.

Výsledky resource-limited experimentu nejsou přímo míchány do hlavního full-data pořadí.

## Použité metriky

Pro regresní úlohu byly použity:

- **MAE** – primární metrika,
- **RMSE**,
- **R²**,
- **MedianAE**.

MAE bylo zvoleno jako hlavní metrika, protože přímo vyjadřuje průměrnou absolutní velikost chyby.

RMSE více penalizuje velké chyby a R² vyjadřuje, jakou část variability cílové proměnné model vysvětluje.

## Výsledky na validation sadě

Nejnižší validační MAE dosáhl Random Forest:

```text
MAE:  1.618119
RMSE: 5.712142
R²:   0.794407
```

Random Forest byl proto vybrán jako finální model ještě před otevřením testovací sady.

## Finální výsledky na testovací sadě

Random Forest dosáhl nejlepších výsledků mezi full-data modely:

```text
MAE:      1.611674
RMSE:     5.277702
R²:       0.839797
MedianAE: 0.424710
```

Druhý nejlepší model podle MAE byl Decision Tree:

```text
MAE:  1.612646
RMSE: 6.138621
R²:   0.783268
```

Rozdíl v MAE mezi Random Forestem a Decision Tree je velmi malý. Random Forest má však výrazně lepší RMSE a vyšší R², takže lépe omezuje velké chyby.

## Porovnání s baseline

Random Forest snížil testovací MAE:

```text
oproti mean baseline:   72.86 %
oproti median baseline: 64.34 %
```

Výsledky tedy ukazují, že použití skutečného modelu má oproti naivní baseline výrazný přínos.

## Overfitting

Random Forest dosáhl:

```text
validation MAE: 1.618119
test MAE:       1.611674
```

Mezi validation a testovací sadou nedošlo k výraznému zhoršení výsledků.

Train chyba je nižší než validation a test chyba, takže určitý generalizační rozdíl existuje. Nejde však o výrazný validation-to-test propad.

Decision Tree má větší rozdíl mezi train a test výkonem a zároveň horší RMSE. To odpovídá vyšší varianci samostatného stromu.

## Diskuze výsledků

Random Forest je nejlepší finální model, protože:

- dosáhl nejnižšího testovacího MAE,
- měl nejlepší RMSE,
- dosáhl nejvyššího R²,
- výrazně překonal oba baseline modely,
- jeho výkon zůstal stabilní mezi validation a test obdobím.

Decision Tree je jednodušší a interpretovatelnější alternativa. Jeho MAE je téměř stejné, ale horší RMSE ukazuje, že u některých pozorování dělá větší chyby.

Složitější Random Forest se proto vyplatil hlavně kvůli lepší stabilitě a menším extrémním chybám.

Reziduální analýza ukázala, že nejobtížnější jsou nejvyšší hodnoty výnosu. Model je častěji podhodnocuje.

## Hlavní soubory

```text
ukol3_evaluace.ipynb
ukol3_evaluace.html

src/run_hw3_pretest_selection.py
src/run_hw3_final_test.py

tests/test_hw3_pretest_selection.py
tests/test_hw3_final_test.py

data/reference/model_feature_manifest.json
data/reference/hw3_frozen_evaluation_plan.json
data/reference/hw3_final_evaluation_record.json

reports/hw3_pretest_full_validation_results.csv
reports/hw3_pretest_resource_validation_results.csv
reports/hw3_final_full_test_results.csv
reports/hw3_final_resource_test_results.csv
reports/hw3_final_evaluation_summary.md
```

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

## Spuštění notebooku

Notebook je možné otevřít přímo v Jupyteru:

```bash
jupyter notebook ukol3_evaluace.ipynb
```

Případně lze z již vytvořených výsledků znovu vytvořit notebook a HTML export:

```bash
python create_hw3_artifacts.py --execute
```

## Testy

```bash
python -m pytest -q -p no:cacheprovider
```

Při poslední kontrole prošlo všech 163 testů.

## Omezení

- Model predikuje výnos, ne ekonomický zisk.
- Výsledky nejsou důkazem kauzálního vlivu počasí.
- Dataset neobsahuje úplné informace o zavlažování, hnojivech, pesticidech, cenách ani odrůdách.
- Výsledky na období 2013–2014 nezaručují stejný výkon na současných datech.
- Nejvyšší hodnoty výnosu jsou pro model nejobtížnější.
- KNN a RBF-SVR byly vyhodnoceny pouze v resource-limited experimentu.
- Testovací sada nesmí být použita pro další tuning.
