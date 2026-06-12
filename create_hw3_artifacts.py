#!/usr/bin/env python3
"""
Vytvorí, spustí a exportuje finálny notebook k MML1 Úlohe 3.

Spúšťaj z koreňa repozitára:
    python create_hw3_artifacts.py --execute

Skript nič netrénuje ani nemení výsledky modelov. Notebook iba načíta
zmrazené validačné a finálne testovacie reporty.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "ukol3_evaluace.ipynb"
HTML_PATH = ROOT / "ukol3_evaluace.html"

REQUIRED_FILES = [
    "reports/hw3_pretest_full_validation_results.csv",
    "reports/hw3_pretest_resource_validation_results.csv",
    "reports/hw3_final_full_test_results.csv",
    "reports/hw3_final_resource_test_results.csv",
    "reports/hw3_validation_test_comparison.csv",
    "reports/hw3_residual_summary_by_target_quartile.csv",
    "reports/hw3_target_distribution_comparison.csv",
    "reports/hw3_worst_predictions.csv",
    "data/reference/hw3_frozen_evaluation_plan.json",
    "data/reference/hw3_final_evaluation_record.json",
    "data/reference/model_feature_manifest.json",
]


def cell_id() -> str:
    return uuid.uuid4().hex[:8]


def md(source: str):
    cell = new_markdown_cell(source.strip() + "\n")
    cell["id"] = cell_id()
    return cell


def code(source: str):
    cell = new_code_cell(source.strip() + "\n")
    cell["id"] = cell_id()
    return cell


def validate_inputs() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Chybí soubory potřebné pro vytvoření notebooku:\n" + formatted
        )


def build_notebook():
    cells = []

    cells.append(md(r"""
# Úkol 3: Komplexní evaluace, komparace a diskuze modelů

**Projekt:** Predikce výnosu zemědělských plodin v Indii  
**Typ úlohy:** regrese  
**Autor:** Peter Briedoň

Cílem notebooku je férově porovnat dosud probírané rodiny modelů, vyhodnotit
jejich generalizaci na chronologicky pozdějším testovacím období a kriticky
interpretovat výsledky. Neuronové sítě nejsou podle zadání zahrnuty.

Notebook **netrénuje nové konfigurace a neprovádí další tuning**. Načítá
výsledky modelů, jejichž seznam, preprocessing, feature set a hyperparametry
byly zmrazeny před zpřístupněním testovací sady.
"""))

    cells.append(md("""
## 1. Task framing

Jednotkou pozorování je kombinace státu, okresu, roku, sezóny a plodiny.
Cílovou proměnnou je `target_yield`, tedy korigovaný výnos vypočtený z
opravené produkce a plochy.

Projekt odpovídá regresní úloze: pro nové pozorování chceme odhadnout spojitou
hodnotu výnosu. Primární metrikou je MAE. Doplňkově používáme RMSE, R² a
MedianAE.

- **MAE** vyjadřuje průměrnou absolutní velikost chyby.
- **RMSE** výrazněji penalizuje velké chyby.
- **R²** porovnává model s konstantní predikcí průměru.
- **MedianAE** popisuje typickou chybu a je méně citlivá na extrémy.
"""))

    cells.append(code(r"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", lambda value: f"{value:.6f}")

ROOT = Path.cwd()
REPORTS = ROOT / "reports"
REFERENCE = ROOT / "data" / "reference"

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def show_image(filename, width=900):
    path = REPORTS / filename
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        print(f"Graf nebyl nalezen: {path}")

frozen_plan = read_json(REFERENCE / "hw3_frozen_evaluation_plan.json")
final_record = read_json(REFERENCE / "hw3_final_evaluation_record.json")
feature_manifest = read_json(REFERENCE / "model_feature_manifest.json")

validation_full = pd.read_csv(REPORTS / "hw3_pretest_full_validation_results.csv")
validation_resource = pd.read_csv(REPORTS / "hw3_pretest_resource_validation_results.csv")
test_full = pd.read_csv(REPORTS / "hw3_final_full_test_results.csv")
test_resource = pd.read_csv(REPORTS / "hw3_final_resource_test_results.csv")
validation_test = pd.read_csv(REPORTS / "hw3_validation_test_comparison.csv")
residual_quartiles = pd.read_csv(REPORTS / "hw3_residual_summary_by_target_quartile.csv")
target_distribution = pd.read_csv(REPORTS / "hw3_target_distribution_comparison.csv")
worst_predictions = pd.read_csv(REPORTS / "hw3_worst_predictions.csv")

print("Načtení reportů dokončeno.")
print(f"Full-data modely: {len(test_full)}")
print(f"Resource-limited modely: {len(test_resource)}")
"""))

    cells.append(md("""
## 2. Data, feature set a ochrana proti leakage

Modelovací dataset používá chronologický split:

- **train:** 1997–2010,
- **validation:** 2011–2012,
- **finální fit:** 1997–2012,
- **test:** 2013–2014.

Při finálním fitu byly train a validation spojeny až po ukončení model
selection. Testovací období zůstalo do zmrazení konfigurací oddělené.

Definitivní target je `target_yield`. Mezi zakázané leakage proměnné patří
původní produkce, korigovaná produkce, původní `yield`, samotný target a
technické sloupce popisující reconciliaci zdrojů.

Pro hlavní srovnání se používá společný feature set `core_without_lag`.
Tím mají full-data modely stejnou informační základnu. Historický lag není
součástí tohoto hlavního srovnání, protože by reprezentoval odlišný aplikační
scénář.
"""))

    cells.append(code(r"""
summary = pd.DataFrame({
    "Položka": [
        "Target",
        "Primární metrika",
        "Train období",
        "Validation období",
        "Finální fit období",
        "Test období",
        "Počet řádků finálního fitu",
        "Počet testovacích řádků",
        "Feature set",
        "Počet features bez lagu",
        "Počet full-data modelů",
        "Počet resource-limited modelů",
    ],
    "Hodnota": [
        feature_manifest["target_column"],
        frozen_plan["primary_metric"],
        frozen_plan["train_period"],
        frozen_plan["validation_period"],
        final_record["final_fit_period"],
        final_record["final_test_period"],
        final_record["final_train_rows"],
        final_record["test_rows"],
        frozen_plan["feature_set"],
        len(feature_manifest["feature_sets"]["core_without_lag"]),
        final_record["full_model_count"],
        final_record["resource_model_count"],
    ],
})
display(summary)
"""))

    cells.append(md("""
## 3. Experimentální protokol

Model selection proběhl pouze na train a validation období. Před obnovením
testovacího souboru byl uložen zmrazený plán obsahující:

- přesný seznam modelů,
- preprocessing pro každou rodinu,
- feature set,
- hyperparametry,
- validační pořadí,
- pravidlo jednorázového finálního testu.

Po zpřístupnění testu se všechny zmrazené konfigurace jednou natrénovaly na
období 1997–2012 a vyhodnotily na letech 2013–2014. Výsledky testu nebyly
použity pro změnu konfigurací.
"""))

    cells.append(code(r"""
protocol = pd.DataFrame({
    "Kontrola": [
        "Konfigurace zmrazená před testem",
        "Test použit pro model selection",
        "Test použit pro tuning hyperparametrů",
        "Proběhl tuning po testu",
        "Model vybraný validací",
        "Model s nejnižším test MAE",
        "Random state",
    ],
    "Hodnota": [
        final_record["configuration_frozen_before_test"],
        final_record["test_used_for_model_selection"],
        final_record["test_used_for_hyperparameter_tuning"],
        final_record["post_test_tuning_performed"],
        final_record["selected_from_validation_run_id"],
        final_record["reported_test_best_run_id"],
        final_record["random_state"],
    ],
})
display(protocol)
"""))

    cells.append(md("""
## 4. Zahrnuté modely

Hlavní full-data experiment zahrnuje:

- DummyRegressor s průměrem,
- DummyRegressor s mediánem,
- Linear Regression,
- Ridge,
- Lasso,
- Elastic Net,
- Decision Tree,
- Random Forest,
- Gradient Boosting,
- LinearSVR.

KNN a kernelový RBF-SVR byly vyhodnoceny odděleně na deterministickém vzorku,
protože jejich výpočetní náročnost je na plném one-hot zakódovaném datasetu
výrazně vyšší. Jejich výsledky proto nelze přímo míchat do hlavního pořadí.
Neuronové sítě nebyly podle zadání zahrnuty.
"""))

    cells.append(code(r"""
model_columns = [
    column for column in [
        "model_run_id", "model_name", "model_family", "preprocessing_family",
        "feature_set", "hyperparameters_json"
    ] if column in validation_full.columns
]
display(validation_full[model_columns].drop_duplicates().reset_index(drop=True))
"""))

    cells.append(md("""
## 5. Baseline

Baseline určuje, zda model přidává skutečnou predikční hodnotu. Použity byly
dvě naivní strategie:

- predikce průměru trénovacího targetu,
- predikce mediánu trénovacího targetu.

Reálný model má smysl pouze tehdy, pokud tyto jednoduché strategie přesvědčivě
překoná.
"""))

    cells.append(code(r"""
baseline_mask = test_full["model_family"].eq("baseline")
baseline_columns = [
    "model_name", "test_mae", "test_rmse", "test_r2", "test_median_ae"
]
display(
    test_full.loc[baseline_mask, baseline_columns]
    .sort_values("test_mae")
    .reset_index(drop=True)
)
"""))

    cells.append(md("""
## 6. Výběr modelu na validation sadě

Hlavní model byl vybrán podle nejnižšího validačního MAE. Testovací data v této
fázi nebyla přítomna. Výběr na validaci je oddělený od následného reportování
testovacího pořadí.
"""))

    cells.append(code(r"""
validation_ranking = (
    validation_full.loc[validation_full["status"].eq("completed")]
    .sort_values("validation_mae")
    .reset_index(drop=True)
)

validation_view = [
    column for column in [
        "model_name", "validation_mae", "validation_rmse",
        "validation_r2", "validation_median_ae"
    ] if column in validation_ranking.columns
]
display(validation_ranking[validation_view])

selected_validation = validation_ranking.iloc[0]
print(
    f"Validací vybraný model: {selected_validation['model_name']} "
    f"(MAE={selected_validation['validation_mae']:.6f})"
)
"""))

    cells.append(code(r"""
show_image("hw3_validation_vs_test_mae.png")
"""))

    cells.append(md("""
### Diskuze validačního výběru

Random Forest dosáhl nejnižšího validačního MAE a byl proto zmrazen jako
finální kandidát. Rozhodovací strom byl druhý. Výsledek naznačuje, že vztahy
v datech jsou výrazně nelineární a obsahují interakce, které lineární modely
nezachycují stejně dobře.

Random Forest používá průměrování většího počtu stromů, což obvykle omezuje
varianci samostatného stromu. Současně je méně interpretovatelný a výpočetně
náročnější.
"""))

    cells.append(md("""
## 7. Finální testovací výsledky

Následující pořadí je pouze výsledkem jednorázového finálního auditu. Nejde o
nové kolo model selection a po zobrazení výsledků nebyly změněny modely,
features ani hyperparametry.
"""))

    cells.append(code(r"""
test_ranking = (
    test_full.loc[test_full["status"].eq("completed")]
    .sort_values("test_mae")
    .reset_index(drop=True)
)

test_view = [
    "model_name", "test_mae", "test_rmse", "test_r2",
    "test_median_ae", "mae_generalization_gap"
]
test_view = [column for column in test_view if column in test_ranking.columns]
display(test_ranking[test_view])
"""))

    cells.append(code(r"""
show_image("hw3_final_test_mae.png")
show_image("hw3_final_test_rmse.png")
show_image("hw3_final_test_r2.png")
"""))

    cells.append(md("""
### Diskuze finálního pořadí

Random Forest zůstal nejlepším modelem i na finálním testu. Samostatný
Decision Tree dosáhl téměř stejného MAE, ale Random Forest měl výrazně nižší
RMSE a vyšší R². To znamená, že při téměř shodné průměrné absolutní chybě
Random Forest lépe omezuje velké chyby a vysvětluje větší část variability.

Lineární a regularizované modely překonaly naivní baseline, ale zaostaly za
stromovými metodami. To podporuje závěr, že vztah mezi lokalitou, plodinou,
sezónou, počasím a výnosem není dobře popsán jedinou globální lineární
funkcí.
"""))

    cells.append(code(r"""
best = test_ranking.iloc[0]
tree = test_ranking.loc[test_ranking["model_name"].eq("Decision Tree")].iloc[0]
mean_baseline = test_ranking.loc[
    test_ranking["model_name"].str.contains("mean", case=False, na=False)
].iloc[0]
median_baseline = test_ranking.loc[
    test_ranking["model_name"].str.contains("median", case=False, na=False)
].iloc[0]

comparison = pd.DataFrame({
    "Porovnání": [
        "Random Forest vs Decision Tree – rozdíl MAE",
        "Random Forest vs Decision Tree – rozdíl RMSE",
        "Random Forest vs mean baseline – absolutní zlepšení MAE",
        "Random Forest vs mean baseline – relativní zlepšení MAE",
        "Random Forest vs median baseline – absolutní zlepšení MAE",
        "Random Forest vs median baseline – relativní zlepšení MAE",
    ],
    "Hodnota": [
        tree["test_mae"] - best["test_mae"],
        tree["test_rmse"] - best["test_rmse"],
        mean_baseline["test_mae"] - best["test_mae"],
        100 * (mean_baseline["test_mae"] - best["test_mae"]) / mean_baseline["test_mae"],
        median_baseline["test_mae"] - best["test_mae"],
        100 * (median_baseline["test_mae"] - best["test_mae"]) / median_baseline["test_mae"],
    ],
})
display(comparison)
"""))

    cells.append(md("""
## 8. Overfitting a stabilita mezi validation a testem

Overfitting nelze posuzovat pouze podle jedné dvojice čísel. V tomto projektu
je navíc split chronologický, takže rozdíl mezi obdobími může vzniknout nejen
přeučením, ale také časovým distribučním posunem.

U Random Forestu je testovací MAE velmi podobné validačnímu MAE. Nevidíme tedy
výrazný propad generalizace mezi validation a testem. Samostatný strom je také
stabilní mezi těmito obdobími, ale jeho horší RMSE ukazuje vyšší citlivost na
některé velké chyby.

Train–test rozdíl je nutné interpretovat opatrně, protože historická trénovací
část obsahuje jiné extrémy targetu než testovací období.
"""))

    cells.append(code(r"""
stability_columns = [
    "model_name", "validation_mae", "test_mae",
    "test_minus_validation_mae", "validation_rmse", "test_rmse",
    "validation_r2", "test_r2"
]
stability_columns = [
    column for column in stability_columns if column in validation_test.columns
]
display(
    validation_test[stability_columns]
    .sort_values("test_mae")
    .reset_index(drop=True)
)
"""))

    cells.append(md("""
## 9. Analýza reziduí nejlepšího modelu

Reziduum je definováno jako:

`skutečná hodnota − predikce`

Kladné reziduum znamená podhodnocení skutečného výnosu. Záporné reziduum
znamená nadhodnocení.
"""))

    cells.append(code(r"""
show_image("hw3_best_actual_vs_predicted.png")
show_image("hw3_best_residuals_vs_predicted.png")
show_image("hw3_best_residual_histogram.png")
"""))

    cells.append(code(r"""
display(residual_quartiles)
"""))

    cells.append(md("""
### Diskuze reziduí

Chyba není rovnoměrná v celém rozsahu targetu. Ve třech nižších kvartilech je
MAE relativně malé, zatímco v nejvyšším kvartilu výrazně roste. U nejvyšších
výnosů převládá podhodnocování.

To vysvětluje, proč může být celkové MAE nízké, ale RMSE zůstává podstatně
vyšší: menší počet extrémních chyb má silný vliv na RMSE. Model je proto
vhodnější pro běžné hodnoty výnosu než pro přesnou predikci nejvyšších
výnosových extrémů.
"""))

    cells.append(md("""
## 10. Nejhorší predikce

Analýza největších absolutních chyb pomáhá zjistit, zda model selhává náhodně,
nebo zda se chyby soustřeďují v určitých plodinách, lokalitách či rozsazích
targetu.
"""))

    cells.append(code(r"""
display(worst_predictions.head(20))
"""))

    cells.append(md("""
Nejhorší případy nelze automaticky označit jako chybu algoritmu. Mohou
zahrnovat neobvyklé klimatické podmínky, lokální faktory chybějící ve features,
historické změny měření, extrémní hodnoty nebo zbytkové problémy kvality dat.
Proto nejsou extrémy po zobrazení testu dodatečně mazány ani winsorizovány.
"""))

    cells.append(md("""
## 11. Distribuce targetu a dataset shift

Chronologické rozdělení znamená, že jednotlivá období nemusí mít stejnou
distribuci. Kontrola targetu pomáhá oddělit možné přeučení od přirozeného
časového posunu.
"""))

    cells.append(code(r"""
display(target_distribution)
show_image("hw3_target_distribution.png")
"""))

    cells.append(md("""
Testovací období má vyšší průměr, medián i horní percentily než starší části
dat. Současně trénovací období obsahuje extrémnější maximum. Tyto rozdíly
potvrzují, že náhodný split by nebyl vhodný: smíchal by časová období a mohl
by vytvořit příliš optimistický obraz generalizace.
"""))

    cells.append(md("""
## 12. Resource-limited experiment: KNN a RBF-SVR

KNN a kernelový SVR jsou zahrnuty kvůli úplnosti srovnání probíraných metod,
ale byly spuštěny na samostatném deterministickém vzorku. Spolu s nimi byly na
stejném vzorku vyhodnoceny baseline a Decision Tree.

Výsledky slouží k porovnání uvnitř resource-limited experimentu. Nesmí být
přímo zařazeny do pořadí full-data modelů.
"""))

    cells.append(code(r"""
resource_view = [
    column for column in [
        "model_name", "test_mae", "test_rmse", "test_r2", "test_median_ae"
    ] if column in test_resource.columns
]
display(
    test_resource.loc[test_resource["status"].eq("completed"), resource_view]
    .sort_values("test_mae")
    .reset_index(drop=True)
)
"""))

    cells.append(md("""
Decision Tree byl na vzorku nejlepší. KNN překonal vzorkové naivní baseline,
ale zaostal za stromem. RBF-SVR měl s danou předem zmrazenou konfigurací vysoké
MAE. Výsledek nelze interpretovat jako obecný důkaz, že KNN nebo kernelový SVM
jsou vždy slabé; platí pouze pro tento dataset, preprocessing, vzorek a
zmrazené hyperparametry.
"""))

    cells.append(md("""
## 13. Vyplatil se složitější model?

Ano. Random Forest snížil MAE přibližně o 73 % proti predikci průměru a o
64 % proti predikci mediánu. To je natolik velký rozdíl, že použití reálného
modelu má jasnou hodnotu.

Proti jedinému rozhodovacímu stromu je rozdíl v MAE zanedbatelný. Přínos
Random Forestu se však ukazuje v nižším RMSE a vyšším R², tedy v lepší kontrole
velkých chyb. Cena za toto zlepšení je nižší interpretovatelnost, vyšší
paměťová náročnost a delší trénink.
"""))

    cells.append(md("""
## 14. Omezení interpretace

- Model predikuje výnos, nikoli ekonomický zisk.
- Výsledky nejsou kauzálním důkazem vlivu počasí nebo lokality.
- Dataset neobsahuje úplné informace o zavlažování, hnojivech, pesticidech,
  pracovní síle, cenách, odrůdách a lokálních zásazích.
- Výkon na letech 2013–2014 nezaručuje stejný výkon v současnosti.
- Nejvyšší výnosy jsou podhodnocovány častěji než běžné hodnoty.
- Resource-limited modely nejsou přímo srovnatelné s full-data pořadím.
- Testovací sada byla použita jednorázově; další tuning podle těchto výsledků
  by porušil její roli nezávislého auditu.
"""))

    cells.append(md("""
## 15. Závěr

Validace vybrala Random Forest a jednorázový finální test tuto volbu podpořil.
Model dosáhl nejnižšího testovacího MAE, nejlepšího RMSE a nejvyššího R² mezi
full-data konfiguracemi.

Samostatný Decision Tree měl prakticky stejné MAE a představuje zajímavou
jednodušší alternativu. Random Forest je však vhodnější jako finální
predikční model, protože výrazně lépe omezuje velké chyby.

Nejdůležitějším výsledkem není pouze pořadí modelů, ale metodika: chronologický
split, preprocessing uvnitř pipeline, explicitní leakage audit, baseline,
zmrazení konfigurací před testem a následná analýza reziduí a limitů.
"""))

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
        },
    )
    return notebook


def write_notebook() -> None:
    validate_inputs()
    notebook = build_notebook()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook vytvořen: {NOTEBOOK_PATH.name}")


def execute_and_export() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            str(NOTEBOOK_PATH),
        ],
        cwd=ROOT,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            str(NOTEBOOK_PATH),
            "--output",
            HTML_PATH.name,
        ],
        cwd=ROOT,
        check=True,
    )

    print(f"Notebook spuštěn: {NOTEBOOK_PATH.name}")
    print(f"HTML export vytvořen: {HTML_PATH.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Po vytvoření notebook spustí a exportuje do HTML.",
    )
    args = parser.parse_args()

    write_notebook()
    if args.execute:
        execute_and_export()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
