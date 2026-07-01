from pathlib import Path
import json

path = Path("notebooks/notebook_hw4.ipynb")
nb = json.loads(path.read_text(encoding="utf-8-sig"))

def lines(text):
    return [line + "\n" for line in text.strip().split("\n")]

markdown_replacements = [
"""# Úkol 4 – MLP síť a jednoduchá ablační studie

V tomto notebooku navazuji na předchozí porovnání klasických modelů. V Úkolu 3 jsem neuronové sítě nepoužíval, protože tam byly ze zadání vynechané. Tady už je zkouším jako jednu z technik z workshopu.

Cílem není tvrdit, že neuronová síť je automaticky nejlepší model pro tabulková data. Chci hlavně ověřit, jestli jednoduché MLP v PyTorchi dokáže na stejných datech rozumně predikovat výnos plodin a jak se výsledek změní při jiné velikosti sítě nebo jiné aktivační funkci.""",

"""## Data a kontrola úniku informací

Používám stejné zpracované datasety jako v předchozích částech projektu:

- train: 1997–2010,
- validation: 2011–2012,
- test: 2013–2014.

Cílová proměnná je výnos plodiny (`target_yield`). Do vstupních proměnných nedávám produkci ani samotný výnos, aby model neměl k dispozici informaci přímo odvozenou z targetu.

Číselné proměnné se doplňují mediánem a škálují pomocí `StandardScaler`. Kategorické proměnné se doplňují nejčastější hodnotou a převádějí přes `OneHotEncoder`. Celý preprocessing se fituje pouze na trénovacích datech.

Kvůli výpočetní náročnosti MLP na CPU netrénuji na úplně všech řádcích. Pro samotný trénink používám reprodukovatelný vzorek 100 000 řádků a pro sledování validační chyby 20 000 řádků. Finální metriky se ale počítají na celé validační a testovací sadě.""",

"""## Co se v experimentu trénuje

V další buňce je celý experiment. Obsahuje načtení dat, přípravu vstupů, baseline modely, definici MLP v PyTorchi, trénování s early stoppingem a výpočet metrik.

Jako baseline používám jednoduchý `DummyRegressor` a také `Ridge`, aby bylo vidět, jestli neuronová síť přidává něco navíc oproti průměrné predikci a oproti lineárnímu modelu.

U MLP porovnávám čtyři varianty:

- malá síť `[64]` s ReLU,
- střední síť `[128, 64]` s ReLU,
- hlubší síť `[256, 128, 64]` s ReLU,
- střední síť `[128, 64]` s Tanh.

Toto beru jako jednoduchou ablační studii: měním velikost sítě a aktivační funkci a sleduji, jestli se tím výsledek zlepší nebo zhorší.""",

"""## Kontrola chyb

Kromě hlavní tabulky výsledků si zobrazuji i graf predikce proti skutečné hodnotě, rezidua a nejhorší predikce. Tyto případy jsou užitečné hlavně proto, že ukazují, kde model nestačí.

U crop yield dat je normální, že některé extrémní výnosy budou predikované hůř. Může jít o lokální podmínky, kvalitu dat, specifickou plodinu, zavlažování nebo jiné faktory, které v datasetu přímo nejsou."""
]

mi = 0
for cell in nb["cells"]:
    if cell.get("cell_type") == "markdown":
        if mi < len(markdown_replacements):
            cell["source"] = lines(markdown_replacements[mi])
        mi += 1

new_conclusion_code = '''from IPython.display import Markdown, display

results = artifacts["results"].copy()
best_name = artifacts["best_mlp"]

best_row = results.loc[results["model"].eq(best_name)].iloc[0]
dummy_row = results.loc[results["model"].eq("Dummy mean")].iloc[0]
ridge_row = results.loc[results["model"].eq("Ridge alpha=1")].iloc[0]

summary = f"""## Stručné shrnutí

Nejlepší MLP varianta podle validačního RMSE byla **{best_name}**. Na testovacích letech dosáhla RMSE `{best_row['test_rmse']:.3f}` a R² `{best_row['test_r2']:.3f}`.

Pro srovnání, jednoduchý `Dummy mean` měl test RMSE `{dummy_row['test_rmse']:.3f}` a lineární `Ridge` měl test RMSE `{ridge_row['test_rmse']:.3f}`. MLP tedy v tomto experimentu překonalo jak naivní baseline, tak jednoduchý lineární model.

Podle mě to dává smysl, protože výnos plodin nezávisí jen na jedné proměnné. Důležitá je kombinace plodiny, regionu, sezóny, plochy a počasí. Takový vztah může být nelineární, a proto zde MLP dokázalo fungovat lépe než Ridge.

Výsledek ale neberu jako důkaz, že neuronové sítě jsou vždy nejlepší volba pro tabulková data. V Úkolu 3 se jako velmi silný ukázal Random Forest. Tuto část proto beru spíš jako samostatné rozšíření z workshopu: vyzkoušel jsem MLP v PyTorchi a porovnal několik jeho variant."""
display(Markdown(summary))
'''

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "## Conclusion" in source or 'artifacts["conclusion"]' in source:
            cell["source"] = lines(new_conclusion_code)
            cell["outputs"] = []
            cell["execution_count"] = None

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("notebook_hw4.ipynb markdown upravený")
