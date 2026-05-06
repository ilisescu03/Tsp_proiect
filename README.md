
# TSP Proiect — Laborator #03 (Backtracking vs Hill Climbing)

Proiect Python structurat pentru rezolvarea problemei Comis-Voiajorului (TSP):

- **Backtracking (optim, exhaustiv + branch-and-bound)**
- **Hill Climbing (euristic) cu `simpleai` + random restarts + vecinătate 2-opt**
- **Experiment comparativ** cu grafic PNG (scară liniară + logaritmică)

## Cerințe

- Python 3.x
- Dependențe: `simpleai`, `matplotlib`, `seaborn`

Instalare dependențe:

```bash
pip install -r requirements.txt
```

## Rulare (Windows)

### 1) Activare venv (opțional, recomandat)

Dacă folosești PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Dacă folosești Command Prompt:

```bat
.venv\Scripts\activate
```

### 2) Comenzi CLI

Scriptul de intrare este: `src/main.py`.

Ajutor:

```bash
python src/main.py --help
python src/main.py solve --help
python src/main.py experiment --help
```

Rezolvare TSP din fișier (backtracking):

```bash
python src/main.py solve <fisier_intrare> --algo bt
```

Rezolvare TSP din fișier (hill climbing):

```bash
python src/main.py solve <fisier_intrare> --algo hc --restarts 30 --iterations 2000 --seed 42
```

Salvare rezultat într-un fișier text:

```bash
python src/main.py solve <fisier_intrare> --algo bt --output rezultat.txt
```

Rulare experiment + generare grafic:

```bash
python src/main.py experiment --output comparare_performanta.png
```

# Laborator #04 (Backtracking moduri + Nearest Neighbor)

 Lab #04 adaugă:

- **Backtracking cu 4 moduri de oprire**: `prima`, `toate`, `timp`, `y_solutii`
- **Nearest Neighbor (NN) manual**: `nn`
- **Nearest Neighbor via AIMA** (wrapper): `nn_aima`
- **Experiment Lab4**: `experiment4`

Notă: pentru compatibilitate, CLI acceptă atât `--algo` cât și alias-ul `--algoritm`.

### Backtracking (Lab4) — moduri de oprire

Prima soluție găsită:

```bash
python src/main.py solve <fisier_intrare> --algoritm bt --mod prima
```

Exhaustiv (optim garantat):

```bash
python src/main.py solve <fisier_intrare> --algoritm bt --mod toate
```

Limită de timp (secunde):

```bash
python src/main.py solve <fisier_intrare> --algoritm bt --mod timp --timp 5
```

După Y soluții complete:

```bash
python src/main.py solve <fisier_intrare> --algoritm bt --mod y_solutii --y 10
```

### Nearest Neighbor (NN) — implementare manuală

NN cu start fix:

```bash
python src/main.py solve <fisier_intrare> --algoritm nn --mod prima --start 0
```

NN multistart (Y = N starturi):

```bash
python src/main.py solve <fisier_intrare> --algoritm nn --mod y_solutii
```

NN în limită de timp (starturi aleatorii până expiră timpul):

```bash
python src/main.py solve <fisier_intrare> --algoritm nn --mod timp --timp 1 --seed 42
```

### NN via AIMA (`nn_aima`)

```bash
python src/main.py solve <fisier_intrare> --algoritm nn_aima --mod prima --start 0
python src/main.py solve <fisier_intrare> --algoritm nn_aima --mod y_solutii
```

Notă: în unele versiuni PyPI ale `aima3`, funcția NN TSP poate lipsi; în acest caz,
implementarea `nn_aima` face fallback automat la NN manual.

### Experiment Lab4

Generează graficul comparativ pentru cazul a) și c):

```bash
python src/main.py experiment4 --output comparare_performanta_lab4.png
```

# Laborator #08 (Simulated Annealing + TSP)

Lab #08 adaugă:

- **Simulated Annealing (Python pur)** pe matricea de distanțe (vecinătate 2-opt)
- **Vizualizările V1–V7** cerute în laborator (generate automat)
- **Comparație `simanneal` vs implementare proprie** (V6)

### Rezolvare din fișier (SA, Python pur)

```bash
python src/main.py solve orase.txt --algo sa --init nn --tmax 10000 --tmin 1 --alpha 0.995 --iters-per-temp 100 --seed 42
```

### Generare vizualizări V1–V7

Generează automat PNG-uri într-un folder (implicit `lab8_out/`):

```bash
python src/main.py lab8 --n 20 --seed 42 --outdir lab8_out
```

Notă: pentru V6 (benchmark), este necesar pachetul `simanneal`.

# Laborator #09 (Algoritmi genetici + TSP)

Lab #09 adaugă o rezolvare TSP folosind **algoritmi genetici** cu biblioteca **PyGAD**:

- Reprezentare: permutare a orașelor
- Fitness: `fitness = -distanta_totală`
- Crossover: **OX (Order Crossover)**
- Mutație: **swap mutation**
- Studii de parametri (task1–task5) + grafice/rapoarte salvate

### Rulare task-uri Lab9

Task 1 (rulare primară + rută + convergență):

```bash
python src/main.py lab9 --mode task1 --outdir lab9_out --seed 42
```

Task 2 (studiu mărime populație):

```bash
python src/main.py lab9 --mode task2 --outdir lab9_out --seed 42
```

Task 3 (studiu rată mutație):

```bash
python src/main.py lab9 --mode task3 --outdir lab9_out --seed 42
```

Task 4 (studiu selecție părinți):

```bash
python src/main.py lab9 --mode task4 --outdir lab9_out --seed 42
```

Task 5 (scalabilitate pe N={15,20,25} orase random):

```bash
python src/main.py lab9 --mode task5 --outdir lab9_out --seed 42
```

## Format fișier de intrare (ex. `orase.txt`)

Fișierul de intrare este un text simplu:

1. Prima linie: `N` (numărul de orașe)
2. Următoarele `N` linii: matricea `N×N` de distanțe (simetrică, diagonala 0)

Exemplu:

```text
4
0 10 15 20
10 0 35 25
15 35 0 30
20 25 30 0
```

## Output

Comanda `solve` afișează:

- numărul de orașe
- algoritmul folosit
- traseul (închis: revine la start)
- costul total
- timpul de execuție

Comanda `experiment` salvează fișierul PNG cu două subploturi (liniar + `semilogy`).
---

## Laborator 10 (NLP) – Clasificarea textelor (20 Newsgroups)

Fiecare sarcină din scriptul nlp_classification_tasks.py explorează o dimensiune specifică a fluxului de procesare NLP:

Sarcina 1 (Naive Bayes): Implementează modelul de bază Multinomial Naive Bayes, utilizat frecvent în clasificarea textelor datorită eficienței sale pe date rare (sparse data).

Sarcina 2 (Comparare Clasificatori): Compară performanța mai multor modele (de exemplu, Naive Bayes vs. Linear SVM) pentru a vedea care arhitectură separă mai bine categoriile de știri.

Sarcina 3 (N-gram Range): Studiază cum influențează contextul (bigrame sau trigrame) acuratețea, trecând de la simple cuvinte izolate la secvențe de cuvinte.

Sarcina 4 (Max Features): Analizează compromisul dintre dimensiunea vocabularului și performanță, limitând numărul de cuvinte cheie păstrate de vectorizator.

Sarcina 5 (Grid Search): Realizează o căutare exhaustivă prin combinarea diferitelor intervale de n-grame și limite de trăsături pentru a identifica configurația optimă.

- `src/utils/nlp_classification_tasks.py`

### Instalare dependențe (dacă nu sunt deja)

```bash
pip install -r requirements.txt
```

> Scriptul folosește `scikit-learn`.

### Rulare: o singură sarcină (cu flag-uri)

Rulare Sarcina 1 (Naive Bayes):
```bat
python src\utils\nlp_classification_tasks.py --task 1
```

Rulare Sarcina 2 (comparare clasificatori):
```bat
python src\utils\nlp_classification_tasks.py --task 2
```

Rulare Sarcina 3 (ngram_range):
```bat
python src\utils\nlp_classification_tasks.py --task 3
```

Rulare Sarcina 4 (max_features):
```bat
python src\utils\nlp_classification_tasks.py --task 4
```

Rulare Sarcina 5 (grid ngram × max_features):
```bat
python src\utils\nlp_classification_tasks.py --task 5
```

### Opțional: fără grafice (pentru backend/chatbot)

```bat
python src\utils\nlp_classification_tasks.py --task 1 --no-plots
```

## Structura proiectului

```text
requirements.txt
src/
	main.py
	utils/
		backtracking.py
		hill_climbing_tsp.py
		io_utils.py
		performance.py
		nearest_neighbor.py
		nn_aima.py
		simulated_annealing_tsp.py
		sa_visualizations.py
		genetic_algorithm_tsp.py
		lab9_visualizations.py
		nlp_classification_tasks.py
		nlp_classification.py
```
