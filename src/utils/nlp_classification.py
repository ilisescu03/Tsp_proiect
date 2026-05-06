"""Laborator 10 – Prelucrarea limbajului natural: Clasificarea textelor

Script unificat ("referință") pentru cerințele laboratorului:
- Sarcina 1: Naive Bayes (MultinomialNB) + TF-IDF cu parametrii impliciți ceruți
- Sarcina 2: Compararea clasificatorilor (NB, LinearSVC, LogisticRegression, RandomForest)
- Sarcina 3: Studiu ngram_range
- Sarcina 4: Studiu max_features
- Sarcina 5 (opțional): Grid search ngram × max_features

Rulare (Sarcina 1 + restul):
    python src/nlp_classification.py

Opțional (fără grid dacă seaborn lipsește):
    pip install seaborn

Notă: Vectorizarea TF-IDF folosește exact: stop_words='english', sublinear_tf=True.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


CATEGORII = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]


@dataclass(frozen=True)
class ModelResult:
    accuracy: float
    pred: np.ndarray
    duration_s: float
    report: str
    cm: np.ndarray


def incarca_date():
    train = fetch_20newsgroups(
        subset="train",
        categories=CATEGORII,
        remove=("headers", "footers", "quotes"),
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=CATEGORII,
        remove=("headers", "footers", "quotes"),
    )
    return train, test


def construieste_pipeline(clasificator, ngram_range=(1, 1), max_features=None):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("clf", clasificator),
        ]
    )


def evalueaza_model(pipeline, train, test, verbose=True) -> tuple[float, np.ndarray, float, str, np.ndarray]:
    start = time.time()
    pipeline.fit(train.data, train.target)
    durata = time.time() - start

    pred = pipeline.predict(test.data)
    acc = accuracy_score(test.target, pred)

    report = classification_report(test.target, pred, target_names=train.target_names)
    cm = confusion_matrix(test.target, pred)

    if verbose:
        print(f"Acuratețe: {acc:.4f}  |  Timp antrenament: {durata:.2f}s")
        print(report)

    return acc, pred, durata, report, cm


def plot_matrice_confuzie(pred, test, titlu="Matricea de confuzie"):
    cm = confusion_matrix(test.target, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test.target_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(titlu)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"matrice_{titlu[:30].replace(' ', '_')}.png", dpi=150)
    plt.show()


def plot_comparatie(etichete, acurateti, titlu, xlabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    culori = plt.cm.tab10(np.linspace(0, 1, len(etichete)))
    bare = ax.bar(etichete, acurateti, color=culori, edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acuratețe")
    ax.set_xlabel(xlabel)
    ax.set_title(titlu)
    for bar, val in zip(bare, acurateti):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{titlu[:30].replace(' ', '_')}.png", dpi=150)
    plt.show()


def interpreteaza_confuzia(cm: np.ndarray, labels: list[str], top_k: int = 4) -> str:
    # cm[i, j] = real i, prezis j
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((cm[i, j], i, j))

    pairs.sort(reverse=True, key=lambda x: x[0])
    pairs = pairs[:top_k]

    sem = {
        "sci.space": "astronomie/spațiu, rachete, misiuni orbitale",
        "rec.sport.hockey": "hochei: echipe, meciuri, reguli și jucători",
        "talk.politics.guns": "dezbatere politică despre arme (guns): legislație, opinii",
        "comp.graphics": "computer graphics: grafică, software, hardware vizual, algoritmi",
    }

    lines = ["Interpretare semantică (cele mai mari erori din matricea de confuzie):"]
    for count, i, j in pairs:
        li = labels[i]
        lj = labels[j]
        lines.append(
            f"- {li} → {lj}: {count} documente real '{li}' prezise greșit ca '{lj}'.\n"
            f"  * Real: {sem.get(li, '')}.\n"
            f"  * Prezis: {sem.get(lj, '')}.\n"
            "  * Explicație: confuzia apare frecvent când vocabularul/temele sunt parțial suprapuse "
            "(termene generice sau contexte similare), sau când modelul nu are suficiente semnale "
            "distinctive între cele două subiecte." 
        )
    return "\n".join(lines)


def studiu_clasificatori(train, test):
    print("\n" + "=" * 60)
    print("SARCINA 2 – Compararea clasificatorilor")
    print("=" * 60)

    clasificatori = {
        "Naive Bayes": MultinomialNB(),
        "LinearSVC": LinearSVC(max_iter=2000),
        "Reg. Logistică": LogisticRegression(max_iter=1000, solver="saga"),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    }

    acurateti, durate, etichete = [], [], []
    rezultate = []

    for nume, clf in clasificatori.items():
        print(f"\n--- {nume} ---")
        pipeline = construieste_pipeline(clf, ngram_range=(1, 1), max_features=None)
        acc, pred, durata, report, cm = evalueaza_model(pipeline, train, test, verbose=False)

        rezultate.append((acc, pred, nume, report, cm))
        acurateti.append(acc)
        durate.append(durata)
        etichete.append(nume)

        # raport succint per model
        print(f"Acuratețe: {acc:.4f} | Timp: {durata:.2f}s")

    print("\n" + "-" * 50)
    print(f"{'Clasificator':<22} {'Acuratețe':>10} {'Timp (s)':>10}")
    print("-" * 50)
    for nume, acc, dur in zip(etichete, acurateti, durate):
        print(f"{nume:<22} {acc:>10.4f} {dur:>10.2f}")

    print("-" * 50)

    cel_mai_bun = max(rezultate, key=lambda x: x[0])
    best_acc, best_pred, best_name, best_report, best_cm = cel_mai_bun

    print(f"\nCel mai bun: {best_name} (acc={best_acc:.4f})")
    plot_matrice_confuzie(best_pred, test, f"{best_name} – Matrice de confuzie")
    plot_comparatie(etichete, acurateti, "Compararea clasificatorilor", "Clasificator")

    # returnăm pentru a putea fi folosit în interpretarea finală (opțional)
    return cel_mai_bun


def studiu_ngram(train, test):
    print("\n" + "=" * 60)
    print("SARCINA 3 – Variația ngram_range")
    print("=" * 60)

    ngram_configs = [(1, 1), (1, 2), (2, 2), (1, 3)]

    acurateti, etichete = [], []

    for ng in ngram_configs:
        pipeline = construieste_pipeline(LinearSVC(max_iter=2000), ngram_range=ng, max_features=None)
        print(f"\nngram_range = {ng}")
        acc, _, _, _, _ = evalueaza_model(pipeline, train, test, verbose=False)
        acurateti.append(acc)
        etichete.append(str(ng))

    plot_comparatie(etichete, acurateti, "Influența ngram_range (SVM)", "ngram_range")
    return list(zip(ngram_configs, acurateti))


def studiu_max_features(train, test):
    print("\n" + "=" * 60)
    print("SARCINA 4 – Variația max_features")
    print("=" * 60)

    valori = [100, 500, 1000, 5000, 10000, None]
    acurateti, etichete = [], []

    for mf in valori:
        pipeline = construieste_pipeline(LinearSVC(max_iter=2000), ngram_range=(1, 1), max_features=mf)
        label = str(mf) if mf is not None else "toate"
        print(f"\nmax_features = {label}")
        acc, _, _, _, _ = evalueaza_model(pipeline, train, test, verbose=False)
        acurateti.append(acc)
        etichete.append(label)

    plot_comparatie(etichete, acurateti, "Influența max_features (SVM)", "max_features")
    return list(zip(valori, acurateti))


def studiu_grid(train, test):
    print("\n" + "=" * 60)
    print("SARCINA 5 (opțional) – Grid ngram × max_features")
    print("=" * 60)

    try:
        import seaborn as sns
    except ImportError:
        print("Instalați seaborn: pip install seaborn")
        return None

    ngrams = [(1, 1), (1, 2), (1, 3)]
    features = [500, 2000, 5000, 10000]
    rezultate = np.zeros((len(ngrams), len(features)))

    for i, ng in enumerate(ngrams):
        for j, mf in enumerate(features):
            pipeline = construieste_pipeline(
                LinearSVC(max_iter=2000),
                ngram_range=ng,
                max_features=mf,
            )
            acc, _, _, _, _ = evalueaza_model(pipeline, train, test, verbose=False)
            rezultate[i, j] = acc
            print(f"  ngram={ng}, max_features={mf}: acc={acc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        rezultate,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=features,
        yticklabels=[str(ng) for ng in ngrams],
        ax=ax,
    )
    ax.set_xlabel("max_features")
    ax.set_ylabel("ngram_range")
    ax.set_title("Acuratețe (SVM) – Grid ngram × max_features")

    plt.tight_layout()
    plt.savefig("grid_ngram_features.png", dpi=150)
    plt.show()

    return rezultate


def main():
    print("Se încarcă datele 20 Newsgroups...")
    train, test = incarca_date()
    print(f"Train: {len(train.data)} documente | Test: {len(test.data)} documente")

    print("\n" + "=" * 60)
    print("SARCINA 1 – Clasificare de bază (Naive Bayes)")
    print("=" * 60)

    # Sarcina 1: exact TF-IDF cu ngram_range=(1,1), max_features=None, stop_words='english'
    pipeline_baza = construieste_pipeline(MultinomialNB(), ngram_range=(1, 1), max_features=None)
    acc, pred, durata, report, cm = evalueaza_model(pipeline_baza, train, test, verbose=True)

    print("\nMatricea de confuzie (rând=real, coloană=prezis):")
    print(cm)

    print("\n" + interpreteaza_confuzia(cm, train.target_names, top_k=4))
    plot_matrice_confuzie(pred, test, "Naive Bayes – Matrice de confuzie")

    # Restul laboratorului
    studiu_clasificatori(train, test)
    studiu_ngram(train, test)
    studiu_max_features(train, test)
    studiu_grid(train, test)


if __name__ == "__main__":
    main()

