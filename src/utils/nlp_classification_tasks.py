"""Laborator 10 – NLP Classification (20 Newsgroups)

Script dedicat pentru rularea unei singure sarcini din laborator prin flag-uri:
    python src\nlp_classification_tasks.py --task 1

Task-uri:
  1 - Sarcina 1: Naive Bayes + TF-IDF (ngram_range=(1,1), max_features=None, stop_words='english')
  2 - Sarcina 2: Compararea clasificatorilor (NB, LinearSVC, LogisticRegression, RandomForest)
  3 - Sarcina 3: Studiu ngram_range (SVM)
  4 - Sarcina 4: Studiu max_features (SVM)
  5 - Sarcina 5 (opțional): Grid ngram × max_features (SVM)

Funcțional: folosește aceeași logică ca `src/nlp_classification.py`, dar izolat pentru front-end.
"""

from __future__ import annotations

import argparse
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
class EvalResult:
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


def evalueaza_model(pipeline, train, test, verbose=True) -> EvalResult:
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

    return EvalResult(accuracy=acc, pred=pred, duration_s=durata, report=report, cm=cm)


def plot_matrice_confuzie(cm: np.ndarray, labels: list[str], titlu: str, no_plots: bool):
    if no_plots:
        return
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(titlu)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{titlu[:40].replace(' ', '_')}.png", dpi=150)
    plt.show()


def plot_comparatie(etichete, acurateti, titlu: str, xlabel: str, no_plots: bool):
    if no_plots:
        return
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
    plt.savefig(f"{titlu[:40].replace(' ', '_')}.png", dpi=150)
    plt.show()


def interpreteaza_confuzia(cm: np.ndarray, labels: list[str], top_k: int = 4) -> str:
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

    lines = ["Interpretare semantică (cele mai mari erori):"]
    for count, i, j in pairs:
        li = labels[i]
        lj = labels[j]
        lines.append(
            f"- {li} → {lj}: {count} documente real '{li}' prezise greșit ca '{lj}'.\n"
            f"  * Real: {sem.get(li, '')}.\n"
            f"  * Prezis: {sem.get(lj, '')}.\n"
            "  * Explicație: confuzia apare când vocabularul/temele sunt parțial suprapuse și semnalele distinctive între clase sunt limitate."
        )
    return "\n".join(lines)


def sarcina_1(train, test, no_plots: bool):
    pipeline_baza = construieste_pipeline(MultinomialNB(), ngram_range=(1, 1), max_features=None)
    res = evalueaza_model(pipeline_baza, train, test, verbose=True)

    print("\nMatricea de confuzie (rând=real, coloană=prezis):")
    print(res.cm)

    print("\n" + interpreteaza_confuzia(res.cm, train.target_names, top_k=4))

    plot_matrice_confuzie(res.cm, list(train.target_names), "NaiveBayes_confusion", no_plots=no_plots)


def sarcina_2(train, test, no_plots: bool):
    print("\n" + "=" * 60)
    print("SARCINA 2 – Compararea clasificatorilor")
    print("=" * 60)

    clasificatori = {
        "Naive Bayes": MultinomialNB(),
        "LinearSVC": LinearSVC(max_iter=2000),
        "Reg. Logistică": LogisticRegression(max_iter=1000, solver="saga"),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    }

    etichete, acurateti, durate = [], [], []
    rezultate = []

    for nume, clf in clasificatori.items():
        print(f"\n--- {nume} ---")
        pipeline = construieste_pipeline(clf, ngram_range=(1, 1), max_features=None)
        res = evalueaza_model(pipeline, train, test, verbose=False)

        print(f"Acuratețe: {res.accuracy:.4f} | Timp: {res.duration_s:.2f}s")
        etichete.append(nume)
        acurateti.append(res.accuracy)
        durate.append(res.duration_s)
        rezultate.append((res.accuracy, nume, res))

    print("\n" + "-" * 50)
    print(f"{'Clasificator':<22} {'Acuratețe':>10} {'Timp (s)':>10}")
    print("-" * 50)
    for n, a, d in zip(etichete, acurateti, durate):
        print(f"{n:<22} {a:>10.4f} {d:>10.2f}")

    best_acc, best_name, best_res = max(rezultate, key=lambda x: x[0])
    print("\n" + "-" * 50)
    print(f"Cel mai bun: {best_name} (acc={best_acc:.4f})")

    plot_matrice_confuzie(best_res.cm, list(train.target_names), f"Best_{best_name}_confusion", no_plots=no_plots)
    plot_comparatie(etichete, acurateti, "Compararea clasificatorilor", "Clasificator", no_plots=no_plots)


def sarcina_3(train, test, no_plots: bool):
    print("\n" + "=" * 60)
    print("SARCINA 3 – Variația ngram_range")
    print("=" * 60)

    configs = [(1, 1), (1, 2), (2, 2), (1, 3)]
    acurateti, etichete = [], []

    for ng in configs:
        print(f"\nngram_range = {ng}")
        pipeline = construieste_pipeline(LinearSVC(max_iter=2000), ngram_range=ng, max_features=None)
        res = evalueaza_model(pipeline, train, test, verbose=False)
        acurateti.append(res.accuracy)
        etichete.append(str(ng))
        print(f"Acuratețe: {res.accuracy:.4f}")

    plot_comparatie(etichete, acurateti, "Influența ngram_range (SVM)", "ngram_range", no_plots=no_plots)


def sarcina_4(train, test, no_plots: bool):
    print("\n" + "=" * 60)
    print("SARCINA 4 – Variația max_features")
    print("=" * 60)

    valori = [100, 500, 1000, 5000, 10000, None]
    acurateti, etichete = [], []

    for mf in valori:
        label = str(mf) if mf is not None else "toate"
        print(f"\nmax_features = {label}")
        pipeline = construieste_pipeline(LinearSVC(max_iter=2000), ngram_range=(1, 1), max_features=mf)
        res = evalueaza_model(pipeline, train, test, verbose=False)
        acurateti.append(res.accuracy)
        etichete.append(label)
        print(f"Acuratețe: {res.accuracy:.4f}")

    plot_comparatie(etichete, acurateti, "Influența max_features (SVM)", "max_features", no_plots=no_plots)


def sarcina_5(train, test, no_plots: bool):
    print("\n" + "=" * 60)
    print("SARCINA 5 (opțional) – Grid ngram × max_features")
    print("=" * 60)

    try:
        import seaborn as sns
    except ImportError:
        print("Instalați seaborn: pip install seaborn")
        return

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
            res = evalueaza_model(pipeline, train, test, verbose=False)
            rezultate[i, j] = res.accuracy
            print(f"  ngram={ng}, max_features={mf}: acc={res.accuracy:.4f}")

    if no_plots:
        return

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    train, test = incarca_date()

    if args.task == 1:
        sarcina_1(train, test, no_plots=args.no_plots)
    elif args.task == 2:
        sarcina_2(train, test, no_plots=args.no_plots)
    elif args.task == 3:
        sarcina_3(train, test, no_plots=args.no_plots)
    elif args.task == 4:
        sarcina_4(train, test, no_plots=args.no_plots)
    elif args.task == 5:
        sarcina_5(train, test, no_plots=args.no_plots)


if __name__ == "__main__":
    main()

