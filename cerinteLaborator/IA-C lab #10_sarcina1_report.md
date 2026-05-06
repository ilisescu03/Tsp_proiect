# Sarcina 1 – Clasificare de bază (Naive Bayes + TF-IDF)

## Ce am implementat
- Script: `src/nlp_classification_sarcina1.py`
- Configurație TF‑IDF (implicită, exact cum cere enunțul):
  - `ngram_range=(1,1)`
  - `max_features=None`
  - `stop_words='english'`
- Clasificator: `MultinomialNB`

## Starea rulării (în acest mediu)
- Instalare/dependețe: `scikit-learn` a fost instalat.
- Rularea scriptului a eșuat înainte de generarea rezultatelor din cauza întreruperii la încărcarea dataset-ului `fetch_20newsgroups()` (`KeyboardInterrupt`).

## Ce trebuie rulat pentru a obține rezultatele cerute (pe mașina ta)
Rulează:
```bat
python src\nlp_classification_sarcina1.py
```

(în cazul în care vrei și imagine PNG pentru matricea de confuzie):
```bat
python src\nlp_classification_sarcina1.py --plot
```

## Formatul raportului (ce va fi afișat de script)
1. **Accuracy (test)**
2. **classification_report** cu precizie/recall/F1 pentru fiecare categorie
3. **Confusion matrix** (rând = real, coloană = prezis)
4. **Interpretare semantică**: perechile de categorii cu cele mai multe confuzii din matricea de confuzie (top-k) și explicație euristică bazată pe suprapunerea vocabularului/temelor.

