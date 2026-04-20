# TSP Performance Graphs - Sarcina C

## Status: ✅ COMPLET

### Implemented:
- [x] Grafic 1: Timp de rulare (BT prima/y_solutii vs NN base/multistart)
  - N_BT: [5,8,10,12]
  - N_NN: [5,8,10,12,15,20,30,50]
  - 2 subplots: linear + log scale
  - Output: `timp_performanta.png`
- [x] Grafic 2: Calitate pentru timp fix T (bonus)
  - Output: `calitate_timp_fix.png`
- [x] Grafic 3: Gap % vs optim (bonus)
  - Output: `gap_optimal.png`

### Files:
- `src/utils/performance.py` - Contains `ruleaza_experiment()` matching spec

### Run:
```bash
cd src/utils && python performance.py
```
Generates PNG files in cwd.
