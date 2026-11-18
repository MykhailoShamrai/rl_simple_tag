# Wytrenowane modele (punkt kontrolny 2)

Kopie `final.pt` wszystkich 36 treningów z nocy 17/18.11.2026 (`run_experiments.py`), razem z
`config.json`, `log.csv` (metryki z każdej iteracji) i `eval.csv` (ewaluacje okresowe).
Checkpointy pośrednie (`ckpt_*k.pt`, `latest.pt`, `best.pt`) zostały lokalnie w `runs/`.

| Katalog | Co |
|---|---|
| `predator_<algo>_adv<N>_spd<v>_s<seed>` | drapieżniki (IPPO/MAPPO) trenowane przeciw losowej ofierze; N drapieżników, prędkość ofiary v× |
| `prey_<algo>_adv3_spd1.0_vs-predator_mappo_adv3_spd1.0_s1_s<seed>` | ofiara trenowana przeciw zamrożonym drapieżnikom `predator_mappo_adv3_spd1.0_s1` |
| `frozen_predator.txt` | dlaczego zamrożono właśnie te drapieżniki |

Katalog modelu można podać wszędzie tam, gdzie katalog z `runs/`:

```bash
python demo.py --right-pred models/predator_mappo_adv3_spd1.0_s1
python demo.py --left-pred models/predator_mappo_adv3_spd1.0_s1 --right-pred models/predator_mappo_adv3_spd1.0_s1 \
               --right-prey models/prey_mappo_adv3_spd1.0_vs-predator_mappo_adv3_spd1.0_s1_s0
python evaluate.py --pred models/predator_ippo_adv3_spd1.0_s0 --prey random
python evaluate.py --suite --runs-dir models --out results/trained_eval.csv   # pełna ewaluacja
python plot_results.py --runs-dir models                                       # krzywe uczenia
python ppo.py --team prey --algo mappo --opponent models/predator_mappo_adv3_spd1.0_s1  # nowy trening ofiary
```

Uwaga: `final.pt` zawiera też krytyka i stan optymalizatora, ale `--resume` wymaga `latest.pt`
(jest tylko w `runs/`). Do dalszego treningu od zera wystarczy `ppo.py` z nowym `--run-dir`.
