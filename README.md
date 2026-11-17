# MPE2 – Simple Tag (temat żółty)

Projekt z przedmiotu *Uczenie ze wzmocnieniem*. Środowisko: [`mpe2.simple_tag_v3`](https://mpe2.farama.org/)
(Multi-Agent Particle Environment, gra drapieżnik–ofiara).

**Pytanie badawcze:** czy scentralizowany krytyk (CTDE, MAPPO) daje lepszą koordynację drapieżników
niż niezależne uczenie (IPPO) i od czego zależy ta różnica (liczba drapieżników, prędkość ofiary,
zasięg obserwacji)? W drugim kroku: czy ofiara nauczy się uciekać przed zamrożonymi, wytrenowanymi
drapieżnikami.

**Podejście sekwencyjne (bez self-play):**
1. drapieżniki (IPPO / MAPPO) uczą się przeciw stałej, losowej ofierze,
2. najlepsze drapieżniki są zamrażane, a ofiara (IPPO / MAPPO) uczy się przeciw nim.

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # CPU wystarcza (małe sieci); torch w wersji CPU:
                                 # pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Uruchomienie wszystkich eksperymentów (na noc)

```bash
python run_experiments.py --dry-run     # lista treningów, nic nie uruchamia
python run_experiments.py --quick       # test całego potoku w ~3 min (runs_quick/, results_quick/)

# właściwe eksperymenty; systemd-inhibit blokuje uśpienie laptopa
mkdir -p runs
nohup systemd-inhibit --what=sleep:idle python run_experiments.py > runs/experiments.log 2>&1 &
tail -f runs/experiments.log            # postęp: start/koniec kolejnych treningów
tail -f runs/*/stdout.log               # szczegóły pojedynczych treningów
```

Kolejno: **faza 1** – 30 treningów drapieżników (5 konfiguracji × IPPO/MAPPO × 3 seedy, po 3 mln kroków),
**faza 2** – zamrożenie najlepszych drapieżników z konfiguracji domyślnej (wybór zapisany w
`runs/frozen_predator.txt`) i 6 treningów ofiary (IPPO/MAPPO × 3 seedy, po 5 mln kroków),
**faza 3** – ewaluacja wszystkich modeli (`results/trained_eval.csv`, `results/summary.md`) i wykresy.
Na 6-rdzeniowym CPU (10 treningów równolegle): ok. 4–5 h.

Skrypt jest idempotentny: ukończone treningi pomija, przerwane wznawia. Zatrzymanie: `Ctrl+C`
(lub `kill <PID>`) – każdy trening zapisuje checkpoint; ponowne `python run_experiments.py` kontynuuje.
Pojedyncze fazy: `--phase 1|2|3`, liczba równoległych procesów: `--parallel N`.

## Pokaz krok po kroku

```bash
# 1. trening na żywo (ok. 2 min) – co 25 iteracji linia z ewaluacją: kolizje/ep i % złapań rosną
python ppo.py --team predator --algo mappo --total-steps 200000 --run-dir runs/demo

# 2. przerwanie (Ctrl+C) zapisuje runs/demo/latest.pt; wznowienie od tego miejsca,
#    z możliwością wydłużenia treningu:
python ppo.py --resume runs/demo --total-steps 300000

# 3. ewaluacja wytrenowanych drapieżników vs losowa ofiara (500 epizodów, te same seedy co baseline)
python evaluate.py --pred runs/predator_mappo_adv3_spd1.0_s0 --prey random
python evaluate.py --pred random --prey random                   # punkt odniesienia
python evaluate.py --pred runs/predator_mappo_adv3_spd1.0_s0 --prey-speed 1.25   # inna prędkość

# 4. ofiara vs zamrożone drapieżniki (nazwa runu ofiary – zob. runs/)
python evaluate.py --pred runs/predator_mappo_adv3_spd1.0_s0 --prey runs/prey_mappo_..._s0

# 5. film: wytrenowane drapieżniki; stan pośredni z treningu (np. ckpt_500k.pt) vs końcowy
python record_agents.py --pred runs/predator_mappo_adv3_spd1.0_s0 --out media/mappo_final.mp4
python record_agents.py --pred runs/predator_mappo_adv3_spd1.0_s0/ckpt_500k.pt --out media/mappo_500k.mp4

# 6. krzywe uczenia i porównania (już wygenerowane przez run_experiments.py)
python plot_results.py            # -> results/*.png, results/summary.md
```

## Checkpointy – zamrażanie i wznawianie

Każdy trening zapisuje w `runs/<nazwa>/`:

| Plik | Zawartość |
|---|---|
| `latest.pt` | ostatni stan (co 10 iteracji i przy przerwaniu): wagi aktora i krytyka, stan optymalizatora Adam, licznik kroków, stany generatorów losowych – **wznowienie przez `--resume`** |
| `ckpt_<N>k.pt` | „zamrożone” stany pośrednie co 500 tys. kroków – do porównania polityki w czasie |
| `best.pt` | najlepszy wynik w ewaluacji okresowej |
| `final.pt` | stan po ukończeniu treningu – **ten plik zamrażamy jako przeciwnika** (`--opponent`) |
| `config.json`, `log.csv`, `eval.csv` | konfiguracja, metryki z każdej iteracji, ewaluacje okresowe |
| `stdout.log` | wyjście konsoli (przy uruchomieniu przez `run_experiments.py`) |

Każdy checkpoint można podać jako przeciwnika (`ppo.py --opponent`), do ewaluacji (`evaluate.py`)
lub do filmu (`record_agents.py`). Zamrożona polityka to sam aktor – w trakcie treningu przeciwnika
nie zmienia się. Po wznowieniu środowiska zaczynają nowe epizody (stan symulacji w połowie epizodu
nie jest zapisywany); wiersze logów zapisane po checkpoincie są usuwane, więc krzywe są ciągłe.

## Metoda

Własna implementacja PPO w PyTorch (`ppo.py`), wspólna dla obu wariantów:

- **współdzielenie parametrów** – jeden aktor i jeden krytyk na drużynę, do wejścia dołączony one-hot
  numeru agenta,
- **aktor** (identyczny w IPPO i MAPPO): MLP 2×128 (tanh), wejście = własna obserwacja + id, 5 akcji,
- **krytyk** – jedyna różnica między wariantami:
  - IPPO: `V(o_i)` – własna obserwacja + id,
  - MAPPO: `V(s)` – stan globalny = konkatenacja obserwacji wszystkich agentów (62 wymiary) + id,
- GAE(λ = 0.95), γ = 0.99, clip 0.2, 10 epok × 4 minibatche, Adam lr 5·10⁻⁴, współczynnik entropii 0.01,
  przycinanie gradientu 0.5, nagrody × 0.1; 16 środowisk × 125 kroków = 2000 kroków (6000 próbek
  przy 3 drapieżnikach) na iterację,
- koniec epizodu po 25 krokach to obcięcie czasowe, nie stan terminalny – wartość ostatniego stanu
  jest bootstrapowana `V(s_T)`,
- przeciwnik: polityka losowa albo zamrożony checkpoint (`--opponent`).

## Metodologia ewaluacji

- **Ewaluacja okresowa** (krzywe uczenia): co 25 iteracji (50 tys. kroków), 100 epizodów, `eval.csv`.
- **Ewaluacja końcowa** (`evaluate.py`): 500 epizodów na porównanie.
- **Te same seedy** (10000–10499) dla wszystkich polityk, w tym losowego baseline'u – każda polityka
  startuje z tych samych pozycji; seedy treningowe są z nimi rozłączne.
- Polityka **stochastyczna** (próbkowanie z rozkładu aktora); `--greedy` = argmax.
- Wynik: średnia ± błąd standardowy po epizodach; między modelami – średnia ± odch. std. po **3 seedach**.
- **Punkty odniesienia:** losowe drapieżniki vs losowa ofiara (w każdej konfiguracji); losowa ofiara
  vs te same zamrożone drapieżniki (dla treningu ofiary).

**Metryki** (`tag_utils.run_episode`): kolizje / epizod, % epizodów ze złapaniem, krok pierwszego
złapania, liczba kroków z kontaktem, średni dystans ofiara–najbliższy drapieżnik, kroki ofiary poza
planszą i kara brzegowa, zwroty obu drużyn.

**Porównania dla różnych ustawień środowiska:**
- liczba drapieżników 2 / 3 / 4 (prędkość ofiary 1.0×) – osobne treningi (wymiar obserwacji
  zależy od liczby agentów),
- prędkość ofiary 0.75 / 1.0 / 1.25× (3 drapieżniki) – osobne treningi **oraz** test modeli
  trenowanych przy 1.0× przy innych prędkościach (uogólnianie).

Wykresy (`results/`): `learning_curves_default.png`, `learning_curves_variants.png`,
`learning_curves_prey.png`, `compare_variants.png`, `speed_generalization.png`; tabele: `summary.md`.

## Pliki

| Plik | Zawartość |
|---|---|
| `tag_utils.py` | tworzenie środowiska (z parametrem `prey_speed`), polityka losowa, epizod z metrykami |
| `ppo.py` | implementacja IPPO / MAPPO, trening, checkpointy, wznawianie |
| `run_experiments.py` | cały zestaw eksperymentów (fazy 1–3), równolegle |
| `evaluate.py` | ewaluacja wytrenowanych / losowych polityk; `--suite` = wszystkie runy |
| `plot_results.py` | krzywe uczenia, wykresy porównawcze, `summary.md` |
| `record_agents.py` | film z wytrenowanymi agentami |
| `baseline_eval.py`, `record_random.py` | ewaluacja i film losowego baseline'u (etap 1) |
| `runs/` | treningi: checkpointy i logi (poza gitem) |
| `results/` | wyniki ewaluacji (CSV) i wykresy |
| `media/` | filmy i GIF-y |
| `presentation/` | prezentacje okresowe |

## Środowisko w skrócie

- **Agenci (domyślnie):** 3 drapieżniki (`adversary_*`, wolniejsze: max_speed 1.0, accel 3.0),
  1 ofiara (`agent_0`, szybsza: 1.3 / 4.0), 2 przeszkody, epizod 25 kroków (`max_cycles`).
- **Akcje:** `Discrete(5)` – brak ruchu, lewo, prawo, dół, góra.
- **Obserwacje:** własna prędkość i pozycja, względne pozycje przeszkód i pozostałych agentów,
  prędkości ofiar – 16 wymiarów dla drapieżnika, 14 dla ofiary.
- **Nagroda:** drapieżniki dostają wspólną nagrodę +10 za każdą parę (drapieżnik, ofiara) w kolizji;
  ofiara −10 za każdą kolizję oraz kara za oddalenie od środka (`|x| > 0.9`, rośnie wykładniczo).

> **Uwaga:** w `mpe2==1.1.1` tryb `curriculum=True` nie działa – `BaseScenario` ma pusty setter
> `curriculum_stage`, więc etap zawsze wynosi 0 (ofiara ma 50% prędkości). Prędkość ofiary
> skalujemy więc ręcznie parametrem `prey_speed` w `tag_utils.make_env`.

## Baseline losowy (3 drapieżniki, 1 ofiara, 25 kroków, 500 epizodów)

| konfiguracja | kolizje / epizod | % epizodów ze złapaniem |
|---|---|---|
| domyślna | 0.46 | 17.6 |
| 2 drapieżniki | 0.32 | 13.4 |
| 4 drapieżniki | 0.67 | 27.2 |
| ofiara 0.75× | 0.44 | 17.4 |
| ofiara 1.25× | 0.44 | 18.6 |

## Plan pracy

1. **Etap 1 (zrobione):** analiza środowiska, losowy baseline, skrypt ewaluacji z metrykami.
2. **Punkt kontrolny (koniec listopada):**
   - własna implementacja IPPO i MAPPO (PPO, PyTorch, współdzielenie parametrów),
   - drapieżniki trenowane przeciw losowej ofierze, krzywe uczenia (3 seedy),
   - wyuczona polityka wyraźnie lepsza od losowej (kolizje, % złapań, filmy),
   - metodologia ewaluacji i metryki (powyżej),
   - wstępne porównania: liczba drapieżników 2–4, prędkość ofiary 0.75–1.25×, IPPO vs MAPPO,
   - pierwszy trening ofiary przeciw zamrożonym drapieżnikom.
3. **Cel końcowy (21/22.01):**
   - częściowa obserwowalność (`num_agent_neighbors`) jako trzecia oś trudności,
   - ofiara trenowana przeciw drapieżnikom IPPO i MAPPO oraz turniej krzyżowy
     (ofiara z jednego treningu vs drapieżniki z innego),
   - miary koordynacji (udział drapieżników w złapaniach, rozłożenie kątowe wokół ofiary),
   - weryfikacja hipotezy: przewaga MAPPO rośnie, gdy złapanie wymaga współpracy,
   - ewaluacja na zbiorze testowym od prowadzącej, raport i skomentowany kod.

## Wykorzystanie AI

Zgodnie z zasadami przedmiotu raport końcowy będzie zawierał opis, do czego i w jakim zakresie
używaliśmy narzędzi AI. Dotychczas: Claude Code (Anthropic) – pomoc przy weryfikacji kodu ewaluacji,
przygotowaniu prezentacji oraz przy pisaniu implementacji PPO, skryptów eksperymentów, ewaluacji
i wykresów.
