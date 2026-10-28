# MPE2 – Simple Tag (temat żółty)

Projekt z przedmiotu *Uczenie ze wzmocnieniem*. Środowisko: [`mpe2.simple_tag_v3`](https://mpe2.farama.org/)
(Multi-Agent Particle Environment, gra drapieżnik–ofiara).

**Pytanie badawcze:** czy scentralizowany krytyk (CTDE, MAPPO) daje lepszą koordynację drapieżników
niż niezależne uczenie (IPPO), gdy obie drużyny uczą się jednocześnie, i od czego zależy ta różnica
(liczba drapieżników, prędkość ofiary, zasięg obserwacji)?

## Uruchomienie

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python baseline_eval.py         # polityki bazowe -> results/*.csv, results/*.png (~kilka minut)
python record_random.py         # film z losowym agentem -> media/random_agent.mp4 / .gif
```

## Pliki

| Plik | Zawartość |
|---|---|
| `tag_utils.py` | tworzenie środowiska (z parametrem `prey_speed`), polityki bazowe (`random`, `greedy`, `flee`), epizod z metrykami |
| `baseline_eval.py` | macierz polityk bazowych + wpływ liczby drapieżników, prędkości ofiary i liczby przeszkód |
| `record_random.py` | nagranie działania losowego agenta |
| `results/` | wyniki ewaluacji (CSV: per epizod i podsumowanie z błędem standardowym) i wykresy |
| `media/` | film i GIF z losowym agentem |
| `presentation/` | prezentacje okresowe |

## Środowisko w skrócie

- **Agenci (domyślnie):** 3 drapieżniki (`adversary_*`, wolniejsze: max_speed 1.0, accel 3.0),
  1 ofiara (`agent_0`, szybsza: 1.3 / 4.0), 2 przeszkody, epizod 25 kroków (`max_cycles`).
- **Akcje:** `Discrete(5)` – brak ruchu, lewo, prawo, dół, góra (opcjonalnie ciągłe `Box(0, 1, (5,))`).
- **Obserwacje:** własna prędkość i pozycja, względne pozycje przeszkód i pozostałych agentów,
  prędkości ofiar – 16 wymiarów dla drapieżnika, 14 dla ofiary.
- **Nagroda:** drapieżniki dostają wspólną nagrodę +10 za każdą parę (drapieżnik, ofiara) w kolizji;
  ofiara −10 za każdą kolizję oraz kara za oddalenie od środka (`|x| > 0.9`, rośnie wykładniczo).
- **Konfiguracja:** `num_good`, `num_adversaries`, `num_obstacles`, `max_cycles`, `continuous_actions`,
  `terminate_on_success`, `num_agent_neighbors` / `num_landmark_neighbors` (częściowa obserwowalność),
  `curriculum`.

> **Uwaga:** w `mpe2==1.1.1` tryb `curriculum=True` nie działa – `BaseScenario` ma pusty setter
> `curriculum_stage`, więc etap zawsze wynosi 0 (ofiara ma 50% prędkości). Prędkość ofiary
> skalujemy więc ręcznie parametrem `prey_speed` w `tag_utils.make_env`.

## Wyniki bazowe (3 drapieżniki, 1 ofiara, 25 kroków, 500 epizodów)

| drapieżnik vs ofiara | kolizje / epizod | % epizodów ze złapaniem |
|---|---|---|
| random vs random | 0.46 | 17.6 |
| random vs flee | 0.16 | 8.4 |
| greedy vs random | 4.68 | 92.6 |
| greedy vs flee | 1.58 | 67.2 |

## Plan pracy

1. **Do 3/4.12 (cel pośredni):** wspólna implementacja PPO w PyTorch z dwoma wariantami krytyka –
   IPPO (krytyk widzi własną obserwację) i MAPPO (krytyk widzi stan globalny); współdzielenie parametrów
   w drużynie. Trening drapieżników przeciw stałej ofierze `flee`, potem obu stron jednocześnie.
   Kryterium: wytrenowane drapieżniki biją heurystykę `greedy` (> 1.58 kolizji / epizod vs `flee`).
2. **Do 21/22.01 (cel końcowy):** kontrolowane porównanie IPPO vs MAPPO (≥ 3 seedy na wariant)
   w zależności od liczby drapieżników (2–4), prędkości ofiary (0.75–1.25) i zasięgu obserwacji;
   turniej krzyżowy (drapieżniki algorytmu A vs ofiara algorytmu B); eksperyment weryfikujący
   hipotezę z wcześniejszych wyników.
3. **Metryki:** kolizje / epizod, % epizodów ze złapaniem, czas do pierwszego złapania, kara brzegowa
   ofiary, średni dystans ofiara–najbliższy drapieżnik, wskaźnik koordynacji (ile drapieżników bierze
   udział w złapaniach, rozłożenie kątowe wokół ofiary), krzywe uczenia z odchyleniem między seedami.

## Wykorzystanie AI

Zgodnie z zasadami przedmiotu raport końcowy będzie zawierał opis, do czego i w jakim zakresie
używaliśmy narzędzi AI. Na razie: pomoc przy weryfikacji kodu ewaluacji i przygotowaniu prezentacji.
