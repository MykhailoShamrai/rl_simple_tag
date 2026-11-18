#!/usr/bin/env bash
# Pokaz krok po kroku: gotowe warianty demo.py uruchamiane po kolei.
# Każdy wariant otwiera okno; Esc / q zamyka je i uruchamia następny wariant.
#
#   ./demo_tour.sh            # wszystkie warianty po kolei
#   ./demo_tour.sh 1 2 9      # tylko wybrane (numery z listy)
#   ./demo_tour.sh -l         # lista wariantów
#   DEMO_ARGS="--greedy --fps 6" ./demo_tour.sh 1   # dodatkowe flagi demo.py dla każdego wariantu
#
# Warianty 3 i 11 używają checkpointów pośrednich z runs/ (nie ma ich w gicie) – gdy ich brak, są pomijane.

set -u
cd "$(dirname "$0")"
PY=python
[ -x .venv/bin/python ] && PY=.venv/bin/python

M=models
P=$M/predator_mappo_adv3_spd1.0_s1                                   # zamrożone drapieżniki (faza 2)
PREY_M=$M/prey_mappo_adv3_spd1.0_vs-predator_mappo_adv3_spd1.0_s1_s0
PREY_I=$M/prey_ippo_adv3_spd1.0_vs-predator_mappo_adv3_spd1.0_s1_s0
RUN_PRED=runs/predator_mappo_adv3_spd1.0_s0
RUN_PREY=runs/prey_mappo_adv3_spd1.0_vs-predator_mappo_adv3_spd1.0_s1_s0

TITLES=(
  ""
  "Losowe drapieżniki vs wytrenowane MAPPO (3 drapieżniki, ofiara 1.0×)"
  "IPPO vs MAPPO (konfiguracja domyślna)"
  "Postęp treningu drapieżników: 0.5 mln kroków vs koniec (3 mln)"
  "2 drapieżniki: losowe vs MAPPO"
  "4 drapieżniki: losowe vs MAPPO"
  "Wolna ofiara 0.75×: losowe vs MAPPO"
  "Szybka ofiara 1.25×: losowe vs MAPPO"
  "Uogólnianie: MAPPO trenowane przy 1.0×, test przy ofierze 1.25×"
  "Losowa vs wyuczona ofiara (MAPPO), te same zamrożone drapieżniki"
  "Ofiara IPPO vs ofiara MAPPO"
  "Postęp treningu ofiary: 0.5 mln kroków vs koniec (5 mln)"
  "Uogólnianie ofiary: drapieżniki z treningu (lewa) vs inne drapieżniki IPPO (prawa)"
)

args_for() {
  case $1 in
    1)  echo "--right-pred $P" ;;
    2)  echo "--left-pred $M/predator_ippo_adv3_spd1.0_s0 --right-pred $M/predator_mappo_adv3_spd1.0_s0" ;;
    3)  echo "--left-pred $RUN_PRED/ckpt_500k.pt --right-pred $RUN_PRED" ;;
    4)  echo "--right-pred $M/predator_mappo_adv2_spd1.0_s0" ;;
    5)  echo "--right-pred $M/predator_mappo_adv4_spd1.0_s0" ;;
    6)  echo "--right-pred $M/predator_mappo_adv3_spd0.75_s0" ;;
    7)  echo "--right-pred $M/predator_mappo_adv3_spd1.25_s0" ;;
    8)  echo "--right-pred $M/predator_mappo_adv3_spd1.0_s0 --prey-speed 1.25" ;;
    9)  echo "--left-pred $P --right-pred $P --right-prey $PREY_M" ;;
    10) echo "--left-pred $P --left-prey $PREY_I --right-pred $P --right-prey $PREY_M" ;;
    11) echo "--left-pred $P --left-prey $RUN_PREY/ckpt_500k.pt --right-pred $P --right-prey $RUN_PREY" ;;
    12) echo "--left-pred $P --left-prey $PREY_M --right-pred $M/predator_ippo_adv3_spd1.0_s0 --right-prey $PREY_M" ;;
  esac
}

N=$(( ${#TITLES[@]} - 1 ))

if [ "${1:-}" = "-l" ]; then
  for i in $(seq 1 $N); do printf "%2d. %s\n" "$i" "${TITLES[$i]}"; done
  exit 0
fi

selected=("$@")
[ ${#selected[@]} -eq 0 ] && selected=($(seq 1 $N))

for i in "${selected[@]}"; do
  if ! [[ $i =~ ^[0-9]+$ ]] || [ "$i" -lt 1 ] || [ "$i" -gt $N ]; then
    echo "nieznany wariant: $i (lista: $0 -l)"; exit 1
  fi
  a=$(args_for "$i")
  if [[ ($i == 3 && ! -d $RUN_PRED) || ($i == 11 && ! -d $RUN_PREY) ]]; then
    echo "[$i] pomijam – brak checkpointów pośrednich w runs/"; continue
  fi
  echo
  echo "[$i/$N] ${TITLES[$i]}"
  echo "       python demo.py $a ${DEMO_ARGS:-}"
  # shellcheck disable=SC2086
  $PY demo.py $a ${DEMO_ARGS:-} || exit $?
done
