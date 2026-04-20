"""Punct de intrare pentru proiectul TSP (laborator #03).

Ofera:
  - rezolvare TSP prin backtracking (optim) sau hill climbing (euristic)
  - rularea experimentului comparativ si generarea graficului PNG
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from utils.backtracking import rezolva_tsp_backtracking, rezolva_tsp_backtracking_extins
from utils.hill_climbing_tsp import rezolva_tsp_hc
from utils.io_utils import citeste_matrice, formateaza_traseu, salveaza_rezultat
from utils.nn_aima import rezolva_tsp_nn_aima, rezolva_tsp_nn_aima_multistart
from utils.nearest_neighbor import rezolva_tsp_nn, rezolva_tsp_nn_multistart, rezolva_tsp_nn_timp
from utils.performance import ruleaza_experiment, ruleaza_experiment_lab4
from utils.simulated_annealing_tsp import SimulatedAnnealingTSP
from utils.sa_visualizations import run_lab8_bundle


def _cmd_solve(args: argparse.Namespace) -> int:
	n, matrix = citeste_matrice(args.input)

	start = time.perf_counter()
	stats = None
	if args.algo == "bt":
		if args.mod is None:
			route, cost = rezolva_tsp_backtracking(n, matrix)
			algo_name = "backtracking"
		else:
			route, cost, nr_solutii, dur = rezolva_tsp_backtracking_extins(
				n,
				matrix,
				mod=args.mod,
				timp_max=args.timp,
				y_max=args.y,
			)
			algo_name = f"backtracking({args.mod})"
			stats = {"nr_solutii": nr_solutii, "durata": dur}
	elif args.algo == "hc":
		route, cost = rezolva_tsp_hc(
			n,
			matrix,
			reporniri=args.restarts,
			iteratii=args.iterations,
			seed=args.seed,
		)
		algo_name = "hill_climbing_random_restarts"
	elif args.algo == "nn":
		mod = args.mod or "prima"
		if mod == "prima":
			route, cost = rezolva_tsp_nn(n, matrix, start=args.start)
			algo_name = f"nearest_neighbor(start={args.start})"
		elif mod == "y_solutii":
			route, cost, _ = rezolva_tsp_nn_multistart(n, matrix)
			algo_name = "nearest_neighbor(multistart)"
		elif mod == "timp":
			route, cost, rulari, dur = rezolva_tsp_nn_timp(n, matrix, timp_max=args.timp, seed=args.seed)
			algo_name = f"nearest_neighbor(timp={args.timp}s)"
			stats = {"rulari": rulari, "durata": dur}
		else:
			raise SystemExit("Modul 'toate' nu este aplicabil pentru NN.")
	elif args.algo == "nn_aima":
		mod = args.mod or "prima"
		if mod == "prima":
			route, cost = rezolva_tsp_nn_aima(n, matrix, start=args.start)
			algo_name = f"nn_aima(start={args.start})"
		elif mod == "y_solutii":
			route, cost, _ = rezolva_tsp_nn_aima_multistart(n, matrix)
			algo_name = "nn_aima(multistart)"
		else:
			raise SystemExit("Pentru nn_aima sunt suportate: prima, y_solutii.")
	elif args.algo == "sa":
		sa = SimulatedAnnealingTSP(
			matrix,  # type: ignore[arg-type]
			t_max=args.tmax,
			t_min=args.tmin,
			alpha=args.alpha,
			iterations_per_temp=args.iters_per_temp,
			seed=args.seed,
			fix_start=True,
		)
		res = sa.solve(init=args.init, start_city=args.start, max_steps=args.max_steps)
		route, cost = res.best_tour, res.best_cost
		algo_name = f"simulated_annealing(init={args.init})"
	else:
		raise SystemExit("Algoritm necunoscut")
	duration = time.perf_counter() - start

	print(f"Numar de orase: {n}")
	print(f"Algoritm: {algo_name}")
	print(f"Traseu: {formateaza_traseu(route)}")
	print(f"Cost: {cost}")
	print(f"Timp: {duration:.6f} secunde")
	if stats is not None:
		if "nr_solutii" in stats:
			print(f"Solutii gasite: {stats['nr_solutii']}")
		if "rulari" in stats:
			print(f"Rulari NN: {stats['rulari']}")

	if args.output:
		# salveaza_rezultat e tipat pe int, dar accepta si float.
		salveaza_rezultat(args.output, route, int(cost) if isinstance(cost, float) and cost.is_integer() else cost, duration, algo_name)  # type: ignore[arg-type]
		print(f"Rezultat salvat in: {args.output}")

	return 0


def _cmd_lab8(args: argparse.Namespace) -> int:
	outdir = Path(args.outdir)
	artifacts = run_lab8_bundle(
		n=args.n,
		seed=args.seed,
		outdir=outdir,
		t_max=args.tmax,
		t_min=args.tmin,
		alpha=args.alpha,
		iterations_per_temp=args.iters_per_temp,
		sim_steps=args.sim_steps,
	)
	print(f"Lab8: grafice generate in: {outdir.resolve()}")
	for p in artifacts:
		print(f"- {p.resolve()}")
	return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
	out = ruleaza_experiment(
		output_png=args.output,
		seed=args.seed,
		reporniri_hc=args.restarts,
		iteratii_hc=args.iterations,
		bt_time_limit_s=args.bt_time_limit,
	)
	out = Path(out)
	print(f"Grafic salvat: {out.resolve()}")
	return 0


def _cmd_experiment4(args: argparse.Namespace) -> int:
	out = ruleaza_experiment_lab4(output_png=args.output, seed=args.seed, timp_nn_s=args.nn_time)
	out = Path(out)
	print(f"Grafic salvat: {out.resolve()}")
	return 0


def build_parser() -> argparse.ArgumentParser:
	"""Construieste parserul CLI."""
	parser = argparse.ArgumentParser(prog="tsp_proiect", description="TSP: backtracking vs hill climbing")
	sub = parser.add_subparsers(dest="command", required=True)

	p_solve = sub.add_parser("solve", help="Rezolva o instanta TSP din fisier")
	p_solve.add_argument("input", help="Fisier input cu matricea NxN")
	p_solve.add_argument(
		"--algo",
		"--algoritm",
		dest="algo",
		choices=["bt", "hc", "nn", "nn_aima", "sa"],
		default="bt",
		help="Algoritm: bt/hc (Lab3), nn/nn_aima (Lab4), sa (Lab8)",
	)
	p_solve.add_argument(
		"--mod",
		choices=["prima", "toate", "timp", "y_solutii"],
		default=None,
		help="Mod oprire (in special pentru bt in Lab4). Daca lipseste, bt ruleaza optim (toate).",
	)
	p_solve.add_argument("--timp", type=float, default=None, help="Limita timp (sec) pentru mod=timp")
	p_solve.add_argument("--y", type=int, default=None, help="Numar solutii pentru mod=y_solutii (bt)")
	p_solve.add_argument("--start", type=int, default=0, help="Oras start (nn/nn_aima, mod=prima)")
	p_solve.add_argument("--restarts", type=int, default=30, help="Reporniri (doar pentru hc)")
	p_solve.add_argument("--iterations", type=int, default=2000, help="Iteratii per repornire (doar pentru hc)")
	p_solve.add_argument("--seed", type=int, default=42, help="Seed pentru hc/experiment")
	# Parametri SA (Lab8)
	p_solve.add_argument("--init", choices=["nn", "random"], default="nn", help="Stare initiala (doar pentru sa)")
	p_solve.add_argument("--tmax", type=float, default=10000.0, help="Temperatura initiala (doar pentru sa)")
	p_solve.add_argument("--tmin", type=float, default=1.0, help="Temperatura minima (doar pentru sa)")
	p_solve.add_argument("--alpha", type=float, default=0.995, help="Rata de racire (doar pentru sa)")
	p_solve.add_argument("--iters-per-temp", type=int, default=100, help="Iteratii per temperatura (doar pentru sa)")
	p_solve.add_argument("--max-steps", type=int, default=None, help="Limita hard pe numarul total de pasi (doar pentru sa)")
	p_solve.add_argument("--output", help="Fisier de iesire (optional)")
	p_solve.set_defaults(func=_cmd_solve)

	p_exp = sub.add_parser("experiment", help="Ruleaza experimentul si genereaza comparare_performanta.png")
	p_exp.add_argument("--output", default="comparare_performanta.png", help="Cale PNG output")
	p_exp.add_argument("--seed", type=int, default=42, help="Seed pentru generarea instantelor")
	p_exp.add_argument("--restarts", type=int, default=30, help="Reporniri hill climbing")
	p_exp.add_argument("--iterations", type=int, default=2000, help="Iteratii per repornire")
	p_exp.add_argument("--bt-time-limit", type=float, default=30.0, help="Prag (sec) pentru backtracking")
	p_exp.set_defaults(func=_cmd_experiment)

	p_exp4 = sub.add_parser("experiment4", help="Ruleaza experimentul Lab4 (bt vs nn) si genereaza comparare_performanta_lab4.png")
	p_exp4.add_argument("--output", default="comparare_performanta_lab4.png", help="Cale PNG output")
	p_exp4.add_argument("--seed", type=int, default=42, help="Seed pentru generarea instantelor")
	p_exp4.add_argument("--nn-time", type=float, default=1.0, help="Timp (sec) folosit in modul NN 'timp' (daca e activat in experiment)")
	p_exp4.set_defaults(func=_cmd_experiment4)

	# Lab8: Simulated Annealing + vizualizari V1-V7
	p_lab8 = sub.add_parser("lab8", help="Genereaza vizualizarile Lab8 (V1–V7) pentru SA+TSP")
	p_lab8.add_argument("--n", type=int, default=20, help="Numar de orase (coordonate random)")
	p_lab8.add_argument("--seed", type=int, default=42, help="Seed reproducibil")
	p_lab8.add_argument("--outdir", default="lab8_out", help="Folder output (PNG-uri)")
	p_lab8.add_argument("--tmax", type=float, default=10000.0, help="Temperatura initiala (SA)")
	p_lab8.add_argument("--tmin", type=float, default=1.0, help="Temperatura minima (SA)")
	p_lab8.add_argument("--alpha", type=float, default=0.995, help="Rata de racire (SA)")
	p_lab8.add_argument("--iters-per-temp", type=int, default=100, help="Iteratii per temperatura (SA)")
	p_lab8.add_argument("--sim-steps", type=int, default=50000, help="Pasi simanneal (V6)")
	p_lab8.set_defaults(func=_cmd_lab8)

	return parser


def main(argv: list[str] | None = None) -> int:
	"""Entry point pentru rulare ca script."""
	parser = build_parser()
	args = parser.parse_args(argv)
	return int(args.func(args))


if __name__ == "__main__":
	raise SystemExit(main())


""" end """
