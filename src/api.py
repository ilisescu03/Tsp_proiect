"""Flask API wrapper for TSP solver.

Provides REST endpoints for solving TSP problems.
"""

import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import argparse
import sys
import os
import io
import re
import contextlib
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import _cmd_solve, _cmd_experiment

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)


@app.route('/api/solve', methods=['POST'])
def solve_tsp():
    """Solve TSP with specified algorithm."""
    data = request.json
    algo = data.get('algo', 'bt')
    input_file = data.get('input_file', 'orase.txt')

    project_root = Path(__file__).parent.parent
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    input_file = str(input_path)
    
    try:
        # Create args namespace
        args = argparse.Namespace(
            input=input_file,
            algo=algo,
            mod=data.get('mod'),
            restarts=data.get('restarts', 30),
            iterations=data.get('iterations', 2000),
            seed=data.get('seed', 42),
            start=data.get('start', 0),
            timp=data.get('timp', 10),
            y=data.get('y', 5),
            tmax=data.get('tmax', 1000),
            tmin=data.get('tmin', 0.1),
            alpha=data.get('alpha', 0.95),
            iters_per_temp=data.get('iters_per_temp', 100),
            init=data.get('init', 'random'),
            max_steps=data.get('max_steps', 10000),
            output=None
        )
        
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _cmd_solve(args)
        out = buf.getvalue()

        def _grab(pattern, cast=str):
            m = re.search(pattern, out)
            return cast(m.group(1)) if m else None

        n_cities = _grab(r"Numar de orase:\s*(\d+)", int)
        algo_name = _grab(r"Algoritm:\s*(.+)")
        traseu_str = _grab(r"Traseu:\s*(.+)")
        cost = _grab(r"Cost:\s*([-\d.]+)", float)
        duration = _grab(r"Timp:\s*([\d.]+)", float)

        route = None
        if traseu_str:
            route = [int(x) for x in re.findall(r"\d+", traseu_str)]

        stats = {}
        for key, label in [("nr_solutii", "Solutii gasite"), ("rulari", "Rulari NN")]:
            val = _grab(rf"{label}:\s*(\d+)", int)
            if val is not None:
                stats[key] = val

        return jsonify({
            'status': 'success',
            'message': f'Solved with {algo}',
            'algorithm': algo_name or algo,
            'numCities': n_cities,
            'route': route,
            'cost': cost,
            'duration': duration,
            'stats': stats,
            'log': out,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'{type(e).__name__}: {e}'
        }), 400


@app.route('/api/experiment', methods=['POST'])
def run_experiment():
    """Run performance comparison experiment."""
    try:
        project_root = Path(__file__).parent.parent
        output_png = str(project_root / 'experiment.png')
        args = argparse.Namespace(
            output=output_png,
            seed=42,
            restarts=30,
            iterations=2000,
            bt_time_limit=30.0,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _cmd_experiment(args)
        
        return jsonify({
            'status': 'success',
            'image': 'experiment.png',
            'message': 'Experiment completed'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'{type(e).__name__}: {e}'
        }), 400


UTILS_DIR = Path(__file__).parent / 'utils'
PROJECT_ROOT = Path(__file__).parent.parent
MAIN_PY = Path(__file__).parent / 'main.py'


def _list_scripts():
    return sorted(p.stem for p in UTILS_DIR.glob('*.py') if p.stem != '__init__')


def _grab_int(prompt: str, *keywords: str, default=None):
    for kw in keywords:
        m = re.search(rf"\b{kw}\s+(\d+)\b", prompt, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return default


def _build_command(prompt: str):
    """Tokenize prompt and decide which command to run.

    Returns (label, cmd_list, cwd) or (None, None, None) if no match.
    """
    p = prompt.lower()
    tokens = set(re.findall(r"[a-z0-9_]+", p))
    src_dir = str(Path(__file__).parent)

    def has(*words):
        return all(w in tokens for w in words)

    # ---- NLP classification ----
    if has('nlp'):
        task = _grab_int(prompt, 'task', 'sarcina')
        if task is not None or 'tasks' in tokens or 'task' in tokens:
            cmd = [sys.executable, str(UTILS_DIR / 'nlp_classification_tasks.py')]
            if task is not None:
                cmd += ['--task', str(task)]
            if re.search(r"\bno[- ]plots?\b", p):
                cmd += ['--no-plots']
            return ('nlp_classification_tasks', cmd, str(PROJECT_ROOT))
        # plain "nlp classification"
        cmd = [sys.executable, str(UTILS_DIR / 'nlp_classification.py')]
        return ('nlp_classification', cmd, str(PROJECT_ROOT))

    # ---- Lab9 (genetic algorithms via main.py) ----
    if 'lab9' in tokens or has('genetic', 'algorithm') or 'pygad' in tokens:
        task = _grab_int(prompt, 'task', 'mode', 'sarcina', default=1)
        seed = _grab_int(prompt, 'seed', default=42)
        cmd = [sys.executable, str(MAIN_PY), 'lab9',
               '--mode', f'task{task}',
               '--outdir', 'lab9_out',
               '--seed', str(seed)]
        return (f'lab9 task{task}', cmd, str(PROJECT_ROOT))

    # ---- Lab8 (simulated annealing visualizations via main.py) ----
    if 'lab8' in tokens or has('sa', 'visualization') or has('sa', 'visualizations') \
            or has('simulated', 'annealing', 'visualization') \
            or has('simulated', 'annealing', 'visualizations'):
        n = _grab_int(prompt, 'n', 'cities', 'orase', default=20)
        seed = _grab_int(prompt, 'seed', default=42)
        cmd = [sys.executable, str(MAIN_PY), 'lab8',
               '--n', str(n), '--seed', str(seed),
               '--outdir', 'lab8_out']
        return ('lab8', cmd, str(PROJECT_ROOT))

    # ---- experiment4 ----
    if 'experiment4' in tokens or (has('experiment', 'lab4')) or (has('experiment', '4')):
        cmd = [sys.executable, str(MAIN_PY), 'experiment4',
               '--output', 'comparare_performanta_lab4.png']
        return ('experiment4', cmd, str(PROJECT_ROOT))

    # ---- Generic fallback: run a utils/ script as a module ----
    direct_aliases = [
        (('genetic', 'algorithm'), 'genetic_algorithm_tsp'),
        (('hill', 'climbing'), 'hill_climbing_tsp'),
        (('nearest', 'neighbor'), 'nearest_neighbor'),
        (('nn', 'aima'), 'nn_aima'),
        (('simulated', 'annealing'), 'simulated_annealing_tsp'),
        (('backtracking',), 'backtracking'),
        (('performance',), 'performance'),
    ]
    for keywords, script in direct_aliases:
        if has(*keywords) and (UTILS_DIR / f'{script}.py').exists():
            cmd = [sys.executable, '-m', f'utils.{script}']
            return (script, cmd, src_dir)

    # 3. direct script-name match
    for script in _list_scripts():
        if script in tokens:
            cmd = [sys.executable, '-m', f'utils.{script}']
            return (script, cmd, src_dir)

    return (None, None, None)


@app.route('/api/dispatch', methods=['POST'])
def dispatch():
    """Tokenize prompt, pick a utils/ script, run it as a subprocess."""
    import subprocess
    data = request.json or {}
    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        return jsonify({'status': 'error', 'message': 'empty prompt'}), 400

    label, cmd, cwd = _build_command(prompt)
    if not cmd:
        return jsonify({
            'status': 'error',
            'message': (
                'No command matched. Try: "nlp classification task 5", '
                '"lab9 task 1 seed 42", "lab8 n 20", "experiment4", '
                f'or any of: {", ".join(_list_scripts())}'
            ),
        }), 400

    import time as _time
    start_ts = _time.time() - 1

    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=cwd,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'message': 'script timed out'}), 504

    # find images created or modified during the run, anywhere under project root
    images = []
    for p in PROJECT_ROOT.rglob('*.png'):
        try:
            if p.stat().st_mtime >= start_ts:
                rel = p.relative_to(PROJECT_ROOT).as_posix()
                images.append(rel)
        except OSError:
            pass

    return jsonify({
        'status': 'success' if proc.returncode == 0 else 'error',
        'script': label,
        'command': ' '.join(cmd),
        'returncode': proc.returncode,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
        'images': images,
        'message': f'Ran {label}',
    }), (200 if proc.returncode == 0 else 400)


@app.route('/api/image/<path:filename>', methods=['GET'])
def serve_image(filename):
    project_root = Path(__file__).parent.parent
    return send_from_directory(str(project_root), filename)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
