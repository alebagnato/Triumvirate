"""
triumvirate.py
--------------
LLM Triumvirate Physical Plausibility Judge
Companion code for: "LLM Triumvirate as Physical Plausibility Judge:
A Benchmark for Physics-Aware Test Verdict Generation"
Submitted to ICTSS 2026 — anonymized for blind review

Three small LLMs vote independently on each state-transition triplet.
Requires LM Studio running locally (OpenAI-compatible API on localhost:1234).

Usage:
    python triumvirate.py
    python triumvirate.py --scenario teleport
    python triumvirate.py --model gemma

Models (configured in LM Studio):
    gemma   : google/gemma-3n-e4b              (6.9B, GGUF Q4_K_M)
    llama   : llama-3.2-3b-instruct            (3B,  GGUF Q4_K_M)
    mistral : mistralai/mistral-7b-instruct-v0.3 (7B, GGUF Q4_K_M)

Hardware: consumer laptop, CPU-only, no GPU required.
"""

import json
import argparse
import requests
import time

# ── Configuration ─────────────────────────────────────────────────────────────

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

MODELS = {
    "gemma"  : "google/gemma-3n-e4b",
    "llama"  : "llama-3.2-3b-instruct",
    "mistral": "mistralai/mistral-7b-instruct-v0.3",
}

# Per-model timeout (seconds)
TIMEOUTS = {
    "gemma"  : 120,
    "llama"  :  90,
    "mistral": 180,
}

# ── System prompt (physics prior) ─────────────────────────────────────────────

PHYSICS_CONTEXT = """You are a physical plausibility judge. You reason about state
transitions in a simple 2D or 3D physical environment.

Physical laws you apply:
- Energy conservation: an object cannot accelerate without an applied force.
- Spatial continuity: an object cannot teleport — position varies continuously.
- Gravity: g = 9.81 m/s² on Earth unless another value is specified in context.
- Friction: objects on the ground decelerate without a driving force.
- Inertia: a moving object continues unless opposed by a force.
- Collisions: post-collision velocity respects conservation of momentum.

State format:
  position   : [x, y] in meters (2D) or [x, y, z] in meters (3D, z = vertical)
  velocity   : [vx, vy] or [vx, vy, vz] in m/s
  on_ground  : boolean
  context    : optional — specifies gravity, environment, or special conditions

Action format:
  force      : [fx, fy] or [fx, fy, fz] in Newtons
  dt         : timestep in seconds

Your response MUST be strict JSON, nothing else:
{
  "plausible": true or false,
  "score": float between 0.0 (impossible) and 1.0 (perfectly plausible),
  "violations": ["list of violated laws, empty if none"],
  "reasoning": "concise explanation (2-3 sentences)"
}
No text before or after the JSON.
"""

# ── Benchmark scenarios ────────────────────────────────────────────────────────

SCENARIOS = {

    # S1 — 2D baseline (correct)
    # Ball at rest, 5N force for 0.1s. Resulting position and velocity
    # are physically consistent. Equivalent to the "no perturbation"
    # condition in LeWM Fig.8.
    "normal": {
        "description": "Ball pushed — correct 2D transition",
        "etat_t" : {"object": "ball", "position": [1.0, 0.05],
                    "velocity": [0.0, 0.0], "on_ground": True},
        "action" : {"force": [5.0, 0.0], "dt": 0.1},
        "etat_t1": {"object": "ball", "position": [1.02, 0.05],
                    "velocity": [0.4, 0.0], "on_ground": True},
        "expected": True
    },

    # S2 — Spatial continuity violation (2D)
    # Jump from x=1.0 to x=3.8 in 0.1s implies 28 m/s — physically impossible.
    # Direct analog of the LeWM teleportation test (Fig.8).
    "teleport": {
        "description": "Teleportation — spatial continuity violation (LeWM Fig.8)",
        "etat_t" : {"object": "block", "position": [1.0, 0.05],
                    "velocity": [0.1, 0.0], "on_ground": True},
        "action" : {"force": [2.0, 0.0], "dt": 0.1},
        "etat_t1": {"object": "block", "position": [3.8, 0.05],
                    "velocity": [0.1, 0.0], "on_ground": True},
        "expected": False
    },

    # S3 — Energy conservation violation (2D)
    # Zero force, velocity jumps from 0.5 to 8.0 m/s — spontaneous acceleration.
    "energy": {
        "description": "Spontaneous acceleration — energy conservation violation",
        "etat_t" : {"object": "ball", "position": [2.0, 0.05],
                    "velocity": [0.5, 0.0], "on_ground": True},
        "action" : {"force": [0.0, 0.0], "dt": 0.1},
        "etat_t1": {"object": "ball", "position": [2.1, 0.05],
                    "velocity": [8.0, 0.0], "on_ground": True},
        "expected": False
    },

    # S4a — Gravity violation on Earth (2D)
    # Cube stationary in mid-air on Earth — impossible without support.
    # Same physical state as S4b (ISS) but opposite expected verdict.
    # Demonstrates that the judge uses context, not numerical values alone.
    # NOTE: single failure case of the triumvirate (split verdict 2/3 PLAUSIBLE).
    "gravity_violation": {
        "description": "Floating object on Earth — gravity violation",
        "etat_t" : {"object": "cube", "position": [1.5, 1.2],
                    "velocity": [0.0, 0.0], "on_ground": False,
                    "context": "Earth surface, g=9.81 m/s², open air, no support"},
        "action" : {"force": [0.0, 0.0], "dt": 0.5},
        "etat_t1": {"object": "cube", "position": [1.5, 1.2],
                    "velocity": [0.0, 0.0], "on_ground": False},
        "expected": False
    },

    # S4b — Same physical state, ISS context (2D)
    # Stationary object in microgravity — plausible.
    # Symmetric counterpart to S4a: identical triplet, opposite verdict.
    # Instantiates a metamorphic relation: verdict must flip with gravity context.
    "gravity_iss": {
        "description": "Floating object on ISS — correct in microgravity",
        "etat_t" : {"object": "cube", "position": [1.5, 1.2],
                    "velocity": [0.0, 0.0], "on_ground": False,
                    "context": "ISS, microgravity environment, g=0 m/s2"},
        "action" : {"force": [0.0, 0.0], "dt": 0.5},
        "etat_t1": {"object": "cube", "position": [1.5, 1.2],
                    "velocity": [0.0, 0.0], "on_ground": False},
        "expected": True
    },

    # S5 — Correct Earth gravity (2D)
    # Cube released from h=2.0m, g=9.81 m/s², dt=0.5s.
    # y_t1 = 2.0 - 0.5*9.81*0.25 = 0.774m | vy_t1 = -4.905 m/s
    "gravity_earth": {
        "description": "Object falling on Earth — correct transition g=9.81 m/s²",
        "etat_t" : {"object": "cube", "position": [1.5, 2.0],
                    "velocity": [0.0, 0.0], "on_ground": False,
                    "context": "Earth gravity g=9.81 m/s², no support below"},
        "action" : {"force": [0.0, 0.0], "dt": 0.5},
        "etat_t1": {"object": "cube", "position": [1.5, 0.774],
                    "velocity": [0.0, -4.905], "on_ground": False},
        "expected": True
    },

    # S6 — Martian gravity, domain transfer (2D)
    # Titanium glass on Mars, g=3.72 m/s², dt=0.5s.
    # y_t1 = 1.0 - 0.5*3.72*0.25 = 0.535m | vy_t1 = -1.86 m/s
    # Tests generalization to a non-standard physical environment.
    "verre_mars": {
        "description": "Glass falling on Mars — Martian gravity g=3.72 m/s²",
        "etat_t" : {"object": "titanium_glass", "position": [0.5, 1.0],
                    "velocity": [0.0, 0.0], "on_ground": False,
                    "context": "Martian gravity g=3.72 m/s², CO2 atmosphere"},
        "action" : {"force": [0.0, 0.0], "dt": 0.5},
        "etat_t1": {"object": "titanium_glass", "position": [0.5, 0.535],
                    "velocity": [0.0, -1.86], "on_ground": False},
        "expected": True
    },

    # S7 — 3D baseline (correct)
    # Ball pushed along x in 3D space, with gravitational component on z.
    # z_t1 = 1.0 - 0.5*9.81*0.01 = 0.951m | coordinates: [x, y, z], z = vertical.
    "normal_3d": {
        "description": "Ball pushed in 3D space — correct transition",
        "etat_t" : {"object": "ball", "position": [1.0, 0.5, 1.0],
                    "velocity": [0.0, 0.0, 0.0], "on_ground": False,
                    "context": "3D space, Earth gravity g=9.81 m/s² along z-axis"},
        "action" : {"force": [5.0, 0.0, 0.0], "dt": 0.1},
        "etat_t1": {"object": "ball", "position": [1.02, 0.5, 0.951],
                    "velocity": [0.4, 0.0, -0.981], "on_ground": False},
        "expected": True
    },

    # S8 — 3D spatial teleportation
    # Impossible jump across all three axes simultaneously.
    # Extension of the S2 spatial continuity test to 3D.
    "teleport_3d": {
        "description": "3D teleportation — spatial continuity violation in 3D",
        "etat_t" : {"object": "block", "position": [1.0, 0.5, 0.5],
                    "velocity": [0.1, 0.0, 0.0], "on_ground": False,
                    "context": "3D space, Earth gravity"},
        "action" : {"force": [2.0, 0.0, 0.0], "dt": 0.1},
        "etat_t1": {"object": "block", "position": [3.8, 2.5, 3.2],
                    "velocity": [0.1, 0.0, 0.0], "on_ground": False},
        "expected": False
    },
}

# ── Model call ────────────────────────────────────────────────────────────────

def call_model(model_key: str, scenario: dict) -> dict:
    """Call a LM Studio model and return the parsed judgment."""
    model_name = MODELS[model_key]
    timeout    = TIMEOUTS.get(model_key, 120)

    prompt = f"""Evaluate the physical plausibility of this state transition.

Scenario: {scenario['description']}

State at t:
{json.dumps(scenario['etat_t'], indent=2)}

Action applied:
{json.dumps(scenario['action'], indent=2)}

State at t+1 (observed):
{json.dumps(scenario['etat_t1'], indent=2)}

Respond only in JSON."""

    # Mistral v0.3 does not support the "system" role — merge with user message
    if "mistral" in model_name.lower():
        messages = [{"role": "user",
                     "content": PHYSICS_CONTEXT + "\n\n" + prompt}]
    else:
        messages = [{"role": "system", "content": PHYSICS_CONTEXT},
                    {"role": "user",   "content": prompt}]

    payload = {
        "model"      : model_name,
        "messages"   : messages,
        "temperature": 0.0,
        "max_tokens" : 512,
    }

    t0 = time.time()
    r  = None

    for attempt in range(2):
        try:
            r = requests.post(LMSTUDIO_URL, json=payload, timeout=timeout)
            if r.status_code == 400 and attempt == 0:
                # LM Studio may unload the model between calls
                time.sleep(5)
                continue
            r.raise_for_status()
            break
        except requests.exceptions.Timeout:
            return {"error": f"timeout after {timeout}s", "_model": model_key}
        except requests.exceptions.ConnectionError:
            return {"error": "LM Studio not running", "_model": model_key}
        except Exception as e:
            if attempt == 1:
                return {"error": str(e), "_model": model_key}
            time.sleep(5)

    if r is None or not r.ok:
        return {"error": "failed after retry", "_model": model_key}

    try:
        elapsed = round(time.time() - t0, 1)
        raw = r.json()["choices"][0]["message"]["content"].strip()

        # Strip <think>...</think> blocks (some reasoning models)
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        # Strip markdown code fences
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        result["_model"]   = model_key
        result["_elapsed"] = elapsed
        return result

    except json.JSONDecodeError:
        return {"error": "invalid JSON", "raw": raw[:200], "_model": model_key}
    except Exception as e:
        return {"error": str(e), "_model": model_key}


# ── Verdict aggregation ───────────────────────────────────────────────────────

def aggregate_verdict(votes: list) -> dict:
    """Simple majority vote. Always returns a 'votes' key."""
    valid = [v for v in votes if "plausible" in v and "error" not in v]

    if not valid:
        return {"verdict": None, "confidence": 0.0,
                "agreement": "no_valid_votes", "votes": "0/0 plausible"}

    plausible_votes = sum(1 for v in valid if v["plausible"])
    total_valid     = len(valid)
    majority        = plausible_votes > total_valid / 2
    agreement       = "unanimous" if plausible_votes in (0, total_valid) else "split"
    avg_score       = sum(v.get("score", 0.5) for v in valid) / total_valid

    return {
        "verdict"   : majority,
        "confidence": round(avg_score, 3),
        "agreement" : agreement,
        "votes"     : f"{plausible_votes}/{total_valid} plausible",
    }


# ── Execution ─────────────────────────────────────────────────────────────────

def run_triumvirate(scenario_name: str,
                    models: list = None,
                    verbose: bool = True) -> dict:
    """Run the triumvirate on a single scenario."""
    if models is None:
        models = list(MODELS.keys())

    scenario = SCENARIOS[scenario_name]
    votes    = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario    : {scenario_name}")
        print(f"Description : {scenario['description']}")
        print(f"{'='*60}")

    for model_key in models:
        if verbose:
            print(f"  [{model_key:7}] running...", end=" ", flush=True)
        result = call_model(model_key, scenario)
        votes.append(result)

        if verbose:
            if "error" in result:
                print(f"ERROR — {result['error']}")
            else:
                symbol = "✓" if result.get("plausible") else "✗"
                print(f"{symbol}  score={result.get('score','?')}  "
                      f"({result.get('_elapsed','?')}s)")
                print(f"  reasoning={result.get('reasoning', 'MISSING')[:160]}")
                print(f"  violations={result.get('violations', 'MISSING')}")

    agg      = aggregate_verdict(votes)
    expected = scenario.get("expected")
    correct  = (agg["verdict"] == expected) if expected is not None else None

    if verbose:
        verdict_str = "PLAUSIBLE" if agg["verdict"] else (
                      "IMPLAUSIBLE" if agg["verdict"] is False else "UNDETERMINED")
        status = "✓ CORRECT" if correct else ("✗ ERROR" if correct is False else "")
        print(f"  {'─'*40}")
        print(f"  Triumvirate verdict : {verdict_str}  "
              f"[{agg['votes']}]  {agg['agreement']}")
        print(f"  Average score       : {agg['confidence']}")
        print(f"  Expected            : {expected}  {status}")

    return {"scenario": scenario_name, "votes": votes,
            "aggregate": agg, "correct": correct}


def run_all(models: list = None, verbose: bool = True) -> None:
    """Run the triumvirate on all benchmark scenarios."""
    results = []
    for name in SCENARIOS:
        r = run_triumvirate(name, models=models, verbose=verbose)
        results.append(r)

    correct   = sum(1 for r in results if r["correct"] is True)
    total     = sum(1 for r in results if r["correct"] is not None)
    unanimous = sum(1 for r in results
                    if r["aggregate"].get("agreement") == "unanimous")

    print(f"\n{'='*60}")
    print(f"GLOBAL RESULT : {correct}/{total} correct")
    print(f"Unanimous     : {unanimous}/{len(results)}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Triumvirate — physical plausibility judge (LM Studio)"
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Scenario to test (default: all)"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Test a single model only"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output"
    )
    args = parser.parse_args()

    models = [args.model] if args.model else None

    if args.scenario == "all":
        run_all(models=models, verbose=not args.quiet)
    else:
        run_triumvirate(args.scenario, models=models, verbose=not args.quiet)
