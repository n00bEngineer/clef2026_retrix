"""
tune_parameters.py

Hyperparameter optimization for a hybrid Information Retrieval pipeline using Random Search. Coordinates query expansion (Python) and document retrieval (Java) to maximize Recall@100.

Workflow:

    Generates unique random parameters (threshold, max_terms, weights).

    Executes sparse_query_expander.py to apply expansion logic.

    Runs the Java retrieval engine (unipd.se.Main) to evaluate performance.

    Parses stdout for "Recall@100" and logs results to CSV.

Search Space:
--threshold (t)      0.6 - 1.0  (Random Uniform)
--max_terms (m)      1 - 10     (Random Integer)
--weights (w1,2,3)   0 - 20     (Random Uniform)

Requirements:

    Compiled Java classes in target/classes

    Dependencies in lib/

    sparse_query_expander.py in the same directory

Usage:
python tune_parameters.py

Output:
tuning_results.csv   ← Logs all trials [thr, max_terms, w1, w2, w3, recall100]
"""

import subprocess
import re
import random
import csv
import os

# --- Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
TARGET_CLASSES = os.path.join(PROJECT_ROOT, "target", "classes")
LIB_DIR = os.path.join(PROJECT_ROOT, "lib", "*")
OUTPUT_LOG = os.path.join(os.path.dirname(__file__), "tuning_results.csv")

PYTHON_EXPAND_SCRIPT = os.path.join(os.path.dirname(__file__), "sparse_query_expander.py")

JAVA_MAIN_CLASS = "unipd.se.Main"
NUM_ATTEMPTS = 200
SEP = ";" if os.name == "nt" else ":"

def run_trial(threshold, max_terms, w1, w2, w3):
    print(f"\n[TEST] Thr: {threshold} | Max: {max_terms} | Weights: [{w1}, {w2}, {w3}]")

    try:
        subprocess.run(["python", PYTHON_EXPAND_SCRIPT, str(threshold), str(max_terms)], check=True)

        classpath = f"{TARGET_CLASSES}{SEP}{LIB_DIR}"

        cmd = (
            f'java -cp "{classpath}" {JAVA_MAIN_CLASS} '
            f'_ _ _ {w1} {w2} {w3}'
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            shell=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            print(f"  [!] Java error (Exit {result.returncode})")
            print(f"  [STDERR]: {result.stderr.strip()}")
            return 0.0

        match = re.search(r"Recall@100:\s+([0-9]+[.,][0-9]+)", result.stdout)

        if match:
            score = float(match.group(1).replace(',', '.'))
            print(f"  -> Recall@100: {score}")
            return score
        else:
            print("  [!] Recall@100 not found. Verify Java output.")
            return 0.0

    except Exception as e:
        print(f"  [!] Error: {e}")
        return 0.0

def main():
    best_score = -1
    best_params = {}
    seen_configs = set()

    os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)

    file_exists = os.path.isfile(OUTPUT_LOG)

    # Loads past trials if the file exists to avoid duplicates between different executions
    if file_exists:
        with open(OUTPUT_LOG, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Salta header
            for row in reader:
                if len(row) >= 5:
                    # Saves as a tuple (thr, max, w1, w2, w3) correctly converted
                    seen_configs.add((float(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4])))

    with open(OUTPUT_LOG, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["threshold", "max_terms", "w1", "w2", "w3", "recall100"])

    for i in range(NUM_ATTEMPTS):
        print(f"\n--- Trial {i+1}/{NUM_ATTEMPTS} ---")

        # Loop to generate unique parameters
        while True:
            t = round(random.uniform(0.6, 1), 2)
            m = random.randint(1, 10)
            w1 = round(random.uniform(0, 20), 2)
            w2 = round(random.uniform(0, 20), 2)
            w3 = round(random.uniform(0, 20), 2)

            config = (t, m, w1, w2, w3)
            if config not in seen_configs:
                seen_configs.add(config)
                break
            else:
                print("  [INFO] Configuration already tested, retrying...")

        score = run_trial(t, m, w1, w2, w3)

        with open(OUTPUT_LOG, "a", newline='') as f:
            csv.writer(f).writerow([t, m, w1, w2, w3, score])

        if score > best_score:
            best_score = score
            best_params = {"thr": t, "max": m, "w1": w1, "w2": w2, "w3": w3}
            print(f"NEW RECORD: {best_score}")

    print("\n" + "="*40)
    print(f"BEST RESULT: {best_score}")
    print(f"PARAMETERS: {best_params}")
    print("="*40)

if __name__ == "__main__":
    main()