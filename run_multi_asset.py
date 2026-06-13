#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_multi_asset.py
Ejecuta dashboard_ms_enhanced.py para multiples activos secuencialmente.

Uso:
    python run_multi_asset.py
    python run_multi_asset.py --assets=BTC-USD,ETH-USD
    python run_multi_asset.py --timeframes=1h,1d
    python run_multi_asset.py --timeout=600
    python run_multi_asset.py --browser  (abre navegador al finalizar cada uno)
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Config
ASSETS = ["BTC-USD", "XRP-USD", "SOL-USD", "ETH-USD"]
TIMEFRAMES = "1h,4h,1d,1wk"
TIMEOUT_PER_ASSET = 360
NO_BROWSER = True

for arg in sys.argv[1:]:
    if arg.startswith("--assets="):
        ASSETS = [a.strip() for a in arg.split("=", 1)[1].split(",")]
    elif arg.startswith("--timeframes="):
        TIMEFRAMES = arg.split("=", 1)[1]
    elif arg.startswith("--timeout="):
        TIMEOUT_PER_ASSET = int(arg.split("=", 1)[1])
    elif arg == "--browser":
        NO_BROWSER = False

SCRIPT_DIR = Path(__file__).parent
DASHBOARD_SCRIPT = SCRIPT_DIR / "dashboard_ms_enhanced.py"


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_asset(asset: str) -> bool:
    print()
    print("=" * 70)
    print(f"  [{timestamp()}] INICIANDO: {asset}")
    print(f"  Timeframes: {TIMEFRAMES}")
    print(f"  Timeout: {TIMEOUT_PER_ASSET}s")
    print("=" * 70)

    cmd = [
        sys.executable,
        str(DASHBOARD_SCRIPT),
        f"--asset={asset}",
        f"--timeframes={TIMEFRAMES}",
    ]
    if NO_BROWSER:
        cmd.append("--no-browser")

    output_path = SCRIPT_DIR / f"hmm_ms_dashboard_{asset}.html"
    if output_path.exists():
        old_size = output_path.stat().st_size
        print(f"  Archivo previo: {output_path.name} ({old_size/1024:.0f} KB)")
    else:
        print(f"  Archivo previo: ninguno")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            timeout=TIMEOUT_PER_ASSET,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] {asset} excedio {TIMEOUT_PER_ASSET}s ({elapsed:.0f}s)")
        return False
    except Exception as e:
        print(f"  [ERROR] {asset}: {e}")
        return False

    elapsed = time.time() - t0
    print(f"\n  Salida ({elapsed:.0f}s, exit_code={result.returncode}):")

    lines = result.stdout.strip().split("\n")
    for line in lines[-20:]:
        print(f"    {line}")

    if result.stderr.strip():
        for line in result.stderr.strip().split("\n")[-10:]:
            print(f"    [STDERR] {line}")

    success = result.returncode == 0
    if success and output_path.exists():
        new_size = output_path.stat().st_size
        print(f"\n  {output_path.name}: {new_size/1024:.0f} KB")
    elif success:
        print(f"\n  Exit OK pero no se encontro {output_path.name}")
    else:
        print(f"\n  Fallo con exit_code={result.returncode}")

    return success


def main():
    print()
    print("=" * 70)
    print("  TradingLatino - Multi-Asset Dashboard Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print(f"  Activos:       {', '.join(ASSETS)}")
    print(f"  Timeframes:    {TIMEFRAMES}")
    print(f"  Timeout/asset: {TIMEOUT_PER_ASSET}s")

    total_start = time.time()
    results = {}

    for asset in ASSETS:
        results[asset] = run_asset(asset)

    total_elapsed = time.time() - total_start
    successes = sum(1 for v in results.values() if v)
    failures = sum(1 for v in results.values() if not v)

    print()
    print("=" * 70)
    print(f"  RESUMEN FINAL")
    print("=" * 70)
    print(f"  Total activos:  {len(ASSETS)}")
    print(f"  Exitosos:       {successes}")
    print(f"  Fallos:         {failures}")
    print(f"  Tiempo total:   {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()

    for asset in ASSETS:
        mark = "OK" if results[asset] else "FAIL"
        path = SCRIPT_DIR / f"hmm_ms_dashboard_{asset}.html"
        size = f"({path.stat().st_size/1024:.0f} KB)" if path.exists() else "(no generado)"
        print(f"    [{mark:4s}] {asset:10s} {size}")

    print()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
