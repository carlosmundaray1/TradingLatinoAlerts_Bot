#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LAUNCHER - Lanzador simple del Dashboard HMM vs Markov Switching

Uso:
    python launcher.py

Selecciona un activo del menu, y el script genera el dashboard.
El archivo HTML se guarda en la misma carpeta.
"""
import os
import sys
import subprocess
import webbrowser
from pathlib import Path

ASSETS = [
    ("1", "BTC-USD",   "Bitcoin"),
    ("2", "ETH-USD",   "Ethereum"),
    ("3", "SOL-USD",   "Solana"),
    ("4", "BNB-USD",   "BNB"),
    ("5", "XRP-USD",   "XRP"),
    ("6", "ADA-USD",   "Cardano"),
    ("7", "DOGE-USD",  "Dogecoin"),
    ("8", "AVAX-USD",  "Avalanche"),
    ("9", "DOT-USD",   "Polkadot"),
    ("10", "LINK-USD", "Chainlink"),
    ("11", "MATIC-USD","Polygon"),
    ("12", "ATOM-USD", "Cosmos"),
    ("13", "UNI-USD",  "Uniswap"),
    ("14", "AAVE-USD", "Aave"),
    ("15", "APT-USD",  "Aptos"),
    ("16", "SUI-USD",  "Sui"),
]

SCRIPT_DIR = Path(__file__).parent
DASHBOARD_SCRIPT = SCRIPT_DIR / "dashboard_ms_enhanced.py"


def mostrar_menu():
    print()
    print("=" * 56)
    print("  TRADINGLATINO - Dashboard HMM vs Markov Switching")
    print("=" * 56)
    print("\n  Selecciona un activo:\n")
    for num, ticker, nombre in ASSETS:
        print(f"    [{num}] {ticker:10s}  {nombre}")
    print()
    print("    [0] Salir\n")


def elegir_activo() -> str:
    while True:
        mostrar_menu()
        opcion = input("  Opcion: ").strip()
        if opcion == "0":
            print("\n  Hasta luego!")
            sys.exit(0)
        for num, ticker, _ in ASSETS:
            if opcion == num:
                return ticker
        print("\n  Opcion invalida. Intenta de nuevo.")


def ejecutar(asset: str):
    output_path = SCRIPT_DIR / f"hmm_ms_dashboard_{asset}.html"
    print()
    print("=" * 56)
    print(f"  Generando dashboard para {asset}...")
    print("=" * 56)
    print()
    cmd = [sys.executable, str(DASHBOARD_SCRIPT), f"--asset={asset}", "--no-browser"]
    try:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("\n  ERROR: El proceso tardo mas de 10 minutos y fue cancelado.")
        return
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return
    if result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-15:]:
            print(f"  {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-5:]:
            if line.strip():
                print(f"  [STDERR] {line}")
    print()
    print("=" * 56)
    if result.returncode == 0 and output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"  LISTO! Dashboard generado:")
        print(f"     Archivo: {output_path.name}")
        print(f"     Tamano:  {size_kb:.0f} KB")
        print(f"     Ruta:    {output_path.resolve()}")
        abrir = input("\n  Abrir en el navegador? (s/n): ").strip().lower()
        if abrir in ("s", "si"):
            webbrowser.open(str(output_path.resolve()))
            print("  Navegador abierto.")
    else:
        print(f"  ERROR al generar el dashboard.")
    print()
    input("  Presiona Enter para volver al menu...")


def main():
    while True:
        asset = elegir_activo()
        ejecutar(asset)


if __name__ == "__main__":
    main()
