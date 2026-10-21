"""Utilitaires communs : configuration, graines, device, journalisation."""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def get_logger(name: str = "crop") -> logging.Logger:
    """Logger console homogène pour tous les scripts."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Charge config.yaml (chemin relatif à la racine du projet si besoin)."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {p}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("project", {}).get("use_system_certs", False):
        enable_system_certs()
    return cfg


def resolve(path: str | Path) -> Path:
    """Chemin relatif à la racine du projet -> chemin absolu."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def enable_system_certs() -> None:
    """Utilise le magasin de certificats de l'OS (utile derrière un proxy/antivirus
    qui inspecte le HTTPS). Sans effet si le paquet `truststore` est absent."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 — optionnel
        pass


def set_seed(seed: int) -> None:
    """Fixe toutes les graines pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(pref: str | int = "auto") -> str:
    """Détecte le device : GPU CUDA si disponible, sinon Apple MPS, sinon CPU."""
    if str(pref) != "auto":
        return str(pref)
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ
