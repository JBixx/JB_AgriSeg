"""Découpage d'une grande image (orthomosaïque) en tuiles avec recouvrement."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def _starts(length: int, tile: int, stride: int) -> list[int]:
    """Positions de départ couvrant [0, length) ; la dernière tuile est collée au bord."""
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] + tile < length:
        s.append(length - tile)
    return s


def make_tiles(width: int, height: int, tile_size: int = 640, overlap: float = 0.25) -> list[Tile]:
    """Grille de tuiles `tile_size` avec un recouvrement fractionnaire `overlap`.

    Le recouvrement garantit qu'une lésion coupée en bord de tuile est vue
    entière dans la tuile voisine ; les doublons sont fusionnés ensuite
    (voir postprocess.merge_instances)."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap doit être dans [0, 1)")
    stride = max(1, int(round(tile_size * (1 - overlap))))
    return [
        Tile(x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in _starts(height, tile_size, stride)
        for x in _starts(width, tile_size, stride)
    ]
