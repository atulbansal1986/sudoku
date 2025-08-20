#!/usr/bin/env python3
from typing import List, Tuple, Dict, Optional, Union
from math import isqrt
from functools import reduce
from operator import mul
from z3 import Int, Solver, And, Or, Distinct, Sum, sat, ModelRef
from z3 import Abs

# ------------------------
# Utility helpers
# ------------------------

def _block_model_on_grid(slv: Solver, X, model: ModelRef, N: int):
    """Add a blocking clause to forbid the current solution."""
    diffs = []
    for r in range(N):
        for c in range(N):
            v = model.eval(X[r][c], model_completion=True)
            diffs.append(X[r][c] != v)
    slv.add(Or(diffs))


def _symbols_to_int_row(s: str) -> List[int]:
    """
    Parse a Sudoku row:
      - digits 1..9
      - A..Z for 10..35 (for 16x16 etc.)
      - '.' or '0' for blanks
    """
    out = []
    for ch in s.strip():
        if ch in "0.":
            out.append(0)
        elif ch.isdigit():
            out.append(int(ch))
        elif "A" <= ch <= "Z":
            out.append(10 + ord(ch) - ord("A"))
        elif "a" <= ch <= "z":
            out.append(10 + ord(ch) - ord("a"))
    return out


def print_grid(grid: List[List[int]]):
    """Pretty-print Sudoku or KenKen grid of any size."""
    N = len(grid)
    n = isqrt(N)
    cellw = 2 if N <= 9 else 3

    def sym(v: int) -> str:
        if v == 0: return "."
        if 1 <= v <= 9: return str(v)
        return chr(ord("A") + (v - 10))

    hsep = "-" * ((cellw + 1) * N + (n - 1) * 2)
    for r in range(N):
        if r % n == 0 and r != 0:
            print(hsep)
        parts = []
        for c in range(N):
            if c % n == 0 and c != 0:
                parts.append("|")
            parts.append(sym(grid[r][c]).rjust(cellw))
        print(" ".join(parts))
    print()


# ------------------------
# Sudoku Solver
# ------------------------

def solve_sudoku_z3(
    grid: List[List[int]],
    *,
    all_solutions: bool = False,
    max_solutions: Optional[int] = None,
) -> Union[Optional[List[List[int]]], List[List[List[int]]]]:
    """
    Solve an N×N Sudoku puzzle using Z3.
    N must be a perfect square (n^2).
    """
    N = len(grid)
    assert all(len(row) == N for row in grid), "Sudoku grid must be square"
    n = isqrt(N)
    if n * n != N:
        raise ValueError(f"N must be a perfect square; got N={N}")

    # Variables
    X = [[Int(f"x_{r}_{c}") for c in range(N)] for r in range(N)]
    s = Solver()

    # Domain constraints: 1..N
    for r in range(N):
        for c in range(N):
            s.add(And(1 <= X[r][c], X[r][c] <= N))

    # Givens
    for r in range(N):
        for c in range(N):
            if grid[r][c] != 0:
                s.add(X[r][c] == grid[r][c])

    # Rows, columns distinct
    for r in range(N):
        s.add(Distinct(X[r]))
    for c in range(N):
        s.add(Distinct([X[r][c] for r in range(N)]))

    # Boxes distinct
    for br in range(n):
        for bc in range(n):
            cells = [X[r][c]
                     for r in range(br * n, br * n + n)
                     for c in range(bc * n, bc * n + n)]
            s.add(Distinct(cells))

    def model_to_grid(m: ModelRef) -> List[List[int]]:
        return [[m.eval(X[r][c], model_completion=True).as_long() for c in range(N)]
                for r in range(N)]

    if not all_solutions:
        if s.check() != sat:
            return None
        return model_to_grid(s.model())

    # Enumerate all solutions
    sols: List[List[List[int]]] = []
    while s.check() == sat:
        m = s.model()
        sols.append(model_to_grid(m))
        _block_model_on_grid(s, X, m, N)
        if max_solutions is not None and len(sols) >= max_solutions:
            break
    return sols


# ------------------------
# KenKen Solver
# ------------------------

Cage = Dict[str, Union[int, str, List[Tuple[int, int]]]]
# cage: {"cells":[(r,c),...], "op": "+|-|*|/|=", "target": int}

def solve_kenken_z3(
    N: int,
    cages: List[Cage],
    *,
    givens: Optional[Dict[Tuple[int, int], int]] = None,
    all_solutions: bool = False,
    max_solutions: Optional[int] = None,
) -> Union[Optional[List[List[int]]], List[List[List[int]]]]:
    """Solve KenKen using Z3."""
    X = [[Int(f"k_{r}_{c}") for c in range(N)] for r in range(N)]
    s = Solver()

    # Domain 1..N
    for r in range(N):
        for c in range(N):
            s.add(And(1 <= X[r][c], X[r][c] <= N))

    # Row/col uniqueness
    for r in range(N):
        s.add(Distinct(X[r]))
    for c in range(N):
        s.add(Distinct([X[r][c] for r in range(N)]))

    # Givens
    if givens:
        for (r1, c1), val in givens.items():
            s.add(X[r1 - 1][c1 - 1] == val)

    # Cages
    for cage in cages:
        cells: List[Tuple[int, int]] = cage["cells"]  # 1-based
        op: str = cage["op"]
        tgt: int = int(cage["target"])
        zcells = [X[r - 1][c - 1] for (r, c) in cells]

        if op == "=":
            assert len(zcells) == 1
            s.add(zcells[0] == tgt)

        elif op == "+":
            s.add(Sum(zcells) == tgt)

        elif op == "*":
            prod = reduce(lambda a, b: a * b, zcells)
            s.add(prod == tgt)


        elif op == "-":
            if len(zcells) == 2:
                a, b = zcells
                s.add(Or(a - b == tgt, b - a == tgt))
            else:
                # exists a pivot i: |xi - sum(others)| == tgt
                terms = []
                for i in range(len(zcells)):
                    others = [zcells[j] for j in range(len(zcells)) if j != i]
                    terms.append(Abs(zcells[i] - Sum(others)) == tgt)
                s.add(Or(*terms))

        elif op == "/":
            if len(zcells) == 2:
                a, b = zcells
                s.add(Or(a == tgt * b, b == tgt * a))
            else:
                # exists a pivot i: xi == tgt * product(others)  or  product(others) == tgt * xi
                terms = []
                for i in range(len(zcells)):
                    others = [zcells[j] for j in range(len(zcells)) if j != i]
                    prod_others = reduce(lambda a, b: a * b, others)
                    terms.append(Or(zcells[i] == tgt * prod_others,
                                    prod_others == tgt * zcells[i]))
                s.add(Or(*terms))
        else:
            raise ValueError(f"Unknown cage op: {op}")

    def model_to_grid(m: ModelRef) -> List[List[int]]:
        return [[m.eval(X[r][c], model_completion=True).as_long() for c in range(N)]
                for r in range(N)]

    if not all_solutions:
        if s.check() != sat:
            return None
        return model_to_grid(s.model())

    # Enumerate solutions
    sols: List[List[List[int]]] = []
    while s.check() == sat:
        m = s.model()
        sols.append(model_to_grid(m))
        _block_model_on_grid(s, X, m, N)
        if max_solutions is not None and len(sols) >= max_solutions:
            break
    return sols


# ------------------------
# Main demo
# ------------------------

if __name__ == "__main__":
    # ---- Sudoku Example ----
    PUZZLE_9 = [
        _symbols_to_int_row("530070000"),
        _symbols_to_int_row("600105000"),
        _symbols_to_int_row("098000060"),
        _symbols_to_int_row("800000003"),
        _symbols_to_int_row("400803001"),
        _symbols_to_int_row("700020006"),
        _symbols_to_int_row("060000280"),
        _symbols_to_int_row("000419005"),
        _symbols_to_int_row("000080079"),
    ]
    print("=== Solving Sudoku ===")
    sudoku_sol = solve_sudoku_z3(PUZZLE_9)
    if sudoku_sol is None:
        print("Sudoku: UNSAT (no solution)")
    else:
        print_grid(sudoku_sol)


    # ---- KenKen Example (4×4, solvable) ----
    # This cage set is consistent and solves to a valid 4×4 Latin square.
    # Cells are 1-based indices (r,c).
    N = 4
    cages = [
        {"cells": [(1,1),(1,2)],               "op": "+", "target": 3},  # 1+2
        {"cells": [(1,3),(1,4)],               "op": "+", "target": 7},  # 3+4
        {"cells": [(2,1),(3,1)],               "op": "+", "target": 7},  # 3+4
        {"cells": [(2,2)],                     "op": "=", "target": 4},  # fixed 4
        {"cells": [(2,3),(3,3),(3,4)],         "op": "+", "target": 4},  # 1+2+1
        {"cells": [(2,4),(3,2),(4,1)],         "op": "+", "target": 7},  # 2+3+2
        {"cells": [(4,2),(4,3),(4,4)],         "op": "*", "target": 12}, # 1*4*3
    ]
    print("=== Solving KenKen ===")
    kenken_sol = solve_kenken_z3(N, cages)
    if kenken_sol is None:
        print("KenKen: UNSAT (no solution)")
    else:
        print_grid(kenken_sol)



    cages = [
        {"cells": [(1,1),(1,2),(2,1),(3,1)],       "op": "*", "target": 120},
        {"cells": [(1,3),(1,4),(2,2),(2,3)],       "op": "+", "target": 14},
        {"cells": [(1,5)],                         "op": "=", "target": 3},
        {"cells": [(1,6),(2,6)],                   "op": "/", "target": 6},
        {"cells": [(2,4),(2,5)],                   "op": "/", "target": 2},
        {"cells": [(3,2),(3,3),(4,3)],             "op": "*", "target": 60},
        {"cells": [(3,4),(4,4)],                   "op": "+", "target": 7},
        {"cells": [(3,5),(4,5),(3,6)],             "op": "+", "target": 12},
        {"cells": [(4,6),(5,5),(5,6),(6,5),(6,6)], "op": "+", "target": 18},
        {"cells": [(4,1),(4,2)],                   "op": "/", "target": 6},
        {"cells": [(5,1),(5,2)],                   "op": "-", "target": 1},
        {"cells": [(5,3),(5,4),(6,4)],             "op": "+", "target": 12},
        {"cells": [(6,1),(6,2),(6,3)],             "op": "*", "target": 24},
    ]

    N = 6
    solution = solve_kenken_z3(N, cages)
    if solution is None:
        print("No solution found.")
    else:
        print_grid(solution)

