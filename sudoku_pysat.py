# sudoku_sat_pysat.py
# Solve a 9x9 Sudoku by encoding to CNF and using PySAT.
# Requires: pip install python-sat[pblib,aiger]

from typing import List, Tuple
from pysat.solvers import Glucose4  # you can swap to Cadical153, Minisat22, etc.

# ----- Variable mapping: x_{r,c,d} -> 1..729
def vid(r: int, c: int, d: int) -> int:
    # r,c,d in 1..9
    return (r - 1) * 81 + (c - 1) * 9 + d

def inv_vid(v: int) -> Tuple[int, int, int]:
    v -= 1
    r = v // 81 + 1
    c = (v % 81) // 9 + 1
    d = v % 9 + 1
    return r, c, d

def exactly_one(vars_):
    """Encoding: pairwise at-most-one + at-least-one."""
    clauses = []
    # at-least-one
    clauses.append(list(vars_))
    # at-most-one
    n = len(vars_)
    for i in range(n):
        for j in range(i + 1, n):
            clauses.append([-vars_[i], -vars_[j]])
    return clauses

def sudoku_cnf(grid: List[List[int]]):
    """Build CNF clauses for standard Sudoku + givens."""
    cls = []

    # 1) Each cell: exactly one digit
    for r in range(1, 10):
        for c in range(1, 10):
            vars_ = [vid(r, c, d) for d in range(1, 10)]
            cls += exactly_one(vars_)

    # 2) Row constraints: each digit appears exactly once per row
    for r in range(1, 10):
        for d in range(1, 10):
            vars_ = [vid(r, c, d) for c in range(1, 10)]
            cls += exactly_one(vars_)

    # 3) Column constraints: each digit appears exactly once per column
    for c in range(1, 10):
        for d in range(1, 10):
            vars_ = [vid(r, c, d) for r in range(1, 10)]
            cls += exactly_one(vars_)

    # 4) Block constraints: each digit appears exactly once per 3x3 block
    for br in range(0, 3):
        for bc in range(0, 3):
            for d in range(1, 10):
                vars_ = []
                for dr in range(1, 4):
                    for dc in range(1, 4):
                        r = br * 3 + dr
                        c = bc * 3 + dc
                        vars_.append(vid(r, c, d))
                cls += exactly_one(vars_)

    # 5) Givens
    for r in range(1, 10):
        for c in range(1, 10):
            d = grid[r - 1][c - 1]
            if d != 0:
                cls.append([vid(r, c, d)])
    return cls

def parse_puzzle(lines: List[str]) -> List[List[int]]:
    grid = []
    for line in lines:
        row = []
        for ch in line.strip():
            if ch in "0.":
                row.append(0)
            elif ch.isdigit():
                row.append(int(ch))
        if len(row) == 9:
            grid.append(row)
    if len(grid) != 9:
        raise ValueError("Expected 9 lines of 9 chars (digits or 0/.)")
    return grid

def solve_sudoku(grid: List[List[int]]) -> List[List[int]]:
    clauses = sudoku_cnf(grid)
    with Glucose4(bootstrap_with=clauses) as solver:
        sat = solver.solve()
        if not sat:
            return []
        model = set(l for l in solver.get_model() if l > 0)

    # Build solved grid: pick the (r,c,d) that are True
    out = [[0] * 9 for _ in range(9)]
    for lit in model:
        r, c, d = inv_vid(lit)
        if out[r - 1][c - 1] == 0:
            out[r - 1][c - 1] = d
    return out

def print_grid(grid: List[List[int]]):
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        row = []
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row.append("|")
            row.append(str(grid[r][c]))
        print(" ".join(row))

if __name__ == "__main__":
    # Same puzzle from before (0/ . = blank)
    PUZZLE = [
        "530070000",
        "600195000",
        "098000060",
        "800060003",
        "400803001",
        "700020006",
        "060000280",
        "000419005",
        "000080079",
    ]
    grid = parse_puzzle(PUZZLE)
    solution = solve_sudoku(grid)
    if not solution:
        print("UNSAT (no solution)")
    else:
        print("Solved:")
        print_grid(solution)

