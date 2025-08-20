from typing import List, Optional, Union
from math import isqrt
from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType

# ---------- encoding helpers ----------
_ENC_MAP = {
    "pairwise": EncType.pairwise,     # O(k^2) AMO, no aux vars
    "seq": EncType.seqcounter,        # sequential/ladder AMO, linear + aux vars
    "cardnet": EncType.cardnetwrk,    # sorting/cardinality networks, strong + aux vars
}

def vid(r: int, c: int, d: int, N: int) -> int:
    """
    1-based indices: r,c in [1..N], d in [1..N]
    Maps (r,c,d) -> {1..N^3}
    """
    return (r - 1) * (N * N) + (c - 1) * N + d

def _exactly_one(cnf: CNF, lits: List[int], enc: EncType) -> None:
    """ sum(lits) == 1  (ALO + AMO via chosen encoding) """
    cnf.append(lits[:])  # ALO
    if enc == EncType.pairwise:
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                cnf.append([-lits[i], -lits[j]])
    else:
        amo = CardEnc.atmost(lits=lits, bound=1, encoding=enc)
        cnf.extend(amo.clauses)

# ---------- solver ----------
def solve_sudoku(
    grid: List[List[int]],
    *,
    all_solutions: bool = True,
    max_solutions: Optional[int] = None,
    encoding: str = "pairwise",
) -> Union[Optional[List[List[int]]], List[List[List[int]]]]:
    """
    Solve an N×N Sudoku where N = n^2 (box size is n×n).
    grid: N×N ints, 0=blank, 1..N=clue
    """
    N = len(grid)
    assert all(len(row) == N for row in grid), "Grid must be square (N×N)."
    n = isqrt(N)
    if n * n != N:
        raise ValueError(f"N must be a perfect square (got N={N}).")

    enc = _ENC_MAP.get(encoding, EncType.pairwise)
    N_PRIMARY = N * N * N  # vars 1..N^3 are the (r,c,d) primaries

    cnf = CNF()

    # 1) exactly one digit per cell
    for r in range(1, N + 1):
        for c in range(1, N + 1):
            lits = [vid(r, c, d, N) for d in range(1, N + 1)]
            _exactly_one(cnf, lits, enc)

    # 2) for each row r and digit d, exactly one column c
    for r in range(1, N + 1):
        for d in range(1, N + 1):
            lits = [vid(r, c, d, N) for c in range(1, N + 1)]
            _exactly_one(cnf, lits, enc)

    # 3) for each column c and digit d, exactly one row r
    for c in range(1, N + 1):
        for d in range(1, N + 1):
            lits = [vid(r, c, d, N) for r in range(1, N + 1)]
            _exactly_one(cnf, lits, enc)

    # 4) for each n×n box and digit d, exactly one cell
    for br in range(n):            # box row 0..n-1
        for bc in range(n):        # box col 0..n-1
            rows = range(br * n + 1, br * n + n + 1)
            cols = range(bc * n + 1, bc * n + n + 1)
            for d in range(1, N + 1):
                lits = [vid(r, c, d, N) for r in rows for c in cols]
                _exactly_one(cnf, lits, enc)

    # 5) clues
    for r0 in range(N):
        for c0 in range(N):
            d = grid[r0][c0]
            if d:
                if not (1 <= d <= N):
                    raise ValueError(f"Clue out of range at ({r0},{c0}): {d} (must be 1..{N})")
                cnf.append([vid(r0 + 1, c0 + 1, d, N)])

    def decode_model(model_pos_set):
        out = [[0] * N for _ in range(N)]
        for r in range(1, N + 1):
            for c in range(1, N + 1):
                for d in range(1, N + 1):
                    if vid(r, c, d, N) in model_pos_set:
                        out[r - 1][c - 1] = d
                        break
        return out

    solutions: List[List[List[int]]] = []
    with Solver(name="g3", bootstrap_with=cnf.clauses) as s:
        if not all_solutions:
            if not s.solve():
                return None
            model = s.get_model()
            model_pos = {l for l in model if l > 0 and l <= N_PRIMARY}
            return decode_model(model_pos)

        while s.solve():
            model = s.get_model()
            model_pos = {l for l in model if l > 0 and l <= N_PRIMARY}
            solutions.append(decode_model(model_pos))

            # block only the primary true literals
            s.add_clause([-l for l in model_pos])

            if max_solutions is not None and len(solutions) >= max_solutions:
                break

    return solutions

# ---------- parsing/printing ----------
def parse_puzzle(lines: List[str]) -> List[List[int]]:
    """
    Accepts:
      - rows of digits with 0/. for blanks (no spaces), e.g. '530070000'
      - or space-separated integers, e.g. '5 3 0 0 7 0 0 0 0'
    Works for any N; validates N is a perfect square.
    """
    rows: List[List[int]] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if " " in s:
            toks = s.split()
            row = [0 if t in ("0", ".") else int(t) for t in toks]
        else:
            row = []
            for ch in s:
                if ch in "0.":
                    row.append(0)
                elif ch.isdigit():
                    row.append(int(ch))
                else:
                    # allow A..Z for 10..35 if someone wants 16x16 input
                    if "A" <= ch <= "Z":
                        row.append(10 + ord(ch) - ord("A"))
                    elif "a" <= ch <= "z":
                        row.append(10 + ord(ch) - ord("a"))
        if row:
            rows.append(row)

    if not rows:
        raise ValueError("No rows parsed.")
    N = len(rows)
    if any(len(r) != N for r in rows):
        raise ValueError("All rows must have equal length (square grid).")
    n = isqrt(N)
    if n * n != N:
        raise ValueError(f"N must be a perfect square (got N={N}).")
    # validate clues range
    for r in range(N):
        for c in range(N):
            v = rows[r][c]
            if v != 0 and not (1 <= v <= N):
                raise ValueError(f"Cell ({r},{c}) value {v} out of range 1..{N}")
    return rows

def _symbol(v: int) -> str:
    # pretty printing up to base-36: 1..9, A..Z
    if v == 0:
        return "."
    if 1 <= v <= 9:
        return str(v)
    return chr(ord("A") + (v - 10))

def print_grid(grid: List[List[int]]) -> None:
    N = len(grid)
    n = isqrt(N)
    line = []
    cellw = 2 if N <= 9 else 3  # wider for 16x16 etc.
    hsep = "-" * ((cellw + 1) * N + (n - 1) * 2)

    LEN = 0
    for r in range(N):
        if r % n == 0 and r != 0:
            print(hsep)
        parts = []
        for c in range(N):
            if c % n == 0 and c != 0:
                parts.append("|")
            parts.append(_symbol(grid[r][c]).rjust(cellw))
        line = " ".join(parts)
        LEN = len(line)
        print(line)
    print(("#" * LEN))
    print(("#" * LEN))

# ---------- demo ----------
if __name__ == "__main__":
    # 9x9 example (n=3)
    PUZZLE_9 = [
        "530070000",
        "600105000",
        "098000060",
        "800000003",
        "400803001",
        "700020006",
        "060000280",
        "000419005",
        "000080079",
    ]
    # 4x4 example (n=2) — solution should be unique
    PUZZLE_4 = [
        "1030",
        "0002",
        "0100",
        "0040",
    ]

    PUZZLE_16 = [
        "1.3.5.7.9.B.D.F.",
        "5.7.9.B.D.F.1.3.",
        "9.B.D.F.1.3.5.7.",
        "D.F.1.3.5.7.9.B.",
        "2.4.6.8.A.C.E.G.",
        "6.8.A.C.E.G.2.4.",
        "A.C.E.G.2.4.6.8.",
        "E.G.2.4.6.8.A.C.",
        "3.5.7.9.B.D.F.1.",
        "7.9.B.D.F.1.3.5.",
        "B.D.F.1.3.5.7.9.",
        "F.1.3.5.7.9.B.D.",
        "4.6.8.A.C.E.G.2.",
        "8.A.C.E.G.2.4.6.",
        "C.E.G.2.4.6.8.A.",
        "G.2.4.6.8.A.C.E.",
    ]



    grid = parse_puzzle(PUZZLE_16)   # swap to PUZZLE_4 to try 4x4
    sols = solve_sudoku(grid, all_solutions=True, max_solutions=5, encoding="pairwise")
    if not sols:
        print("UNSAT (no solution)")
    else:
        print("Num solutions:", len(sols))
        for g in sols:
            print_grid(g)

