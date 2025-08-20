from typing import List, Optional, Union
from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType

# ----- core mapping: (row, col, digit) -> SAT var id in [1..729]
def vid(r: int, c: int, d: int) -> int:
    # r,c,d are 1-based (1..9)
    return (r - 1) * 81 + (c - 1) * 9 + d

# optional: map a friendly name to pysat's EncType
_ENC_MAP = {
    "pairwise": EncType.pairwise,     # O(n^2) binary AMO, no aux vars
    "seq": EncType.seqcounter,        # sequential/ladder AMO, linear size + aux vars
    "cardnet": EncType.cardnetwrk,    # cardinality/sorting networks, strong + aux vars
}

def _exactly_one(cnf: CNF, lits: List[int], enc: EncType) -> None:
    """
    Add CNF for sum(lits) == 1.
    Always add ALO as one big clause; choose AMO encoding via 'enc'.
    """
    # At-Least-One
    cnf.append(lits[:])

    # At-Most-One
    if enc == EncType.pairwise:
        # simple and strong for n<=9 (no auxiliaries)
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                cnf.append([-lits[i], -lits[j]])
    else:
        amo = CardEnc.atmost(lits=lits, bound=1, encoding=enc)
        cnf.extend(amo.clauses)

def solve_sudoku(
    grid: List[List[int]],
    *,
    all_solutions: bool = True,
    max_solutions: Optional[int] = None,
    encoding: str = "pairwise",
) -> Union[Optional[List[List[int]]], List[List[List[int]]]]:
    """
    Solve a 9x9 Sudoku via SAT.
    - grid: 9x9 ints, 0=blank, 1..9=clue
    - all_solutions: if False (default), return a single solution or None.
                     if True, return a list of all solutions (possibly empty).
    - max_solutions: cap how many to enumerate when all_solutions=True (None = no cap).
    - encoding: 'pairwise' | 'seq' | 'cardnet'
    """
    enc = _ENC_MAP.get(encoding, EncType.pairwise)

    # Primary variables are exactly 1..729
    N_PRIMARY = 9 * 9 * 9

    cnf = CNF()

    # 1) Exactly one digit per cell
    for r in range(1, 10):
        for c in range(1, 10):
            lits = [vid(r, c, d) for d in range(1, 10)]
            _exactly_one(cnf, lits, enc)

    # 2) For each row r and digit d, exactly one column c
    for r in range(1, 10):
        for d in range(1, 10):
            lits = [vid(r, c, d) for c in range(1, 10)]
            _exactly_one(cnf, lits, enc)

    # 3) For each column c and digit d, exactly one row r
    for c in range(1, 10):
        for d in range(1, 10):
            lits = [vid(r, c, d) for r in range(1, 10)]
            _exactly_one(cnf, lits, enc)

    # 4) For each 3x3 box and digit d, exactly one cell
    for br in range(0, 3):
        for bc in range(0, 3):
            rows = range(3 * br + 1, 3 * br + 4)
            cols = range(3 * bc + 1, 3 * bc + 4)
            for d in range(1, 10):
                lits = [vid(r, c, d) for r in rows for c in cols]
                _exactly_one(cnf, lits, enc)

    # 5) Clues as unit clauses
    for r0 in range(9):
        for c0 in range(9):
            d = grid[r0][c0]
            if d:
                cnf.append([vid(r0 + 1, c0 + 1, d)])

    def decode_model(model_pos_set):
        """Turn a model (set of positive ints) into a 9x9 grid."""
        out = [[0] * 9 for _ in range(9)]
        for r in range(1, 10):
            for c in range(1, 10):
                for d in range(1, 10):
                    if vid(r, c, d) in model_pos_set:
                        out[r - 1][c - 1] = d
                        break
        return out

    solutions = []
    with Solver(name="g3", bootstrap_with=cnf.clauses) as s:
        if not all_solutions:
            if not s.solve():
                return None
            model = s.get_model()
            model_pos = {l for l in model if l > 0 and l <= N_PRIMARY}
            return decode_model(model_pos)

        # Enumerate all (or up to max_solutions)
        while s.solve():
            model = s.get_model()
            model_pos = {l for l in model if l > 0 and l <= N_PRIMARY}
            solutions.append(decode_model(model_pos))

            # Block only the assignment on *primary* vars that are True.
            # This guarantees we don't accidentally block equivalent Sudoku
            # solutions just because aux vars differ.
            blocking_clause = [-l for l in model_pos]
            s.add_clause(blocking_clause)

            if max_solutions is not None and len(solutions) >= max_solutions:
                break

    return solutions

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
    print("\n\n")

if __name__ == "__main__":
    # Same puzzle from before (0/ . = blank)
    PUZZLE = [
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
    grid = parse_puzzle(PUZZLE)
    solutions = solve_sudoku(grid)
    if not solutions:
        print("UNSAT (no solution)")
    else:
        print("Solved. Numsolution: ", len(solutions))
        for solution in solutions:
            print_grid(solution)

