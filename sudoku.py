# sudoku_sat_solver.py
# Solve Sudoku via SAT (CNF) with a minimal DPLL solver (unit propagation + backtracking).
# No external libraries required.

from typing import List, Dict, Tuple, Optional

# --- Variable mapping: x_{r,c,d} -> integer in [1..729]
def var_id(r: int, c: int, d: int) -> int:
    # r,c,d are 1..9
    return (r - 1) * 81 + (c - 1) * 9 + d

def inv_var_id(v: int) -> Tuple[int, int, int]:
    v0 = v - 1
    r = v0 // 81 + 1
    c = (v0 % 81) // 9 + 1
    d = v0 % 9 + 1
    return r, c, d

# --- CNF helpers
def at_least_one(vars_: List[int]) -> List[List[int]]:
    return [vars_]

def at_most_one(vars_: List[int]) -> List[List[int]]:
    clauses = []
    n = len(vars_)
    for i in range(n):
        for j in range(i + 1, n):
            clauses.append([-vars_[i], -vars_[j]])
    return clauses

def exactly_one(vars_: List[int]) -> List[List[int]]:
    return at_least_one(vars_) + at_most_one(vars_)

# --- Sudoku -> CNF
def sudoku_cnf(grid: List[List[int]]) -> List[List[int]]:
    clauses: List[List[int]] = []

    # 1) Each cell has exactly one digit
    for r in range(1, 10):
        for c in range(1, 10):
            vars_ = [var_id(r, c, d) for d in range(1, 10)]
            clauses += exactly_one(vars_)

    # 2) Each digit appears exactly once in each row
    for r in range(1, 10):
        for d in range(1, 10):
            vars_ = [var_id(r, c, d) for c in range(1, 10)]
            clauses += exactly_one(vars_)

    # 3) Each digit appears exactly once in each column
    for c in range(1, 10):
        for d in range(1, 10):
            vars_ = [var_id(r, c, d) for r in range(1, 10)]
            clauses += exactly_one(vars_)

    # 4) Each digit appears exactly once in each 3x3 block
    for br in range(0, 3):
        for bc in range(0, 3):
            for d in range(1, 10):
                vars_ = []
                for dr in range(1, 4):
                    for dc in range(1, 4):
                        r = br * 3 + dr
                        c = bc * 3 + dc
                        vars_.append(var_id(r, c, d))
                clauses += exactly_one(vars_)

    # 5) Givens
    for r in range(1, 10):
        for c in range(1, 10):
            d = grid[r - 1][c - 1]
            if d != 0:
                clauses.append([var_id(r, c, d)])

    return clauses

# --- Minimal SAT solver (DPLL with unit propagation)
def simplify(clauses: List[List[int]], lit: int) -> Optional[List[List[int]]]:
    """Assign literal 'lit' = True; return simplified clauses or None on conflict."""
    new_clauses: List[List[int]] = []
    for clause in clauses:
        satisfied = False
        new_clause = []
        for l in clause:
            if l == lit:
                satisfied = True
                break
            elif l == -lit:
                # remove the false literal
                continue
            else:
                new_clause.append(l)
        if satisfied:
            continue
        if len(new_clause) == 0:
            return None  # conflict
        new_clauses.append(new_clause)
    return new_clauses

def unit_propagate(clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[Tuple[List[List[int]], Dict[int, bool]]]:
    """Repeatedly apply unit propagation; return (simplified_clauses, assignment) or None on conflict."""
    changed = True
    while changed:
        changed = False
        unit_literals = [cl[0] for cl in clauses if len(cl) == 1]
        if not unit_literals:
            break
        for lit in unit_literals:
            var = abs(lit)
            val = (lit > 0)
            if var in assignment:
                if assignment[var] != val:
                    return None  # conflict
            else:
                assignment[var] = val
                clauses = simplify(clauses, lit)
                if clauses is None:
                    return None
                changed = True
    return clauses, assignment

def choose_literal(clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[int]:
    """Pick a literal from the first clause that still has unassigned vars."""
    for clause in clauses:
        for lit in clause:
            if abs(lit) not in assignment:
                return lit
    return None

def dpll(clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
    up = unit_propagate(clauses, assignment)
    if up is None:
        return None
    clauses, assignment = up
    if not clauses:
        return assignment  # SAT

    lit = choose_literal(clauses, assignment)
    if lit is None:
        return assignment

    var = abs(lit)
    # Try True then False
    for val in (True, False):
        lit_try = var if val else -var
        new_clauses = simplify(clauses, lit_try)
        if new_clauses is None:
            continue
        new_assignment = assignment.copy()
        new_assignment[var] = val
        res = dpll(new_clauses, new_assignment)
        if res is not None:
            return res
    return None  # UNSAT

# --- Utilities
def parse_puzzle(lines: List[str]) -> List[List[int]]:
    grid: List[List[int]] = []
    for line in lines:
        row: List[int] = []
        for ch in line.strip():
            if ch in '0.':
                row.append(0)
            elif ch.isdigit():
                row.append(int(ch))
        if len(row) == 9:
            grid.append(row)
    if len(grid) != 9:
        raise ValueError("Puzzle must be 9 lines of 9 chars (digits or '.'/0).")
    return grid

def solve_sudoku(grid: List[List[int]]) -> Optional[List[List[int]]]:
    clauses = sudoku_cnf(grid)
    sol = dpll(clauses, {})
    if sol is None:
        return None
    # Build solved grid from true literals
    out = [[0] * 9 for _ in range(9)]
    for v, val in sol.items():
        if val:
            r, c, d = inv_var_id(v)
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
    if solution is None:
        print("UNSAT (no solution).")
    else:
        print("Solved:")
        print_grid(solution)

