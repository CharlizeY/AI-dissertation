import re
import subprocess
from typing import List


# Manually tested


# Uses clingo to get all answer sets of a program
# Returns a number of outer lists equal to the number answer sets
# List i contains number of atoms in i-th answer set
def clingo_solve(*files: str) -> List[List[str]]:
    out = subprocess.run(['clingo', '-n', '0', *files], stdout=subprocess.PIPE)
    lines = out.stdout.decode('utf-8').splitlines()

    sols = []
    for i, line in enumerate(lines):
        if re.match(r'Answer: (\d+)', line):
            # Matches any possible bracketed predicate or just atom.
            # ?: is a non-capturing group
            atoms = re.findall(r'\w+(?:\([^)]*\))?', lines[i + 1])
            sols.append(atoms)

    return sols
