"""Generate C0 wppm sweep job decks - headline *Static formulation on the
canonical 4 x 4 mm 1200-grain (v3) mesh.

Forked from build_inp_static.py. The only substantive differences are the
mesh file, the applied displacement (1.0 mm for 25% engineering strain on a
4 mm gauge), and the output directory. All cohesive parameters and the
Serebrinsky-based per-deck sigma_c assignment follow build_inp_static.py.

Decks land in Simulation/jobs_v3_static/.
"""

import os
import re
import math

# ---- C0 sweep (wppm) -------------------------------------------------------
C0_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# ---- Paths (relative to this script) ---------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_FILE  = os.path.join(SCRIPT_DIR, '..', 'Mesh', 'inconel600_4mm_v3_final.inp')
JOBS_DIR   = os.path.join(SCRIPT_DIR, 'jobs_v3_static')

# ---- Bulk material  INCONEL600  (mm-N-tonne-MPa) ---------------------------
E, NU = 207000.0, 0.29
RHO   = 8.47e-9

# ---- Cohesive  Shi (2026) Table 5-7 Case 1  (mm-N-tonne-MPa) ---------------
K_COH   = 1.0e6
SIGMA_C = 1000.0
DELTA_F = 9.36e-6
VISC    = 5.0e-2

# ---- Loading (v3: 4 mm domain, 25% strain) ---------------------------------
DISP   = 1.0
STEP_T = 1.0

# ---- wppm -> atoms/mm^3 conversion (mass-based; M_H = 1.008e-6 tonne/mol) -
N_A   = 6.022e23
M_H   = 1.008e-6
WPPM_TO_ATOMS_PER_MM3 = 1e-6 * RHO / M_H * N_A

# ---- Langmuir-McLean theta(C_L) -------------------------------------------
K_LM = (1.0 / 8.07e19) * math.exp(3.0e7 / (8.314e3 * 298.0))

# ---- Serebrinsky polynomial sigma_c(theta) --------------------------------
def sigma_c(theta):
    return SIGMA_C * (1.0 - 1.0467 * theta + 0.1687 * theta * theta)

# ============================================================================
# Step 1  Load and prepare mesh content (once)
# ============================================================================
with open(MESH_FILE, 'r') as f:
    mesh = f.read()

# 1a. Revert bulk element type CPE3T -> CPE3 (no temp DOF in *Static).
mesh, n_swap = re.subn(r'\*Element,\s*type=CPE3T\b',
                       '*Element, type=CPE3', mesh)
assert n_swap == 1, f'Expected 1 CPE3T element-type block, swapped {n_swap}'

# Cohesive elements stay COH2D4 on disk - no promotion to COH2D4T.

# 1b. Collect part-level elsets.
face_sets  = re.findall(r'\*Elset,\s*elset=(face\d+)',  mesh)
bound_sets = re.findall(r'\*Elset,\s*elset=(bound\d+)', mesh)
print(f'[mesh] face sets = {len(face_sets)}, bound sets = {len(bound_sets)}')
assert len(face_sets)  > 0
assert len(bound_sets) > 0

# 1c. Sanity-check sequential node numbering (informational).
node_block = re.search(r'\*Node\s*\n(.*?)\*Element', mesh, re.DOTALL).group(1)
node_ids   = [int(line.split(',')[0]) for line in node_block.splitlines() if line.strip()]
n_max      = max(node_ids)
print(f'[mesh] {n_max} nodes; CPE3T -> CPE3 applied')

# ============================================================================
# Step 2  Splice fragments (once)
# ============================================================================
def elset_block(name, members, per_line=16):
    lines = [f'*Elset, elset={name}']
    for i in range(0, len(members), per_line):
        lines.append(', '.join(members[i:i+per_line]) + ',')
    return '\n'.join(lines) + '\n'

bulk_elset = elset_block('ELSET_BULK', face_sets)
coh_elset  = elset_block('ELSET_COH',  bound_sets)

part_inserts = (
    bulk_elset + coh_elset +
    '*Solid Section, elset=ELSET_BULK, material=INCONEL600\n'
    ',\n'
    '*Cohesive Section, elset=ELSET_COH, material=COHESIVE_MAT, '
    'response=TRACTION SEPARATION, thickness=SPECIFIED, controls=COH_CONTROLS\n'
    '1.,\n'
)

# Assembly-level elsets for step-output requests.
assembly_inserts = (
    '*Elset, elset=OUTPUT_BULK, instance=tess\n'
    'ELSET_BULK,\n'
    '*Elset, elset=OUTPUT_COH, instance=tess\n'
    'ELSET_COH,\n'
)

mesh_modified = (
    mesh
    .replace('*End Part',     part_inserts     + '*End Part',     1)
    .replace('*End Assembly', assembly_inserts + '*End Assembly', 1)
)

# ============================================================================
# Step 3  Section Controls block
# ============================================================================
section_controls_block = (
    '**\n** SECTION CONTROLS  (must precede *Cohesive Section reference)\n**\n'
    f'*Section Controls, name=COH_CONTROLS, viscosity={VISC}\n'
)

# ============================================================================
# Step 4  Materials block (per-deck, sigma_c varies)
# ============================================================================
def make_materials(sigma_c_deck):
    """Materials block with uniform per-deck sigma_c."""
    return f"""**
** MATERIALS
**
*Material, name=INCONEL600
*Elastic
{E}, {NU}
*Density
{RHO}
**
*Material, name=COHESIVE_MAT
*Elastic, type=TRACTION
{K_COH}, {K_COH}
*Damage Initiation, criterion=MAXS
{sigma_c_deck:.4f}, {sigma_c_deck:.4f}, {sigma_c_deck:.4f}
*Damage Evolution, type=DISPLACEMENT
{DELTA_F}
*Density
{RHO}
"""

# ============================================================================
# Step 5  Per-C0 deck loop
# ============================================================================
os.makedirs(JOBS_DIR, exist_ok=True)

def c0_label(c0):
    if c0 == 0:
        return '0'
    return str(c0).replace('.', 'p')

written = []
for c0 in C0_VALUES:
    label    = c0_label(c0)
    fname    = os.path.join(JOBS_DIR, f'c0_{label}.inp')
    c0_atoms = c0 * WPPM_TO_ATOMS_PER_MM3

    theta = (c0_atoms * K_LM) / (1.0 + c0_atoms * K_LM)
    if theta < 0.0:
        theta = 0.0

    sigma_c_deck = sigma_c(theta)

    step = f"""**
** STEP  uniaxial tension, STATIC
**
*Step, name=Loading, nlgeom=YES, inc=10000
*Static, stabilize=0.002
1e-06, {STEP_T}, 1e-09, 0.05
**
** BOUNDARY CONDITIONS
**
*Boundary
y0, 2, 2, 0.
x0y0, 1, 1, 0.
**
*Boundary, type=DISPLACEMENT
y1, 2, 2, {DISP}
**
** OUTPUT REQUESTS
**
*Output, field, frequency=20
*Node Output
U, RF
*Element Output, elset=OUTPUT_BULK, directions=YES
S
*Element Output, elset=OUTPUT_COH
SDEG
*Output, history, frequency=1
*Node Output, nset=y1
RF2
*Node Output, nset=y0
RF2
*End Step
"""

    heading = (f'*Heading\n'
               f'HEDE Inconel 600 STATIC v3 -- C0 = {c0} wppm '
               f'(theta = {theta:.4e}, sigma_c = {sigma_c_deck:.2f} MPa)\n')

    deck = (heading + section_controls_block + mesh_modified
            + make_materials(sigma_c_deck) + step)
    with open(fname, 'w', newline='\n') as f:
        f.write(deck)
    written.append((fname, theta, sigma_c_deck))
    print(f'[deck] wrote {fname}  theta = {theta:.4e}  sigma_c = {sigma_c_deck:.2f} MPa')

# ============================================================================
# Step 6  Sanity checks per generated deck
# ============================================================================
print('\n[sanity] per-deck checks:')
for fname, theta, sigma_c_deck in written:
    with open(fname, 'r') as f:
        text = f.read()

    must_be_one = ['*End Part', '*End Assembly', '*Step,', '*End Step',
                   '*Section Controls, name=COH_CONTROLS',
                   '*Static',
                   '*Damage Initiation, criterion=MAXS',
                   'controls=COH_CONTROLS']
    must_have   = ['ELSET_BULK', 'ELSET_COH', 'INCONEL600', 'COHESIVE_MAT',
                   '*Element, type=CPE3',
                   '*Element, type=COH2D4',
                   'OUTPUT_BULK', 'OUTPUT_COH',
                   'elset=OUTPUT_BULK', 'elset=OUTPUT_COH']
    must_not    = ['*Coupled Temperature-Displacement',
                   '*Initial Conditions, type=TEMPERATURE',
                   '*Initial Conditions, type=FIELD',
                   '*User Defined Field',
                   '*Depvar',
                   '*Element, type=CPE3T',
                   '*Element, type=COH2D4T',
                   '*Conductivity',
                   '*Specific Heat',
                   'dependencies=1',
                   'tess.ELSET_BULK', 'tess.ELSET_COH']

    ok = True
    for kw in must_be_one:
        n = text.count(kw)
        if n != 1:
            print(f'  FAIL {fname}: {kw!r} count = {n}, expected 1')
            ok = False
    for kw in must_have:
        if kw not in text:
            print(f'  FAIL {fname}: missing {kw!r}')
            ok = False
    for kw in must_not:
        if kw in text:
            print(f'  FAIL {fname}: forbidden {kw!r} present')
            ok = False

    if ok:
        print(f'  OK   {fname}')

print(f'\n[done] {len(written)} v3 static decks written to {JOBS_DIR}')
print(f'       K = {K_COH:.0e}  visc = {VISC}  DISP = {DISP} mm  STEP_T = {STEP_T} s')
print(f'       per-deck sigma_c:')
for fname, theta, sigma_c_deck in written:
    print(f'         {os.path.basename(fname):<14}  theta = {theta:.4e}  sigma_c = {sigma_c_deck:7.2f} MPa')
