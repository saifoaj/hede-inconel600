"""Generate C0 wppm sweep job decks for the coupled HEDE / Inconel 600
reference formulation on the reduced 300-grain (v_smoke) mesh.

Mirrors build_inp.py but on the 1.732 x 1.732 mm 300-grain mesh, with K
dropped to 1e6 MPa/mm (so delta_0 = sigma_c/K is 100x larger than the
original calibration) and the cohesive viscosity bumped from 1e-2 to
5e-2. sigma_c, G_c and delta_F are unchanged.

Decks land in Simulation/jobs_smoke/.
"""

import os
import re
import math

# ---- C0 sweep (wppm) -------------------------------------------------------
C0_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# ---- Paths (relative to this script) ---------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_FILE  = os.path.join(SCRIPT_DIR, '..', 'Mesh', 'inconel600_1p732mm_smoke_final.inp')
JOBS_DIR   = os.path.join(SCRIPT_DIR, 'jobs_smoke')

# ---- Bulk material  INCONEL600  (mm-N-tonne-MPa) ---------------------------
E, NU = 207000.0, 0.29
RHO   = 8.47e-9
COND  = 14.9
CP    = 4.44e8

# ---- Cohesive  Shi (2026) Table 5-7 Case 1  (mm-N-tonne-MPa) ---------------
K_COH   = 1.0e6
SIGMA_C = 1000.0
DELTA_F = 9.36e-6
# Cohesive viscosity is bounded at roughly 1-2% artificial energy dissipation
# at this value (Gao & Bower 2004; Hu & Belytschko 2010 give a 5% safety
# bound) and is identical across all 7 decks so ML trends are preserved.
VISC    = 5.0e-2

# ---- Loading ---------------------------------------------------------------
# Domain height is 1.732 mm (v_smoke; was 4 mm in v3). Keep 25% engineering
# strain target -> applied displacement = 1.732 * 0.25 = 0.433 mm.
DISP   = 0.433
STEP_T = 1.0

# ---- wppm -> atoms/mm^3 conversion (mass-based; M_H = 1.008e-6 tonne/mol) -
N_A   = 6.022e23
M_H   = 1.008e-6
WPPM_TO_ATOMS_PER_MM3 = 1e-6 * RHO / M_H * N_A

# ---- Langmuir-McLean theta(C_L)  (must match usdfld.f exactly) ------------
K_LM = (1.0 / 8.07e19) * math.exp(3.0e7 / (8.314e3 * 298.0))

# ============================================================================
# Step 1  Load and prepare mesh content (once)
# ============================================================================
with open(MESH_FILE, 'r') as f:
    mesh = f.read()

mesh, n_swap = re.subn(r'\*Element,\s*type=COH2D4(?!T)',
                       '*Element, type=COH2D4T', mesh)
assert n_swap == 1, f'Expected 1 COH2D4 element-type block, swapped {n_swap}'

face_sets  = re.findall(r'\*Elset,\s*elset=(face\d+)',  mesh)
bound_sets = re.findall(r'\*Elset,\s*elset=(bound\d+)', mesh)
# Smoke build: don't hard-assert counts (mesh is different from v3).
print(f'[mesh] face sets = {len(face_sets)}, bound sets = {len(bound_sets)}')
assert len(face_sets)  > 0, 'No face elsets found in mesh'
assert len(bound_sets) > 0, 'No bound elsets found in mesh'

node_block = re.search(r'\*Node\s*\n(.*?)\*Element', mesh, re.DOTALL).group(1)
node_ids   = [int(line.split(',')[0]) for line in node_block.splitlines() if line.strip()]
n_max      = max(node_ids)
assert sorted(node_ids) == list(range(1, n_max + 1)), 'Node IDs not sequential 1..N'
print(f'[mesh] {n_max} nodes; COH2D4 -> COH2D4T applied')

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

assembly_inserts = (
    f'*Nset, nset=ALL_NODES, instance=tess, generate\n1, {n_max}, 1\n'
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
# Step 3  Serebrinsky polynomial sigma_c(theta) table (once)
# ============================================================================
THETAS = [round(0.1 * i, 1) for i in range(11)]
def sigma_c(theta):
    return SIGMA_C * (1.0 - 1.0467 * theta + 0.1687 * theta * theta)

damage_init_lines = ['*Damage Initiation, criterion=MAXS, dependencies=1']
for t in THETAS:
    s = sigma_c(t)
    damage_init_lines.append(f'{s:.4f}, {s:.4f}, , {t:.1f}')
damage_init_block = '\n'.join(damage_init_lines) + '\n'

# ============================================================================
# Step 4  Section Controls block
# ============================================================================
section_controls_block = (
    '**\n** SECTION CONTROLS  (must precede *Cohesive Section reference)\n**\n'
    f'*Section Controls, name=COH_CONTROLS, viscosity={VISC}\n'
)

# ============================================================================
# Step 5  Materials block (once)
# ============================================================================
materials = f"""**
** MATERIALS
**
*Material, name=INCONEL600
*Elastic
{E}, {NU}
*Density
{RHO}
*Conductivity
{COND}
*Specific Heat
{CP}
**
*Material, name=COHESIVE_MAT
*Depvar
1
*Elastic, type=TRACTION
{K_COH}, {K_COH}
{damage_init_block}*Damage Evolution, type=DISPLACEMENT
{DELTA_F}
*User Defined Field
*Density
{RHO}
*Conductivity
{COND}
*Specific Heat
{CP}
"""

# ============================================================================
# Step 6  Per-C0 deck loop
# ============================================================================
os.makedirs(JOBS_DIR, exist_ok=True)

def c0_label(c0):
    if c0 == 0:
        return '0'
    # str() is deterministic in Py3 and handles 0.01, 0.05, 0.1, 0.2,
    # 0.5, 1.0 etc. without losing precision (which f'{c0:.1f}' would).
    return str(c0).replace('.', 'p')

written = []
for c0 in C0_VALUES:
    label    = c0_label(c0)
    fname    = os.path.join(JOBS_DIR, f'c0_{label}.inp')
    c0_atoms = c0 * WPPM_TO_ATOMS_PER_MM3

    theta_init = (c0_atoms * K_LM) / (1.0 + c0_atoms * K_LM)
    if theta_init < 1.0e-12:
        theta_init = 1.0e-12

    initial = (
        '**\n** INITIAL CONDITIONS  H pre-charge + FV1 declaration\n**\n'
        '*Initial Conditions, type=TEMPERATURE\n'
        f'ALL_NODES, {c0_atoms:.6e}\n'
        '*Initial Conditions, type=FIELD, variable=1\n'
        f'ALL_NODES, {theta_init:.6e}\n'
    )

    step = f"""**
** STEP  uniaxial tension, coupled temp-disp
**
*Step, name=Loading, nlgeom=YES, inc=10000
*Coupled Temperature-Displacement, stabilize=0.002, deltmx=1.0
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
               f'HEDE Inconel 600 SMOKE -- C0 = {c0} wppm '
               f'(= {c0_atoms:.4e} atoms/mm^3, theta_init = {theta_init:.4e})\n')

    deck = heading + section_controls_block + mesh_modified + materials + initial + step
    with open(fname, 'w', newline='\n') as f:
        f.write(deck)
    written.append(fname)
    print(f'[deck] wrote {fname}  C0_atoms = {c0_atoms:.4e}  theta_init = {theta_init:.4e}')

# ============================================================================
# Step 7  Sanity checks per generated deck
# ============================================================================
print('\n[sanity] per-deck checks:')
for fname in written:
    with open(fname, 'r') as f:
        text = f.read()

    must_be_one = ['*End Part', '*End Assembly', '*Step,', '*End Step',
                   '*Section Controls, name=COH_CONTROLS',
                   '*Initial Conditions, type=FIELD, variable=1',
                   'controls=COH_CONTROLS']
    must_have   = ['ELSET_BULK', 'ELSET_COH', 'INCONEL600', 'COHESIVE_MAT',
                   '*User Defined Field', '*Depvar',
                   '*Coupled Temperature-Displacement',
                   'ALL_NODES', '*Initial Conditions, type=TEMPERATURE',
                   '*Element, type=COH2D4T',
                   'OUTPUT_BULK', 'OUTPUT_COH',
                   'elset=OUTPUT_BULK', 'elset=OUTPUT_COH']
    must_not    = ['*Static', '*Damage Stabilization',
                   'tess.ELSET_BULK', 'tess.ELSET_COH']

    coh2d4_orphans = re.findall(r'\*Element,\s*type=COH2D4(?!T)', text)

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
    if coh2d4_orphans:
        print(f'  FAIL {fname}: {len(coh2d4_orphans)} unswapped COH2D4 element-type blocks')
        ok = False

    if ok:
        print(f'  OK   {fname}')

print(f'\n[done] {len(written)} smoke decks written to {JOBS_DIR}')
print(f'       K = {K_COH:.0e}  visc = {VISC}  DISP = {DISP} mm  STEP_T = {STEP_T} s')
