# Mesh Generation

The cohesive-element-ready mesh used in the parametric sweep is
generated externally by Neper 4.10.1 rather than committed to the
repository. The mesh file is ~100 MB and exceeds the GitHub file-size
limit; regeneration takes under a minute on a standard workstation.

## Regenerate the v3 mesh

```
neper -T -n 1200 -id 5 -dim 2 -domain "square(4,4)" \
  -morpho "diameq:lognormal(0.0967,0.0876,from=0.030)" \
  -morphooptistop "val=1e-2,itermax=1000" \
  -reg 1 \
  -format tess,geo,ori \
  -o inconel600_4mm_v3

neper -M inconel600_4mm_v3.tess \
  -rcl 0.5 -elttype tri -order 1 \
  -interface cohesive \
  -format msh,inp \
  -o inconel600_4mm_v3_tri1
```

The bulk element type is emitted as CPE3 by Neper. For the headline
*Static formulation no further substitution is required. For the
coupled temperature-displacement reference formulation, substitute
CPE3 -> CPE3T in the generated .inp file before running.

## Regenerate the v_smoke mesh

The v_smoke realisation is generated identically with `-id 6 -n 300
-domain "square(1.732,1.732)"` and is otherwise unchanged.

The frozen mesh files used for the dissertation submission can be made
available on request.
