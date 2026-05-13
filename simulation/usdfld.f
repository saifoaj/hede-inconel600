C  USDFLD - hydrogen-coverage field for HEDE coupling in Inconel 600
C
C  Reads the local lattice hydrogen concentration C_L from the
C  temperature DOF NT11 under the thermal-mass diffusion analogy
C  (Wong 2015) and writes the surface coverage theta given by the
C  Langmuir-McLean isotherm to FIELD(1). The cohesive material
C  table tabulates the Serebrinsky 2004 polynomial sigma_c(theta)
C  against this field.
C
C  Author:     Saif Abu Joudeh (11317648)
C  Supervisor: Dr Andrey Jivkov
C
C  References:
C    Serebrinsky, Carter, Ortiz (2004) JMPS 52: 2403-2430
C    Wong (2015) Eng. Fract. Mech. 142: 81-101
C    Shi (2026) PhD thesis, University of Manchester
C
C  Unit system: mm-N-tonne-MPa
C
      SUBROUTINE USDFLD(FIELD,STATEV,PNEWDT,DIRECT,T,CELENT,
     1     TIME,DTIME,CMNAME,ORNAME,NFIELD,NSTATV,NOEL,NPT,LAYER,
     2     KSPT,KSTEP,KINC,NDI,NSHR,COORD,JMAC,JMATYP,MATLAYO,LACCFLG)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME, ORNAME
      DIMENSION FIELD(NFIELD), STATEV(NSTATV), DIRECT(3,3), T(2),
     1          TIME(2), COORD(*), JMAC(*), JMATYP(*)
C
C  Grain-boundary segregation free energy (Serebrinsky 2004), N.mm/mol.
      DOUBLE PRECISION DGB
      PARAMETER (DGB = 3.0D7)
C
C  Universal gas constant, N.mm/(mol.K).
      DOUBLE PRECISION R_GAS
      PARAMETER (R_GAS = 8.314D3)
C
C  Simulation temperature, K.
      DOUBLE PRECISION T_K
      PARAMETER (T_K = 2.98D2)
C
C  FCC nickel lattice site density, sites/mm^3.
      DOUBLE PRECISION AN_L
      PARAMETER (AN_L = 8.07D19)
C
C  Clipping bounds for theta. Keeps the *DAMAGE INITIATION table
C  interpolation off the {0, 1} endpoints.
      DOUBLE PRECISION TH_MIN, TH_MAX
      PARAMETER (TH_MIN = 1.0D-12)
      PARAMETER (TH_MAX = 1.0D0 - 1.0D-12)
C
C  Locals. ABA_PARAM.INC imposes IMPLICIT REAL*8 (A-H,O-Z) and
C  INTEGER (I-N); declarations below override the implicit typing.
      DOUBLE PRECISION C_L, ANUM, ADEN, THETA, K_LM
C
C  Langmuir-McLean equilibrium constant.
      K_LM = (1.0D0 / AN_L) * EXP(DGB / (R_GAS * T_K))
C
C  Step 1. Read end-of-increment lattice concentration from NT11.
      C_L = T(1) + T(2)
C
C  Clip negative C_L from numerical noise during convergence cutbacks.
      IF (C_L .LT. 0.0D0) THEN
          FIELD(1) = 0.0D0
          RETURN
      ENDIF
C
C  Step 2. Closed-form Langmuir-McLean isotherm.
      ANUM  = C_L * K_LM
      ADEN  = 1.0D0 + ANUM
      THETA = ANUM / ADEN
C
C  Clip theta into the interior of [0, 1].
      IF (THETA .LT. TH_MIN) THETA = TH_MIN
      IF (THETA .GT. TH_MAX) THETA = TH_MAX
C
C  Step 3. Expose theta as the user-defined field for the cohesive
C  damage-initiation table.
      FIELD(1) = THETA
C
      RETURN
      END
