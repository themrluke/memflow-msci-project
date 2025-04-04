      SUBROUTINE M0_SMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=16)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)

C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
C     
C     GLOBAL VARIABLES
C     
      INTEGER USERHEL
      COMMON/M0_HELUSERCHOICE/USERHEL
C     ----------
C     BEGIN CODE
C     ----------
      USERHEL=HEL
      CALL M0_SMATRIX(P,ANS)
      USERHEL=-1

      END

      SUBROUTINE M0_SMATRIX(P,ANS)
C     
C     Generated by MadGraph5_aMC@NLO v. 3.4.1, 2022-09-01
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     
C     MadGraph5_aMC@NLO StandAlone Version
C     
C     Returns amplitude squared summed/avg over colors
C     and helicities
C     for the point in phase space P(0:3,NEXTERNAL)
C     
C     Process: g g > h t t~ WEIGHTED<=4 @1
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=16)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=4)
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
C     
C     LOCAL VARIABLES 
C     
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
C     put in common block to expose this variable to python interface
      COMMON/M0_PROCESS_NHEL/NHEL
      REAL*8 T
      REAL*8 M0_MATRIX
      INTEGER IHEL,IDEN, I, J
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/2,2/
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./

C     
C     GLOBAL VARIABLES
C     
      INTEGER USERHEL
      COMMON/M0_HELUSERCHOICE/USERHEL
      DATA USERHEL/-1/
      LOGICAL HELRESET
      COMMON/M0_HELRESET/HELRESET
      DATA HELRESET/.TRUE./

      DATA (NHEL(I,   1),I=1,5) /-1,-1, 0,-1, 1/
      DATA (NHEL(I,   2),I=1,5) /-1,-1, 0,-1,-1/
      DATA (NHEL(I,   3),I=1,5) /-1,-1, 0, 1, 1/
      DATA (NHEL(I,   4),I=1,5) /-1,-1, 0, 1,-1/
      DATA (NHEL(I,   5),I=1,5) /-1, 1, 0,-1, 1/
      DATA (NHEL(I,   6),I=1,5) /-1, 1, 0,-1,-1/
      DATA (NHEL(I,   7),I=1,5) /-1, 1, 0, 1, 1/
      DATA (NHEL(I,   8),I=1,5) /-1, 1, 0, 1,-1/
      DATA (NHEL(I,   9),I=1,5) / 1,-1, 0,-1, 1/
      DATA (NHEL(I,  10),I=1,5) / 1,-1, 0,-1,-1/
      DATA (NHEL(I,  11),I=1,5) / 1,-1, 0, 1, 1/
      DATA (NHEL(I,  12),I=1,5) / 1,-1, 0, 1,-1/
      DATA (NHEL(I,  13),I=1,5) / 1, 1, 0,-1, 1/
      DATA (NHEL(I,  14),I=1,5) / 1, 1, 0,-1,-1/
      DATA (NHEL(I,  15),I=1,5) / 1, 1, 0, 1, 1/
      DATA (NHEL(I,  16),I=1,5) / 1, 1, 0, 1,-1/
      DATA IDEN/256/

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/M0_BORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/

C     
C     FUNCTIONS
C     
      LOGICAL M0_IS_BORN_HEL_SELECTED

C     ----------
C     Check if helreset mode is on
C     ---------
      IF (HELRESET) THEN
        NTRY = 0
        DO I=1,NCOMB
          GOODHEL(I) = .FALSE.
        ENDDO
        HELRESET = .FALSE.
      ENDIF

C     ----------
C     BEGIN CODE
C     ----------
      IF(USERHEL.EQ.-1) NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
        JC(IHEL) = +1
      ENDDO
C     When spin-2 particles are involved, the Helicity filtering is
C      dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points
C      have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero.
C      But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS
C      helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the
C      helicity sum is).
C     For this reason, we simply remove the filterin when there is
C      only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        DO IHEL=1,NCOMB
          GOODHEL(IHEL)=.TRUE.
        ENDDO
      ENDIF
      ANS = 0D0
      DO IHEL=1,NCOMB
        IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
          IF (GOODHEL(IHEL) .OR. NTRY .LT. 20.OR.USERHEL.NE.-1) THEN
            IF(NTRY.GE.2.AND.POLARIZATIONS(0,0).NE.
     $       -1.AND.(.NOT.M0_IS_BORN_HEL_SELECTED(IHEL))) THEN
              CYCLE
            ENDIF
            T=M0_MATRIX(P ,NHEL(1,IHEL),JC(1))
            IF(POLARIZATIONS(0,0).EQ.
     $       -1.OR.M0_IS_BORN_HEL_SELECTED(IHEL)) THEN
              ANS=ANS+T
            ENDIF
            IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
              GOODHEL(IHEL)=.TRUE.
            ENDIF
          ENDIF
        ENDIF
      ENDDO
      ANS=ANS/DBLE(IDEN)
      IF(USERHEL.NE.-1) THEN
        ANS=ANS*HELAVGFACTOR
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).NE.-1) THEN
            ANS=ANS*BEAMS_HELAVGFACTOR(J)
            ANS=ANS/POLARIZATIONS(J,0)
          ENDIF
        ENDDO
      ENDIF
      END


      REAL*8 FUNCTION M0_MATRIX(P,NHEL,IC)
C     
C     Generated by MadGraph5_aMC@NLO v. 3.4.1, 2022-09-01
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     
C     Returns amplitude squared -- no average over initial
C      state/symmetry factor
C     for the point with external lines W(0:6,NEXTERNAL)
C     
C     Process: g g > h t t~ WEIGHTED<=4 @1
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=8)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=9, NCOLOR=2)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), TMP_JAMP(0)
      COMPLEX*16 W(20,NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0D0, 0D0), (1D0, 0D0)/
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'coupl.inc'

C     
C     COLOR DATA
C     
      DATA (CF(I,  1),I=  1,  2) /5.333333333333333D+00,
     $ -6.666666666666666D-01/
C     1 T(1,2,4,5)
      DATA (CF(I,  2),I=  1,  2) /-6.666666666666666D-01
     $ ,5.333333333333333D+00/
C     1 T(2,1,4,5)
C     ----------
C     BEGIN CODE
C     ----------
      CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
      CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
      CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
      CALL OXXXXX(P(0,4),MDL_MT,NHEL(4),+1*IC(4),W(1,4))
      CALL IXXXXX(P(0,5),MDL_MT,NHEL(5),-1*IC(5),W(1,5))
      CALL VVV1P0_1(W(1,1),W(1,2),GC_10,ZERO,ZERO,W(1,6))
      CALL FFS4_1(W(1,4),W(1,3),GC_94,MDL_MT,MDL_WT,W(1,7))
C     Amplitude(s) for diagram number 1
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),GC_11,AMP(1))
      CALL FFS4_2(W(1,5),W(1,3),GC_94,MDL_MT,MDL_WT,W(1,8))
C     Amplitude(s) for diagram number 2
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),GC_11,AMP(2))
      CALL FFV1_1(W(1,4),W(1,1),GC_11,MDL_MT,MDL_WT,W(1,6))
      CALL FFV1_2(W(1,5),W(1,2),GC_11,MDL_MT,MDL_WT,W(1,9))
C     Amplitude(s) for diagram number 3
      CALL FFS4_0(W(1,9),W(1,6),W(1,3),GC_94,AMP(3))
C     Amplitude(s) for diagram number 4
      CALL FFV1_0(W(1,8),W(1,6),W(1,2),GC_11,AMP(4))
      CALL FFV1_2(W(1,5),W(1,1),GC_11,MDL_MT,MDL_WT,W(1,6))
      CALL FFV1_1(W(1,4),W(1,2),GC_11,MDL_MT,MDL_WT,W(1,5))
C     Amplitude(s) for diagram number 5
      CALL FFS4_0(W(1,6),W(1,5),W(1,3),GC_94,AMP(5))
C     Amplitude(s) for diagram number 6
      CALL FFV1_0(W(1,6),W(1,7),W(1,2),GC_11,AMP(6))
C     Amplitude(s) for diagram number 7
      CALL FFV1_0(W(1,8),W(1,5),W(1,1),GC_11,AMP(7))
C     Amplitude(s) for diagram number 8
      CALL FFV1_0(W(1,9),W(1,7),W(1,1),GC_11,AMP(8))
      JAMP(1)=+IMAG1*AMP(1)+IMAG1*AMP(2)-AMP(3)-AMP(4)-AMP(8)
      JAMP(2)=-IMAG1*AMP(1)-IMAG1*AMP(2)-AMP(5)-AMP(6)-AMP(7)

      M0_MATRIX = 0.D0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
        ENDDO
        M0_MATRIX = M0_MATRIX+ZTEMP*DCONJG(JAMP(I))
      ENDDO

      END

      SUBROUTINE M0_GET_VALUE(P, ALPHAS, NHEL ,ANS)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS
      REAL*8 PI
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
C     ROUTINE FOR F2PY to read the benchmark point.    
C     the include file with the values of the parameters and masses 
      INCLUDE 'coupl.inc'

      PI = 3.141592653589793D0
      G = 2* DSQRT(ALPHAS*PI)
      CALL UPDATE_AS_PARAM()
      IF (NHEL.NE.0) THEN
        CALL M0_SMATRIXHEL(P, NHEL, ANS)
      ELSE
        CALL M0_SMATRIX(P, ANS)
      ENDIF
      RETURN
      END

      SUBROUTINE M0_INITIALISEMODEL(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.    
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters    
      RETURN
      END

      LOGICAL FUNCTION M0_IS_BORN_HEL_SELECTED(HELID)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NCOMB
      PARAMETER (NCOMB=16)
C     
C     ARGUMENTS
C     
      INTEGER HELID
C     
C     LOCALS
C     
      INTEGER I,J
      LOGICAL FOUNDIT
C     
C     GLOBALS
C     
      INTEGER HELC(NEXTERNAL,NCOMB)
      COMMON/M0_PROCESS_NHEL/HELC

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/M0_BORN_BEAM_POL/POLARIZATIONS
C     ----------
C     BEGIN CODE
C     ----------

      M0_IS_BORN_HEL_SELECTED = .TRUE.
      IF (POLARIZATIONS(0,0).EQ.-1) THEN
        RETURN
      ENDIF

      DO I=1,NEXTERNAL
        IF (POLARIZATIONS(I,0).EQ.-1) THEN
          CYCLE
        ENDIF
        FOUNDIT = .FALSE.
        DO J=1,POLARIZATIONS(I,0)
          IF (HELC(I,HELID).EQ.POLARIZATIONS(I,J)) THEN
            FOUNDIT = .TRUE.
            EXIT
          ENDIF
        ENDDO
        IF(.NOT.FOUNDIT) THEN
          M0_IS_BORN_HEL_SELECTED = .FALSE.
          RETURN
        ENDIF
      ENDDO

      RETURN
      END

