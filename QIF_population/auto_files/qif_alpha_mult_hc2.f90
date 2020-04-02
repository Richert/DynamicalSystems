!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   qifa : Homoclinic bifurcations in the QIF model with adaptation
!----------------------------------------------------------------------
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM,ICP(*),IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM),DFDP(NDIM,*)

      DOUBLE PRECISION I,D,J,tau,alph,taua,PI,R,V,A,B

!        F(1)=PAR(3) * ( PAR(2)*U(2) - U(1)**3 + 3.0*U(1) - PAR(1) )
!        F(2)=U(1) - 2*U(2) + U(3)
!        F(3)=U(2) - U(3)

	I = PAR(1)
	D = PAR(2)
	J = PAR(3)
	tau = PAR(4)
	alph = PAR(5)
	taua = PAR(6)
	PI = 4.0*ATAN(1.0D0)

	R = U(1)
	V = U(2)
	A = U(3)
	B = U(4)

	F(1) = D/(PI*tau*tau) + 2.0*R*V/tau
	F(2) = (V*V + I)/tau + J*R*(1.0-A) - PI*PI*R*R*tau
	F(3) = B
	F(4) = (alph*R - 2.0*B - A/taua)/taua

      IF(IJAC.EQ.0)RETURN

!        DFDU(1,1)=PAR(3)*(-3.0*U(1)**2 + 3.0)
!        DFDU(1,2)=PAR(3)*PAR(2)
!        DFDU(1,3)=0.0
!
!        DFDU(2,1)=1.0
!        DFDU(2,2)=-2.0
!        DFDU(2,3)=1.0
!
!        DFDU(3,1)=0.0
!        DFDU(3,2)=1.0
!        DFDU(3,3)=-1.0

	DFDU(1,1) = 2.0*V/tau
	DFDU(1,2) = 2.0*R/tau
	DFDU(1,3) = 0.0
	DFDU(1,4) = 0.0

	DFDU(2,1) = J*(1.0-A)-2.0*R*tau*PI**2
	DFDU(2,2) = 2.0*V/tau
	DFDU(2,3) = -J*R
	DFDU(2,4) = 0.0

	DFDU(3,1) = 0.0
	DFDU(3,2) = 0.0
	DFDU(3,3) = 0.0
	DFDU(3,4) = 1.0

	DFDU(4,1) = alph/taua
	DFDU(4,2) = 0.0
	DFDU(4,3) = -1.0/(taua*taua)
	DFDU(4,4) = -2.0/taua

      IF(IJAC.EQ.1)RETURN

!        DFDP(1,1)=- PAR(3)
!        DFDP(2,1)=0.d0
!        DFDP(3,1)=0.d0
!
!        DFDP(1,2)=PAR(3) *U(2)
!        DFDP(2,2)=0.d0
!        DFDP(3,2)=0.d0
!
!        DFDP(1,3)=PAR(2)*U(2) - U(1)**3 + 3.0*U(1) - PAR(1)
!        DFDP(2,3)=0.d0
!        DFDP(3,3)=0.d0

	DFDP(1,1) = 0.0
	DFDP(1,2) = 1.0/(PI*tau**2)
	DFDP(1,3) = 0.0
	DFDP(1,4) = -2.0*D/(PI*tau**3) -2.0*R*V/(tau**2)
	DFDP(1,5) = 0.0
	DFDP(1,6) = 0.0

	DFDP(2,1) = 1.0/tau
	DFDP(2,2) = 0.0
	DFDP(2,3) = R*(1.0-A)
	DFDP(2,4) = -(V**2+I)/(tau**2) -PI**2*R**2
	DFDP(2,5) = 0.0
	DFDP(2,6) = 0.0

	DFDP(3,1) = 0.0
	DFDP(3,2) = 0.0
	DFDP(3,3) = 0.0
	DFDP(3,4) = 0.0
	DFDP(3,5) = 0.0
	DFDP(3,6) = 0.0

	DFDP(4,1) = 0.0
	DFDP(4,2) = 0.0
	DFDP(4,3) = 0.0
	DFDP(4,4) = 0.0
	DFDP(4,5) = R/taua
	DFDP(4,6) = (2.0*B - alph*R + 2.0*A/taua)/(taua*taua)

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ----------------

! Sets parameter values for homoclinic bifurcation analysis (IPS=9).

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

! COMMON block needed if IPS=9 (homoclinic bifurcations) :
      INTEGER ITWIST,ISTART,IEQUIB,NFIXED,NPSI,NUNSTAB,NSTAB,NREV
      COMMON /BLHOM/ ITWIST,ISTART,IEQUIB,NFIXED,NPSI,NUNSTAB,NSTAB,NREV

!----------------------------------------------------------------------
! Problem parameters (only PAR(1-9) are available to the user) :

!        PAR(1) = -1.851185124d0    ! lambda
!        PAR(2) = -0.15D0         ! kappa
!        PAR(3) = 10.0d0          ! 1/epsilon_1
!
!        PAR(11)=  0.1d0          ! truncated time interval

	PAR(1) = -5.6810470245
	PAR(2) = 2.0
	PAR(3) = 15.0*SQRT(2.0)
	PAR(4) = 1.0
	PAR(5) = 0.05
	PAR(6) = 10.0
    PAR(11) = 0.1
	U(1) = 3.2006263160E-001
	U(2) = -9.9452374240E-001
	U(3) = 1.6003128705E-001
	U(4) = 0.0

!----------------------------------------------------------------------
! If IEQUIB=1 then put the equilibrium in PAR(11+i), i=1,...,NDIM :

        IF (IEQUIB.NE.0) THEN
          ! node
          !PAR(12) = 2.3709287018E-001
          !PAR(13) = -1.3425535991E+000
          !PAR(14) = 1.1854641379E-001
          !PAR(15) = 0.0

          ! saddle
          PAR(12) = 3.2006263160E-001
          PAR(13) = -9.9452374240E-001
          PAR(14) = 1.6003128705E-001
          PAR(15) = 0.0

          ! focus
          !PAR(12) = 5.8953463310E-001
          !PAR(13) = -5.3993415876E-001
          !PAR(14) = 2.9476726359E-001
          !PAR(15) = 0.0000000000E+000

        ENDIF
!----------------------------------------------------------------------
! Distance along the unstable manifold :

        IF (ISTART.EQ.3) THEN
          PAR(12+NDIM*IEQUIB)=-0.00001
        ENDIF
!----------------------------------------------------------------------

      END SUBROUTINE STPNT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT
