!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
!   qif :     QIF neural mass model with synaptic adaptation
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION tau,eta,J,D,PI,R,V,A,S,alpha,tau_a,tau_s

       eta  = PAR(1)
       J  = PAR(2)
       alpha = PAR(3)
       tau = PAR(4)
       D  = PAR(5)
       tau_a = PAR(6)
       tau_s = PAR(7)
       PI = 4*ATAN(1.0D0)

       R=U(1)
       V=U(2)
       A=U(3)
       S=U(4)

       F(1) = (D/(PI*tau) + 2.0*R*V)/tau
       F(2) = (V*V + S*tau - A - PI*PI*R*R*tau*tau + eta)/tau
       F(3) = -A/tau_a + alpha*R
       F(4) = -S/tau_s + J*R

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION eta,J,alpha,tau,D,tau_a,tau_s

       eta  = -10.0
       alpha  = 0.0
       tau  = 10.0
       tau_a = 100.0
       tau_s = 6.0
       D  = 1.0
       J  = 10.0

       PAR(1)=eta
       PAR(2)=J
       PAR(3)=alpha
       PAR(4)=tau
       PAR(5)=D
       PAR(6)=tau_a
       PAR(7)=tau_s

       U(1)=0.114741
       U(2)=-2.774150
       U(3)=0.0
       U(4)=0.0

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
