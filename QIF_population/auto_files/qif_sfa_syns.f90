!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
!   qif :     QIF neural mass model with synaptic adaptation
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

      INCLUDE 'qif_module.f90'

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----
      USE QIF
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION eta,delta,J,PI,r,v,a,x,alpha,tau_a,tau_s,t

       eta  = PAR(1)
       J  = PAR(2)
       alpha = PAR(3)
       tau_a = PAR(4)
       tau_s = PAR(5)
       delta  = PAR(6)
       PI = 4*ATAN(1.0D0)
       t = 0.0

       call FRHS(t,U,F,eta,delta,J,alpha,tau_a,tau_s)

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION eta,delta,J,alpha,tau_a,tau_s

       eta  = -0.5
       alpha  = -0.01
       tau_a  = 10.0
       tau_s = 1.0
       delta = 0.1
       J  = 0.0

       PAR(1)=eta
       PAR(2)=J
       PAR(3)=alpha
       PAR(4)=tau_a
       PAR(5)=tau_s
       PAR(6)=delta

       !U(1)=1.457484
       !U(2)=-0.218397
       !U(1)=0.67026
       !U(2)=-0.4749
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
