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

      DOUBLE PRECISION eta,delta,J,D,PI,r,v,a,x,alpha,tau_a,tau_s

       eta  = PAR(1)
       J  = PAR(2)
       alpha = PAR(3)
       tau_a = PAR(4)
       tau_s = PAR(5)
       delta  = PAR(6)
       D = PAR(7)
       PI = 4*ATAN(1.0D0)

       r=U(1)
       v=U(2)
       a=U(3)
       x=U(4)

       F(1) = (delta + sqrt(D)/PI)/PI + 2.0*r*v
       F(2) = v*v + eta - a + x - PI*PI*r*r
       F(3) = alpha*r - a/tau_a
       F(4) = J*r - x/tau_s

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION eta,delta,J,alpha,tau_a,D,tau_s

       eta  = -10.0
       alpha  = 0.0
       tau_a  = 10.0
       tau_s = 0.8
       D  = 0.0
       delta = 2.0
       J  = 15.0*SQRT(delta)

       PAR(1)=eta
       PAR(2)=J
       PAR(3)=alpha
       PAR(4)=tau_a
       PAR(5)=tau_s
       PAR(6)=delta
       PAR(7)=D

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
