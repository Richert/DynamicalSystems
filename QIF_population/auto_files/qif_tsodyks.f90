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

      DOUBLE PRECISION eta,J,D,PI,R,V,E,X,alpha,tau_f,tau_d,E0

       eta  = PAR(1)
       J  = PAR(2)
       alpha = PAR(3)
       tau_f = PAR(4)
       tau_d = PAR(5)
       E0 = PAR(6)
       D  = PAR(7)
       PI = 4*ATAN(1.0D0)

       R=U(1)
       V=U(2)
       E=U(3)
       X=U(4)

       F(1) = D/PI + 2.0*R*V
       F(2) = V*V + J*R*E*X - PI*PI*R*R + eta
       F(3) = (E0 - E)/tau_f + E0*(1-E)*r
       F(4) = (1-X)/tau_d - alpha*E*X*R

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION eta,J,alpha,tau_f,tau_d,D,E0

       eta  = -10.0
       J  = 15.0*SQRT(2.0)
       alpha  = 0.0
       tau_f  = 20.0
       tau_d  = 50.0
       D  = 2.0
       E0 = 1.0

       PAR(1)=eta
       PAR(2)=J
       PAR(3)=alpha
       PAR(4)=tau_f
       PAR(5)=tau_d
       PAR(6)=E0
       PAR(7)=D

       U(1)=0.114741
       U(2)=-2.774150
       U(3)=1.0
       U(4)=1.0

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
