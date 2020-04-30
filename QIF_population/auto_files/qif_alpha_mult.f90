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

      DOUBLE PRECISION eta,J,D,PI,R,V,E,A,alpha,tau

       eta  = PAR(1) 
       J  = PAR(2) 
       alpha = PAR(3) 
       tau = PAR(4) 
       D  = PAR(5)
       PI = 4*ATAN(1.0D0)

       R=U(1)
       V=U(2)
       E=U(3)
       A=U(4)  

       F(1) = D/PI + 2.0*R*V
       F(2) = V*V + J*(1.0-E)*R - PI*PI*R*R + eta
       F(3) = A   
       F(4) = alpha*R/tau - 2.0*A/tau - E/(tau*tau)  

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)  
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION eta,J,alpha,tau,D

       eta  = -10.0
       J  = 15.0*SQRT(2.0)
       alpha  = 0.0
       tau  = 10.0
       D  = 2.0

       PAR(1)=eta
       PAR(2)=J
       PAR(3)=alpha
       PAR(4)=tau
       PAR(5)=D
       
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
