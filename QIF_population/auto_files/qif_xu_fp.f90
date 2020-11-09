!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
!   qif :      Periodically forced QIF neural mass model
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP) 
!     ---------- ---- 

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      INTEGER :: n, M
      DOUBLE PRECISION :: R((NDIM)/3),V((NDIM)/3),XA((NDIM)/3),UA((NDIM)/3)
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION I,J,A,W,D,taux,tauu,alph,U0,PI,RM

       I    = PAR(1) 
       J    = PAR(2) 
       A    = PAR(3) 
       W    = PAR(4) 
       D    = PAR(5)
       taux = PAR(6)
       tauu = PAR(7)
       alph = PAR(8)
       U0   = PAR(9)
       PI   = 4*ATAN(1.0D0)
       M    = (NDIM)/3

       do n=1,M
         R(n) = U(n)
         V(n) = U(n+M)
         XA(n) = U(n+2*M)
       end do 
       RM = 0
       do n=1,M
         RM = RM + XA(n)*R(n)/M
       end do
       
       do n=1,M
         F(n) = D*(TAN(0.5*PI*(2*n-M-0.5)/M)-TAN(0.5*PI*(2*n-M-1.5)/M)) + 2.0*R(n)*V(n)
         F(n+M) = V(n)*V(n) + J*RM - PI*PI*R(n)*R(n) + I+D*TAN(0.5*PI*(2*n-M-1)/M)
         F(n+2*M) = (1-XA(n))/taux - alph*XA(n)*R(n)
       end do

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      INTEGER :: n,M
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION I,J,A,W,D,taux,tauu,alph,U0,TPI

       I  = -20.0
       J  = 80
       A  = 0.0
       W  = 10
       !W = 5.0*8.0*ATAN(1.0D0)/50.0
       D  = 1.0
       taux = 10.0
       tauu = 40.0
       alph = 1.0
       U0 = 1.0

       PAR(1)=I
       PAR(2)=J
       PAR(3)=A
       PAR(4)=W
       PAR(5)=D
       PAR(6)=taux
       PAR(7)=tauu
       PAR(8)=alph
       PAR(9)=U0
       TPI=8.0*ATAN(1.0D0)
       M  =(NDIM)/3

       !U(1)=1.457484
       !U(2)=-0.218397
       !U(1)=0.67026
       !U(2)=-0.4749
       !U(1)=0.114741
       !U(2)=-2.774150
       do n=1,M
         U(n) = 0
         U(n+M) = -SQRT(-(I-D+(n-0.5)*(2.0*D/M)))
         U(n+2*M) = 1;
       end do

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS