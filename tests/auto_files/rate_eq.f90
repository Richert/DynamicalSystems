!----------------------------------------------------------------------------------
!----------------------------------------------------------------------------------
!   str : ODE-approximation of delayed rate equation
!----------------------------------------------------------------------------------
!----------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION r, r_1, r_2, r_3, r_4, r_5
      DOUBLE PRECISION k

       k  = PAR(1)
       k = 1.0

       r = U(1)
       r_1 = U(2)
       r_2 = U(3)
       r_3 = U(4)
       r_4 = U(5)
       r_5 = U(6)

       F(1) = -k*r_5
       F(2) = 5*(r-r_1)
       F(3) = 5*(r_1-r_2)
       F(4) = 5*(r_2-r_3)
       F(5) = 5*(r_3-r_4)
       F(6) = 5*(r_4-r_5)

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION r,r_1,r_2,r_3,r_4,r_5
      DOUBLE PRECISION k

       k = 1.0
       PAR(1)=k
       U(1)=0.0

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS