!----------------------------------------------------------------------------------
!----------------------------------------------------------------------------------
!   str : rate-based mean-field model of striatal microcircuit (3 populations)
!----------------------------------------------------------------------------------
!----------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION V_e,V_i
      DOUBLE PRECISION eta_e,eta_i,k,r_ei,r_io
      DOUBLE PRECISION tau_e,tau_i,s_e,s_i,max_e,max_i

       eta_e  = PAR(1)
       eta_i  = PAR(2)
       k = PAR(3)
       r_ei = PAR(4)
       r_io = PAR(5)

       tau_e = 0.006
       tau_i = 0.014
       s_e = 7.0
       s_i = 0.5
       max_e = 300.0
       max_i = 400.0

       V_e=U(1)
       V_i=U(2)

       F(1) = (eta_e - k*(max_i/((1.0 + EXP(-s_i*V_i)))) - V_e)/tau_e
       F(2) = (eta_i + k*r_ei*(max_e/((1.0 + EXP(-s_e*V_e)))) - k*r_io*(max_i/((1.0 + EXP(-s_i*V_i)))) - V_i)/tau_i

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION eta_e,eta_i,k,r_ei,r_io
      DOUBLE PRECISION tau_e,tau_i,s_e,s_i,max_e,max_i

       eta_e = 0.23
       eta_i = -3.0
       k = 0.01
       r_ei = 1.0
       r_io = 1.0

       PAR(1)=eta_e
       PAR(2)=eta_i
       PAR(3)=k
       PAR(4)=r_ei
       PAR(5)=r_io

       U(1)=-0.382669901458847
       U(2)=-3.419942028854723

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
