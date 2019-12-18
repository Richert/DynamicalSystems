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

      DOUBLE PRECISION V_e,V_i,I_ei,I_ie,I_ii,X_ei,X_ie,X_ii,A_ei,Z_ei,A_ii,Z_ii
      DOUBLE PRECISION eta_e,eta_i,k,r_ei,r_io,alpha,delta
      DOULBE PRECISION tau_e,tau_i,tau_ei,tau_ie,tau_ii,tau_a,s_e,s_i,max_e,max_i

       eta_e  = PAR(1)
       eta_i  = PAR(2)
       k = PAR(3)
       r_ei = PAR(4)
       r_io = PAR(5)
       alpha = PAR(6)
       kappa = PAR(7)

       tau_e = 0.006
       tau_i = 0.014
       tau_ei = 0.006
       tau_ie = 0.006
       tau_ii = 0.004
       tau_a = 0.5
       s_e = 7.0
       s_i = 0.5
       max_e = 300.0
       max_i = 400.0

       V_e=U(1)
       V_i=U(2)
       I_ei=U(3)
       X_ei=U(4)
       I_ie=U(5)
       X_ie=U(6)
       I_ii=U(7)
       X_ii=U(8)
       A_ei=U(9)
       Z_ei=U(10)
       A_ii=U(11)
       Z_ii=U(12)

       F(1) = (eta_e - I_ei*(1.0-A_i) - V_e)/tau_e
       F(2) = (eta_i + I_ie - I_ii*(1.0-A_i) - V_i - E_i)/tau_i
       F(3) = X_ei
       F(4) = k*(max_i/((1.0 + EXP(-s_i*V_i))))/tau_ei - 2.0*X_ei/tau_ei - I_ei/(tau_ei*tau_ei)
       F(5) = X_ie
       F(6) = k*r_ei*(max_e/((1.0 + EXP(-s_e*V_e))))/tau_ie - 2.0*X_ie/tau_ie - I_ie/(tau_ie*tau_ie)
       F(7) = X_ii
       F(8) = k*r_io*(max_i/((1.0 + EXP(-s_i*V_i))))/tau_ii - 2.0*X_ii/tau_ii - I_ii/(tau_ii*tau_ii)
       F(9) = Z_ei
       F(10) = alpha*I_ei/tau_a - 2.0*Z_ei/tau_a - A_ei/(tau_a*tau_a)
       F(11) = Z_ii
       F(12) = alpha*I_ii/tau_a - 2.0*Z_ii/tau_a - A_ii/(tau_a*tau_a)

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION eta_e,eta_i,k,r_ei,r_io,alpha,delta
      DOULBE PRECISION tau_e,tau_i,tau_ei,tau_ie,tau_ii,tau_a,s_e,s_i,max_e,max_i

       eta_e = 1.2
       eta_i = -2.0
       k = 4.0
       r_ei = 0.7
       r_io = 1.5
       alpha = 0.0
       delta = 0.0

       s_e = 7.0
       s_i = 0.5
       max_e = 300.0
       max_i = 400.0
       tau_e = 0.006
       tau_i = 0.014
       tau_ei = 0.006
       tau_ie = 0.006
       tau_ii = 0.004
       tau_a = 0.5

       PAR(1)=eta_e
       PAR(2)=eta_i
       PAR(3)=k
       PAR(4)=r_ei
       PAR(5)=r_io
       PAR(6)=alpha
       PAR(7)=kappa

       U(1)=-0.379683040599987
       U(2)=-3.249507445042881
       U(3)=1.579683040599820
       U(4)=0.0
       U(5)=0.330175595554070
       U(6)=0.0
       U(7)=1.579683040599358
       U(8)=0.0
       U(9)=0.0
       U(10)=0.0
       U(11)=0.0
       U(12)=0.0

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
