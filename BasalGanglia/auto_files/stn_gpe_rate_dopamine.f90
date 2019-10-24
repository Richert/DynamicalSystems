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

      DOUBLE PRECISION V_e,V_i,I_ei,I_ie,I_ii,X_ei,X_ie,X_ii,A_i,Z_i
      DOUBLE PRECISION k_e,k_i,tau_e,tau_i,tau_ei,tau_ie,tau_ii,eta_e,eta_i,s_e,s_i,max_e,max_i,alpha,tau_a,delta

       eta_e  = PAR(1)
       eta_i  = PAR(2)
       k_e = PAR(3)
       k_i = PAR(4)
       alpha = PAR(5)
       tau_e = PAR(6)
       tau_i = PAR(7)
       tau_ei = PAR(8)
       tau_ie = PAR(9)
       tau_ii = PAR(10)
       tau_a = PAR(13)
       s_e = PAR(14)
       s_i = PAR(15)
       max_e = PAR(16)
       max_i = PAR(17)
       delta = PAR(18)

       V_e=U(1)
       V_i=U(2)
       I_ei=U(3)
       X_ei=U(4)
       I_ie=U(5)
       X_ie=U(6)
       I_ii=U(7)
       X_ii=U(8)
       A_i=U(9)
       Z_i=U(10)

       F(1) = (eta_e*(1.0-0.5*delta) - I_ei - V_e)/tau_e
       F(2) = (eta_i*(1.0+2.0*delta) + I_ie - I_ii - V_i)/tau_i
       F(3) = X_ei
       F(4) = k_i*(1.0-0.5*delta)*max_i/((1.0 + EXP(-s_i*V_i))*tau_ei) - 2.0*X_ei/tau_ei - I_ei/(tau_ei*tau_ei)
       F(5) = X_ie
       F(6) = k_e*max_e/((1.0 + EXP(-s_e*V_e))*tau_ie) - 2.0*X_ie/tau_ie - I_ie/(tau_ie*tau_ie)
       F(7) = X_ii
       F(8) = k_i*max_i/((1.0 + EXP(-s_i*V_i))*tau_ii) - 2.0*X_ii/tau_ii - I_ii/(tau_ii*tau_ii)
       F(9) = Z_i
       F(10) = alpha*max_i/((1.0 + EXP(-s_i*V_i))*tau_a) - 2.0*Z_i/tau_a - A_i/(tau_a*tau_a)

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION k_e,k_i,tau_e,tau_i,tau_ei,tau_ie,tau_ii,eta_e,eta_i,s_e,s_i,max_e,max_i,alpha,tau_a

       eta_e = 1.0
       eta_i = -3.0
       k_e = 3.7
       k_i = 3.0
       s_e = 7.0
       s_i = 0.5
       max_e = 300.0
       max_i = 400.0
       alpha = 0.0
       tau_e = 0.006
       tau_i = 0.014
       tau_ei = 0.006
       tau_ie = 0.006
       tau_ii = 0.004
       tau_a = 0.1

       PAR(1)=eta_e
       PAR(2)=eta_i
       PAR(3)=k_e
       PAR(4)=k_i
       PAR(5)=alpha
       PAR(6)=tau_e
       PAR(7)=tau_i
       PAR(8)=tau_ei
       PAR(9)=tau_ie
       PAR(10)=tau_ii
       PAR(13)=tau_a
       PAR(14)=s_e
       PAR(15)=s_i
       PAR(16)=max_e
       PAR(17)=max_i

       U(1)=-0.3784634
       U(2)=-3.3884044
       U(3)=1.3784544
       U(4)=0.0
       U(5)=0.35658336
       U(6)=0.0
       U(7)=0.74509746
       U(8)=0.0

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
