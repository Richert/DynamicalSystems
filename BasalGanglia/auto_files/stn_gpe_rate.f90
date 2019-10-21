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

      DOUBLE PRECISION V_e,V_i,I_ei,I_ie,I_ii,X_ei,X_ie,X_ii
      DOUBLE PRECISION k_ei,k_ie,k_ii,tau_e,tau_i,tau_ei,tau_ie,tau_ii,eta_e,eta_i,s_e,s_i,max_e,max_i

       eta_e  = PAR(1)
       eta_i  = PAR(2)
       k_ei = PAR(3)
       k_ie = PAR(4)
       k_ii = PAR(5)
       tau_e = PAR(6)
       tau_i = PAR(7)
       tau_ei = PAR(8)
       tau_ie = PAR(9)
       tau_ii = PAR(10)
       s_e = PAR(13)
       s_i = PAR(14)
       max_e = PAR(15)
       max_i = PAR(16)

       V_e=U(1)
       V_i=U(2)
       I_ei=U(3)
       X_ei=U(4)
       I_ie=U(5)
       X_ie=U(6)
       I_ii=U(7)
       X_ii=U(8)

       F(1) = (eta_e - I_ei - V_e)/tau_e
       F(2) = (eta_i + I_ie - I_ii - V_i)/tau_i
       F(3) = X_ei
       F(4) = k_ei*max_i/((1.0 + EXP(-s_i*V_i))*tau_ei) - 2.0*X_ei/tau_ei - I_ei/(tau_ei*tau_ei)
       F(5) = X_ie
       F(6) = k_ie*max_e/((1.0 + EXP(-s_e*V_e))*tau_ie) - 2.0*X_ie/tau_ie - I_ie/(tau_ie*tau_ie)
       F(7) = X_ii
       F(8) = k_ii*max_i/((1.0 + EXP(-s_i*V_i))*tau_ii) - 2.0*X_ii/tau_ii - I_ii/(tau_ii*tau_ii)

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION k_ei,k_ie,k_ii,tau_e,tau_i,tau_ei,tau_ie,tau_ii,eta_e,eta_i,s_e,s_i,max_e,max_i

       eta_e = 1.0
       eta_i = -1.0
       k_ei = 3.0
       k_ie = 1.0
       k_ii = 1.0
       s_e = 1.0
       s_i = 1.0
       max_e = 200.0
       max_i = 300.0
       tau_e = 0.005
       tau_i = 0.005
       tau_ei = 0.020
       tau_ie = 0.010
       tau_ii = 0.010

       PAR(1)=eta_e
       PAR(2)=eta_i
       PAR(3)=k_ei
       PAR(4)=k_ie
       PAR(5)=k_ii
       PAR(6)=tau_e
       PAR(7)=tau_i
       PAR(8)=tau_ei
       PAR(9)=tau_ie
       PAR(10)=tau_ii
       PAR(13)=s_e
       PAR(14)=s_i
       PAR(15)=max_e
       PAR(16)=max_i

       U(1)=-2.484569809129215
       U(2)=-1.426867638831595
       U(3)=3.484569809128105
       U(4)=0.0
       U(5)=0.153893996023352
       U(6)=0.0
       U(7)=0.580761634854392
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
