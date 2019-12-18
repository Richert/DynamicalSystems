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

      DOUBLE PRECISION Ve,Vi,Re,Ri,Iee,Iei,Iie,Iii,Xee,Xei,Xie,Xii,Iia,Xia
      DOUBLE PRECISION Re_1,Re_2,Re_3,Re_4,Re_5,Re_6,Re_7,Re_8,Re_9,Re_10,Re_11,Re_12,Re_13,Re_14,Re_15,Re_16
      DOUBLE PRECISION Ri_1,Ri_2,Ri_3,Ri_4,Ri_5,Ri_6,Ri_7,Ri_8,Ri_9,Ri_10,Ri_11,Ri_12,Ri_13,Ri_14,Ri_15,Ri_16
      DOUBLE PRECISION eta_e,eta_i,eta_str,eta_tha,k,k_ee,k_ei,k_ie,k_ii,alpha,delta_e,delta_i,pi,delay
      DOUBLE PRECISION tau_e,tau_i,tau_ee_r,tau_ee_d,tau_ei_r,tau_ei_d,tau_ie_r,tau_ie_d,tau_ii_r, tau_ii_d,tau_a

       eta_e  = PAR(1)
       eta_i  = PAR(2)
       eta_str = PAR(3)
       eta_tha = PAR(4)
       k_ee = PAR(5)
       k_ei = PAR(6)
       k_ie = PAR(7)
       k_ii = PAR(8)
       alpha = PAR(9)
       tau_e = PAR(10)
       tau_i = PAR(11)
       tau_ee_r = PAR(12)
       tau_ee_d = PAR(13)
       tau_ei_r = PAR(14)
       tau_ei_d = PAR(15)
       tau_ie_r = PAR(16)
       tau_ie_d = PAR(17)
       tau_ii_r = PAR(18)
       tau_ii_d = PAR(19)
       tau_a = PAR(20)
       delta_e = PAR(21)
       delta_i = PAR(22)
       delay = PAR(23)
       k = 1000.0
       pi = 4*ATAN(1.0D0)
       beta = 16.0/delay

       Re=U(1)
       Ve=U(2)
       Ri=U(3)
       Vi=U(4)
       Iee=U(5)
       Xee=U(6)
       Iei=U(7)
       Xei=U(8)
       Iie=U(9)
       Xie=U(10)
       Iii=U(11)
       Xii=U(12)
       Iia=U(13)
       Xia=U(14)

       F(1) = (delta_e/(pi*tau_e) + 2.0*Re*Ve)/tau_e
       F(2) = (Ve*Ve + eta_e)/tau_e + Iee - Iei - pi*pi*Re*Re*tau_e
       F(3) = (delta_i/(pi*tau_i) + 2.0*Ri*Vi)/tau_i
       F(4) = (Vi*Vi + eta_i + eta_str + eta_tha)/tau_i + Iie - Iii - Iia - pi*pi*Ri*Ri*tau_i

       F(5) = ((tau_ee_d*tau_ee_r)**(tau_ee_r/(tau_ee_d-tau_ee_r))*Xee - Iee)/tau_ee_r
       F(6) = k_ie*k*Re - X_ee/tau_ee_d
       F(7) = ((tau_ei_d*tau_ei_r)**(tau_ei_r/(tau_ei_d-tau_ei_r))*Xei - Iei)/tau_ei_r
       F(8) = k_ei*k*Ri_16 - X_ei/tau_ei_d
       F(9) = ((tau_ie_d*tau_ie_r)**(tau_ie_r/(tau_ie_d-tau_ie_r))*Xie - Iie)/tau_ie_r
       F(10) = k_ie*k*Re_16 - X_ie/tau_ie_d
       F(11) = ((tau_ii_d*tau_ii_r)**(tau_ii_r/(tau_ii_d-tau_ii_r))*Xii - Iii)/tau_ii_r
       F(12) = k_ii*k*Ri - X_ii/tau_ii_d

       F(13) = Xia
       F(14) = (alpha*Ri - 2.0*Xia - Iia/tau_a)/tau_a

       F(15) = (Re-Re_1)*beta
       F(16) = (Re_1-Re_2)*beta
       F(17) = (Re_2-Re_3)*beta
       F(18) = (Re_3-Re_4)*beta
       F(19) = (Re_4-Re_5)*beta
       F(20) = (Re_5-Re_6)*beta
       F(21) = (Re_6-Re_7)*beta
       F(22) = (Re_7-Re_8)*beta
       F(23) = (Re_8-Re_9)*beta
       F(24) = (Re_9-Re_10)*beta
       F(25) = (Re_10-Re_11)*beta
       F(26) = (Re_11-Re_12)*beta
       F(27) = (Re_12-Re_13)*beta
       F(28) = (Re_13-Re_14)*beta
       F(29) = (Re_14-Re_15)*beta
       F(30) = (Re_15-Re_16)*beta

       F(31) = (Ri-Ri_1)*beta
       F(32) = (Ri_1-Ri_2)*beta
       F(33) = (Ri_2-Ri_3)*beta
       F(34) = (Ri_3-Ri_4)*beta
       F(35) = (Ri_4-Ri_5)*beta
       F(36) = (Ri_5-Ri_6)*beta
       F(37) = (Ri_6-Ri_7)*beta
       F(38) = (Rie_7-Ri_8)*beta
       F(39) = (Ri_8-Ri_9)*beta
       F(40) = (Ri_9-Ri_10)*beta
       F(41) = (Ri_10-Ri_11)*beta
       F(42) = (Ri_11-Ri_12)*beta
       F(43) = (Ri_12-Ri_13)*beta
       F(44) = (Ri_13-Ri_14)*beta
       F(45) = (Ri_14-Ri_15)*beta
       F(46) = (Ri_15-Ri_16)*beta

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)  
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION eta_e,eta_i,eta_str,eta_tha,k,k_ee,k_ei,k_ie,k_ii,alpha,delta_e,delta_i,pi,delay
      DOUBLE PRECISION tau_e,tau_i,tau_ee_r,tau_ee_d,tau_ei_r,tau_ei_d,tau_ie_r,tau_ie_d,tau_ii_r, tau_ii_d,tau_a

       eta_e = -4.39
       eta_i = 0.31
       eta_str = -1.0
       eta_tha = 10.0
       k_ee = 14.18
       k_ei = 11.36
       k_ie = 47.71
       k_ii = 7.61
       alpha = 6.9
       tau_e = 0.006
       tau_i = 0.014
       tau_ee_r = 0.002
       tau_ee_d = 0.002
       tau_ei_r = 0.002
       tau_ei_d = 0.002
       tau_ie_r = 0.002
       tau_ie_d = 0.002
       tau_ii_r = 0.002
       tau_ii_d = 0.002
       tau_a = 0.2
       delta_e = 2.0
       delta_i = 2.0
       delay = 0.004

       PAR(1)=eta_e
       PAR(2)=eta_i
       PAR(3)=eta_str
       PAR(4)=eta_tha
       PAR(5)=k_ee
       PAR(6)=k_ei
       PAR(7)=k_ie
       PAR(8)=k_ii
       PAR(9)=alpha
       PAR(10)=tau_e
       PAR(11)=tau_i
       PAR(12)=tau_ee_r
       PAR(13)=tau_ee_d
       PAR(14)=tau_ei_r
       PAR(15)=tau_ei_d
       PAR(16)=tau_ie_r
       PAR(17)=tau_ie_d
       PAR(18)=tau_ii_r
       PAR(19)=tau_ii_d
       PAR(20)=tau_a
       PAR(21)=delta_e
       PAR(22)=delta_i
       PAR(23)=delay

       U(1)=20.165978
       U(2)=-2.630749
       U(3)=60.183210
       U(4)=-0.377783
       U(5)=20.165897
       U(6)=60.183252
       U(7)=35.989579
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
