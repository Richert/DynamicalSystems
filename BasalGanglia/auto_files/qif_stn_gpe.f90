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

      DOUBLE PRECISION V_e,V_i,R_e,R_i,I_ee,I_ei,I_ie,I_ii,X_ee,X_ei,X_ie,X_ii,I_a,X_a
      DOUBLE PRECISION R_e1,R_e2,R_e3,R_e4,R_e5,R_e6,R_e7,R_e8,R_e9,R_e10,R_e11,R_e12,R_e13,R_e14,R_e15,R_e16
      DOUBLE PRECISION R_i1,R_i2,R_i3,R_i4,R_i5,R_i6,R_i7,R_i8,R_i9,R_i10,R_i11,R_i12,R_i13,R_i14,R_i15,R_i16
      DOUBLE PRECISION eta_e,eta_i,eta_str,eta_tha,k_ee,k_ei,k_ie,k_ii,alpha,delta_e,delta_i,pi,d_e,d_i
      DOUBLE PRECISION tau_e,tau_i,tau_ee_r,tau_ee_d,tau_ei_r,tau_ei_d,tau_ie_r,tau_ie_d,tau_ii_r,tau_ii_d,tau_a

       eta_e  = PAR(1)
       eta_i  = PAR(2)
       eta_str = PAR(3)
       eta_tha = PAR(4)
       k_ee = PAR(5)
       k_ei = PAR(6)
       k_ie = PAR(7)
       k_ii = PAR(8)
       alpha = PAR(9)
       tau_e = 6
       tau_i = 14
       tau_ee_r = 0.8
       tau_ee_d = 3.7
       tau_ei_r = 0.8
       tau_ei_d = 10.0
       tau_ie_r = 0.8
       tau_ie_d = 3.7
       tau_ii_r = 0.5
       tau_ii_d = 5.0
       tau_a = 200.0
       delta_e = 2.0
       delta_i = 2.0
       d_e = 4.0
       d_i = 4.0
       PI = 4*ATAN(1.0D0)

       R_e=U(1)
       V_e=U(2)
       R_i=U(3)
       V_i=U(4)
       I_ee=U(5)
       X_ee=U(6)
       I_ei=U(7)
       X_ei=U(8)
       I_ie=U(9)
       X_ie=U(10)
       I_ii=U(11)
       X_ii=U(12)
       I_a=U(13)
       X_a=U(14)
       R_e1=U(15)
       R_e2=U(16)
       R_e3=U(17)
       R_e4=U(18)
       R_e5=U(19)
       R_e6=U(20)
       R_e7=U(21)
       R_e8=U(22)
       R_e9=U(23)
       R_e10=U(24)
       R_e11=U(25)
       R_e12=U(26)
       R_e13=U(27)
       R_e14=U(28)
       R_e15=U(29)
       R_e16=U(30)
       R_i1=U(31)
       R_i2=U(32)
       R_i3=U(33)
       R_i4=U(34)
       R_i5=U(35)
       R_i6=U(36)
       R_i7=U(37)
       R_i8=U(38)
       R_i9=U(39)
       R_i10=U(40)
       R_i11=U(41)
       R_i12=U(42)
       R_i13=U(43)
       R_i14=U(44)
       R_i15=U(45)
       R_i16=U(46)

       F(1) = delta_e/(PI*tau_e*tau_e) + 2.*R_e*V_e/tau_e
       F(2) = (V_e*V_e + eta_e)/tau_e + I_ee - I_ei - tau_e*PI*PI*R_e*R_e
       F(3) = delta_i/(PI*tau_i*tau_i) + 2.*R_i*V_i/tau_i
       F(4) = (V_i*V_i + eta_i + eta_str + eta_tha)/tau_i + I_ie - I_ii - I_a - tau_i*PI*PI*R_i*R_i
       F(5) = X_ee
       F(6) = (k_ee*R_e - X_ee*(tau_ee_r+tau_ee_d) - I_ee)/(tau_ee_r*tau_ee_d)
       F(7) = X_ei
       F(8) = (k_ei*R_i16 - X_ei*(tau_ei_r+tau_ei_d) - I_ei)/(tau_ei_r*tau_ei_d)
       F(9) = X_ie
       F(10) = (k_ie*R_e16 - X_ie*(tau_ie_r+tau_ie_d) - I_ie)/(tau_ie_r*tau_ie_d)
       F(11) = X_ii
       F(12) = (k_ii*R_i - X_ii*(tau_ii_r+tau_ii_d) - I_ii)/(tau_ii_r*tau_ii_d)
       F(13) = X_a
       F(14) = (alpha*R_i - 2.0*X_a - I_a/tau_a)/tau_a
       F(15) = (R_e - R_e1)*16.0/d_e
       F(16) = (R_e1 - R_e2)*16.0/d_e
       F(17) = (R_e2 - R_e3)*16.0/d_e
       F(18) = (R_e3 - R_e4)*16.0/d_e
       F(19) = (R_e4 - R_e5)*16.0/d_e
       F(20) = (R_e5 - R_e6)*16.0/d_e
       F(21) = (R_e6 - R_e7)*16.0/d_e
       F(22) = (R_e7 - R_e8)*16.0/d_e
       F(23) = (R_e8 - R_e9)*16.0/d_e
       F(24) = (R_e9 - R_e10)*16.0/d_e
       F(25) = (R_e10 - R_e11)*16.0/d_e
       F(26) = (R_e11 - R_e12)*16.0/d_e
       F(27) = (R_e12 - R_e13)*16.0/d_e
       F(28) = (R_e13 - R_e14)*16.0/d_e
       F(29) = (R_e14 - R_e15)*16.0/d_e
       F(30) = (R_e15 - R_e16)*16.0/d_e
       F(31) = (R_i - R_i1)*16.0/d_i
       F(32) = (R_i1 - R_i2)*16.0/d_i
       F(33) = (R_i2 - R_i3)*16.0/d_i
       F(34) = (R_i3 - R_i4)*16.0/d_i
       F(35) = (R_i4 - R_i5)*16.0/d_i
       F(36) = (R_i5 - R_i6)*16.0/d_i
       F(37) = (R_i6 - R_i7)*16.0/d_i
       F(38) = (R_i7 - R_i8)*16.0/d_i
       F(39) = (R_i8 - R_i9)*16.0/d_i
       F(40) = (R_i9 - R_i10)*16.0/d_i
       F(41) = (R_i10 - R_i11)*16.0/d_i
       F(42) = (R_i11 - R_i12)*16.0/d_i
       F(43) = (R_i12 - R_i13)*16.0/d_i
       F(44) = (R_i13 - R_i14)*16.0/d_i
       F(45) = (R_i14 - R_i15)*16.0/d_i
       F(46) = (R_i15 - R_i16)*16.0/d_i

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)  
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION eta_e,eta_i,eta_str,eta_tha,k,k_ee,k_ei,k_ie,k_ii,alpha,delta_e,delta_i,pi,d_e,d_i
      DOUBLE PRECISION tau_e,tau_i,tau_ee_r,tau_ee_d,tau_ei_r,tau_ei_d,tau_ie_r,tau_ie_d,tau_ii_r, tau_ii_d,tau_a

       eta_e = -10.0
       eta_i = -5.0
       eta_str = 0.0
       eta_tha = 0.0
       k_ee = 1.0
       k_ei = 10.0
       k_ie = 10.0
       k_ii = 5.0
       alpha = 0.0

       PAR(1)=eta_e
       PAR(2)=eta_i
       PAR(3)=eta_str
       PAR(4)=eta_tha
       PAR(5)=k_ee
       PAR(6)=k_ei
       PAR(7)=k_ie
       PAR(8)=k_ii
       PAR(9)=alpha

       U(1)=16.222179
       U(2)=-3.270351
       U(3)=11.653199
       U(4)=-1.951145
       U(5)=16.221917
       U(6)=0.0
       U(7)=11.652416
       U(8)=0.0
       U(9)=16.221917
       U(10)=0.0
       U(11)=11.652416
       U(12)=0.0
       U(13)=11.652416
       U(14)=0.0
       U(15)=16.221917
       U(16)=16.221917
       U(17)=16.221917
       U(18)=16.221917
       U(19)=16.221917
       U(20)=16.221917
       U(21)=16.221917
       U(22)=16.221917
       U(23)=16.221917
       U(24)=16.221917
       U(25)=16.221917
       U(26)=16.221917
       U(27)=16.221917
       U(28)=16.221917
       U(29)=16.221917
       U(30)=16.221917
       U(31)=11.652416
       U(32)=11.652416
       U(33)=11.652416
       U(34)=11.652416
       U(35)=11.652416
       U(36)=11.652416
       U(37)=11.652416
       U(38)=11.652416
       U(39)=11.652416
       U(40)=11.652416
       U(41)=11.652416
       U(42)=11.652416
       U(43)=11.652416
       U(44)=11.652416
       U(45)=11.652416
       U(46)=11.652416


      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
