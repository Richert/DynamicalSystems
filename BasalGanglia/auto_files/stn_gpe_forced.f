
	subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)
	implicit none
	integer, intent(in) :: ndim, icp(*), ijac
	double precision, intent(in) :: y(ndim), args(*)
	double precision, intent(out) :: y_delta(ndim)
	double precision, intent(inout) :: dfdu(ndim,ndim), 
     & dfdp(ndim,*)
	double precision R_e,V_e,R_i,V_i,I_ee,I_ei,I_ie,I_ii,I_a,
     & X_ee,X_ei,X_ie,X_ii,X_a,X1,X2,eta_e,eta_i,eta_str,
     & eta_tha,k_ee,k_ei,k_ie,k_ii,k,k_i,tau_e,tau_i,
     & tau_ee_r,tau_ee_d,tau_ei_r,tau_ei_d,tau_ie_r,
     & tau_ie_d,tau_ii_r,tau_ii_d,tau_a,alpha,delta_e,
     & delta_i,d_e,d_i,beta,omega,R_e1,R_e2,R_e3,R_e4,R_e5,
     & R_e6,R_e7,R_e8,R_e9,R_e10,R_e11,R_e12,R_e13,R_e14,
     & R_e15,R_e16,R_i1,R_i2,R_i3,R_i4,R_i5,R_i6,R_i7,R_i8,
     & R_i9,R_i10,R_i11,R_i12,R_i13,R_i14,R_i15,R_i16,pi

	! declare constants
	eta_e = args(1)
	eta_i = args(2)
	eta_str = args(3)
	eta_tha = args(4)
	k_ee = args(5)
	k_ei = args(6)
	k_ie = args(7)
	k_ii = args(8)
	k = args(9)
	k_i = args(10)
	tau_e = args(15)
	tau_i = args(16)
	tau_ee_r = args(17)
	tau_ee_d = args(18)
	tau_ei_r = args(19)
	tau_ei_d = args(20)
	tau_ie_r = args(21)
	tau_ie_d = args(22)
	tau_ii_r = args(23)
	tau_ii_d = args(24)
	tau_a = args(25)
	alpha = args(26)
	delta_e = args(27)
	delta_i = args(28)
	d_e = args(29)
	d_i = args(30)
	beta = args(31)
	omega = args(32)
	pi = 4*atan(1.0D0)

	! extract state variables from input vector
	R_e = y(1)
	V_e = y(2)
	R_i = y(3)
	V_i = y(4)
	I_ee = y(5)
	I_ei = y(7)
	I_ie = y(9)
	I_ii = y(11)
	I_a = y(13)
	X_ee = y(6)
	X_ei = y(8)
	X_ie = y(10)
	X_ii = y(12)
	X_a = y(14)
	X1 = y(47)
	X2 = y(48)
	R_e1 = y(15)
	R_e2 = y(16)
	R_e3 = y(17)
	R_e4 = y(18)
	R_e5 = y(19)
	R_e6 = y(20)
	R_e7 = y(21)
	R_e8 = y(22)
	R_e9 = y(23)
	R_e10 = y(24)
	R_e11 = y(25)
	R_e12 = y(26)
	R_e13 = y(27)
	R_e14 = y(28)
	R_e15 = y(29)
	R_e16 = y(30)
	R_i1 = y(31)
	R_i2 = y(32)
	R_i3 = y(33)
	R_i4 = y(34)
	R_i5 = y(35)
	R_i6 = y(36)
	R_i7 = y(37)
	R_i8 = y(38)
	R_i9 = y(39)
	R_i10 = y(40)
	R_i11 = y(41)
	R_i12 = y(42)
	R_i13 = y(43)
	R_i14 = y(44)
	R_i15 = y(45)
	R_i16 = y(46)

	! calculate right-hand side update of equation system
	y_delta(1) = delta_e / (pi * tau_e ** 2)
     & + (2.0 * R_e * V_e) / tau_e
	y_delta(2) = (V_e ** 2 + eta_e) / tau_e + I_ee 
     & - I_ei * (1.0 - I_a) - tau_e * (pi * R_e)
     & ** 2
	y_delta(3) = delta_i / (pi * tau_i ** 2)
     & + (2.0 * R_i * V_i) / tau_i
	y_delta(4) = (V_i ** 2 
     & + eta_i + eta_str + eta_tha + beta * X1) / tau_i + I_ie 
     & - I_ii * (1.0 - I_a) - tau_i * (pi * R_i)
     & ** 2
	y_delta(5) = X_ee
	y_delta(6) = (k * k_ee * R_e - X_ee * (tau_ee_r 
     & + tau_ee_d) - I_ee) / (tau_ee_r * tau_ee_d)
	y_delta(7) = X_ei
	y_delta(8) = (k * k_i * k_ei * R_i16 - X_ei * (tau_ei_r 
     & + tau_ei_d) - I_ei) / (tau_ei_r * tau_ei_d)
	y_delta(9) = X_ie
	y_delta(10) = (k * k_ie * R_e16 - X_ie * (tau_ie_r 
     & + tau_ie_d) - I_ie) / (tau_ie_r * tau_ie_d)
	y_delta(11) = X_ii
	y_delta(12) = (k * k_i * k_ii * R_i - X_ii * (tau_ii_r 
     & + tau_ii_d) - I_ii) / (tau_ii_r * tau_ii_d)
	y_delta(13) = X_a
	y_delta(14) = (alpha * R_i - 2.0 * X_a - I_a 
     & / tau_a) / tau_a
	y_delta(15) = ((R_e - R_e1) * 16.0) / d_e
	y_delta(16) = ((R_e1 - R_e2) * 16.0) / d_e
	y_delta(17) = ((R_e2 - R_e3) * 16.0) / d_e
	y_delta(18) = ((R_e3 - R_e4) * 16.0) / d_e
	y_delta(19) = ((R_e4 - R_e5) * 16.0) / d_e
	y_delta(20) = ((R_e5 - R_e6) * 16.0) / d_e
	y_delta(21) = ((R_e6 - R_e7) * 16.0) / d_e
	y_delta(22) = ((R_e7 - R_e8) * 16.0) / d_e
	y_delta(23) = ((R_e8 - R_e9) * 16.0) / d_e
	y_delta(24) = ((R_e9 - R_e10) * 16.0) / d_e
	y_delta(25) = ((R_e10 - R_e11) * 16.0) / d_e
	y_delta(26) = ((R_e11 - R_e12) * 16.0) / d_e
	y_delta(27) = ((R_e12 - R_e13) * 16.0) / d_e
	y_delta(28) = ((R_e13 - R_e14) * 16.0) / d_e
	y_delta(29) = ((R_e14 - R_e15) * 16.0) / d_e
	y_delta(30) = ((R_e15 - R_e16) * 16.0) / d_e
	y_delta(31) = ((R_i - R_i1) * 16.0) / d_i
	y_delta(32) = ((R_i1 - R_i2) * 16.0) / d_i
	y_delta(33) = ((R_i2 - R_i3) * 16.0) / d_i
	y_delta(34) = ((R_i3 - R_i4) * 16.0) / d_i
	y_delta(35) = ((R_i4 - R_i5) * 16.0) / d_i
	y_delta(36) = ((R_i5 - R_i6) * 16.0) / d_i
	y_delta(37) = ((R_i6 - R_i7) * 16.0) / d_i
	y_delta(38) = ((R_i7 - R_i8) * 16.0) / d_i
	y_delta(39) = ((R_i8 - R_i9) * 16.0) / d_i
	y_delta(40) = ((R_i9 - R_i10) * 16.0) / d_i
	y_delta(41) = ((R_i10 - R_i11) * 16.0) / d_i
	y_delta(42) = ((R_i11 - R_i12) * 16.0) / d_i
	y_delta(43) = ((R_i12 - R_i13) * 16.0) / d_i
	y_delta(44) = ((R_i13 - R_i14) * 16.0) / d_i
	y_delta(45) = ((R_i14 - R_i15) * 16.0) / d_i
	y_delta(46) = ((R_i15 - R_i16) * 16.0) / d_i
	y_delta(47) = X1 + omega * X2 - X1 * (X1 ** 2 + X2 ** 2)
	y_delta(48) = X2 - omega * X1 - X2 * (X1 ** 2 + X2 ** 2)

	end subroutine func
 
 
	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision R_e,V_e,R_i,V_i,I_ee,I_ei,I_ie,I_ii,I_a,
     & X_ee,X_ei,X_ie,X_ii,X_a,X1,X2,eta_e,eta_i,eta_str,
     & eta_tha,k_ee,k_ei,k_ie,k_ii,k,k_i,tau_e,tau_i,
     & tau_ee_r,tau_ee_d,tau_ei_r,tau_ei_d,tau_ie_r,
     & tau_ie_d,tau_ii_r,tau_ii_d,tau_a,alpha,delta_e,
     & delta_i,d_e,d_i,beta,omega,R_e1,R_e2,R_e3,R_e4,R_e5,
     & R_e6,R_e7,R_e8,R_e9,R_e10,R_e11,R_e12,R_e13,R_e14,
     & R_e15,R_e16,R_i1,R_i2,R_i3,R_i4,R_i5,R_i6,R_i7,R_i8,
     & R_i9,R_i10,R_i11,R_i12,R_i13,R_i14,R_i15,R_i16


	eta_e = -3.0
	eta_i = 16.6
	eta_str = -4.0
	eta_tha = 18.8
	k_ee = 2.9
	k_ei = 30.4
	k_ie = 97.9
	k_ii = 7.2
	k = 1.0
	k_i = 1.0
	tau_e = 6.0
	tau_i = 14.0
	tau_ee_r = 0.8
	tau_ee_d = 3.7
	tau_ei_r = 0.8
	tau_ei_d = 10.0
	tau_ie_r = 0.8
	tau_ie_d = 3.7
	tau_ii_r = 0.5
	tau_ii_d = 5.0
	tau_a = 500.0
	alpha = 0.0
	delta_e = 2.8
	delta_i = 1.5
	d_e = 4.0
	d_i = 4.0
	beta = 0.0
	omega = 20.0


	args(1) = eta_e
	args(2) = eta_i
	args(3) = eta_str
	args(4) = eta_tha
	args(5) = k_ee
	args(6) = k_ei
	args(7) = k_ie
	args(8) = k_ii
	args(9) = k
	args(10) = k_i
	args(11) = 8.0*atan(1.0D0)/omega
	args(15) = tau_e
	args(16) = tau_i
	args(17) = tau_ee_r
	args(18) = tau_ee_d
	args(19) = tau_ei_r
	args(20) = tau_ei_d
	args(21) = tau_ie_r
	args(22) = tau_ie_d
	args(23) = tau_ii_r
	args(24) = tau_ii_d
	args(25) = tau_a
	args(26) = alpha
	args(27) = delta_e
	args(28) = delta_i
	args(29) = d_e
	args(30) = d_i
	args(31) = beta
	args(32) = omega


	y(1) = 0.014043194241821766
	y(2) = 0.014043194241821766
	y(3) = 0.13786283135414124
	y(4) = 0.0
	y(5) = 0.04072526469826698
	y(7) = 4.191030025482178
	y(9) = 1.374828815460205
	y(11) = 0.0
	y(13) = 0.0
	y(6) = 9.736920007012775e-18
	y(8) = -3.8963520867653736e-16
	y(10) = 0.0
	y(12) = 0.0
	y(14) = 0.0
	y(47) = sin(8.0*atan(1.0D0)*t)
	y(48) = cos(8.0*atan(1.0D0)*t)
	y(15) = 0.0
	y(16) = 0.0
	y(17) = 0.0
	y(18) = 0.0
	y(19) = 0.0
	y(20) = 0.0
	y(21) = 0.0
	y(22) = 0.0
	y(23) = 0.0
	y(24) = 0.0
	y(25) = 0.0
	y(26) = 0.0
	y(27) = 0.0
	y(28) = 0.0
	y(29) = 0.0
	y(30) = 0.0
	y(31) = 0.0
	y(32) = 0.0
	y(33) = 0.0
	y(34) = 0.0
	y(35) = 0.0
	y(36) = 0.0
	y(37) = 0.0
	y(38) = 0.0
	y(39) = 0.0
	y(40) = 0.0
	y(41) = 0.0
	y(42) = 0.0
	y(43) = 0.0
	y(44) = 0.0
	y(45) = 0.0
	y(46) = 0.0


	end subroutine stpnt

	subroutine bcnd
	end subroutine bcnd

	subroutine icnd
	end subroutine icnd

	subroutine fopt
	end subroutine fopt

	subroutine pvls
	end subroutine pvls

