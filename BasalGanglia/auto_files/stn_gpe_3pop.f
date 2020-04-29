
	subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)
	implicit none
	integer, intent(in) :: ndim, icp(*), ijac
	double precision, intent(in) :: y(ndim), args(*)
	double precision, intent(out) :: y_delta(ndim)
	double precision, intent(inout) :: dfdu(ndim,ndim),
     & dfdp(ndim,*)
	double precision R_e,R_i,k_ei,gpe_p,k_ie,stn_0,k_ii,gpe_p_0
	double precision R_a,k_ia,gpe_a,R_s,k_is,str,k_ae,stn_1,k_ai
	double precision k_aa,gpe_a_0,k_as,str_0,mu,tau_s,V_e,eta_e
	double precision delta_e,d,ctx,V_i,eta_i,tau_i,delta_i,V_a
	double precision eta_a,tau_a,delta_a,gpe_p_1,tau_e,stn
	double precision R_e_d1,R_e_d2,R_e_d3,R_e_d4,R_e_d5,R_e_d6,R_e_d7
	double precision R_e_d8,R_e_d9,R_e_d10,R_e_d11,R_e_d12,R_e_d13
	double precision R_e_d14,R_e_d15,R_e_d16
	double precision R_ei_d1,R_ei_d2,R_ei_d3,R_ei_d4,R_ei_d5,R_ei_d6
	double precision R_ei_d7,R_ei_d8,R_ei_d9,R_ei_d10,R_ei_d11
	double precision R_ei_d12,R_ei_d13,R_ei_d14,R_ei_d15,R_ei_d16
	double precision R_i_d1,R_i_d2,R_i_d3,R_i_d4,R_i_d5,R_i_d6,R_i_d7
	double precision R_i_d8,R_i_d9,R_i_d10,R_i_d11,R_i_d12,R_i_d13
	double precision R_i_d14,R_i_d15,R_i_d16
	double precision R_a_d1,R_a_d2,R_a_d3,R_a_d4,R_a_d5,R_a_d6,R_a_d7
	double precision R_a_d8,R_a_d9,R_a_d10,R_a_d11,R_a_d12,R_a_d13
	double precision R_a_d14,R_a_d15,R_a_d16
	double precision k_e_d,k_ei_d,k_i_d,k_a_d,k_ee_d,k_ee
	double precision R_ee_d1,R_ee_d2,R_ee_d3,R_ee_d4,R_ee_d5,R_ee_d6
	double precision R_ee_d7,R_ee_d8,R_ee_d9,R_ee_d10,R_ee_d11
	double precision R_ee_d12,R_ee_d13,R_ee_d14,R_ee_d15,R_ee_d16

	! declare parameters
	d = args(1)
	k_ei = args(2)
	k_ie = args(3)
	k_ii = args(4)
	k_ia = args(5)
	k_is = args(6)
	k_ae = args(7)
	k_ai = args(8)
	k_aa = args(9)
	k_as = args(10)
	delta_e = args(15)
	delta_i = args(16)
	delta_a = args(17)
	eta_e = args(18)
	eta_i = args(19)
	eta_a = args(20)
	tau_e = args(21)
	tau_i = args(22)
	tau_a = args(23)
	k_ee = args(24)

	! declare constants
	tau_s = 1.0
	ctx = 0.0
	mu = 0.002
	k_e_d = 4
	k_ei_d = 5
	k_i_d = 10
	k_a_d = 10
	k_ee_d = 10

	delta_e = delta_e*tau_e*tau_e
	delta_i = delta_i*tau_i*tau_i/d
	delta_a = delta_a*tau_a*tau_a/d

	eta_e = eta_e*delta_e
	eta_i = eta_i*delta_i
	eta_a = eta_a*delta_a

	k_ee = k_ee*sqrt(delta_e)
	k_ei = k_ei*sqrt(delta_e)
	k_ie = k_ie*sqrt(delta_i)
	k_ii = k_ii*sqrt(delta_i)
	k_ia = k_ia*sqrt(delta_i)
	k_is = k_is*sqrt(delta_i)
	k_ae = k_ae*sqrt(delta_a)
	k_ai = k_ai*sqrt(delta_a)*d
	k_aa = k_aa*sqrt(delta_a)*d
	k_as = k_as*sqrt(delta_a)

	! extract state variables from input vector
	R_s = y(1)
	R_e = y(2)
	V_e = y(3)
	R_i = y(4)
	V_i = y(5)
	R_a = y(6)
	V_a = y(7)
	R_e_d1 = y(8)
	R_e_d2 = y(9)
	R_e_d3 = y(10)
	R_e_d4 = y(11)
	R_e_d5 = y(12)
	R_e_d6 = y(13)
	R_e_d7 = y(14)
	R_e_d8 = y(15)
	R_e_d9 = y(16)
	R_e_d10 = y(17)
	R_e_d11 = y(18)
	R_e_d12 = y(19)
	R_e_d13 = y(20)
	R_e_d14 = y(21)
	R_e_d15 = y(22)
	R_e_d16 = y(23)
	R_ei_d1 = y(24)
	R_ei_d2 = y(25)
	R_ei_d3 = y(26)
	R_ei_d4 = y(27)
	R_ei_d5 = y(28)
	R_ei_d6 = y(29)
	R_ei_d7 = y(30)
	R_ei_d8 = y(31)
	R_ei_d9 = y(32)
	R_ei_d10 = y(33)
	R_ei_d11 = y(34)
	R_ei_d12 = y(35)
	R_ei_d13 = y(36)
	R_ei_d14 = y(37)
	R_ei_d15 = y(38)
	R_ei_d16 = y(39)
	R_i_d1 = y(40)
	R_i_d2 = y(41)
	R_i_d3 = y(42)
	R_i_d4 = y(43)
	R_i_d5 = y(44)
	R_i_d6 = y(45)
	R_i_d7 = y(46)
	R_i_d8 = y(47)
	R_i_d9 = y(48)
	R_i_d10 = y(49)
	R_i_d11 = y(50)
	R_i_d12 = y(51)
	R_i_d13 = y(52)
	R_i_d14 = y(53)
	R_i_d15 = y(54)
	R_i_d16 = y(55)
	R_a_d1 = y(56)
	R_a_d2 = y(57)
	R_a_d3 = y(58)
	R_a_d4 = y(59)
	R_a_d5 = y(60)
	R_a_d6 = y(61)
	R_a_d7 = y(62)
	R_a_d8 = y(63)
	R_a_d9 = y(64)
	R_a_d10 = y(65)
	R_a_d11 = y(66)
	R_a_d12 = y(67)
	R_a_d13 = y(68)
	R_a_d14 = y(69)
	R_a_d15 = y(70)
	R_a_d16 = y(71)
	R_ee_d1 = y(72)
	R_ee_d2 = y(73)
	R_ee_d3 = y(74)
	R_ee_d4 = y(75)
	R_ee_d5 = y(76)
	R_ee_d6 = y(77)
	R_ee_d7 = y(78)
	R_ee_d8 = y(79)
	R_ee_d9 = y(80)
	R_ee_d10 = y(81)
	R_ee_d11 = y(82)
	R_ee_d12 = y(83)
	R_ee_d13 = y(84)
	R_ee_d14 = y(85)
	R_ee_d15 = y(86)
	R_ee_d16 = y(87)

	! calculate right-hand side update of equation system
	stn = R_ee_d16 * k_ee
	gpe_p = R_ei_d16 * k_ei
	stn_0 = R_e_d16 * k_ie
	gpe_p_0 = R_i_d16 * k_ii
	gpe_a = R_a_d16 * k_ia
	str = R_s * k_is
	stn_1 = R_e_d16 * k_ae
	gpe_p_1 = R_i_d16 * k_ai
	gpe_a_0 = R_a_d16 * k_aa
	str_0 = R_s * k_as

    ! dummy STR
	y_delta(1) = (mu - R_s) / tau_s

	! STN
	y_delta(2) = delta_e / (3.141592653589793 * tau_e ** 2)
     & + (2.0 * R_e * V_e) / tau_e
	y_delta(3) = (V_e**2 + eta_e) / tau_e + ctx - gpe_p
     & + stn - tau_e * (3.141592653589793 * R_e) ** 2

	! GPe-p
	y_delta(4) = delta_i / (3.141592653589793 * tau_i ** 2)
     & + (2.0 * R_i * V_i) / tau_i
	y_delta(5) = (V_i**2 + eta_i)/tau_i + stn_0 - gpe_p_0
     & - gpe_a - str - tau_i
     & * (3.141592653589793 * R_i) ** 2

    ! GPe-a
	y_delta(6) = delta_a / (3.141592653589793 * tau_a ** 2)
     & + (2.0 * R_a * V_a) / tau_a
	y_delta(7) = (V_a**2 + eta_a)/tau_a + stn_1 - gpe_p_1
     & - gpe_a_0 - str_0 - tau_a
     & * (3.141592653589793 * R_a) ** 2

	! STN to GPe-p
	y_delta(8) = k_e_d * (R_e - R_e_d1)
	y_delta(9) = k_e_d * (R_e_d1 - R_e_d2)
	y_delta(10) = k_e_d * (R_e_d2 - R_e_d3)
	y_delta(11) = k_e_d * (R_e_d3 - R_e_d4)
	y_delta(12) = k_e_d * (R_e_d4 - R_e_d5)
	y_delta(13) = k_e_d * (R_e_d5 - R_e_d6)
	y_delta(14) = k_e_d * (R_e_d6 - R_e_d7)
	y_delta(15) = k_e_d * (R_e_d7 - R_e_d8)
	y_delta(16) = k_e_d * (R_e_d8 - R_e_d9)
	y_delta(17) = k_e_d * (R_e_d9 - R_e_d10)
	y_delta(18) = k_e_d * (R_e_d10 - R_e_d11)
	y_delta(19) = k_e_d * (R_e_d11 - R_e_d12)
	y_delta(20) = k_e_d * (R_e_d12 - R_e_d13)
	y_delta(21) = k_e_d * (R_e_d13 - R_e_d14)
	y_delta(22) = k_e_d * (R_e_d14 - R_e_d15)
	y_delta(23) = k_e_d * (R_e_d15 - R_e_d16)

	! GPe-p to STN
	y_delta(24) = k_ei_d * (R_i - R_ei_d1)
	y_delta(25) = k_ei_d * (R_ei_d1 - R_ei_d2)
	y_delta(26) = k_ei_d * (R_ei_d2 - R_ei_d3)
	y_delta(27) = k_ei_d * (R_ei_d3 - R_ei_d4)
	y_delta(28) = k_ei_d * (R_ei_d4 - R_ei_d5)
	y_delta(29) = k_ei_d * (R_ei_d5 - R_ei_d6)
	y_delta(30) = k_ei_d * (R_ei_d6 - R_ei_d7)
	y_delta(31) = k_ei_d * (R_ei_d7 - R_ei_d8)
	y_delta(32) = k_ei_d * (R_ei_d8 - R_ei_d9)
	y_delta(33) = k_ei_d * (R_ei_d9 - R_ei_d10)
	y_delta(34) = k_ei_d * (R_ei_d10 - R_ei_d11)
	y_delta(35) = k_ei_d * (R_ei_d11 - R_ei_d12)
	y_delta(36) = k_ei_d * (R_ei_d12 - R_ei_d13)
	y_delta(37) = k_ei_d * (R_ei_d13 - R_ei_d14)
	y_delta(38) = k_ei_d * (R_ei_d14 - R_ei_d15)
	y_delta(39) = k_ei_d * (R_ei_d15 - R_ei_d16)

	! Gpe-p to both GPes
	y_delta(40) = k_i_d * (R_i - R_i_d1)
	y_delta(41) = k_i_d * (R_i_d1 - R_i_d2)
	y_delta(42) = k_i_d * (R_i_d2 - R_i_d3)
	y_delta(43) = k_i_d * (R_i_d3 - R_i_d4)
	y_delta(44) = k_i_d * (R_i_d4 - R_i_d5)
	y_delta(45) = k_i_d * (R_i_d5 - R_i_d6)
	y_delta(46) = k_i_d * (R_i_d6 - R_i_d7)
	y_delta(47) = k_i_d * (R_i_d7 - R_i_d8)
	y_delta(48) = k_i_d * (R_i_d8 - R_i_d9)
	y_delta(49) = k_i_d * (R_i_d9 - R_i_d10)
	y_delta(50) = k_i_d * (R_i_d10 - R_i_d11)
	y_delta(51) = k_i_d * (R_i_d11 - R_i_d12)
	y_delta(52) = k_i_d * (R_i_d12 - R_i_d13)
	y_delta(53) = k_i_d * (R_i_d13 - R_i_d14)
	y_delta(54) = k_i_d * (R_i_d14 - R_i_d15)
	y_delta(55) = k_i_d * (R_i_d15 - R_i_d16)

	! Gpe-a to both GPes
	y_delta(56) = k_a_d * (R_a - R_a_d1)
	y_delta(57) = k_a_d * (R_a_d1 - R_a_d2)
	y_delta(58) = k_a_d * (R_a_d2 - R_a_d3)
	y_delta(59) = k_a_d * (R_a_d3 - R_a_d4)
	y_delta(60) = k_a_d * (R_a_d4 - R_a_d5)
	y_delta(61) = k_a_d * (R_a_d5 - R_a_d6)
	y_delta(62) = k_a_d * (R_a_d6 - R_a_d7)
	y_delta(63) = k_a_d * (R_a_d7 - R_a_d8)
	y_delta(64) = k_a_d * (R_a_d8 - R_a_d9)
	y_delta(65) = k_a_d * (R_a_d9 - R_a_d10)
	y_delta(66) = k_a_d * (R_a_d10 - R_a_d11)
	y_delta(67) = k_a_d * (R_a_d11 - R_a_d12)
	y_delta(68) = k_a_d * (R_a_d12 - R_a_d13)
	y_delta(69) = k_a_d * (R_a_d13 - R_a_d14)
	y_delta(70) = k_a_d * (R_a_d14 - R_a_d15)
	y_delta(71) = k_a_d * (R_a_d15 - R_a_d16)

    ! STN to STN
	y_delta(72) = k_ee_d * (R_e - R_ee_d1)
	y_delta(73) = k_ee_d * (R_ee_d1 - R_ee_d2)
	y_delta(74) = k_ee_d * (R_ee_d2 - R_ee_d3)
	y_delta(75) = k_ee_d * (R_ee_d3 - R_ee_d4)
	y_delta(76) = k_ee_d * (R_ee_d4 - R_ee_d5)
	y_delta(77) = k_ee_d * (R_ee_d5 - R_ee_d6)
	y_delta(78) = k_ee_d * (R_ee_d6 - R_ee_d7)
	y_delta(79) = k_ee_d * (R_ee_d7 - R_ee_d8)
	y_delta(80) = k_ee_d * (R_ee_d8 - R_ee_d9)
	y_delta(81) = k_ee_d * (R_ee_d9 - R_ee_d10)
	y_delta(82) = k_ee_d * (R_ee_d10 - R_ee_d11)
	y_delta(83) = k_ee_d * (R_ee_d11 - R_ee_d12)
	y_delta(84) = k_ee_d * (R_ee_d12 - R_ee_d13)
	y_delta(85) = k_ee_d * (R_ee_d13 - R_ee_d14)
	y_delta(86) = k_ee_d * (R_ee_d14 - R_ee_d15)
	y_delta(87) = k_ee_d * (R_ee_d15 - R_ee_d16)

	end subroutine func


	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision d,k_ei,k_ie,k_ii,k_ia,k_is,k_ae,k_ai,k_aa,k_as
	double precision delta_e,delta_i,delta_a,eta_e,eta_i,eta_a
	double precision tau_e,tau_i,tau_a,k_ee

	d = 1.0

	tau_e = 13.0
	tau_i = 25.0
	tau_a = 20.0

	!delta_e = 0.75
	!delta_i = 0.89
	!delta_a = 0.72

	delta_e = 0.1
	delta_i = 0.4
	delta_a = 0.1

	!eta_e = -3.7
	!eta_i = -4.7
	!eta_a = -11.6

	eta_e = 0.0
	eta_i = -4.0
	eta_a = 0.0

	!k_ee = 10.9
	!k_ei = 487.7
	!k_ie = 352.0
	!k_ii = 81.4
	!k_ia = 120.2
	!k_is = 516.0
	!k_ae = 292.4
	!k_ai = 65.6
	!k_aa = 417.8
	!k_as = 143.6

	k_ee = 4.0
	k_ei = 20.0
	k_ie = 130.0
	k_ii = 4.0
	k_ia = 60.0
	k_is = 80.0
	k_ae = 40.0
	k_ai = 20.0
	k_aa = 4.0
	k_as = 160.0

	args(1) = d
	args(2) = k_ei
	args(3) = k_ie
	args(4) = k_ii
	args(5) = k_ia
	args(6) = k_is
	args(7) = k_ae
	args(8) = k_ai
	args(9) = k_aa
	args(10) = k_as
	args(15) = delta_e
	args(16) = delta_i
	args(17) = delta_a
	args(18) = eta_e
	args(19) = eta_i
	args(20) = eta_a
	args(21) = tau_e
	args(22) = tau_i
	args(23) = tau_a

	y(2) = 0.016097187995910645
	y(4) = 0.03679664060473442
	y(6) = 0.02945549227297306
	y(1) = 0.0020000000949949026
	y(3) = -15.160260200500488
	y(5) = -6.614039421081543
	y(7) = -6.889124393463135

	end subroutine stpnt

	subroutine bcnd
	end subroutine bcnd

	subroutine icnd
	end subroutine icnd

	subroutine fopt
	end subroutine fopt

	subroutine pvls
	end subroutine pvls
