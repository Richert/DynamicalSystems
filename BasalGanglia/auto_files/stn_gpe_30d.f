
	subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)
	implicit none
	integer, intent(in) :: ndim, icp(*), ijac
	double precision, intent(in) :: y(ndim), args(*)
	double precision, intent(out) :: y_delta(ndim)
	double precision, intent(inout) :: dfdu(ndim,ndim), 
     & dfdp(ndim,*)
	double precision gpe,R_e,V_e,I_ampa,I_gabaa,X_ampa,X_gabaa,
     & eta_e,k_ee,k_ei,k_ctx,tau_e,tau_ampa_r,tau_ampa_d,
     & tau_gabaa_r,tau_gabaa_d,delta_e,ctx,R_e_d1,k_d1,
     & R_e_d2,k_d2,R_e_d3,k_d3,R_e_d4,k_d4,R_e_d5,k_d5,
     & R_e_d6,k_d6,R_e_d7,k_d7,R_e_d8,k_d8,R_e_d9,k_d9,
     & R_e_buffered,stn,R_i,V_i,I_ampa_0,I_gabaa_0,X_ampa_0,
     & X_gabaa_0,eta_i,eta_tha,k_ie,k_ii,k_str,tau_i,
     & tau_ampa_r_0,tau_ampa_d_0,tau_gabaa_r_0,
     & tau_gabaa_d_0,delta_i,str,R_i_d1,k_d1_0,R_i_d2,
     & k_d2_0,R_i_d3,k_d3_0,R_i_d4,k_d4_0,R_i_d5,k_d5_0,
     & R_i_d6,k_d6_0,R_i_d7,k_d7_0,R_i_d8,k_d8_0,R_i_d9,
     & k_d9_0,R_i_buffered,weight,weight_0,k,k_i

	! declare constants
	gpe = args(1)
	eta_e = args(2)
	k_ee = args(3)
	k_ei = args(4)
	k_ctx = args(5)
	tau_e = args(6)
	tau_ampa_r = args(7)
	tau_ampa_d = args(8)
	tau_gabaa_r = args(9)
	tau_gabaa_d = args(10)
	delta_e = args(15)
	ctx = args(16)
	k_d1 = args(17)
	k_d2 = args(18)
	k_d3 = args(19)
	k_d4 = args(20)
	k_d5 = args(21)
	k_d6 = args(22)
	k_d7 = args(23)
	k_d8 = args(24)
	k_d9 = args(25)
	R_e_buffered = args(26)
	stn = args(27)
	eta_i = args(28)
	eta_tha = args(29)
	k_ie = args(30)
	k_ii = args(31)
	k_str = args(32)
	tau_i = args(33)
	tau_ampa_r_0 = args(34)
	tau_ampa_d_0 = args(35)
	tau_gabaa_r_0 = args(36)
	tau_gabaa_d_0 = args(37)
	delta_i = args(38)
	str = args(39)
	k_d1_0 = args(40)
	k_d2_0 = args(41)
	k_d3_0 = args(42)
	k_d4_0 = args(43)
	k_d5_0 = args(44)
	k_d6_0 = args(45)
	k_d7_0 = args(46)
	k_d8_0 = args(47)
	k_d9_0 = args(48)
	R_i_buffered = args(49)
	weight = args(50)
	weight_0 = args(51)
	k = args(52)
	k_i = args(53)

	! extract state variables from input vector
	R_e = y(1)
	V_e = y(2)
	I_ampa = y(3)
	I_gabaa = y(5)
	X_ampa = y(4)
	X_gabaa = y(6)
	R_e_d1 = y(7)
	R_e_d2 = y(8)
	R_e_d3 = y(9)
	R_e_d4 = y(10)
	R_e_d5 = y(11)
	R_e_d6 = y(12)
	R_e_d7 = y(13)
	R_e_d8 = y(14)
	R_e_d9 = y(15)
	R_i = y(16)
	V_i = y(17)
	I_ampa_0 = y(18)
	I_gabaa_0 = y(20)
	X_ampa_0 = y(19)
	X_gabaa_0 = y(21)
	R_i_d1 = y(22)
	R_i_d2 = y(23)
	R_i_d3 = y(24)
	R_i_d4 = y(25)
	R_i_d5 = y(26)
	R_i_d6 = y(27)
	R_i_d7 = y(28)
	R_i_d8 = y(29)
	R_i_d9 = y(30)

	! calculate right-hand side update of equation system
	R_e_buffered = R_e_d9
	R_i_buffered = R_i_d9
	gpe = R_i_buffered * weight
	stn = R_e_buffered * weight_0
	y_delta(1) = delta_e / (3.141592653589793 * tau_e ** 2) 
     & + (2.0 * R_e * V_e) / tau_e
	y_delta(2) = (V_e ** 2 + eta_e) / tau_e + I_ampa 
     & - I_gabaa - tau_e * (3.141592653589793 * R_e) ** 2
	y_delta(3) = X_ampa
	y_delta(4) = (k*k_ee * R_e + k*k_ctx * ctx
     & - X_ampa * (tau_ampa_r + tau_ampa_d) - I_ampa) 
     & / (tau_ampa_r * tau_ampa_d)
	y_delta(5) = X_gabaa
	y_delta(6) = (k*k_i*k_ei * gpe - X_gabaa * (tau_gabaa_r
     & + tau_gabaa_d) - I_gabaa) / (tau_gabaa_r * tau_gabaa_d)
	y_delta(7) = k_d1 * (R_e - R_e_d1)
	y_delta(8) = k_d2 * (R_e_d1 - R_e_d2)
	y_delta(9) = k_d3 * (R_e_d2 - R_e_d3)
	y_delta(10) = k_d4 * (R_e_d3 - R_e_d4)
	y_delta(11) = k_d5 * (R_e_d4 - R_e_d5)
	y_delta(12) = k_d6 * (R_e_d5 - R_e_d6)
	y_delta(13) = k_d7 * (R_e_d6 - R_e_d7)
	y_delta(14) = k_d8 * (R_e_d7 - R_e_d8)
	y_delta(15) = k_d9 * (R_e_d8 - R_e_d9)
	y_delta(16) = delta_i / (3.141592653589793 * tau_i ** 2) 
     & + (2.0 * R_i * V_i) / tau_i
	y_delta(17) = (V_i ** 2 + eta_i + eta_tha) 
     & / tau_i + I_ampa_0 - I_gabaa_0 - tau_i 
     & * (3.141592653589793 * R_i) ** 2
	y_delta(18) = X_ampa_0
	y_delta(19) = (k*k_ie * stn - X_ampa_0 * (tau_ampa_r_0
     & + tau_ampa_d_0) - I_ampa_0) / (tau_ampa_r_0 * tau_ampa_d_0)
	y_delta(20) = X_gabaa_0
	y_delta(21) = (k*k_i*k_ii * R_i + k_i*k*k_str * str
     & - X_gabaa_0 * (tau_gabaa_r_0 + tau_gabaa_d_0) - I_gabaa_0) 
     & / (tau_gabaa_r_0 * tau_gabaa_d_0)
	y_delta(22) = k_d1_0 * (R_i - R_i_d1)
	y_delta(23) = k_d2_0 * (R_i_d1 - R_i_d2)
	y_delta(24) = k_d3_0 * (R_i_d2 - R_i_d3)
	y_delta(25) = k_d4_0 * (R_i_d3 - R_i_d4)
	y_delta(26) = k_d5_0 * (R_i_d4 - R_i_d5)
	y_delta(27) = k_d6_0 * (R_i_d5 - R_i_d6)
	y_delta(28) = k_d7_0 * (R_i_d6 - R_i_d7)
	y_delta(29) = k_d8_0 * (R_i_d7 - R_i_d8)
	y_delta(30) = k_d9_0 * (R_i_d8 - R_i_d9)

	end subroutine func
 
 
	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision gpe,R_e,V_e,I_ampa,I_gabaa,X_ampa,X_gabaa,
     & eta_e,k_ee,k_ei,k_ctx,tau_e,tau_ampa_r,tau_ampa_d,
     & tau_gabaa_r,tau_gabaa_d,delta_e,ctx,R_e_d1,k_d1,
     & R_e_d2,k_d2,R_e_d3,k_d3,R_e_d4,k_d4,R_e_d5,k_d5,
     & R_e_d6,k_d6,R_e_d7,k_d7,R_e_d8,k_d8,R_e_d9,k_d9,
     & R_e_buffered,stn,R_i,V_i,I_ampa_0,I_gabaa_0,X_ampa_0,
     & X_gabaa_0,eta_i,eta_tha,k_ie,k_ii,k_str,tau_i,
     & tau_ampa_r_0,tau_ampa_d_0,tau_gabaa_r_0,
     & tau_gabaa_d_0,delta_i,str,R_i_d1,k_d1_0,R_i_d2,
     & k_d2_0,R_i_d3,k_d3_0,R_i_d4,k_d4_0,R_i_d5,k_d5_0,
     & R_i_d6,k_d6_0,R_i_d7,k_d7_0,R_i_d8,k_d8_0,R_i_d9,
     & k_d9_0,R_i_buffered,weight,weight_0,k,k_i


	gpe = 0.0
	eta_e = 29.600000381469727
	k_ee = 5.5
	k_ei = 107.69999694824219
	k_ctx = 10.0
	tau_e = 12.800000190734863
	tau_ampa_r = 0.800000011920929
	tau_ampa_d = 3.700000047683716
	tau_gabaa_r = 0.800000011920929
	tau_gabaa_d = 10.0
	delta_e = 10.0
	ctx = 0.0
	k_d1 = 3.0
	k_d2 = 3.0
	k_d3 = 3.0
	k_d4 = 3.0
	k_d5 = 3.0
	k_d6 = 3.0
	k_d7 = 3.0
	k_d8 = 3.0
	k_d9 = 3.0
	R_e_buffered = 0.0
	stn = 0.0
	eta_i = 25.0
	eta_tha = 32.70000076293945
	k_ie = 131.89999389648438
	k_ii = 43.0
	k_str = 470.20001220703125
	tau_i = 25.399999618530273
	tau_ampa_r_0 = 0.800000011920929
	tau_ampa_d_0 = 3.700000047683716
	tau_gabaa_r_0 = 0.5
	tau_gabaa_d_0 = 5.0
	delta_i = 11.899999618530273
	str = 0.0020000000949949026
	k_d1_0 = 6.0
	k_d2_0 = 6.0
	k_d3_0 = 6.0
	k_d4_0 = 6.0
	k_d5_0 = 6.0
	k_d6_0 = 6.0
	k_d7_0 = 6.0
	k_d8_0 = 6.0
	k_d9_0 = 6.0
	R_i_buffered = 0.0
	weight = 1.0
	weight_0 = 1.0
	k = 1.0
	k_i = 1.0


	args(1) = gpe
	args(2) = eta_e
	args(3) = k_ee
	args(4) = k_ei
	args(5) = k_ctx
	args(6) = tau_e
	args(7) = tau_ampa_r
	args(8) = tau_ampa_d
	args(9) = tau_gabaa_r
	args(10) = tau_gabaa_d
	args(15) = delta_e
	args(16) = ctx
	args(17) = k_d1
	args(18) = k_d2
	args(19) = k_d3
	args(20) = k_d4
	args(21) = k_d5
	args(22) = k_d6
	args(23) = k_d7
	args(24) = k_d8
	args(25) = k_d9
	args(26) = R_e_buffered
	args(27) = stn
	args(28) = eta_i
	args(29) = eta_tha
	args(30) = k_ie
	args(31) = k_ii
	args(32) = k_str
	args(33) = tau_i
	args(34) = tau_ampa_r_0
	args(35) = tau_ampa_d_0
	args(36) = tau_gabaa_r_0
	args(37) = tau_gabaa_d_0
	args(38) = delta_i
	args(39) = str
	args(40) = k_d1_0
	args(41) = k_d2_0
	args(42) = k_d3_0
	args(43) = k_d4_0
	args(44) = k_d5_0
	args(45) = k_d6_0
	args(46) = k_d7_0
	args(47) = k_d8_0
	args(48) = k_d9_0
	args(49) = R_i_buffered
	args(50) = weight
	args(51) = weight_0
	args(52) = k
	args(53) = k_i


	y(1) = 0.01681089960038662
	y(2) = -7.3963799476623535
	y(3) = 0.09245993942022324
	y(5) = 6.643198490142822
	y(4) = -4.4567837931673214e-18
	y(6) = 1.815610463758383e-15
	y(7) = 0.01681089960038662
	y(8) = 0.01681089960038662
	y(9) = 0.01681089960038662
	y(10) = 0.01681089960038662
	y(11) = 0.01681089960038662
	y(12) = 0.01681089960038662
	y(13) = 0.01681089960038662
	y(14) = 0.01681089960038662
	y(15) = 0.01681089960038662
	y(16) = 0.06168243661522865
	y(17) = -1.2088483572006226
	y(18) = 2.2173573970794678
	y(20) = 3.592744827270508
	y(19) = 6.522149458693376e-16
	y(21) = -5.588892792573324e-17
	y(22) = 0.06168243661522865
	y(23) = 0.06168243661522865
	y(24) = 0.06168243661522865
	y(25) = 0.06168243661522865
	y(26) = 0.06168243661522865
	y(27) = 0.06168243661522865
	y(28) = 0.06168243661522865
	y(29) = 0.06168243661522865
	y(30) = 0.06168243661522865


	end subroutine stpnt

	subroutine bcnd
	end subroutine bcnd

	subroutine icnd
	end subroutine icnd

	subroutine fopt
	end subroutine fopt

	subroutine pvls
	end subroutine pvls

