
	subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)
	implicit none
	integer, intent(in) :: ndim, icp(*), ijac
	double precision, intent(in) :: y(ndim), args(*)
	double precision, intent(out) :: y_delta(ndim)
	double precision, intent(inout) :: dfdu(ndim,ndim),
     & dfdp(ndim,*)
	double precision r_e, v_e, r_p, v_p, r_a, v_a, r_s
	double precision r_xe, r_ep, r_xp, r_xa, r_ee
	double precision E_e, x_e, I_e, y_e
	double precision E_p, x_p, I_p, y_p
	double precision E_a, x_a, I_a, y_a
	double precision eta_e, eta_p, eta_a, eta_s, eta
	double precision k_ee, k_pe, k_ae
	double precision k_ep, k_pp, k_ap
	double precision k_pa, k_aa
	double precision k_ps, k_as, k
	double precision delta_e, delta_p, delta_a, delta
	double precision tau_e, tau_p, tau_a, tau_s, tau_stn
	double precision tau_ampa_r, tau_ampa_d, tau_gabaa_r, tau_gabaa_d
	double precision PI, k_gp, k_p, k_i, k_d

	! declare parameters
	eta_e = args(1)
	eta_p = args(2)
	eta_a = args(3)
	k_ee = args(4)
	k_pe = args(5)
	k_ae = args(6)
	k_ep = args(7)
	k_pp = args(8)
	k_ap = args(9)
	k_pa = args(10)
	k_aa = args(15)
	delta_e = args(16)
	delta_p = args(17)
	delta_a = args(18)
	k_gp = args(19)
	k_p = args(20)
	k_i = args(21)
	tau_e = args(22)
	tau_p = args(23)
	tau_a = args(24)
	tau_ampa_r = args(25)
	tau_ampa_d = args(26)
	tau_gabaa_r = args(27)
	tau_gabaa_d = args(28)
	tau_stn = args(29)
	eta = args(30)
	delta = args(31)
	k = args(32)

	! declare constants
	tau_s = 1.0
	eta_s = 0.002
	k_ps = 20.0
	k_as = 20.0
	k_d = 3.0
	PI = 3.141592653589793

	delta_e = delta_e*delta
	delta_p = delta_p*delta
	delta_a = delta_a*delta

	eta_e = eta_e*eta
	eta_p = eta_p*eta
	eta_a = eta_a*eta

	k_ee = k_ee*k
	k_pe = k_pe*k
	k_ae = k_ae*k
	k_ep = k_ep*k
	k_pp = k_pp*k*k_gp*k_p/k_i
	k_ap = k_ap*k*k_gp*k_i*k_p
	k_pa = k_pa*k*k_gp*k_i/k_p
	k_aa = k_aa*k*k_gp/(k_i*k_p)
	k_ps = k_ps*k
	k_as = k_as*k

	! extract state variables from input vector
	r_e = y(1)
	v_e = y(2)
	r_p = y(3)
	v_p = y(4)
	r_a = y(5)
	v_a = y(6)
	r_s = y(7)
	E_e = y(8)
	x_e = y(9)
	I_e = y(10)
	y_e = y(11)
	E_p = y(12)
	x_p = y(13)
	I_p = y(14)
	y_p = y(15)
	E_a = y(16)
	x_a = y(17)
	I_a = y(18)
	y_a = y(19)
	r_ee = y(22)
	r_xe = y(28)
	r_xp = y(31)
	r_ep = y(37)
	r_xa = y(40)

	! calculate right-hand side update of equation system

    ! 1. population updates

	! STN
	y_delta(1) = delta_e / (PI*tau_e**2) + (2.0*r_e*v_e) / tau_e
	y_delta(2) = (v_e**2 + eta_e + (E_e-I_e)*tau_e
     & - (tau_e*PI*r_e)**2) / tau_e

	! GPe-p
	y_delta(3) = delta_p / (PI*tau_p**2) + (2.0*r_p*v_p) / tau_p
	y_delta(4) = (v_p**2 + eta_p + (E_p-I_p)*tau_p
     & - (tau_p*PI*r_p)**2) / tau_p

    ! GPe-a
	y_delta(5) = delta_a / (PI*tau_a**2) + (2.0*r_a*v_a) / tau_a
	y_delta(6) = (v_a**2 + eta_a + (E_a-I_a)*tau_a
     & - (tau_a*PI*r_a)**2) / tau_a

	! STR
	y_delta(7) = (eta_s - r_s) / tau_s

	! 2. synapse dynamics

	! at STN
	y_delta(8) = x_e
	y_delta(9) = (k_ee*r_ee - x_e*(tau_ampa_r+tau_ampa_d) - E_e)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(10) = y_e
	y_delta(11) = (k_ep*r_ep - y_e*tau_stn*(tau_gabaa_r+tau_gabaa_d) -
     & I_e) / (tau_gabaa_r*tau_gabaa_d*tau_stn*tau_stn)

	! at GPe-p
	y_delta(12) = x_p
	y_delta(13) = (k_pe*r_xe - x_p*(tau_ampa_r+tau_ampa_d) - E_p)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(14) = y_p
	y_delta(15) = (k_pp*r_xp + k_pa*r_xa + k_ps*r_s
     & - y_p*(tau_gabaa_r+tau_gabaa_d) - I_p)/(tau_gabaa_r*tau_gabaa_d)

	! at GPe-a
	y_delta(16) = x_a
	y_delta(17) = (k_ae*r_xe - x_a*(tau_ampa_r+tau_ampa_d) - E_a)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(18) = y_a
	y_delta(19) = (k_ap*r_xp + k_aa*r_xa + k_as*r_s
     & - y_a*(tau_gabaa_r+tau_gabaa_d) - I_a)/(tau_gabaa_r*tau_gabaa_d)

	! STN output
	y_delta(20) = k_d * (r_e - y(20))
	y_delta(21) = k_d * (y(20) - y(21))
	y_delta(22) = k_d * (y(21) - y(22))
	y_delta(23) = k_d * (y(22) - y(23))
	y_delta(24) = k_d * (y(23) - y(24))
	y_delta(25) = k_d * (y(24) - y(25))
	y_delta(26) = k_d * (y(25) - y(26))
	y_delta(27) = k_d * (y(26) - y(27))
	y_delta(28) = k_d * (y(27) - y(28))

	! GPe-p output
	y_delta(29) = k_d * (r_p - y(29))
	y_delta(30) = k_d * (y(29) - y(30))
	y_delta(31) = k_d * (y(30) - y(31))
	y_delta(32) = k_d * (y(31) - y(32))
	y_delta(33) = k_d * (y(32) - y(33))
	y_delta(34) = k_d * (y(33) - y(34))
	y_delta(35) = k_d * (y(34) - y(35))
	y_delta(36) = k_d * (y(35) - y(36))
	y_delta(37) = k_d * (y(36) - y(37))

	! Gpe-a output
	y_delta(38) = k_d * (r_a - y(38))
	y_delta(39) = k_d * (y(38) - y(39))
	y_delta(40) = k_d * (y(39) - y(40))

	end subroutine func


	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision eta_e, eta_p, eta_a, eta
	double precision k_ee, k_pe, k_ae
	double precision k_ep, k_pp, k_ap
	double precision k_pa, k_aa
	double precision delta_e, delta_p, delta_a, delta
	double precision k_gp, k_p, k_i, k
	double precision tau_e, tau_p, tau_a, tau_s, tau_stn
	double precision tau_ampa_r, tau_ampa_d, tau_gabaa_r, tau_gabaa_d
	
	tau_e = 13.0
	tau_p = 25.0
	tau_a = 20.0
	
	tau_ampa_r = 0.8
	tau_ampa_d = 3.7
	tau_gabaa_r = 0.5
	tau_gabaa_d = 5.0
	tau_stn = 2.0

	k_gp = 3.0
	k_p = 1.5
	k_i = 1.0

	delta_e = 0.3
	delta_p = 0.9
	delta_a = 1.2

	eta_e = 0.0
	eta_p = 0.0
	eta_a = 0.0

	k_ee = 0.8
	k_pe = 4.0
	k_ae = 0.0
	k_ep = 10.0
	k_pp = 1.0
	k_ap = 1.0
	k_pa = 1.0
	k_aa = 1.0
	
	eta = 100.0
	delta = 100.0
	k = 100.0

	args(1) = eta_e
	args(2) = eta_p
	args(3) = eta_a
	args(4) = k_ee
	args(5) = k_pe
	args(6) = k_ae
	args(7) = k_ep
	args(8) = k_pp
	args(9) = k_ap
	args(10) = k_pa
	args(15) = k_aa
	args(16) = delta_e
	args(17) = delta_p
	args(18) = delta_a
	args(19) = k_gp
	args(20) = k_p
	args(21) = k_i
	args(22) = tau_e
	args(23) = tau_p
	args(24) = tau_a
	args(25) = tau_ampa_r
	args(26) = tau_ampa_d
	args(27) = tau_gabaa_r
	args(28) = tau_gabaa_d
	args(29) = tau_stn
	args(30) = eta
	args(31) = delta
	args(32) = k

	y(1) = 0.02
	y(3) = 0.06
	y(5) = 0.03
	y(7) = 0.002
	y(2) = -4.0
	y(4) = -2.0
	y(6) = -4.0

	end subroutine stpnt

	subroutine bcnd
	end subroutine bcnd

	subroutine icnd
	end subroutine icnd

	subroutine fopt
	end subroutine fopt

	subroutine pvls
	end subroutine pvls
