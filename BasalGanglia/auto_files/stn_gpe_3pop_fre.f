
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
	double precision tau_e, tau_p, tau_a, tau_s
	double precision tau_ampa_r, tau_ampa_d, tau_gabaa_r, tau_gabaa_d
	double precision tau_gabaa_r2, tau_gabaa_d2
	double precision k_ee_d, k_pe_d, k_ep_d, k_p_d, k_a_d
	double precision PI, k_gp, k_p, k_i, k_stn

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
	k_ps = args(16)
	k_as = args(17)
	delta_e = args(18)
	delta_p = args(19)
	delta_a = args(20)
	eta_s = args(21)
	k_gp = args(22)
	k_p = args(23)
	k_i = args(24)
	k_stn = args(25)

	! declare constants
	tau_e = 13.0
	tau_p = 25.0
	tau_a = 20.0
	tau_s = 1.0
	tau_ampa_r = 0.8
	tau_ampa_d = 3.7
	tau_gabaa_r = 0.5
	tau_gabaa_d = 5.0
	tau_gabaa_r2 = 0.8
	tau_gabaa_d2 = 10.0
	k_ee_d = 1.14
	k_pe_d = 2.67
	k_ep_d = 2.0
	k_p_d = 1.33
	k_a_d = 1.33
	PI = 3.141592653589793
	delta = 10.0
	k = 100.0
	eta = 100.0

	delta_e = delta_e*delta
	delta_p = delta_p*delta
	delta_a = delta_a*delta

	eta_e = eta_e*eta
	eta_p = eta_p*eta
	eta_a = eta_a*eta

	k_ee = k_ee*k
	k_pe = k_pe*k*k_stn
	k_ae = k_ae*k/k_stn
	k_ep = k_ep*k
	k_pp = k_pp*k*k_gp*k_p/k_i
	k_ap = k_ap*k*k_gp*k_i*k_p
	k_pa = k_pa*k*k_gp*k_i/k_p
	k_aa = k_aa*k*k_gp/(k_i*k_p)
	k_ps = k_ps*k
	k_as = k_as*k

	! extract state variables from input vector
	r_e = y(1)
	r_p = y(2)
	r_a = y(3)
	r_s = y(4)
	E_e = y(5)
	x_e = y(6)
	I_e = y(7)
	y_e = y(8)
	E_p = y(9)
	x_p = y(10)
	I_p = y(11)
	y_p = y(12)
	E_a = y(13)
	x_a = y(14)
	I_a = y(15)
	y_a = y(16)
	r_xe = y(24)
	r_ep = y(28)
	r_xp = y(30)
	r_xa = y(32)
	r_ee = y(34)

	! calculate right-hand side update of equation system

    ! 1. population updates

	! STN
	y_delta(1) = -r_e / tau_e + (1.0/(SQRT(2.0)*PI*tau_e*tau_e))
     & * SQRT(eta_e + (E_e-I_e)*tau_e + SQRT((eta_e+(E_e-I_e)*tau_e)**2
     & + delta_e*delta_e))

	! GPe-p
	y_delta(2) = -r_p / tau_p + (1.0/(SQRT(2.0)*PI*tau_p*tau_p))
     & * SQRT(eta_p + (E_p-I_p)*tau_p + SQRT((eta_p+(E_p-I_p)*tau_p)**2
     & + delta_p*delta_p))

    ! GPe-a
	y_delta(3) = -r_a / tau_a + (1.0/(SQRT(2.0)*PI*tau_a*tau_a))
     & * SQRT(eta_a + (E_a-I_a)*tau_a + SQRT((eta_a+(E_a-I_a)*tau_a)**2
     & + delta_a*delta_a))

	! STR
	y_delta(4) = (eta_s - r_s) / tau_s

	! 2. synapse dynamics

	! at STN
	y_delta(5) = x_e
	y_delta(6) = (k_ee*r_ee - x_e*(tau_ampa_r+tau_ampa_d) - E_e)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(7) = y_e
	y_delta(8) = (k_ep*r_ep - y_e*(tau_gabaa_r2+tau_gabaa_d2) - I_e)
     & / (tau_gabaa_r2*tau_gabaa_d2)

	! at GPe-p
	y_delta(9) = x_p
	y_delta(10) = (k_pe*r_xe - x_p*(tau_ampa_r+tau_ampa_d) - E_p)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(11) = y_p
	y_delta(12) = (k_pp*r_xp + k_pa*r_xa + k_ps*r_s
     & - y_p*(tau_gabaa_r+tau_gabaa_d) - I_p)/(tau_gabaa_r*tau_gabaa_d)

	! at GPe-a
	y_delta(13) = x_a
	y_delta(14) = (k_ae*r_xe - x_a*(tau_ampa_r+tau_ampa_d) - E_a)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(15) = y_a
	y_delta(16) = (k_ap*r_xp + k_aa*r_xa + k_as*r_s
     & - y_a*(tau_gabaa_r+tau_gabaa_d) - I_a)/(tau_gabaa_r*tau_gabaa_d)

	! STN to both GPe
	y_delta(17) = k_pe_d * (r_e - y(17))
	y_delta(18) = k_pe_d * (y(17) - y(18))
	y_delta(19) = k_pe_d * (y(18) - y(19))
	y_delta(20) = k_pe_d * (y(19) - y(20))
	y_delta(21) = k_pe_d * (y(20) - y(21))
	y_delta(22) = k_pe_d * (y(21) - y(22))
	y_delta(23) = k_pe_d * (y(22) - y(23))
	y_delta(24) = k_pe_d * (y(23) - y(24))

	! GPe-p to STN
	y_delta(25) = k_ep_d * (r_p - y(25))
	y_delta(26) = k_ep_d * (y(25) - y(26))
	y_delta(27) = k_ep_d * (y(26) - y(27))
	y_delta(28) = k_ep_d * (y(27) - y(28))

	! Gpe-p to both GPes
	y_delta(29) = k_p_d * (r_p - y(29))
	y_delta(30) = k_p_d * (y(29) - y(30))

	! Gpe-a to both GPes
	y_delta(31) = k_a_d * (r_a - y(31))
	y_delta(32) = k_a_d * (y(31) - y(32))

	! STN to STN
	y_delta(33) = k_ee_d * (r_e - y(33))
	y_delta(34) = k_ee_d * (y(33) - y(34))

	end subroutine func


	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision eta_e, eta_p, eta_a, eta_s
	double precision k_ee, k_pe, k_ae
	double precision k_ep, k_pp, k_ap
	double precision k_pa, k_aa
	double precision k_ps, k_as
	double precision delta_e, delta_p, delta_a
	double precision k_gp, k_p, k_i, k_stn

	k_gp = 3.0
	k_p = 1.5
	k_i = 1.0
	k_stn = 1.0

	delta_e = 3.0
	delta_p = 9.0
	delta_a = 12.0

	eta_e = 3.0
	eta_p = 0.0
	eta_a = 0.0
	eta_s = 0.002

	k_ee = 1.0
	k_pe = 5.0
	k_ae = 0.0
	k_ep = 10.0
	k_pp = 1.0
	k_ap = 1.0
	k_pa = 1.0
	k_aa = 1.0
	k_ps = 20.0
	k_as = 20.0

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
	args(16) = k_ps
	args(17) = k_as
	args(18) = delta_e
	args(19) = delta_p
	args(20) = delta_a
	args(21) = eta_s
	args(22) = k_gp
	args(23) = k_p
	args(24) = k_i
	args(25) = k_stn

	y(1) = 0.02
	y(2) = 0.06
	y(3) = 0.03
	y(4) = 0.002

	end subroutine stpnt

	subroutine bcnd
	end subroutine bcnd

	subroutine icnd
	end subroutine icnd

	subroutine fopt
	end subroutine fopt

	subroutine pvls
	end subroutine pvls
