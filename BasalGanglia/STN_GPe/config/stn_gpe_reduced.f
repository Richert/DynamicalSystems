
	subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)
	implicit none
	integer, intent(in) :: ndim, icp(*), ijac
	double precision, intent(in) :: y(ndim), args(*)
	double precision, intent(out) :: y_delta(ndim)
	double precision, intent(inout) :: dfdu(ndim,ndim),
     & dfdp(ndim,*)
	double precision r_e, v_e, r_p, v_p, r_s
	double precision r_pe, r_ep, r_pp, r_ee
	double precision E_e, x_e, I_e, y_e
	double precision E_p, x_p, I_p, y_p
	double precision eta_e, eta_p, eta_s
	double precision k_ee, k_pe, k_ep, k_pp, k_ps
	double precision delta_e, delta_p
	double precision tau_e, tau_p, tau_s
	double precision tau_ampa_r, tau_ampa_d, tau_gabaa_r, tau_gabaa_d
	double precision tau_gabaa_r2, tau_gabaa_d2
	double precision k_ee_d, k_pe_d, k_ep_d, k_pp_d
	double precision PI, k_gp, k_gp_intra, k_gp_inh, d

	! declare parameters
	eta_e = args(1)
	eta_p = args(2)
	k_ee = args(3)
	k_pe = args(4)
	k_ep = args(5)
	k_pp = args(6)
	k_ps = args(7)
	delta_e = args(8)
	delta_p = args(9)
	k_gp = args(10)
	eta_s = args(15)
	k_gp_intra = args(16)
	k_gp_inh = args(17)
	d = args(18)

	! declare constants
	tau_e = 13.0
	tau_p = 25.0
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
	k_pp_d = 1.33
	PI = 3.141592653589793

	delta_e = delta_e*tau_e*tau_e
	delta_p = delta_p*tau_p*tau_p

	eta_e = eta_e*delta_e
	eta_p = eta_p*delta_p

	k_ee = k_ee*sqrt(delta_e)
	k_pe = k_pe*sqrt(delta_p)*k_gp
	k_ep = k_ep*sqrt(delta_e)
	k_pp = k_pp*sqrt(delta_p)*k_gp*k_gp_inh
	k_ps = k_ps*sqrt(delta_p)*k_gp*k_gp_inh

	! extract state variables from input vector
	r_e = y(1)
	v_e = y(2)
	r_p = y(3)
	v_p = y(4)
	r_s = y(5)
	E_e = y(6)
	x_e = y(7)
	I_e = y(8)
	y_e = y(9)
	E_p = y(10)
	x_p = y(11)
	I_p = y(12)
	y_p = y(13)
	r_pe = y(21)
	r_ep = y(25)
	r_pp = y(27)
	r_ee = y(29)

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

	! STR
	y_delta(5) = (eta_s - r_s) / tau_s

	! 2. synapse dynamics

	! at STN
	y_delta(6) = x_e
	y_delta(7) = (k_ee*r_ee - x_e*(tau_ampa_r+tau_ampa_d) - E_e)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(8) = y_e
	y_delta(9) = (k_ep*r_ep - y_e*(tau_gabaa_r2+tau_gabaa_d2) - I_e)
     & / (tau_gabaa_r2*tau_gabaa_d2)

	! at GPe-p
	y_delta(10) = x_p
	y_delta(11) = (k_pe*r_pe - x_p*(tau_ampa_r+tau_ampa_d) - E_p)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(12) = y_p
	y_delta(13) = (k_pp*r_pp + k_ps*r_s
     & - y_p*(tau_gabaa_r+tau_gabaa_d) - I_p)/(tau_gabaa_r*tau_gabaa_d)

	! STN to both GPe
	y_delta(14) = k_pe_d * (r_e - y(14))
	y_delta(15) = k_pe_d * (y(14) - y(15))
	y_delta(16) = k_pe_d * (y(15) - y(16))
	y_delta(17) = k_pe_d * (y(16) - y(17))
	y_delta(18) = k_pe_d * (y(17) - y(18))
	y_delta(19) = k_pe_d * (y(18) - y(19))
	y_delta(20) = k_pe_d * (y(19) - y(20))
	y_delta(21) = k_pe_d * (y(20) - y(21))

	! GPe-p to STN
	y_delta(22) = k_ep_d * (r_p - y(22))
	y_delta(23) = k_ep_d * (y(22) - y(23))
	y_delta(24) = k_ep_d * (y(23) - y(24))
	y_delta(25) = k_ep_d * (y(24) - y(25))

	! Gpe-p to both GPes
	y_delta(26) = k_pp_d * (r_p - y(26))
	y_delta(27) = k_pp_d * (y(26) - y(27))

	! STN to STN
	y_delta(28) = k_ee_d * (r_e - y(28))
	y_delta(29) = k_ee_d * (y(28) - y(29))

	end subroutine func


	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision eta_e, eta_p, eta_s
	double precision k_ee, k_pe, k_ep, k_pp, k_ps
	double precision delta_e, delta_p
	double precision k_gp, k_gp_intra, k_gp_inh, d

	k_gp = 1.0
	k_gp_intra = 1.0
	k_gp_inh = 1.0
	d = 1.0

	delta_e = 0.10
	delta_p = 0.25

	eta_e = 0.8
	eta_p = 0.46
	eta_s = 0.002

	k_ee = 5.9
	k_pe = 158.9
	k_ep = 44.2
	k_pp = 18.5
	k_ps = 116.9

	args(1) = eta_e
	args(2) = eta_p
	args(3) = k_ee
	args(4) = k_pe
	args(5) = k_ep
	args(6) = k_pp
	args(7) = k_ps
	args(8) = delta_e
	args(9) = delta_p
	args(10) = k_gp
	args(15) = eta_s
	args(16) = k_gp_intra
	args(17) = k_gp_inh
	args(18) = d

	y(1) = 0.02
	y(3) = 0.06
	y(2) = -3.0
	y(4) = -1.0

	end subroutine stpnt

	subroutine bcnd
	end subroutine bcnd

	subroutine icnd
	end subroutine icnd

	subroutine fopt
	end subroutine fopt

	subroutine pvls
	end subroutine pvls
