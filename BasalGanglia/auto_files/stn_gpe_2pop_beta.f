
	subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)
	implicit none
	integer, intent(in) :: ndim, icp(*), ijac
	double precision, intent(in) :: y(ndim), args(*)
	double precision, intent(out) :: y_delta(ndim)
	double precision, intent(inout) :: dfdu(ndim,ndim),
     & dfdp(ndim,*)
	double precision r_e, v_e, r_p, v_p, r_s
	double precision E_e, x_e, I_e, y_e
	double precision E_p, x_p, I_p, y_p
	double precision eta_e, eta_p, eta_s, eta
	double precision k_ee, k_pe, k_ae
	double precision k_ep, k_pp
	double precision k_ps, k, k_d
	double precision r_ee, r_pe, r_pp, r_ep
	double precision delta_e, delta_p, delta
	double precision tau_e, tau_p, tau_s
	double precision tau_ampa_r, tau_ampa_d, tau_gabaa_r, tau_gabaa_d
	double precision PI, k_gp, k_p, k_i, tau_stn

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
	eta_s = args(10)
	k_gp = args(15)
	k_p = args(16)
	k_i = args(17)
	tau_stn = args(18)
	tau_e = args(19)
	tau_p = args(20)
	tau_ampa_r = args(21)
	tau_ampa_d = args(22)
	tau_gabaa_r = args(23)
	tau_gabaa_d = args(24)
	eta = args(25)
	delta = args(26)
	k = args(27)

	! declare constants
	tau_s = 1.0
	PI = 3.141592653589793
	k_d = 3.0

	delta_e = delta_e*delta
	delta_p = delta_p*delta

	eta_e = eta_e*eta
	eta_p = eta_p*eta

	k_ee = k_ee*k
	k_pe = k_pe*k
	k_ep = k_ep*k
	k_pp = k_pp*k*k_gp*k_p/k_i
	k_ps = k_ps*k

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
	r_ee = y(16)
	r_pe = y(22)
	r_pp = y(25)
	r_ep = y(31)

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
	y_delta(9) = (k_ep*r_ep - y_e*tau_stn*(tau_gabaa_r+tau_gabaa_d)
     & - I_e) / (tau_gabaa_r*tau_gabaa_d*tau_stn*tau_stn)

	! at GPe-p
	y_delta(10) = x_p
	y_delta(11) = (k_pe*r_pe - x_p*(tau_ampa_r+tau_ampa_d) - E_p)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(12) = y_p
	y_delta(13) = (k_pp*r_pp + k_ps*r_s
     & - y_p*(tau_gabaa_r+tau_gabaa_d) - I_p)/(tau_gabaa_r*tau_gabaa_d)

	! STN output
	y_delta(14) = k_d * (r_e - y(14))
	y_delta(15) = k_d * (y(14) - y(15))
	y_delta(16) = k_d * (y(15) - y(16))
	y_delta(17) = k_d * (y(16) - y(17))
	y_delta(18) = k_d * (y(17) - y(18))
	y_delta(19) = k_d * (y(18) - y(19))
	y_delta(20) = k_d * (y(19) - y(20))
	y_delta(21) = k_d * (y(20) - y(21))
	y_delta(22) = k_d * (y(21) - y(22))

	! GPe-p output
	y_delta(23) = k_d * (r_p - y(23))
	y_delta(24) = k_d * (y(23) - y(24))
	y_delta(25) = k_d * (y(24) - y(25))
	y_delta(26) = k_d * (y(25) - y(26))
	y_delta(27) = k_d * (y(26) - y(27))
	y_delta(28) = k_d * (y(27) - y(28))
	y_delta(29) = k_d * (y(28) - y(29))
	y_delta(30) = k_d * (y(29) - y(30))
	y_delta(31) = k_d * (y(30) - y(31))

	end subroutine func


	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision eta_e, eta_p, eta_s
	double precision k_ee, k_pe
	double precision k_ep, k_pp
	double precision k_ps, delta, eta, k
	double precision delta_e, delta_p
	double precision tau_e, tau_p
	double precision tau_ampa_r, tau_ampa_d, tau_gabaa_r, tau_gabaa_d
	double precision k_gp, k_p, k_i, tau_stn

	k_gp = 3.0
	k_p = 2.0
	k_i = 1.5

	delta = 39.2
	delta_e = 0.3
	delta_p = 0.9

	eta = 20.0
	eta_e = -20.0
	eta_p = -20.0
	eta_s = 0.002

	k = 21.2
	k_ee = 0.8
	k_pe = 8.0
	k_ep = 3.0
	k_pp = 0.6
	k_ps = 20.0

	tau_e = 19.9
	tau_p = 37.1

	tau_ampa_r = 0.7
	tau_ampa_d = 5.6
	tau_gabaa_r = 1.3
	tau_gabaa_d = 15.8

	tau_stn = 1.5

	args(1) = eta_e
	args(2) = eta_p
	args(3) = k_ee
	args(4) = k_pe
	args(5) = k_ep
	args(6) = k_pp
	args(7) = k_ps
	args(8) = delta_e
	args(9) = delta_p
	args(10) = eta_s
	args(15) = k_gp
	args(16) = k_p
	args(17) = k_i
	args(18) = tau_stn
	args(19) = tau_e
	args(20) = tau_p
	args(21) = tau_ampa_r
	args(22) = tau_ampa_d
	args(23) = tau_gabaa_r
	args(24) = tau_gabaa_d
	args(25) = eta
	args(26) = delta
	args(27) = k

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