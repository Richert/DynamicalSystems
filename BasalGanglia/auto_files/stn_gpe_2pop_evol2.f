
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
	double precision k_ee, k_pe
	double precision k_ep, k_pp
	double precision k, k_gp, k_ps
	double precision delta_e, delta_p, delta
	double precision tau_e, tau_p, tau_s
	double precision tau_ampa_r, tau_ampa_d, tau_gabaa_r, tau_gabaa_d
	double precision PI, tau_stn

	! declare parameters
	tau_e = args(1)
	tau_p = args(2)
	tau_ampa_r = args(3)
	tau_ampa_d = args(4)
	tau_gabaa_r = args(5)
	tau_gabaa_d = args(6)
	tau_stn = args(7)
	eta = args(8)
	delta = args(9)
	k = args(10)
	eta_e = args(15)
	eta_p = args(16)
	k_pe = args(17)
	k_ep = args(18)
	k_pp = args(19)
	k_gp = args(20)

	! declare constants
	tau_s = 1.0
	PI = 3.141592653589793
	eta_s = 0.002

	delta_e = 0.3*delta
	delta_p = 0.9*delta

	eta_e = eta_e*eta
	eta_p = eta_p*eta

	k_ee = 0.8*k
	k_pe = k_pe*k
	k_ep = k_ep*k
	k_pp = k_pp*k*k_gp
	k_ps = 20.0*k

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
	y_delta(7) = (k_ee*r_e - x_e*(tau_ampa_r+tau_ampa_d) - E_e)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(8) = y_e
	y_delta(9) = (k_ep*r_p - y_e*tau_stn*(tau_gabaa_r+tau_gabaa_d)
     & - I_e) / (tau_gabaa_r*tau_gabaa_d*tau_stn*tau_stn)

	! at GPe-p
	y_delta(10) = x_p
	y_delta(11) = (k_pe*r_e - x_p*(tau_ampa_r+tau_ampa_d) - E_p)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(12) = y_p
	y_delta(13) = (k_pp*r_p + k_ps*r_s
     & - y_p*(tau_gabaa_r+tau_gabaa_d) - I_p)/(tau_gabaa_r*tau_gabaa_d)

	end subroutine func


	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision eta_e, eta_p, eta
	double precision k_pe, k_ep, k_pp
	double precision k_ps, k_gp, k
	double precision delta_e, delta_p, delta
	double precision tau_e, tau_p, tau_s, tau_stn
	double precision tau_ampa_r, tau_ampa_d, tau_gabaa_r, tau_gabaa_d

	tau_e = 18.79
	tau_p = 45.86

	tau_ampa_r = 1.11
	tau_ampa_d = 8.79
	tau_gabaa_r = 1.65
	tau_gabaa_d = 13.62
	tau_stn = 2.97

	eta = 724.67
	delta = 329.6
	k = 999.93
	k_gp = 3.0

	eta_e = -10.0
	eta_p = -10.0

	k_pe = 7.15
	k_ep = 7.66
	k_pp = 3.34/k_gp

	args(1) = tau_e
	args(2) = tau_p
	args(3) = tau_ampa_r
	args(4) = tau_ampa_d
	args(5) = tau_gabaa_r
	args(6) = tau_gabaa_d
	args(7) = tau_stn
	args(8) = eta
	args(9) = delta
	args(10) = k
	args(15) = eta_e
	args(16) = eta_p
	args(17) = k_pe
	args(18) = k_ep
	args(19) = k_pp
	args(20) = k_gp

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
