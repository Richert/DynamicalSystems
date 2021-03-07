
	subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)
	implicit none
	integer, intent(in) :: ndim, icp(*), ijac
	double precision, intent(in) :: y(ndim), args(*)
	double precision, intent(out) :: y_delta(ndim)
	double precision, intent(inout) :: dfdu(ndim,ndim),
     & dfdp(ndim,*)
	double precision r_p, v_p, r_a, v_a, r_s, r_e
	double precision r_xp, r_xa
	double precision E_p, x_p, I_p, y_p
	double precision E_a, x_a, I_a, y_a
	double precision eta_p, eta_a, eta_s, eta_e
	double precision k_pe, k_ae, k_pp, k_ap, k_pa, k_aa, k_ps, k_as
	double precision delta_p, delta_a
	double precision tau_e, tau_p, tau_a, tau_s
	double precision tau_ampa_r, tau_ampa_d, tau_gabaa_r, tau_gabaa_d
	double precision PI, k_gp, k, k_d

	! declare parameters
	eta_e = args(1)
	eta_p = args(2)
	eta_a = args(3)
	k_pe = args(4)
	k_ae = args(5)
	k_pp = args(6)
	k_ap = args(7)
	k_pa = args(8)
	k_aa = args(9)
	k_ps = args(10)
	k_as = args(15)
	delta_p = args(16)
	delta_a = args(17)
	eta_s = args(18)
	k_gp = args(19)

	! declare constants
	tau_e = 13.00
	tau_p = 18.00
	tau_a = 32.00
	tau_s = 20.00
	tau_ampa_r = 0.80
	tau_ampa_d = 3.70
	tau_gabaa_r = 0.50
	tau_gabaa_d = 5.00
	k_d = 3.00
	PI = 3.141592653589793
	k = 10.00

	k_pe = k_pe*k
	k_ae = k_ae*k
	k_ap = k_ap*k_gp*k
	k_pa = k_pa*k_gp*k
	k_pp = k_pp*k_gp*k
	k_aa = k_aa*k_gp*k
	k_ps = k_ps*k
	k_as = k_as*k

	! extract state variables from input vector
	r_e = y(1)
	r_p = y(2)
	v_p = y(3)
	r_a = y(4)
	v_a = y(5)
	r_s = y(6)
	E_p = y(7)
	x_p = y(8)
	I_p = y(9)
	y_p = y(10)
	E_a = y(11)
	x_a = y(12)
	I_a = y(13)
	y_a = y(14)
	r_xp = y(17)
	r_xa = y(20)

	! calculate right-hand side update of equation system

    ! 1. population updates

	! STN
	y_delta(1) = (eta_e - r_e) / tau_e

	! GPe-p
	y_delta(2) = delta_p / (PI*tau_p**2) + (2.0*r_p*v_p) / tau_p
	y_delta(3) = (v_p**2 + eta_p + (E_p-I_p)*tau_p
     & - (tau_p*PI*r_p)**2) / tau_p

    ! GPe-a
	y_delta(4) = delta_a / (PI*tau_a**2) + (2.0*r_a*v_a) / tau_a
	y_delta(5) = (v_a**2 + eta_a + (E_a-I_a)*tau_a
     & - (tau_a*PI*r_a)**2) / tau_a

	! STR
	y_delta(6) = (eta_s - r_s) / tau_s

	! 2. synapse dynamics

	! at GPe-p
	y_delta(7) = x_p
	y_delta(8) = (k_pe*r_e - x_p*(tau_ampa_r+tau_ampa_d) - E_p)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(9) = y_p
	y_delta(10) = (k_pp*r_xp + k_pa*r_xa + k_ps*r_s
     & - y_p*(tau_gabaa_r+tau_gabaa_d) - I_p)/(tau_gabaa_r*tau_gabaa_d)

	! at GPe-a
	y_delta(11) = x_a
	y_delta(12) = (k_ae*r_e - x_a*(tau_ampa_r+tau_ampa_d) - E_a)
     & / (tau_ampa_r*tau_ampa_d)
	y_delta(13) = y_a
	y_delta(14) = (k_ap*r_xp + k_aa*r_xa + k_as*r_s
     & - y_a*(tau_gabaa_r+tau_gabaa_d) - I_a)/(tau_gabaa_r*tau_gabaa_d)

	! Gpe-p to both GPes
	y_delta(15) = k_d * (r_p - y(15))
	y_delta(16) = k_d * (y(15) - y(16))
	y_delta(17) = k_d * (y(16) - y(17))

	! Gpe-a to both GPes
	y_delta(18) = k_d * (r_a - y(18))
	y_delta(19) = k_d * (y(18) - y(19))
	y_delta(20) = k_d * (y(19) - y(20))

	end subroutine func


	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision eta_e, eta_p, eta_a, eta_s
	double precision k_pe, k_ae
	double precision k_pp, k_ap
	double precision k_pa, k_aa
	double precision k_ps, k_as
	double precision delta_p, delta_a
	double precision k_gp

	k_gp = 1.0

	delta_p = 10.0
	delta_a = 3.0

	eta_e = 0.02
	eta_p = 0.0
	eta_a = 0.0
	eta_s = 0.002

	k_pe = 5.0
	k_ae = 1.5
	k_pp = 0.0
	k_ap = 0.0
	k_pa = 0.0
	k_aa = 0.0
	k_ps = 10.0
	k_as = 2.0

	args(1) = eta_e
	args(2) = eta_p
	args(3) = eta_a
	args(4) = k_pe
	args(5) = k_ae
	args(6) = k_pp
	args(7) = k_ap
	args(8) = k_pa
	args(9) = k_aa
	args(10) = k_ps
	args(15) = k_as
	args(16) = delta_p
	args(17) = delta_a
	args(18) = eta_s
	args(19) = k_gp

	y(2) = 0.06
	y(4) = 0.01
	y(6) = 0.002
	y(1) = 0.02
	y(3) = -2.0
	y(5) = -4.0

	end subroutine stpnt

	subroutine bcnd
	end subroutine bcnd

	subroutine icnd
	end subroutine icnd

	subroutine fopt
	end subroutine fopt

	subroutine pvls
	end subroutine pvls
