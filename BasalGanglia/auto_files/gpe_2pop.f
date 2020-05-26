
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
	double precision k_p_d, k_a_d
	double precision PI, k_gp, k_i, k_p, k_po

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
	k_p = args(20)
	k_i = args(21)
	k_po = args(22)

	! declare constants
	tau_e = 13.0
	tau_p = 25.0
	tau_a = 20.0
	tau_s = 20.0
	tau_ampa_r = 0.8
	tau_ampa_d = 3.7
	tau_gabaa_r = 0.5
	tau_gabaa_d = 5.0
	k_p_d = 1.33
	k_a_d = 1.33
	PI = 3.141592653589793

	delta_p = delta_p*tau_p*tau_p
	delta_a = delta_a*tau_a*tau_a

	eta_p = eta_p*delta_p
	eta_a = eta_a*delta_a

	k_pe = k_pe*sqrt(delta_p)
	k_ae = k_ae*sqrt(delta_a)
	k_pp = k_pp*sqrt(delta_p)*k_gp*k_p*k_po/k_i
	k_ap = k_ap*sqrt(delta_a)*k_gp*k_i*k_p*k_po
	k_pa = k_pa*sqrt(delta_p)*k_gp*k_i/k_p
	k_aa = k_aa*sqrt(delta_a)*k_gp/(k_i*k_p)
	k_ps = k_ps*sqrt(delta_p)
	k_as = k_as*sqrt(delta_a)

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
	r_xp = y(16)
	r_xa = y(18)

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
	y_delta(15) = k_p_d * (r_p - y(15))
	y_delta(16) = k_p_d * (y(15) - y(16))

	! Gpe-a to both GPes
	y_delta(17) = k_a_d * (r_a - y(17))
	y_delta(18) = k_a_d * (y(17) - y(18))

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
	double precision k_gp, k_p, k_i, k_po

	k_gp = 1.0
	k_p = 1.0
	k_i = 1.0
	k_po = 1.0

	delta_p = 0.1
	delta_a = 0.2

	eta_e = 0.02
	eta_p = 0.0
	eta_a = 0.0
	eta_s = 0.002

	k_pe = 100
	k_ae = 100
	k_pp = 1
	k_ap = 1
	k_pa = 1
	k_aa = 1
	k_ps = 200
	k_as = 200

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
	args(20) = k_p
	args(21) = k_i
	args(22) = k_po

	y(2) = 0.06
	y(4) = 0.03
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
