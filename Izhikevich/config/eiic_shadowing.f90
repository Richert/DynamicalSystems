module eiic_shadowing

double precision :: PI = 4.0*atan(1.0)

contains


subroutine eiic_run(t,y,dy,v_t,v_r,k,g_gabaa,g_ampa,Delta,C,E_gabaa,&
     & E_ampa,I_ext,b,a,d,tau_ampa,tau_gabaa,v_t_0,v_r_0,k_0,g_gabaa_0,&
     & g_ampa_0,Delta_0,C_0,E_gabaa_0,E_ampa_0,I_ext_0,b_0,a_0,d_0,&
     & tau_ampa_0,tau_gabaa_0,v_t_1,v_r_1,k_1,g_gabaa_1,g_ampa_1,&
     & Delta_1,C_1,E_gabaa_1,E_ampa_1,I_ext_1,b_1,a_1,d_1,tau_ampa_1,&
     & tau_gabaa_1,weight,weight_in0,weight_in1,weight_0,weight_in0_0,&
     & weight_in1_0,weight_1,weight_2)

implicit none

double precision, intent(in) :: t
double precision, intent(in) :: y(15)
double precision :: r
double precision :: v
double precision :: u
double precision :: s_ampa
double precision :: s_gabaa
double precision :: r_in0
double precision :: v_0
double precision :: u_0
double precision :: s_ampa_0
double precision :: s_gabaa_0
double precision :: r_in1
double precision :: v_1
double precision :: u_1
double precision :: s_ampa_1
double precision :: s_gabaa_1
double precision :: r_e
double precision :: r_i_in0
double precision :: r_i_in1
double precision :: r_i
double precision :: r_e_0
double precision :: r_i_in0_0
double precision :: r_i_in1_0
double precision :: r_i_0
double precision :: r_e_1
double precision :: r_i_1
double precision, intent(inout) :: dy(15)
double precision, intent(in) :: v_t
double precision, intent(in) :: v_r
double precision, intent(in) :: k
double precision, intent(in) :: g_gabaa
double precision, intent(in) :: g_ampa
double precision, intent(in) :: Delta
double precision, intent(in) :: C
double precision, intent(in) :: E_gabaa
double precision, intent(in) :: E_ampa
double precision, intent(in) :: I_ext
double precision, intent(in) :: b
double precision, intent(in) :: a
double precision, intent(in) :: d
double precision, intent(in) :: tau_ampa
double precision, intent(in) :: tau_gabaa
double precision, intent(in) :: v_t_0
double precision, intent(in) :: v_r_0
double precision, intent(in) :: k_0
double precision, intent(in) :: g_gabaa_0
double precision, intent(in) :: g_ampa_0
double precision, intent(in) :: Delta_0
double precision, intent(in) :: C_0
double precision, intent(in) :: E_gabaa_0
double precision, intent(in) :: E_ampa_0
double precision, intent(in) :: I_ext_0
double precision, intent(in) :: b_0
double precision, intent(in) :: a_0
double precision, intent(in) :: d_0
double precision, intent(in) :: tau_ampa_0
double precision, intent(in) :: tau_gabaa_0
double precision, intent(in) :: v_t_1
double precision, intent(in) :: v_r_1
double precision, intent(in) :: k_1
double precision, intent(in) :: g_gabaa_1
double precision, intent(in) :: g_ampa_1
double precision, intent(in) :: Delta_1
double precision, intent(in) :: C_1
double precision, intent(in) :: E_gabaa_1
double precision, intent(in) :: E_ampa_1
double precision, intent(in) :: I_ext_1
double precision, intent(in) :: b_1
double precision, intent(in) :: a_1
double precision, intent(in) :: d_1
double precision, intent(in) :: tau_ampa_1
double precision, intent(in) :: tau_gabaa_1
double precision, intent(in) :: weight
double precision, intent(in) :: weight_in0
double precision, intent(in) :: weight_in1
double precision, intent(in) :: weight_0
double precision, intent(in) :: weight_in0_0
double precision, intent(in) :: weight_in1_0
double precision, intent(in) :: weight_1
double precision, intent(in) :: weight_2


r = y(1)
v = y(2)
u = y(3)
s_ampa = y(4)
s_gabaa = y(5)
r_in0 = y(6)
v_0 = y(7)
u_0 = y(8)
s_ampa_0 = y(9)
s_gabaa_0 = y(10)
r_in1 = y(11)
v_1 = y(12)
u_1 = y(13)
s_ampa_1 = y(14)
s_gabaa_1 = y(15)
r_e = r*weight
r_i_in0 = r_in0*weight_in0
r_i_in1 = r_in1*weight_in1
r_i = r_i_in0 + r_i_in1
r_e_0 = r*weight_0
r_i_in0_0 = r_in0*weight_in0_0
r_i_in1_0 = r_in1*weight_in1_0
r_i_0 = r_i_in0_0 + r_i_in1_0
r_e_1 = r*weight_1
r_i_1 = r_in0*weight_2

dy(1) = (r*(-g_ampa*s_ampa - g_gabaa*s_gabaa &
     & + k*(2.0*v - v_r - v_t)) + Delta*k**2*(v - v_r)/(pi*C))/C
dy(2) = (-pi*C*r*(pi*C*r/k &
     & + Delta) + I_ext + g_ampa*s_ampa*(E_ampa &
     & - v) + g_gabaa&
     & *s_gabaa*(E_gabaa - v) + k*v*(v - v_r - v_t) + k*v_r*v_t - u)/C
dy(3) = a*(b*(v - v_r) - u) + d*r
dy(4) = r_e - s_ampa/tau_ampa
dy(5) = r_i - s_gabaa/tau_gabaa
dy(6) = (r_in0*(-g_ampa_0*s_ampa_0 - g_gabaa_0*s_gabaa_0 &
     & + k_0*(2.0*v_0 - v_r_0 - v_t_0)) + Delta_0*k_0**2*(v_0 - v_r_0)&
     & /(pi*C_0))/C_0
dy(7) = (-pi*C_0*r_in0*(pi*C_0*r_in0/k_0 &
     & + Delta_0) + I_ext_0 + g_ampa_0*s_ampa_0*(E_ampa_0 &
     & - v_0) + g_gabaa_0&
     & *s_gabaa_0*(E_gabaa_0 - v_0) &
     & + k_0*v_0*(v_0 - v_r_0 - v_t_0) + k_0*v_r_0*v_t_0 - u_0)/C_0
dy(8) = a_0*(b_0*(v_0 - v_r_0) - u_0) + d_0*r_in0
dy(9) = r_e_0 - s_ampa_0/tau_ampa_0
dy(10) = r_i_0 - s_gabaa_0/tau_gabaa_0
dy(11) = (r_in1*(-g_ampa_1*s_ampa_1 - g_gabaa_1*s_gabaa_1 &
     & + k_1*(2.0*v_1 - v_r_1 - v_t_1)) + Delta_1*k_1**2*(v_1 - v_r_1)&
     & /(pi*C_1))/C_1
dy(12) = (-pi*C_1*r_in1*(pi*C_1*r_in1/k_1 &
     & + Delta_1) + I_ext_1 + g_ampa_1*s_ampa_1*(E_ampa_1 &
     & - v_1) + g_gabaa_1&
     & *s_gabaa_1*(E_gabaa_1 - v_1) &
     & + k_1*v_1*(v_1 - v_r_1 - v_t_1) + k_1*v_r_1*v_t_1 - u_1)/C_1
dy(13) = a_1*(b_1*(v_1 - v_r_1) - u_1) + d_1*r_in1
dy(14) = r_e_1 - s_ampa_1/tau_ampa_1
dy(15) = r_i_1 - s_gabaa_1/tau_gabaa_1

end subroutine


end module


subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)

use eiic_shadowing
implicit none
integer, intent(in) :: ndim, icp(*), ijac
double precision, intent(in) :: y(ndim), args(*)
double precision, intent(out) :: dy(ndim)
double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)

call eiic_run(args(14), y, dy, args(1), args(2), args(3), args(4), &
     & args(5), args(6), args(7), args(8), args(9), args(15), args(16),&
     &  args(17), args(18), args(19), args(20), args(21), args(22), &
     & args(23), args(24), args(25), args(26), args(27), args(28), &
     & args(29), args(30), args(31), args(32), args(33), args(34), &
     & args(35), args(36), args(37), args(38), args(39), args(40), &
     & args(41), args(42), args(43), args(44), args(45), args(46), &
     & args(47), args(48), args(49), args(50), args(51), args(52), &
     & args(53), args(54), args(55), args(56), args(57), args(58))

end subroutine func


subroutine stpnt(ndim, y, args, t)

implicit None
integer, intent(in) :: ndim
double precision, intent(inout) :: y(ndim), args(*)
double precision, intent(in) :: t

args(1) = -40.0  ! v_t
args(2) = -60.0  ! v_r
args(3) = 0.7  ! k
args(4) = 1.0  ! g_gabaa
args(5) = 1.0  ! g_ampa
args(6) = 0.5  ! Delta
args(7) = 100.0  ! C
args(8) = -65.0  ! E_gabaa
args(9) = 0.0  ! E_ampa
args(15) = 0.0  ! I_ext
args(16) = -2.0  ! b
args(17) = 0.03  ! a
args(18) = 100.0  ! d
args(19) = 6.0  ! tau_ampa
args(20) = 8.0  ! tau_gabaa
args(21) = -40.0  ! v_t_0
args(22) = -55.0  ! v_r_0
args(23) = 1.0  ! k_0
args(24) = 1.0  ! g_gabaa_0
args(25) = 1.0  ! g_ampa_0
args(26) = 0.8  ! Delta_0
args(27) = 20.0  ! C_0
args(28) = -65.0  ! E_gabaa_0
args(29) = 0.0  ! E_ampa_0
args(30) = 30.0  ! I_ext_0
args(31) = 0.025  ! b_0
args(32) = 0.2  ! a_0
args(33) = 0.0  ! d_0
args(34) = 6.0  ! tau_ampa_0
args(35) = 8.0  ! tau_gabaa_0
args(36) = -42.0  ! v_t_1
args(37) = -56.0  ! v_r_1
args(38) = 1.0  ! k_1
args(39) = 1.0  ! g_gabaa_1
args(40) = 1.0  ! g_ampa_1
args(41) = 0.5  ! Delta_1
args(42) = 100.0  ! C_1
args(43) = -65.0  ! E_gabaa_1
args(44) = 0.0  ! E_ampa_1
args(45) = 0.0  ! I_ext_1
args(46) = 8.0  ! b_1
args(47) = 0.03  ! a_1
args(48) = 20.0  ! d_1
args(49) = 6.0  ! tau_ampa_1
args(50) = 8.0  ! tau_gabaa_1
args(51) = 16.0  ! weight
args(52) = 16.0  ! weight_in0
args(53) = 16.0  ! weight_in1
args(54) = 4.0  ! weight_0
args(55) = 4.0  ! weight_in0_0
args(56) = 4.0  ! weight_in1_0
args(57) = 4.0  ! weight_1
args(58) = 4.0  ! weight_2
y(1) = 0.02  ! r
y(2) = -45.0  ! v
y(3) = 0.0  ! u
y(4) = 0.0  ! s_ampa
y(5) = 0.0  ! s_gabaa
y(6) = 0.0  ! r_in0
y(7) = -60.0  ! v_0
y(8) = 0.0  ! u_0
y(9) = 0.0  ! s_ampa_0
y(10) = 0.0  ! s_gabaa_0
y(11) = 0.0  ! r_in1
y(12) = -60.0  ! v_1
y(13) = 0.0  ! u_1
y(14) = 0.0  ! s_ampa_1
y(15) = 0.0  ! s_gabaa_1

end subroutine stpnt



subroutine bcnd
end subroutine bcnd


subroutine icnd
end subroutine icnd


subroutine fopt
end subroutine fopt


subroutine pvls
end subroutine pvls