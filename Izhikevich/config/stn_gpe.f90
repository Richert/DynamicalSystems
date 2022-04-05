module stn_gpe

double precision :: PI = 4.0*atan(1.0)

contains


subroutine stn_gpe_run(t,y,dy,v_t,v_r,k,g,q,Delta,C,v_z,v_p,E_r,I_ext,&
     & b,a,d,tau_s,v_t_0,v_r_0,k_0,g_gabaa,g_ampa,q_0,Delta_0,C_0,&
     & E_gabaa,E_ampa,v_z_0,v_p_0,I_ext_0,b_0,a_0,d_0,tau_ampa,&
     & tau_gabaa,weight,weight_0,weight_1)

implicit none

double precision, intent(in) :: t
double precision, intent(in) :: y(9)
double precision :: r_0
double precision :: v
double precision :: u
double precision :: s
double precision :: r
double precision :: v_0
double precision :: u_0
double precision :: s_ampa
double precision :: s_gabaa
double precision :: r_in
double precision :: r_e
double precision :: r_i
double precision, intent(inout) :: dy(9)
double precision, intent(in) :: v_t
double precision, intent(in) :: v_r
double precision, intent(in) :: k
double precision, intent(in) :: g
double precision, intent(in) :: q
double precision, intent(in) :: Delta
double precision, intent(in) :: C
double precision, intent(in) :: v_z
double precision, intent(in) :: v_p
double precision, intent(in) :: E_r
double precision, intent(in) :: I_ext
double precision, intent(in) :: b
double precision, intent(in) :: a
double precision, intent(in) :: d
double precision, intent(in) :: tau_s
double precision, intent(in) :: v_t_0
double precision, intent(in) :: v_r_0
double precision, intent(in) :: k_0
double precision, intent(in) :: g_gabaa
double precision, intent(in) :: g_ampa
double precision, intent(in) :: q_0
double precision, intent(in) :: Delta_0
double precision, intent(in) :: C_0
double precision, intent(in) :: E_gabaa
double precision, intent(in) :: E_ampa
double precision, intent(in) :: v_z_0
double precision, intent(in) :: v_p_0
double precision, intent(in) :: I_ext_0
double precision, intent(in) :: b_0
double precision, intent(in) :: a_0
double precision, intent(in) :: d_0
double precision, intent(in) :: tau_ampa
double precision, intent(in) :: tau_gabaa
double precision, intent(in) :: weight
double precision, intent(in) :: weight_0
double precision, intent(in) :: weight_1

r_0 = y(1)
v = y(2)
u = y(3)
s = y(4)
r = y(5)
v_0 = y(6)
u_0 = y(7)
s_ampa = y(8)
s_gabaa = y(9)

r_in = r*weight
r_e = r_0*weight_0
r_i = r*weight_1

dy(1) = (r_0*(-g*s + k*(2.0*v - v_r - v_t) - q) + Delta*k**2*(v - v_r)&
     & /(pi*C))/C
dy(2) = (-pi*C*r_0*(pi*C*r_0/k &
     & + Delta) + C*q*r_0*log(v_p/v_z)/k + I_ext + g*s*(E_r &
     & - v) + k*v*(v - v_r - v_t) + k*v_r*v_t - u)/C
dy(3) = a*(b*(v - v_r) - u) + d*r_0
dy(4) = r_in - s/tau_s
dy(5) = (r*(-g_ampa*s_ampa - g_gabaa*s_gabaa &
     & + k_0*(2.0*v_0 - v_r_0 - v_t_0) - q_0) + 10.0*Delta_0*k_0**2&
     & /(pi*C_0))/C_0
dy(6) = (-pi**2*C_0**2*r**2/k_0 &
     & + C_0*q_0*r*log(v_p_0&
     & /v_z_0)/k_0 + I_ext_0 + g_ampa*s_ampa*(E_ampa &
     & - v_0) + g_gabaa&
     & *s_gabaa*(E_gabaa - v_0) &
     & + k_0*v_0*(v_0 - v_r_0 - v_t_0) + k_0*v_r_0*v_t_0 - u_0)/C_0
dy(7) = a_0*(b_0*(v_0 - v_r_0) - u_0) + d_0*r
dy(8) = r_e - s_ampa/tau_ampa
dy(9) = r_i - s_gabaa/tau_gabaa

end subroutine


end module


subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)

use stn_gpe
implicit none
integer, intent(in) :: ndim, icp(*), ijac
double precision, intent(in) :: y(ndim), args(*)
double precision, intent(out) :: dy(ndim)
double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)

call stn_gpe_run(args(14), y, dy, args(1), args(2), args(3), args(4), &
     & args(5), args(6), args(7), args(8), args(9), args(15), args(16),&
     &  args(17), args(18), args(19), args(20), args(21), args(22), &
     & args(23), args(24), args(25), args(26), args(27), args(28), &
     & args(29), args(30), args(31), args(32), args(33), args(34), &
     & args(35), args(36), args(37), args(38), args(39), args(40), &
     & args(41))

end subroutine func


subroutine stpnt(ndim, y, args, t)

implicit None
integer, intent(in) :: ndim
double precision, intent(inout) :: y(ndim), args(*)
double precision, intent(in) :: t

args(1) = -40.0  ! v_t
args(2) = -55.0  ! v_r
args(3) = 0.45  ! k
args(4) = 0.5  ! g
args(5) = 0.0  ! q
args(6) = 1.0  ! Delta
args(7) = 25.0  ! C
args(8) = 50.0  ! v_z
args(9) = 15.0  ! v_p
args(15) = -65.0  ! E_r
args(16) = 0.0  ! I_ext
args(17) = 4.0  ! b
args(18) = 0.02  ! a
args(19) = 15.0  ! d
args(20) = 8.0  ! tau_s
args(21) = -45.0  ! v_t_0
args(22) = -55.0  ! v_r_0
args(23) = 0.95  ! k_0
args(24) = 0.8  ! g_gabaa
args(25) = 1.5  ! g_ampa
args(26) = 0.0  ! q_0
args(27) = 2.0  ! Delta_0
args(28) = 70.0  ! C_0
args(29) = -65.0  ! E_gabaa
args(30) = 0.0  ! E_ampa
args(31) = 60.0  ! v_z_0
args(32) = 25.0  ! v_p_0
args(33) = 0.0  ! I_ext_0
args(34) = 4.0  ! b_0
args(35) = 0.005  ! a_0
args(36) = 0.4  ! d_0
args(37) = 2.0  ! tau_ampa
args(38) = 5.0  ! tau_gabaa
args(39) = 15.0  ! weight
args(40) = 15.0  ! weight_0
args(41) = 15.0  ! weight_1
y(1) = 0.0  ! r_0
y(2) = -60.0  ! v
y(3) = 0.0  ! u
y(4) = 0.0  ! s
y(5) = 0.0  ! r
y(6) = -60.0  ! v_0
y(7) = 0.0  ! u_0
y(8) = 0.0  ! s_ampa
y(9) = 0.0  ! s_gabaa

end subroutine stpnt



subroutine bcnd
end subroutine bcnd


subroutine icnd
end subroutine icnd


subroutine fopt
end subroutine fopt


subroutine pvls
end subroutine pvls
