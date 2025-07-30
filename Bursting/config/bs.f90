module bs

double precision :: PI = 4.0*atan(1.0)

contains


subroutine bs_run(t,y,dy,v_t,v_r,k,g,Delta,C,E_r,I_ext,b,tau_u,kappa,tau_s,tau_x)

implicit none

double precision, intent(in) :: t
double precision, intent(in) :: y(6)
double precision :: r
double precision :: v
double precision :: u
double precision :: w
double precision :: x
double precision :: s
double precision, intent(inout) :: dy(6)
double precision, intent(in) :: v_t
double precision, intent(in) :: v_r
double precision, intent(in) :: k
double precision, intent(in) :: g
double precision, intent(in) :: Delta
double precision, intent(in) :: C
double precision, intent(in) :: E_r
double precision, intent(in) :: I_ext
double precision, intent(in) :: b
double precision, intent(in) :: tau_u
double precision, intent(in) :: kappa
double precision, intent(in) :: tau_s
double precision, intent(in) :: tau_x

r = y(1)
v = y(2)
u = y(3)
w = y(4)
x = y(5)
s = y(6)

dy(1) = (W*k/(pi*C) + r*(k*(2.0*v-v_r-v_t) - g*s)) / C
dy(2) = (k*(v-v_r)*(v-v_t) + I_ext + g*s*(E_r-v) - u &
     & - pi*C*r*pi*C*r/k)/C
dy(3) = (b*(v - v_r) - u)/tau_u + kappa*x
dy(4) = (Delta*abs(v-v_r) - w) / tau_u
dy(5) = (r-x)/tau_x
dy(6) = -s/tau_s + r

end subroutine


end module


subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)

use bs
implicit none
integer, intent(in) :: ndim, icp(*), ijac
double precision, intent(in) :: y(ndim), args(*)
double precision, intent(out) :: dy(ndim)
double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)

call bs_run(args(14), y, dy, args(1), args(2), args(3), args(4), &
     & args(5), args(6), args(7), args(8), args(9), args(15), args(16),&
     &  args(17), args(18))

end subroutine func


subroutine stpnt(ndim, y, args, t)

implicit None
integer, intent(in) :: ndim
double precision, intent(inout) :: y(ndim), args(*)
double precision, intent(in) :: t

args(1) = -40.0  ! v_t
args(2) = -60.0  ! v_r
args(3) = 0.7  ! k
args(4) = 1.0  ! g
args(5) = 0.2  ! Delta
args(6) = 100.0  ! C
args(7) = 0.0  ! E_r
args(8) = 0.0  ! I_ext
args(9) = -3.0  ! b
args(15) = 50.0  ! tau_u
args(16) = 0.0  ! kappa
args(17) = 6.0  ! tau_s
args(18) = 300.0  ! tau_x

y(1) = 0.0  ! r
y(2) = -60.0  ! v
y(3) = 0.0  ! u
y(4) = 0.0  ! w
y(4) = 0.01  ! x
y(5) = 0.0  ! s

end subroutine stpnt



subroutine bcnd
end subroutine bcnd


subroutine icnd
end subroutine icnd


subroutine fopt
end subroutine fopt


subroutine pvls
end subroutine pvls
