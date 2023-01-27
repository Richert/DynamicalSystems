module izhikevich_biexp

double precision :: PI = 4.0*atan(1.0)

contains


subroutine ik_biexp_run(t,y,dy,v_t,v_r,k,g,Delta,C,E_r,I_ext,&
     & b,a,d,tau_r,tau_d)

implicit none

double precision, intent(in) :: t
double precision, intent(in) :: y(5)
double precision :: r
double precision :: v
double precision :: u
double precision :: s
double precision :: s_in
double precision, intent(inout) :: dy(5)
double precision, intent(in) :: v_t
double precision, intent(in) :: v_r
double precision, intent(in) :: k
double precision, intent(in) :: g
double precision, intent(in) :: Delta
double precision, intent(in) :: C
double precision, intent(in) :: E_r
double precision, intent(in) :: I_ext
double precision, intent(in) :: b
double precision, intent(in) :: a
double precision, intent(in) :: d
double precision, intent(in) :: tau_r
double precision, intent(in) :: tau_d

r = y(1)
v = y(2)
u = y(3)
s = y(4)
s_in = y(5)

dy(1) = (r*(-g*s + k*(2.0*v - v_r - v_t)) + Delta*k**2*abs(v-v_r)&
     & /(pi*C))/C
dy(2) = (-pi*C*r*(pi*C*r/k + Delta*sign(dble(1), v-v_r)) + I_ext &
     & + g*s*(E_r-v) + k*v*(v - v_r - v_t) + k*v_r*v_t - u)/C
dy(3) = a*(b*(v - v_r) - u) + d*r
dy(4) = s_in - s/tau_d
dy(5) = r/(tau_r*tau_d) - s_in/tau_r

end subroutine


end module


subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)

use izhikevich_biexp
implicit none
integer, intent(in) :: ndim, icp(*), ijac
double precision, intent(in) :: y(ndim), args(*)
double precision, intent(out) :: dy(ndim)
double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)

call ik_biexp_run(args(14), y, dy, args(1), args(2), args(3), args(4), &
     & args(5), args(6), args(7), args(8), args(9), args(15), args(16),&
     & args(17), args(18))

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
args(5) = 1.0  ! Delta
args(6) = 100.0  ! C
args(7) = 0.0  ! E_r
args(8) = 0.0  ! I_ext
args(9) = -2.0  ! b
args(15) = 0.03  ! a
args(16) = 10.0  ! d
args(17) = 1.0  ! tau_r
args(18) = 6.0 ! tau_d
y(1) = 0.0  ! r
y(2) = -60.0  ! v
y(3) = 0.0  ! u
y(4) = 0.0  ! s
y(5) = 0.0  ! s_in

end subroutine stpnt



subroutine bcnd
end subroutine bcnd


subroutine icnd
end subroutine icnd


subroutine fopt
end subroutine fopt


subroutine pvls
end subroutine pvls
