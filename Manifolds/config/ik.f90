module izhikevich_exc

double precision :: PI = 4.0*atan(1.0)

contains


subroutine ik_run(t,y,dy,v_t,v_r,k,g,eta,Delta,C,E_r,b,a,d,tau_s)

implicit none

double precision, intent(in) :: t
double precision, intent(in) :: y(4)
double precision :: r
double precision :: v
double precision :: u
double precision :: s
double precision :: r_in
double precision, intent(inout) :: dy(4)
double precision, intent(in) :: v_t
double precision, intent(in) :: v_r
double precision, intent(in) :: k
double precision, intent(in) :: g
double precision, intent(in) :: eta
double precision, intent(in) :: Delta
double precision, intent(in) :: C
double precision, intent(in) :: E_r
double precision, intent(in) :: b
double precision, intent(in) :: a
double precision, intent(in) :: d
double precision, intent(in) :: tau_s

r = y(1)
v = y(2)
u = y(3)
s = y(4)

r_in = r

dy(1) = (r*(-g*s + k*(2.0*v - v_r - v_t)) + Delta*k**2/(pi*C))/C
dy(2) = (-(pi*C*r)**2/k + eta + g*s*(E_r-v) &
     & + k*v*(v - v_r - v_t) + k*v_r*v_t - u)/C
dy(3) = a*(b*(v - v_r) - u) + d*r
dy(4) = r_in - s/tau_s

end subroutine


end module


subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)

use izhikevich_exc
implicit none
integer, intent(in) :: ndim, icp(*), ijac
double precision, intent(in) :: y(ndim), args(*)
double precision, intent(out) :: dy(ndim)
double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)

call ik_run(args(14), y, dy, args(1), args(2), args(3), args(4), &
     & args(5), args(6), args(7), args(8), args(9), args(15), args(16),&
     &  args(17))

end subroutine func


subroutine stpnt(ndim, y, args, t)

implicit None
integer, intent(in) :: ndim
double precision, intent(inout) :: y(ndim), args(*)
double precision, intent(in) :: t

args(1) = -40.0  ! v_t
args(2) = -60.0  ! v_r
args(3) = 0.7  ! k
args(4) = 10.0  ! g
args(5) = 0.0  ! eta
args(6) = 2.0  ! Delta
args(7) = 100.0  ! C
args(8) = 0.0  ! E_r
args(9) = -2.0  ! b
args(15) = 0.03  ! a
args(16) = 100.0  ! d
args(17) = 6.0  ! tau_s
y(1) = 0.0  ! r
y(2) = -60.0  ! v
y(3) = 0.0  ! u
y(4) = 0.0  ! s

end subroutine stpnt



subroutine bcnd
end subroutine bcnd


subroutine icnd
end subroutine icnd


subroutine fopt
end subroutine fopt


subroutine pvls
end subroutine pvls
