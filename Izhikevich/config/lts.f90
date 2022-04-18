module izhikevich_inh

double precision :: PI = 4.0*atan(1.0)

contains


subroutine ik_run(t,y,dy,v_t,v_r,k,g,q,Delta,C,v_z,v_p,E_r,I_ext,b,a,d,&
     & tau_s)

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

r = y(1)
v = y(2)
u = y(3)
s = y(4)

r_in = r

dy(1) = (r*(-g*s + k*(2.0*v - v_r - v_t) - q) + Delta*k**2*(v - v_r)&
     & /(pi*C))/C
dy(2) = (-pi*C*r*(pi*C*r/k &
     & + Delta) + C*q*r*log(v_p/v_z)/k + I_ext + g*s*(E_r &
     & - v) + k*v*(v - v_r - v_t) + k*v_r*v_t - u)/C
dy(3) = a*(b*(v - v_r) - u) + d*r
dy(4) = r_in - s/tau_s

end subroutine


end module


subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)

use izhikevich_inh
implicit none
integer, intent(in) :: ndim, icp(*), ijac
double precision, intent(in) :: y(ndim), args(*)
double precision, intent(out) :: dy(ndim)
double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)

call ik_run(args(14), y, dy, args(1), args(2), args(3), args(4), &
     & args(5), args(6), args(7), args(8), args(9), args(15), args(16),&
     &  args(17), args(18), args(19), args(20))

end subroutine func


subroutine stpnt(ndim, y, args, t)

implicit None
integer, intent(in) :: ndim
double precision, intent(inout) :: y(ndim), args(*)
double precision, intent(in) :: t

args(1) = -42.0  ! v_t
args(2) = -56.0  ! v_r
args(3) = 1.0  ! k
args(4) = 1.0  ! g
args(5) = 0.0  ! q
args(6) = 1.0  ! Delta
args(7) = 100.0  ! C
args(8) = 53.0  ! v_z
args(9) = 40.0  ! v_p
args(15) = -65.0  ! E_r
args(16) = 0.0  ! I_ext
args(17) = 8.0  ! b
args(18) = 0.03  ! a
args(19) = 20.0  ! d
args(20) = 6.0  ! tau_s
y(1) = 0.0  ! r
y(2) = -56.0  ! v
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
