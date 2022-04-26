module izhikevich_exc

double precision :: PI = 4.0*atan(1.0)

contains


subroutine ik_run(t,ndim,y,dy,v_t,v_r,k,g,q,Delta,C,v_z,v_p,E_r,I_ext,&
     & b,a,d,tau_s)

implicit none

double precision, intent(in) :: t
integer, intent(in) :: ndim
double precision, intent(in) :: y(ndim)
double precision :: r((ndim-1)/3)
double precision :: v((ndim-1)/3)
double precision :: u((ndim-1)/3)
double precision :: s
double precision :: r_in
double precision, intent(inout) :: dy(ndim)
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
double precision :: Delta_m, v_tm
integer :: n, m

m = (ndim-1)/3
do n=1,m
    r(n) = y(n)
    v(n) = y(n+m)
    u(n) = y(n+2*m)
end do
s = y(ndim)

r_in = 0
do n=1,m
    r_in = r_in + r(n)
end do

do n=1,m

    v_tm = v_t + Delta*tan(0.5*pi*(2*n-m-1)/(m+1))
    Delta_m = Delta*(tan(0.5*pi*(2*n-m-0.5)/(m+1))-tan(0.5*pi*(2*n-m-1.5)/(m+1)))

    dy(n) = (r(n)*(-g*s + k*(2.0*v(n) - v_r - v_tm) - q) + Delta_m*k**2*(v(n) - v_r)&
         & /(pi*C))/C
    dy(n+m) = (-pi*C*r(n)*(pi*C*r(n)/k &
         & + Delta_m) + C*q*r(n)*log(v_p/v_z)/k + I_ext + g*s*(E_r &
         & - v(n)) + k*v(n)*(v(n) - v_r - v_tm) + k*v_r*v_tm - u(n))/C
    dy(n+2*m) = a*(b*(v(n) - v_r) - u(n)) + d*r(n)

end do
dy(ndim) = r_in/m - s/tau_s

end subroutine


end module


subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)

use izhikevich_exc
implicit none
integer, intent(in) :: ndim, icp(*), ijac
double precision, intent(in) :: y(ndim), args(*)
double precision, intent(out) :: dy(ndim)
double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)

call ik_run(args(14), ndim, y, dy, args(1), args(2), args(3), args(4), &
     & args(5), args(6), args(7), args(8), args(9), args(15), args(16),&
     &  args(17), args(18), args(19), args(20))

end subroutine func


subroutine stpnt(ndim, y, args, t)

implicit None
integer, intent(in) :: ndim
double precision, intent(inout) :: y(ndim), args(*)
double precision, intent(in) :: t
integer :: m, n

args(1) = -40.0  ! v_t
args(2) = -60.0  ! v_r
args(3) = 0.7  ! k
args(4) = 1.0  ! g
args(5) = 0.0  ! q
args(6) = 1.0  ! Delta
args(7) = 100.0  ! C
args(8) = 60.0  ! v_z
args(9) = 40.0  ! v_p
args(15) = 0.0  ! E_r
args(16) = 0.0  ! I_ext
args(17) = -2.0  ! b
args(18) = 0.03  ! a
args(19) = 10.0  ! d
args(20) = 6.0  ! tau_s

m = (ndim-1)/3
do n=1,m
    y(n) = 0.0  ! r
    y(n+m) = -60.0  ! v
    y(n+2*m) = 0.0  ! u
end do
y(ndim) = 0.0  ! s

end subroutine stpnt

subroutine bcnd
end subroutine bcnd


subroutine icnd
end subroutine icnd


subroutine fopt
end subroutine fopt


subroutine pvls
end subroutine pvls
