MODULE QIF

CONTAINS

      SUBROUTINE FRHS(T,U,F,eta,delta,J,alpha,tau_a,tau_s)

      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN) :: U(4)
      DOUBLE PRECISION, INTENT(OUT) :: F(4)
      DOUBLE PRECISION, INTENT(IN) :: T,eta,delta,J,alpha,tau_a,tau_s
      DOUBLE PRECISION :: PI,r,v,a,x

      PI = 4*ATAN(1.0D0)

      r=U(1)
      v=U(2)
      a=U(3)
      x=U(4)

      F(1) = delta/PI + 2.0*r*v
      F(2) = v*v + eta - a + x - PI*PI*r*r
      F(3) = alpha*r - a/tau_a
      F(4) = J*r - x/tau_s

      END SUBROUTINE FRHS

END MODULE QIF