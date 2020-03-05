!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
!   qif :     QIF neural mass model with synaptic adaptation
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION Ve,Vi,Re,Ri,Ae,Ai,Je,Ji,Ta,Tb,Ee,Ei,Te,Ti,Xe,Xi,alpha,beta,D,PI

       Ee  = PAR(1)
       Ei  = PAR(2)
       Je = PAR(3)
       Ji = PAR(4)
       Te = PAR(5)
       Ti = PAR(6)
       Ta = PAR(7)
       Tb = PAR(8)
       alpha = PAR(9)
       beta = PAR(10)
       D = 4.0
       PI = 4*ATAN(1.0D0)

       Re=U(1)
       Ve=U(2)
       Ri=U(3)
       Vi=U(4)
       Ae=U(5)
       Xe=U(6)
       Ai=U(7)
       Xi=U(8)

       F(1) = D/(PI*Te*Te) + 2.0*Re*Ve/Te
       F(2) = (Ve*Ve + Ee)/Te - Ji*Ri - PI*PI*Re*Re*Te
       F(3) = D/(PI*Ti*Ti) + 2.0*Ri*Vi/Ti
       F(4) = (Vi*Vi + Ei)/Ti + Je*Re - Ji*Ri*(1.0-Ae)*0.2 - Ai - PI*PI*Ri*Ri*Ti

       F(5) = Xe
       F(6) = (alpha*Ri - 2.0*Xe - Ae/Ta)/Ta
       F(7) = Xi
       F(8) = (beta*Ri - 2.0*Xi - Ai/Tb)/Tb

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION Ve,Vi,Re,Ri,Ae,Ai,Je,Ji,Ta,Tb,Ee,Ei,Te,Ti,Xe,Xi,alpha,beta

       Ee = -4.0
       Ei = 16.0
       Je = 100.0
       Ji = 40.0
       Te = 6.0
       Ti = 14.0
       Ta = 400.0
       Tb = 600.0
       alpha = 0.0
       beta =  0.1

       PAR(1)=Ee
       PAR(2)=Ei
       PAR(3)=Je
       PAR(4)=Ji
       PAR(5)=Te
       PAR(6)=Ti
       PAR(7)=Ta
       PAR(8)=Tb
       PAR(9)=alpha
       PAR(10)=beta

       U(1)=18.971274
       U(2)=-2.796420
       U(3)=62.999583
       U(4)=-0.360898
       U(5)=0.03794255
       U(6)=0.0
       U(7)=0.12599917
       U(8)=0.0

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
