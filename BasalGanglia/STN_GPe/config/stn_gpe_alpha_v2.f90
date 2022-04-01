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

      DOUBLE PRECISION Ve,Vi,Re,Ri,Ae,Ai,Je,Ji,Ta,Tb,Ee,Ei,Te,Ti,Xe,Xi,alpha,beta,D,PI,k

       Ee  = PAR(1)
       Ei  = PAR(2)
       Je = PAR(3)
       Ji = PAR(4)
       Te = PAR(5)
       Ti = PAR(6)
       Ta = PAR(7)
       Tb = PAR(8)
       alpha = PAR(9)
       k = PAR(10)
       beta = 2.0
       D = 2.0
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
       F(2) = (Ve*Ve + Ee)/Te - k*Ji*Ri*(1.0-Ai) - PI*PI*Re*Re*Te
       F(3) = D/(PI*Ti*Ti) + 2.0*Ri*Vi/Ti
       F(4) = (Vi*Vi + Ei)/Ti + k*Je*Re*(1.0-Ae) - k*Ji*Ri*(1.0-Ai)*0.3 - PI*PI*Ri*Ri*Ti

       F(5) = Xe
       F(6) = (alpha*Re - 2.0*Xe - Ae/Ta)/Ta
       F(7) = Xi
       F(8) = (alpha*beta*Ri - 2.0*Xi - Ai/Tb)/Tb

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION Ve,Vi,Re,Ri,Ae,Ai,Je,Ji,Ta,Tb,Ee,Ei,Te,Ti,Xe,Xi,alpha,k

       Ee = 4.0
       Ei = 6.0
       Je = 40.0
       Ji = 40.0
       Te = 6.0
       Ti = 14.0
       Ta = 100.0
       Tb = 200.0
       alpha = 0.0
       k = 1.0

       !Ee = 2.0
       !Ei = 10.0
       !Je = 20.0
       !Ji = 30.0
       !Te = 6.0
       !Ti = 14.0
       !Ta = 100.0
       !Tb = 200.0
       !alpha = 0.0
       !k = 1.0

       !Ee = -2.0
       !Ei = 10.0
       !Je = 30.0
       !Ji = 30.0
       !Te = 6.0
       !Ti = 14.0
       !Ta = 100.0
       !Tb = 200.0
       !alpha = 0.0
       !k = 1.0

       PAR(1)=Ee
       PAR(2)=Ei
       PAR(3)=Je
       PAR(4)=Ji
       PAR(5)=Te
       PAR(6)=Ti
       PAR(7)=Ta
       PAR(8)=Tb
       PAR(9)=alpha
       PAR(10)=k

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
