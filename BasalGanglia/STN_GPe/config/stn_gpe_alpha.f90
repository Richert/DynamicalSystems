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

      DOUBLE PRECISION Ve,Vi,Re,Ri,Ae,Ai,Je,Ji,Ta,D,PI,Ee,Ei,Te,Ti,Xe,Xi,alpha

       Ee  = PAR(1)
       Ei  = PAR(2)
       Je = PAR(3)
       Ji = PAR(4)
       Te = PAR(5)
       Ti = PAR(6)
       Ta = PAR(7)
       alpha = PAR(8)
       D = PAR(9)
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
       F(2) = (Ve*Ve + Ee)/Te + Je*Re*(1.0-Ae)/10.0 - Ji*Ri*(1.0-Ai) - PI*PI*Re*Re*Te
       F(3) = D/(PI*Ti*Ti) + 2.0*Ri*Vi/Ti
       F(4) = (Vi*Vi + Ei)/Ti + Je*Re*(1.0-Ae) - Ji*Ri*(1.0-Ai)/2.0 - PI*PI*Ri*Ri*Ti

       F(5) = Xe
       F(6) = alpha*Re/Ta - 2.0*Xe/Ta - Ae/(Ta*Ta)
       F(7) = Xi
       F(8) = alpha*Ri/Ta - 2.0*Xi/Ta - Ai/(Ta*Ta)

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION Ee,Ei,Je,Ji,alpha,D,Te,Ti,Ta

       Ee = -1.0
       Ei = 5.0
       Je = 50.0
       Ji = 10.0
       Te = 0.01
       Ti = 0.02
       Ta = 0.2
       alpha = 0.05
       D = 2.0

       PAR(1)=Ee
       PAR(2)=Ei
       PAR(3)=Je
       PAR(4)=Ji
       PAR(5)=Te
       PAR(6)=Ti
       PAR(7)=Ta
       PAR(8)=alpha
       PAR(9)=D

       U(1)=18.6210391
       U(2)=-1.70941
       U(3)=62.7459448
       U(4)=-0.25364977
       U(5)=0.186210391
       U(6)=0.0
       U(7)=0.627459448
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
