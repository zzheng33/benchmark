! CLASS = B
!  
!  
!  This file is generated automatically by the setparams utility.
!  It sets the number of processors and the class of the NPB
!  in this directory. Do not modify it by hand.
!  
        integer nx, ny, nz, maxdim, niter_default
        parameter (nx=512, ny=256, nz=256, maxdim=512)
        parameter (niter_default=20)
        logical  convertdouble
        parameter (convertdouble = .false.)
        character*11 compiletime
        parameter (compiletime='26 Feb 2024')
        character*5 npbversion
        parameter (npbversion='3.4.2')
        character*6 cs1
        parameter (cs1='mpif90')
        character*8 cs2
        parameter (cs2='$(MPIFC)')
        character*6 cs3
        parameter (cs3='(none)')
        character*6 cs4
        parameter (cs4='(none)')
        character*9 cs5
        parameter (cs5='-O3 -fPIE')
        character*9 cs6
        parameter (cs6='$(FFLAGS)')
        character*6 cs7
        parameter (cs7='randi8')
