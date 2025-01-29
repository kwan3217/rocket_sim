KPL/MK

   This meta-kernel lists the NH SPICE kernels providing coverage for
   from around 19.March, 2007 through August, 2014, the end of archive
   Release 0002.

   All of the kernels listed below are archived in the NH SPICE data
   set (DATA_SET_ID = "NH-J/P/SS-SPICE-6-V1.0"). This set of files and
   the order in which they are listed were picked to provide the best
   available data and the most complete coverage based on the
   information about the kernels available at the time this meta-kernel
   was made. For detailed information about the kernels listed below
   refer to the internal comments included in the kernels and the
   documentation accompanying the NH SPICE data set.
 
   It is recommended that users make a local copy of this file and
   modify the value of the PATH_VALUES keyword to point to the actual
   location of the NH SPICE data set's ``data'' directory on their
   system. Replacing ``/'' with ``\'' and converting line terminators
   to the format native to the user's system may also be required if
   this meta-kernel is to be used on a non-UNIX workstation.
 
   This file was created 2024-02-07 by Chris Jeppesen

   The original name of this file was nh_v02.tm.

   \begindata

      PATH_VALUES       = (
                           '/home/jeppesen/workspace/Data/spice/generic/',
                           '/home/jeppesen/workspace/rocket_sim/data/'
                           '/home/jeppesen/workspace/rocket_sim/products/'
                          )
      PATH_SYMBOLS      = (
                           'GENERIC',
                           'PUBLISHED',
                           'KWAN'
                          )
      KERNELS_TO_LOAD   = ('$GENERIC/lsk/naif0012.tls',
                           '$GENERIC/pck/pck00011.tpc',
                           '$PUBLISHED/vgr2_st.bsp',
                           '$KWAN/vgr2_horizons_vectors.bsp',
                           '$KWAN/vgr2_pm.bsp',
                           '$KWAN/vgr2_centaur2.bsp',
                          )


   \begintext

