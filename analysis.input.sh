#!/bin/bash

# -------------------------------------------------------
# - change parameters from here
# -------------------------------------------------------

T=
LX=
LY=
LZ=
Nconf=
cutradius=
cutangle=
sourceid=
sourceid2=
Nsave=
format=
BCangleT=
BCangleX=
BCangleY=
BCangleZ=
filename_prefix=
filename_prefix2=
gaugefilename_prefix=
resume=
subtract=
cutdir=
# -------------------------------------------------------
# - to here
# -------------------------------------------------------
IN=cvc.input

echo $T         >  $IN
echo $LX        >> $IN
echo $LY        >> $IN
echo $LZ        >> $IN
echo $Nconf     >> $IN
echo $cutradius >> $IN
echo $cutangle  >> $IN
echo $sourceid  >> $IN
echo $sourceid2 >> $IN
echo $Nsave     >> $IN
echo $format    >> $IN
echo $BCangleT  >> $IN
echo $BCangleX  >> $IN
echo $BCangleY  >> $IN
echo $BCangleZ  >> $IN
echo $filename_prefix  >> $IN
echo $filename_prefix2 >> $IN
echo $gaugefilename_prefix >> $IN
echo $resume    >> $IN
echo $subtract  >> $IN
echo $cutdir >> $IN

