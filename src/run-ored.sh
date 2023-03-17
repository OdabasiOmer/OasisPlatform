#!/bin/bash
# Script authored and last modified by Omer Odabasi on 23/02/2023
# Should be called from a model run directory a.k.a., "model_run_dir".
# Omer Odabasi
###################################################################

###################################################
# FUNCTIONS
###################################################

set_n_sim_redloss(){
	# Set the name of the text file
	file_name="redloss.cf"
	nS=$1
	# Search for the line containing "OPT_OUTDIR" in the text file
	line=$(grep "OPT_SIMACC" $file_name)

	# Check if the line was found
	if [ -n "$line" ]
	then
	  echo $line
	  # Extract the part of the line after the comma
	  value=$(echo $line | awk -F, '{print $2}')
	  newLine="OPT_SIMACC,$nS" 
      #value=$(echo $value | sed 's+\/+\\\/+g')

	  # Replace the original line with new
	  sed -i "s+${line}+${newLine}+g" redloss.cf
      echo "INFO > Set OPT_SIMACC to $nS."
	fi
}

###################################################
# INPUT VARIABLES
###################################################
username=$(whoami)
p1="/home/${username}/REDCat/bin"
p2="/usr/lib/x86_64-linux-gnu"
ORED_EXP_KEYS_DIR=~/GitHub/ored-exposure-mapping/oed_exposure/
nS=10
# Commenting out below line, fixing nS at 10 for current version of platform
#read -p "Number of samples -- as is set in redloss.cf? [00-99] " -n 2 -r nS
#echo

###################################################
# PROGRAM
###################################################

echo '==================== STAGE-0: Set up requisite directories  ======================='
echo " > Note that this program expects the latest REDCat executables to reside in ${p1}"
if ! echo "$PATH" | grep -q "$p1"
then
    echo " > Adding to path > ${p1}"
    export PATH=$PATH:${p1}
fi

if ! echo "$PATH" | grep -q "$p2"
then
    echo " > Adding to path >> ${p2}"
    export PATH=$PATH:${p2}
    if [ ! -f "$p2/libgfortran.so.4" ]; then
        # If the file does not exist, copy it from the current working directory to p2
        cp "$p1/libf2c.so.0" $p2
        cp "$p1/libgfortran.so.4" $p2
        echo " > Files 'libf2c.so.0 and libgfortran.so.4' have been copied to $p2"
    else
        echo " > C++ libraries already exist in $p2. Moving on..."
    fi
fi

echo ''
echo 'INFO > Copying .cf files over to cwd...'
cp ../../redexp.cf .
cp ../../redhaz.cf .
cp ../../redloss.cf .
set_n_sim_redloss "$nS"

# Step-1: Call oredexp to convert location.csv into portfolio.csv to stream into REDExp.
#  > Define folder containing OED-REDEXP conversion files. In my device its located in the below dir.
###########################################################################################

echo '==================== STAGE-1: Input portfolio conversion: OED to REDCat ======================='
echo ''
echo 'INFO: Oasis-REDExp conversion keys data directory is set to ' $ORED_EXP_KEYS_DIR
echo ''
# Copy over the necessary key files
cp $ORED_EXP_KEYS_DIR/occupancy_codes.csv ./input
cp $ORED_EXP_KEYS_DIR/construction_codes.csv ./input
cp $ORED_EXP_KEYS_DIR/oed_fields.csv ./input

read -p "Want to convert OED input to REDCat ones? [y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    #Call oredexp
    oredexp -i ./input -o ./input
    echo
fi

echo '==================== STAGE-2a: Call REDExp ======================='
echo ' NOTE that below steps of the program expects three REDCat configuration files (.cf) for REDExp, REDHaz, and REDLoss to be herein present.'
echo ''
read -p "Want to (re)run REDExp? [y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    REDExp -f redexp.cf
    # Wait for the process to finishread -p "Want to (re)run REDExp? [y/n] " -n 1 -r
    wait $!
    mv REDExp.out log/
    echo
    echo " INFO> REDEXp execution completed. Moving the log file..."
fi

echo '==================== STAGE-2b: Call REDHaz ======================='
echo ''
read -p "Want to (re)run REDHaz? [y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -r ./work/hazout 
    mkdir ./work/hazout/
    REDHaz -f redhaz.cf
    wait $!
    echo "Moving files..."
    mv ./REDHaz.out log/
    mv occurrence.bin ./input/
    mv occurrence.csv ./input/
    echo "INFO> REDHaz execution completed"
fi

# echo " INFO> IMPORTANT: Assuming 10000y catalog length. May be subject to change. It is currently hard-coded to 10000."
# occurrencetobin -D -P 10000 < ./input/occurrence.csv > ./input/occurrence.bin

echo "==================== STAGE-2c: Call REDLoss ======================="
read -p "Want to (re)run REDLoss? [y/n]" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    REDLoss -f redloss.cf
    wait $!
    echo "Moving files..."
    mv ./REDLoss.out log/
fi

read -p "[Optional step] Want to convert gulcalc binary to csv?" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "INFO > [Verification] Convert GUP_ALL.gup of REDLoss to CSV..."
    echo "INFO > Saving to ./work/gulcalc.csv"
    gultocsv < ./work/lossout/GUP_ALL.gup > ./work/gulcalc.csv
fi

echo '==================== STAGE-3: Ktools ======================='
read -p "Do you Want to follow through with the Ktools pipeline?" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo 'INFO > fmcalc to generate the fmcalc file at ./work.'

    # [DEBUGGING] In case of faulty .gup file:
    # gultobin -S0 < ./work/lossout/GUP_ALL.csv > ./work/gulcalc.bin
    # fmcalc -a2 < ./work/gulcalc.bin > ./work/fmcalc-a2.bin
    # ********

    # NOTE: THE FOLLOWING (fmcalc | summarycalc | eltcalc... etc) WILL LATER CONDENNSE THE FOLLOWING INTO ONE SINGLE PIPE.
    #fmcalc < ./work/lossout/GUP_ALL.gup > ./work/fmcalc.bin
    #fmtocsv < ./work/fmcalc.bin > ./work/fmcalc.csv

    #fmcalc -a1 < ./work/lossout/GUP_ALL.gup > ./work/fmcalc-a1.bin
    #fmtocsv < ./work/fmcalc-a1.bin > ./work/fmcalc-a1.csv

    fmcalc -a2 < ./work/lossout/GUP_ALL.gup > ./work/fmcalc-a2.bin
    fmtocsv < ./work/fmcalc-a2.bin > ./work/fmcalc-a2.csv

    echo 'INFO > Running Summarycalc -f1, f2, and f3,,,, using fmcalc-a2.bin'
    mkdir work/summary1/
    summarycalc -f -1 ./work/summary1/summarycalc1.bin < ./work/fmcalc-a2.bin
    #summarycalc -f -2 ./work/fmsummarycalc-f2.bin < ./work/fmcalc-a2.bin
    #summarycalc -f -3 ./work/fmsummarycalc-f3.bin < ./work/fmcalc-a2.bin
    echo
    
    echo 'INFO > Running summarycalc -i for ground up loss at ./work/summary-g1/'
    mkdir work/summaryg1/
    summarycalc -i -1 ./work/summaryg1/summarycalci.bin < ./work/lossout/GUP_ALL.gup
    echo ''

    echo 'INFO > [Optional pipe] Running eltcalc using (il & gul) summarycalc1.bin'
    echo ''
    eltcalc < ./work/summary1/summarycalc1.bin > ./work/summary1/il_elt.csv
    eltcalc < ./work/summaryg1/summarycalci.bin > ./work/summaryg1/gul_elt.csv

    # Then run leccalc, pointing to the specified sub-directory of work containing summarycalc binaries.
    # leccalc -Ksummary1 -F lec_full_uncertainty_agg.csv -f lec_full_uncertainty_occ.csv 
    echo "INFO > (1/2) Running leccalc for IL..."
    leccalc -r -Ksummary1 -F output/il_S"$nS"_leccalc_full_uncertainty_agg.csv -f output/il_S"$nS"_leccalc_full_uncertainty_occ.csv 
    echo "INFO > (2/2) Running leccalc for GUL..."
    leccalc -r -Ksummaryg1 -F output/gul_S"$nS"_leccalc_full_uncertainty_agg.csv -f output/gul_S"$nS"_leccalc_full_uncertainty_occ.csv 
    echo ""
    
    echo "INFO > Running aalcalc..."
    aalcalc -Ksummary1 > output/il_S"$nS"_aalcalc.csv 
    aalcalc -Ksummaryg1 > output/gul_S"$nS"_aalcalc.csv 
    echo ""
fi

# ***********************************************
# gulcalc item stream and coverage stream
#eve 1 1 | getmodel | gulcalc -S1 -a0 -i ./out/gulcalci.bin
#gultocsv < ./out/gulcalci.bin > ./out/gulcalci.csv 
#
## fmcalc
#fmcalc < ./out/gulcalci.bin > ./out/fmcalc.bin
#fmtocsv > ./out/fmcalci.csv < ./out/fmcalc.bin
#
## summarycalc
#summarycalc -i -2 /out/gulsummarycalc2.bin < ./out/gulcalci.bin
#summarycalc -f -1 ./out/fmsummarycalc1.bin < ./out/fmcalc.bin
#
## selt
#summarycalctocsv -o > ./out/gulselt1.csv < ./out/gulsummarycalc1.bin
#
## eltcalc (prints out csv)
#eltcalc < ./out//gulsummarycalc2.bin > ./out/gulelt2.csv
#eltcalc < ./out/fmsummarycalc1.bin > ./out/fmelt1.csv
