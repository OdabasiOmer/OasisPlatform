
mkdir ./work
mkdir ./outputs 

mkfifo ./work/1/gul
mkfifo ./work/1/gul_summary1
mkfifo ./work/1/gul_summary1_output1
mkfifo ./work/1/gul_summary1_output2
mkfifo ./work/1/gul_summary1_output3

mkfifo ./work/2/gul
mkfifo ./work/2/gul_summary1
mkfifo ./work/2/gul_summary1_output1
mkfifo ./work/2/gul_summary1_output2
mkfifo ./work/2/gul_summary1_output3

mkfifo ./work/3/gul
mkfifo ./work/3/gul_summary1
mkfifo ./work/3/gul_summary1_output1
mkfifo ./work/3/gul_summary1_output2
mkfifo ./work/3/gul_summary1_output3

mkfifo ./work/4/gul
mkfifo ./work/4/gul_summary1
mkfifo ./work/4/gul_summary1_output1
mkfifo ./work/4/gul_summary1_output2
mkfifo ./work/4/gul_summary1_output3

mkfifo ./work/5/gul
mkfifo ./work/5/gul_summary1
mkfifo ./work/5/gul_summary1_output1
mkfifo ./work/5/gul_summary1_output2
mkfifo ./work/5/gul_summary1_output3

mkfifo ./work/1/il
mkfifo ./work/1/il_summary1
mkfifo ./work/1/il_summary1_output1
mkfifo ./work/1/il_summary1_output2
mkfifo ./work/1/il_summary1_output3

mkfifo ./work/2/il
mkfifo ./work/2/il_summary1
mkfifo ./work/2/il_summary1_output1
mkfifo ./work/2/il_summary1_output2
mkfifo ./work/2/il_summary1_output3

mkfifo ./work/3/il
mkfifo ./work/3/il_summary1
mkfifo ./work/3/il_summary1_output1
mkfifo ./work/3/il_summary1_output2
mkfifo ./work/3/il_summary1_output3

mkfifo ./work/4/il
mkfifo ./work/4/il_summary1
mkfifo ./work/4/il_summary1_output1
mkfifo ./work/4/il_summary1_output2
mkfifo ./work/4/il_summary1_output3

mkfifo ./work/5/il
mkfifo ./work/5/il_summary1
mkfifo ./work/5/il_summary1_output1
mkfifo ./work/5/il_summary1_output2
mkfifo ./work/5/il_summary1_output3

eltcalc < ./work/1/gul_summary1_output1 > ./work/1/gul_1_elt.csv &
eltcalc < ./work/2/gul_summary1_output1 > ./work/2/gul_1_elt.csv &
eltcalc < ./work/3/gul_summary1_output1 > ./work/3/gul_1_elt.csv &
eltcalc < ./work/4/gul_summary1_output1 > ./work/4/gul_1_elt.csv &
eltcalc < ./work/5/gul_summary1_output1 > ./work/5/gul_1_elt.csv &
eltcalc < ./work/1/il_summary1_output1 > ./work/1/il_1_elt.csv &
eltcalc < ./work/2/il_summary1_output1 > ./work/2/il_1_elt.csv &
eltcalc < ./work/3/il_summary1_output1 > ./work/3/il_1_elt.csv &
eltcalc < ./work/4/il_summary1_output1 > ./work/4/il_1_elt.csv &
eltcalc < ./work/5/il_summary1_output1 > ./work/5/il_1_elt.csv &

leccalc -F -p 1 -P 1000000 -k ./work/1/gul_aep/1 < ./work/1/gul_summary1_output2 &
leccalc -F -p 2 -P 1000000 -k ./work/2/gul_aep/1 < ./work/2/gul_summary1_output2 &
leccalc -F -p 3 -P 1000000 -k ./work/3/gul_aep/1 < ./work/3/gul_summary1_output2 &
leccalc -F -p 4 -P 1000000 -k ./work/4/gul_aep/1 < ./work/4/gul_summary1_output2 &
leccalc -F -p 5 -P 1000000 -k ./work/5/gul_aep/1 < ./work/5/gul_summary1_output2 &
leccalc -F -p 1 -P 1000000 -k ./work/1/il_aep/1 < ./work/1/il_summary1_output2 &
leccalc -F -p 2 -P 1000000 -k ./work/2/il_aep/1 < ./work/2/il_summary1_output2 &
leccalc -F -p 3 -P 1000000 -k ./work/3/il_aep/1 < ./work/3/il_summary1_output2 &
leccalc -F -p 4 -P 1000000 -k ./work/4/il_aep/1 < ./work/4/il_summary1_output2 &
leccalc -F -p 5 -P 1000000 -k ./work/5/il_aep/1 < ./work/5/il_summary1_output2 &


leccalc -f -p 1 -P 1000000 -k ./work/1/gul_oep/1 < ./work/1/gul_summary1_output3 &
leccalc -f -p 2 -P 1000000 -k ./work/2/gul_oep/1 < ./work/2/gul_summary1_output3 &
leccalc -f -p 3 -P 1000000 -k ./work/3/gul_oep/1 < ./work/3/gul_summary1_output3 &
leccalc -f -p 4 -P 1000000 -k ./work/4/gul_oep/1 < ./work/4/gul_summary1_output3 &
leccalc -f -p 5 -P 1000000 -k ./work/5/gul_oep/1 < ./work/5/gul_summary1_output3 &
leccalc -f -p 1 -P 1000000 -k ./work/1/il_oep/1 < ./work/1/il_summary1_output3 &
leccalc -f -p 2 -P 1000000 -k ./work/2/il_oep/1 < ./work/2/il_summary1_output3 &
leccalc -f -p 3 -P 1000000 -k ./work/3/il_oep/1 < ./work/3/il_summary1_output3 &
leccalc -f -p 4 -P 1000000 -k ./work/4/il_oep/1 < ./work/4/il_summary1_output3 &
leccalc -f -p 5 -P 1000000 -k ./work/5/il_oep/1 < ./work/5/il_summary1_output3 &


tee < ./work/1/gul_summary1 ./work/1/gul_summary1_output1 ./work/1/gul_summary1_output2 ./work/1/gul_summary1_output3 > /dev/null &
tee < ./work/2/gul_summary1 ./work/2/gul_summary1_output1 ./work/2/gul_summary1_output2 ./work/1/gul_summary1_output3 > /dev/null &
tee < ./work/3/gul_summary1 ./work/3/gul_summary1_output1 ./work/3/gul_summary1_output2 ./work/1/gul_summary1_output3 > /dev/null &
tee < ./work/4/gul_summary1 ./work/4/gul_summary1_output1 ./work/4/gul_summary1_output2 ./work/1/gul_summary1_output3 > /dev/null &
tee < ./work/5/gul_summary1 ./work/5/gul_summary1_output1 ./work/5/gul_summary1_output2 ./work/1/gul_summary1_output3 > /dev/null &
tee < ./work/1/il_summary1 ./work/1/il_summary1_output1 ./work/1/il_summary1_output2 ./work/1/il_summary1_output3 > /dev/null &
tee < ./work/2/il_summary1 ./work/2/il_summary1_output1 ./work/2/il_summary1_output2 ./work/1/il_summary1_output3 > /dev/null &
tee < ./work/3/il_summary1 ./work/3/il_summary1_output1 ./work/3/il_summary1_output2 ./work/1/il_summary1_output3 > /dev/null &
tee < ./work/4/il_summary1 ./work/4/il_summary1_output1 ./work/4/il_summary1_output2 ./work/1/il_summary1_output3 > /dev/null &
tee < ./work/5/il_summary1 ./work/5/il_summary1_output1 ./work/5/il_summary1_output2 ./work/1/il_summary1_output3 > /dev/null &


summarycalc -g -1 ./work/1/gul_summary1 < ./work/1/gul &
summarycalc -g -1 ./work/2/gul_summary1 < ./work/2/gul &
summarycalc -g -1 ./work/3/gul_summary1 < ./work/3/gul &
summarycalc -g -1 ./work/4/gul_summary1 < ./work/4/gul &
summarycalc -g -1 ./work/5/gul_summary1 < ./work/5/gul &
summarycalc -g -1 ./work/1/il_summary1 < ./work/1/il &
summarycalc -g -1 ./work/2/il_summary1 < ./work/2/il &
summarycalc -g -1 ./work/3/il_summary1 < ./work/3/il &
summarycalc -g -1 ./work/4/il_summary1 < ./work/4/il &
summarycalc -g -1 ./work/5/il_summary1 < ./work/5/il &


eve 1 5 | getmodel | gulcalc -S100 -R1000000 -c ./work/1/gul | fmcalc -S100 -R1000000 -c ./work/1/il
eve 2 5 | getmodel | gulcalc -S100 -R1000000 -c ./work/2/gul | fmcalc -S100 -R1000000 -c ./work/2/il
eve 3 5 | getmodel | gulcalc -S100 -R1000000 -c ./work/3/gul | fmcalc -S100 -R1000000 -c ./work/3/il
eve 4 5 | getmodel | gulcalc -S100 -R1000000 -c ./work/4/gul | fmcalc -S100 -R1000000 -c ./work/4/il
eve 5 5 | getmodel | gulcalc -S100 -R1000000 -c ./work/5/gul | fmcalc -S100 -R1000000 -c ./work/5/il

cat ./work/1/gul_1_elt.csv ./work/2/gul_1_elt.csv ./work/3/gul_1_elt.csv ./work/4/gul_1_elt.csv ./work/5/gul_1_elt.csv > ./outputs/gul_1_elt.csv
cat ./work/1/il_1_elt.csv ./work/2/il_1_elt.csv ./work/3/il_1_elt.csv ./work/4/il_1_elt.csv ./work/5/il_1_elt.csv > ./outputs/il_1_elt.csv
lecsummary –A -k ./work/1/gul_aep/1 > ./output/gul_1_AEP.csv
lecsummary –A -k ./work/1/il_aep/1 > ./output/il_1_AEP.csv
lecsummary –A -k ./work/1/gul_oep/1 > ./output/gul_1_OEP.csv
lecsummary –A -k ./work/1/il_oep/1 > ./output/il_1_OEP.csv