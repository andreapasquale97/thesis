#!/bin/bash
INTEGRAND=$1
INTEGRAND1="${INTEGRAND}1"
DIM=$2
OUTFILE=$3

for integrator in "VegasFlow" "VegasFlowPlus"
do
    for rtol in 0.01 0.001 0.0001
    do
        for ncalls in 1000 10000 100000 1000000
            do 
                python script.py --integrator $integrator --train True --adaptive False --integrand $INTEGRAND --dim $DIM --ncalls $ncalls --accuracy $rtol --outfile $OUTFILE
            done
    done
done


for rtol in 0.01 0.001 0.0001
do
    for ncalls in 1000 10000 100000 1000000
        do 
            python script.py --integrator VegasFlowPlus --train True --adaptive True --integrand $INTEGRAND1 --dim $DIM --ncalls $ncalls --accuracy $rtol --outfile $OUTFILE
        done
done

