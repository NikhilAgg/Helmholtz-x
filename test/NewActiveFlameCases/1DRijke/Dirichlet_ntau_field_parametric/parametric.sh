start=`date +%s`

par_start=-0.15
par_end=0.7
par_step=0.02

flame_location=0.25

for parameter in $(seq $par_start $par_step $par_end); do
    echo " "
    echo "Calculating eigenvalues for x_f =  $parameter +$flame_location"
    python3 main_dim.py $parameter
done


end=`date +%s`
runtime=$((end-start))

echo "Total runtime is: $runtime seconds."

