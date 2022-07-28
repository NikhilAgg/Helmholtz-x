start=`date +%s`

par_start=1
par_end=8
par_step=1


for parameter in $(seq $par_start $par_step $par_end); do
    echo " "
    echo "Calculating eigenvalues "
    mpirun -np $parameter python3 main.py
done


end=`date +%s`
runtime=$((end-start))

echo "Total runtime is: $runtime seconds."

