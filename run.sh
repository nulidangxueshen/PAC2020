for i in $(seq 1 10)
do
	OMP_WAIT_POLICY=ACTIVE KMP_AFFINITY=compact granularity=fine OMP_BIND_PROC=true ./logVS
done
