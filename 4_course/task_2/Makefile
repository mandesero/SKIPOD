all:
	mpicc -std=c99 -o run mpi_redb_2d.c -lm
# run:
# 	mpirun -np 6 --mca shmem posix --mca opal_event_include poll --map-by :OVERSUBSCRIBE --with-ft ulfm ./run