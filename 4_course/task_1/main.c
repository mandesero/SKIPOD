#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define MATRIX_SIZE 8

int main(int argc, char **argv)
{
    int process_rank, num_processes;
    int time_start = 100;
    int time_transfer = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int coords[2];
    int dimensions[2] = {MATRIX_SIZE, MATRIX_SIZE};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, process_rank, 2, coords);

    srand(time(NULL) + process_rank);
    int num = rand() % 100;

    if (process_rank != 0)
    {
        int dest_coords[2] = {0, 0};
        int dest_rank;
        MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);
        MPI_Send(&num, 1, MPI_INT, dest_rank, 0, cart_comm);
    }
    else
    {
        int received_nums[MATRIX_SIZE * MATRIX_SIZE] = {0};
        received_nums[0] = num;

        for (int i = 1; i < num_processes; i++)
        {
            int source_coords[2];
            if (MPI_Recv(&received_nums[i], 1, MPI_INT, MPI_ANY_SOURCE, 0, cart_comm, MPI_STATUS_IGNORE) == MPI_SUCCESS)
            {
                MPI_Cart_coords(cart_comm, i, 2, source_coords);
                printf("Процесс (%d, %d) отправил число %d\n", source_coords[0], source_coords[1], received_nums[i]);
            }
        }

        int max_num = received_nums[0];
        for (int i = 1; i < num_processes; i++)
        {
            if (received_nums[i] > max_num)
            {
                max_num = received_nums[i];
            }
        }

        printf("Максимальное число: %d\n", max_num);
    }

    MPI_Finalize();
    return 0;
}