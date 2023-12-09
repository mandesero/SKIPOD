#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>

#define MAX(x, y) x > y ? x : y
#define UP(coords) \
    (int[]) { coords[0] - 1, coords[1] }
#define DOWN(coords) \
    (int[]) { coords[0] + 1, coords[1] }
#define RIGHT(coords) \
    (int[]) { coords[0], coords[1] + 1 }
#define LEFT(coords) \
    (int[]) { coords[0], coords[1] - 1 }

#define MATRIX_SIZE 4

MPI_Request send(int *coords, MPI_Comm comm, int *data)
{
    int target_rank;
    MPI_Cart_rank(comm, coords, &target_rank);

    MPI_Request request;
    MPI_Isend(data, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD, &request);
    return request;
}

int recv(int *coords, MPI_Comm comm)
{
    int target_rank;
    MPI_Cart_rank(comm, coords, &target_rank);

    int data;
    MPI_Recv(&data, 1, MPI_INT, target_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return data;
}

int main(int argc, char **argv)
{
    int process_rank, num_processes;

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
    int num = rand() % 255;
    printf("--- Number in (%d, %d) = %d\n", coords[0], coords[1], num);
    sleep(1);

    int recv_num;
    if (coords[0] != 0)
    {
        if (coords[0] != MATRIX_SIZE - 1)
        {
            int *from = DOWN(coords);
            recv_num = recv(from, cart_comm);
            // printf("(%d, %d) get %d from (%d, %d)\n", coords[0], coords[1], recv_num, from[0], from[1]);
            num = MAX(num, recv_num);
        }
        int *to = UP(coords);
        // printf("(%d, %d) send %d to (%d, %d)\n", coords[0], coords[1], recv_num, to[0], to[1]);
        MPI_Request request = send(to, cart_comm, &num);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    else
    {
        recv_num = recv(DOWN(coords), cart_comm);
        num = MAX(num, recv_num);

        if (coords[1] != MATRIX_SIZE - 1)
        {
            int *from = RIGHT(coords);
            recv_num = recv(from, cart_comm);
            // printf("(%d, %d) get %d from (%d, %d)\n", coords[0], coords[1], recv_num, from[0], from[1]);
            num = MAX(num, recv_num);
        }

        if (coords[1] != 0)
        {
            int *to = LEFT(coords);
            // printf("(%d, %d) send %d to (%d, %d)\n", coords[0], coords[1], recv_num, to[0], to[1]);
            MPI_Request request = send(to, cart_comm, &num);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
    }

    if (process_rank == 0)
        printf("\nMax number is %d\n", num);

    MPI_Finalize();
    return 0;
}