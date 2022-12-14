#include "mpi.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define Max(a, b) ((a)>(b)?(a):(b))

#define N (256*64+2)
float maxeps = 0.1e-7;
int itmax = 100;
float w = 0.5;
float eps, local_eps;

float A[N][N];
float tmp_A_row[N];

int rank, num_workers, rc;
int first_row, last_row, n_rows;

void relax();
void init();
void verify();
int master_job();

int main(int an, char **as) {

    if ((rc = MPI_Init(&an, &as))) {
        printf("Execution Error%d", rc);
        MPI_Abort(MPI_COMM_WORLD, rc);
        return rc;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_workers);

    n_rows = (N-2) / num_workers;
    first_row = n_rows * rank + 1;

    if (rank != num_workers-1){
        last_row = first_row + n_rows;
    } else {
        last_row = N-1;
    }

    // printf("rank %d: first_row %d, last_row %d\n", rank, first_row, last_row);

    struct timeval start, stop;
    double secs;

    if (!rank){
        gettimeofday(&start, NULL);
    }
    
    init();

    for (int it = 1; it <= itmax; it++) {
        eps = 0.;
        local_eps = 0.;
        relax();
        // if (!rank) {printf("it=%4i   eps=%f\n", it, eps);}
        if (eps < maxeps) break;
    }
    
    verify();

    if (!rank) {
        gettimeofday(&stop, NULL);
        secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
        printf("thread=%d, N=%d: %f seconds\n", num_workers, N, secs);
    }

    MPI_Finalize();

    return 0;
}

void init() {
    for (int i = 0; i <= N - 1; i++)
        for (int j = 0; j <= N - 1; j++){
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) A[i][j] = 0.;
            else A[i][j] = (1. + i + j);
        }
}


void relax() {
    MPI_Status status;  
    MPI_Barrier(MPI_COMM_WORLD); 

    int up_send_tag = 2, down_send_tag = 3;

    for (int i = first_row; i < last_row; i++)
        for (int j = 1; j <= N - 2; j++){
            if ((i + j) % 2 == 1) {
                float b;
                b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
                local_eps = Max(fabs(b), local_eps);
                A[i][j] = A[i][j] + b;
            }}

    if (rank!=0) {
        MPI_Send(A[first_row], N, MPI_FLOAT, rank-1, up_send_tag, MPI_COMM_WORLD);
    }

    if (rank!=num_workers-1) {
        MPI_Recv(tmp_A_row, N, MPI_FLOAT, rank+1, up_send_tag, MPI_COMM_WORLD, &status);

        for (int j=1+(last_row % 2); j<=N-2; j+=2) {
            A[last_row][j] = tmp_A_row[j];
        }
    }

    if (rank!=num_workers-1) {
        MPI_Send(A[last_row-1], N, MPI_FLOAT, rank+1, down_send_tag, MPI_COMM_WORLD);
    }

    if (rank!=0){
        MPI_Recv(tmp_A_row, N, MPI_FLOAT, rank-1, down_send_tag, MPI_COMM_WORLD, &status);
        for (int j=1+((first_row-1) % 2); j<=N-2; j+=2) {A[first_row-1][j] = tmp_A_row[j];}
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    for (int i = first_row; i < last_row; i++)
        for (int j = 1; j <= N - 2; j++)
            if ((i + j) % 2 == 0) {
                float b;
                b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
                A[i][j] = A[i][j] + b;
            }

    if (rank!=0) {
        MPI_Send(A[first_row], N, MPI_FLOAT, rank-1, up_send_tag, MPI_COMM_WORLD);
    }

    if (rank!=num_workers-1) {
        
        MPI_Recv(tmp_A_row, N, MPI_FLOAT, rank+1, up_send_tag, MPI_COMM_WORLD, &status);

        for (int j=(last_row % 2); j<=N-2; j+=2) {
            A[last_row][j] = tmp_A_row[j];
        }
    }

    if (rank!=num_workers-1) {
        MPI_Send(A[last_row-1], N, MPI_FLOAT, rank+1, down_send_tag, MPI_COMM_WORLD);
    }

    if (rank!=0) {
        MPI_Recv(tmp_A_row, N, MPI_FLOAT, rank-1, down_send_tag, MPI_COMM_WORLD, &status);
        for (int j=((first_row-1) % 2); j<=N-2; j+=2) {
            A[first_row-1][j] = tmp_A_row[j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    MPI_Allreduce(&local_eps, &eps, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
}


void verify() {
    float local_sum, sum;

    local_sum = 0.;
    int begin = first_row;
    int end = last_row;
    if (first_row == 1) {begin = 0;}
    if (last_row == N - 1) {end = N - 1;}
    
    for (int i = begin; i <= end; i++)
        for (int j = 0; j <= N - 1; j++){
            local_sum = local_sum + A[i][j] * (i + 1) * (j + 1) / (N * N);
        }

    MPI_Reduce(&local_sum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) {printf("  S = %f\n", sum);}
}

// mpicc -std=c99 -o run_1 redb_2d.c -lm
