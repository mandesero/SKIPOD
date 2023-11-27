#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <mpi-ext.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/time.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (256 * 64 + 2)

float maxeps = 0.1e-7;
int itmax = 100;
float w = 0.5;
float eps, local_eps;

float A[N][N];
float copy_A[N][N];
float tmp_A_row[N];

int rank, num_workers, rc;
int first_row, last_row, n_rows;

int event = 0;
int KILL_PROC_RANK = -1;
MPI_Comm __MPI_COMM_WORLD;
jmp_buf jbuf;

void relax();
void init();
void verify();
int master_job();

void getMatrixCopy(float from_matrix[N][N], float to_matrix[N][N])
{
    for (int i = first_row; i < last_row; i++)
        for (int j = 1; j <= N - 2; j++)
            to_matrix[i][j] = from_matrix[i][j];
}

void init_params(int num_workers, int rank)
{
    n_rows = (N - 2) / num_workers;
    first_row = n_rows * rank + 1;
    if (rank != num_workers - 1)
        last_row = first_row + n_rows;
    else
        last_row = N - 1;
    printf("num_workers: %d, rank %d: first_row %d, last_row %d\n",
           num_workers, rank, first_row, last_row);
}

static void errorHandler(MPI_Comm *comm, int *err, ...)
{
    event = 1;

    int amount_f, len;
    int old_size;
    int old_rank;
    char errstr[MPI_MAX_ERROR_STRING];

    MPI_Group group_f;
    MPI_Group group_norm;
    MPI_Comm_rank(__MPI_COMM_WORLD, &old_rank);       // Получение ранга текущего процесса в группе __MPI_COMM_WORLD.
    MPI_Comm_size(__MPI_COMM_WORLD, &old_size);       // Получение количества процессов в группе __MPI_COMM_WORLD.
    int *norm_ranks = malloc(sizeof(int) * old_size); // Выделение памяти для массива norm_ranks.
    int *f_ranks = malloc(sizeof(int) * amount_f);    // Выделение памяти для массива f_ranks.

    if (old_rank == 0)
    {
        MPI_Comm_group(__MPI_COMM_WORLD, &group_norm);           // Получение группы процессов, связанных с коммуникатором __MPI_COMM_WORLD.
        MPIX_Comm_failure_ack(__MPI_COMM_WORLD);                 // Отметка сбоя процессов, связанных с коммуникатором __MPI_COMM_WORLD.
        MPIX_Comm_failure_get_acked(__MPI_COMM_WORLD, &group_f); // Получение всех сбойных процессы и сохранение их группы в group_f.
        MPI_Group_size(group_f, &amount_f);                      // Получение размера группы group_f и сохранение его в amount_f.
        for (int i = 0; i < amount_f; i++)
            f_ranks[i] = i;                                                            // Заполнение массива f_ranks значениями от 0 до amount_f - 1.
        MPI_Group_translate_ranks(group_f, amount_f, f_ranks, group_norm, norm_ranks); // Перевод рангов в группе group_f в ранги в группе group_norm и сохранение результатов в массив norm_ranks.
    }

    MPI_Error_string(*err, errstr, &len); // Преобразование кода ошибки *err в строку и сохранение её в errstr, а также сохранение длины строки в len.

    MPIX_Comm_shrink(*comm, &__MPI_COMM_WORLD);    // Сокращение коммуникатора *comm до __MPI_COMM_WORLD.
    MPI_Comm_rank(__MPI_COMM_WORLD, &rank);        // Получение ранга текущего процесса в группе __MPI_COMM_WORLD.
    MPI_Comm_size(__MPI_COMM_WORLD, &num_workers); // Получение количества процессов в группе __MPI_COMM_WORLD.
    MPI_Barrier(__MPI_COMM_WORLD);                 // Блокирование выполнения до тех пор, пока все процессы в коммуникаторе __MPI_COMM_WORLD не достигнут этой точки в программе.

    init_params(num_workers, rank);
    getMatrixCopy(copy_A, A);      // Копирование матрицы A в матрицу copy_A.
    MPI_Barrier(__MPI_COMM_WORLD); // Блокирование выполнения до тех пор, пока все процессы в коммуникаторе __MPI_COMM_WORLD не достигнут этой точки в программе.

    longjmp(jbuf, 1); // Вызов длинного перехода на установленную точку программы.
    free(norm_ranks);
    free(f_ranks);
}

int main(int an, char **as)
{
    srand(time(NULL));
    if (KILL_PROC_RANK == -1)
        KILL_PROC_RANK = rand() % 6 + 1;

    if ((rc = MPI_Init(&an, &as)))
    {
        printf("Execution Error%d", rc);
        MPI_Abort(MPI_COMM_WORLD, rc);
        return rc;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);        // Получение ранга текущего процесса в группе __MPI_COMM_WORLD.
    MPI_Comm_size(MPI_COMM_WORLD, &num_workers); // Получение количества процессов в группе __MPI_COMM_WORLD.
    __MPI_COMM_WORLD = MPI_COMM_WORLD;

    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(errorHandler, &errh);
    MPI_Comm_set_errhandler(__MPI_COMM_WORLD, errh);
    MPI_Barrier(__MPI_COMM_WORLD);

    init_params(num_workers, rank);

    struct timeval start, stop;
    double secs;
    if (!rank)
        gettimeofday(&start, NULL);

    init();
    getMatrixCopy(A, copy_A);
    for (int it = 1; it <= itmax; it++)
    {
        setjmp(jbuf); // checkpoint (goto if error)
        eps = 0.;
        local_eps = 0.;
        relax();
        if (eps < maxeps)
            break;
        getMatrixCopy(A, copy_A);
    }
    verify();

    if (!rank)
    {
        gettimeofday(&stop, NULL);
        secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
        printf("thread=%d, N=%d: %f seconds\n", num_workers, N, secs);
    }

    MPI_Finalize();

    return 0;
}

void init()
{
    for (int i = 0; i <= N - 1; i++)
        for (int j = 0; j <= N - 1; j++)
        {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
                A[i][j] = 0.;
            else
                A[i][j] = (1. + i + j);
        }
}

void relax()
{
    MPI_Status status;
    MPI_Barrier(__MPI_COMM_WORLD);

    int up_send_tag = 2, down_send_tag = 3;

    for (int i = first_row; i < last_row; i++)
        for (int j = 1; j <= N - 2; j++)
        {
            if ((i + j) % 2 == 1)
            {
                float b;
                b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
                local_eps = Max(fabs(b), local_eps);
                A[i][j] = A[i][j] + b;
            }
        }

    if (rank != 0)
        MPI_Send(A[first_row], N, MPI_FLOAT, rank - 1, up_send_tag, __MPI_COMM_WORLD);
    if (rank != num_workers - 1)
    {
        MPI_Recv(tmp_A_row, N, MPI_FLOAT, rank + 1, up_send_tag, __MPI_COMM_WORLD, &status);
        for (int j = 1 + (last_row % 2); j <= N - 2; j += 2)
            A[last_row][j] = tmp_A_row[j];
    }

    if ((event == 0) && (rank == KILL_PROC_RANK))
    {
        printf("killed proc with proc_rank: %d\n", KILL_PROC_RANK);
        raise(SIGKILL);
    }

    if (rank != num_workers - 1)
        MPI_Send(A[last_row - 1], N, MPI_FLOAT, rank + 1, down_send_tag, __MPI_COMM_WORLD);
    if (rank != 0)
    {
        MPI_Recv(tmp_A_row, N, MPI_FLOAT, rank - 1, down_send_tag, __MPI_COMM_WORLD, &status);
        for (int j = 1 + ((first_row - 1) % 2); j <= N - 2; j += 2)
            A[first_row - 1][j] = tmp_A_row[j];
    }

    MPI_Barrier(__MPI_COMM_WORLD);

    for (int i = first_row; i < last_row; i++)
        for (int j = 1; j <= N - 2; j++)
            if ((i + j) % 2 == 0)
            {
                float b;
                b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
                A[i][j] = A[i][j] + b;
            }

    if (rank != 0)
        MPI_Send(A[first_row], N, MPI_FLOAT, rank - 1, up_send_tag, __MPI_COMM_WORLD);
    if (rank != num_workers - 1)
    {
        MPI_Recv(tmp_A_row, N, MPI_FLOAT, rank + 1, up_send_tag, __MPI_COMM_WORLD, &status);
        for (int j = (last_row % 2); j <= N - 2; j += 2)
            A[last_row][j] = tmp_A_row[j];
    }
    if (rank != num_workers - 1)
        MPI_Send(A[last_row - 1], N, MPI_FLOAT, rank + 1, down_send_tag, __MPI_COMM_WORLD);
    if (rank != 0)
    {
        MPI_Recv(tmp_A_row, N, MPI_FLOAT, rank - 1, down_send_tag, __MPI_COMM_WORLD, &status);
        for (int j = ((first_row - 1) % 2); j <= N - 2; j += 2)
            A[first_row - 1][j] = tmp_A_row[j];
    }

    MPI_Barrier(__MPI_COMM_WORLD);

    MPI_Allreduce(&local_eps, &eps, 1, MPI_FLOAT, MPI_MAX, __MPI_COMM_WORLD);
}

void verify()
{
    float local_sum, sum;

    local_sum = 0.;
    int begin = first_row;
    int end = last_row;
    if (first_row == 1)
        begin = 0;
    if (last_row == N - 1)
        end = N - 1;

    for (int i = begin; i <= end; i++)
        for (int j = 0; j <= N - 1; j++)
            local_sum = local_sum + A[i][j] * (i + 1) * (j + 1) / (N * N);

    MPI_Reduce(&local_sum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, __MPI_COMM_WORLD);
    if (!rank)
        printf("  S = %f\n", sum);
}
