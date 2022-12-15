#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))


#include <omp.h>


#define N (256 * 64 + 2)
float   maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
float w = 0.5;
float eps;

float A [N][N];

void relax();
void init();
void verify();
int main(int an, char **as) {
    int status;
    double time = omp_get_wtime();

    status = run_read_black_2d();

    printf("nThreads = %d\n", omp_get_max_threads());
    printf("time = %f\n", omp_get_wtime() - time);
    return status;
}

int run_read_black_2d(){
    int it;
    init();

    for (it = 1; it <= itmax; it++) {
        eps = 0.;
        relax();
        // printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps) break;
    }

    verify();

    return 0;
}


void init() {

	#pragma omp parallel for collapse(2) schedule(static)
	for(j=0; j<=N-1; j++)
		for(i=0; i<=N-1; i++) {
			if(i==0 || i==N-1 || j==0 || j==N-1)
				A[i][j]= 0.;
			else
				A[i][j]= ( 1. + i + j);
		}
}


void relax() {

	#pragma omp parallel for  schedule(static) collapse(2) reduction(max : eps)
	for(j=1; j<=N-2; j++)
		for(i=1; i<=N-2; i++)
			if ((i + j) % 2 == 1) {

				float b;
				b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
				eps =  Max(fabs(b),eps);
				A[i][j] = A[i][j] + b;
			}


	#pragma omp parallel for collapse(2) schedule(static)
	for(j=1; j<=N-2; j++)
		for(i=1; i<=N-2; i++)
			if ((i + j) % 2 == 0) {
				float b;
				b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
				A[i][j] = A[i][j] + b;
			}
}


void verify() {

	float s;
	s=0.;

	#pragma omp parallel for  schedule(static) collapse(2) reduction (+: s)
	for(j=0; j<=N-1; j++)
		for(i=0; i<=N-1; i++) {
			s=s+A[i][j]*(i+1)*(j+1)/(N*N);
		}

	printf("  S = %f\n",s);
}
