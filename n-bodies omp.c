#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100 /**Numero de particulas**/
#define T 1e4 /**Numero de iteraciones**/
#define DIM 2
#define X 0 
#define Y 1
#define thread_count 4 /**Numero de threads**/
const double G = 6.673e-11f; /**Constante de Gravedad**/
const double delta_t = 1.0; /**Intervalo de tiempo**/
const double EARTH_MASS	= 5.9742e24;	// kg
const double EARTH_DIAMETER	= 12756000.32;	// meters


float Ranf( float, float );

int main()
{
	double forces[N][DIM];
	double loc_forces[thread_count][N][DIM];
	double pos[N][DIM];
	double masses[N];
	double vel[N][DIM];
	double force_qk[thread_count][DIM];
	double x_diff[thread_count], y_diff[thread_count], dist[thread_count], dist_cubed[thread_count];
	double time0 = omp_get_wtime();
	/**inicializacion de las particulas con valores 
	 * aleatorios ***/
	#pragma omp parallel for schedule(static, N/thread_count)
	for(int i=0;i<N;i++){
		for(int j=0;j<2;j++){
			forces[i][j]=0.0;
			pos[i][j]=EARTH_DIAMETER*Ranf( -100.f, 100.f );
			vel[i][j]=Ranf( -100.f, 100.f );
		}
		masses[i] = EARTH_MASS*Ranf( 0.5f, 10.f );
	}
	#pragma omp parallel for schedule(static, N/thread_count)
	for(int t=0;t<thread_count;t++){
		for(int i=0;i<N;i++){
			for(int j=0;j<2;j++){
				loc_forces[t][i][j]=0;
			}
		}
	}
	/**iteraciones por cada intervalo de tiempo ***/
	for(int t=0;t<T;t++){			
		/**Impresion de las posiciones y velocidades de
		 * las particulas cada cierto intervalo de tiempo ***/
		/*
		if(!(t%100)){
			#pragma omp single					
			for(int q=0;q<N;q++){
				printf("particula nro: %d\tx: %.4f\ty: %.4f\tvx: %.4f\tvy: %.4f\n",q+1,pos[q][X],pos[q][Y],vel[q][X],vel[q][Y]);
			}
			#pragma omp single
			printf("timestep output: %d\n",t);
		}
		*/
		/**Calculo de la fuerza total que actua en
		 * cada una de las particulas que interctuan
		 * entre si ***/
		#pragma omp parallel for schedule(static, N/thread_count)
		for(int q=0;q<N;q++){
			for(int k=0;k<N;k++){
				if(k>q){
					int my_rank = omp_get_thread_num();
					x_diff[my_rank] = pos[q][X]-pos[k][X];
					y_diff[my_rank] = pos[q][Y]-pos[k][Y];
					dist[my_rank] = sqrt(x_diff[my_rank]*x_diff[my_rank] + y_diff[my_rank]*y_diff[my_rank]);
					dist_cubed[my_rank] = dist[my_rank]*dist[my_rank]*dist[my_rank];
					force_qk[my_rank][X] = G*masses[q]*masses[k]/dist_cubed[my_rank] * x_diff[my_rank];
					force_qk[my_rank][Y] = G*masses[q]*masses[k]/dist_cubed[my_rank] * y_diff[my_rank];
					
					loc_forces[my_rank][q][X] += force_qk[my_rank][X]; 
					loc_forces[my_rank][q][Y] += force_qk[my_rank][Y];
					loc_forces[my_rank][k][X] -= force_qk[my_rank][X];
					loc_forces[my_rank][k][Y] -= force_qk[my_rank][Y];
				}
			}
		}
		/**Segunda fase suma las fuerzas de cada
		 * particula ***/
		# pragma omp parallel for schedule(static, N/thread_count)
		for(int q=0;q<N;q++){
			forces[q][X]=0.0;
			forces[q][Y]=0.0;
			for(int t=0;t<thread_count;t++){
				forces[q][X]+=loc_forces[t][q][X];
				forces[q][Y]+=loc_forces[t][q][Y];
			}
		}
		/**Calculo de la posicion y velocidad de cada
		 * particula ***/
		# pragma omp parallel for schedule(static, N/thread_count)
		for(int q=0;q<N;q++){
			pos[q][X] += delta_t*vel[q][X];
			pos[q][Y] += delta_t*vel[q][Y];
			vel[q][X] += delta_t/masses[q]*forces[q][X];
			vel[q][Y] += delta_t/masses[q]*forces[q][Y];
		}
	}
	double time1 = omp_get_wtime();
	/**Impresion de las posiciones y velocidades de
	 * las particulas ***/
	for(int q=0;q<N;q++){
		printf("particula nro: %d\tx: %.4f\ty: %.4f\tvx: %.4f\tvy: %.4f\n",q+1,pos[q][X],pos[q][Y],vel[q][X],vel[q][Y]);
		//printf("particula nro: %d\tfx: %.4f\tfy: %.4f\tmasa: %.4f\n",q+1,forces[q][X],forces[q][Y],masses[q]);
	}
	printf("timestep output: %.0f\n",T);
	printf("duracion: %f\n",time1-time0);

	return 0;
}

float Ranf( float low, float high ) {
	float r = (float) rand();		// 0 - RAND_MAX
	return(low + r*(high-low) / (float)RAND_MAX);
}
