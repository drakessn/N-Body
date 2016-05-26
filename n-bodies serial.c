#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define N 100 /**Numero de particulas**/
#define T 1e5 /**Numero de iteraciones**/
#define X 0 
#define Y 1
#define G 6.673e-11f /**Constante de Gravedad**/
#define delta_t 1e-2f /**Intervalo de tiempo**/
//#define G 0.00000000006673

int main()
{
	double forces[N][2];
	double pos[N][2];
	double masses[N];
	double vel[N][2];
	double x_diff, y_diff, dist, dist_cubed;
	/**inicializacion de las particulas con valores 
	 * aleatorios ***/
	for(int i=0;i<N;i++){
		for(int j=0;j<2;j++){
			forces[i][j]=0.0;
			pos[i][j]=rand()%N;
			vel[i][j]=rand()%N;
		}
		masses[i]=rand()%N+1;
	}
	/**iteraciones por cada intervalo de tiempo ***/
	for(int t=0;t<T;t++){			
		/**Impresion de las posiciones y velocidades de
		 * las particulas cada cierto intervalo de tiempo ***/
		if(!(t%100)){					
			for(int q=0;q<N;q++){
				printf("particula nro: %d\tx: %.4f\ty: %.4f\tvx: %.4f\tvy: %.4f\n",q+1,pos[q][X],pos[q][Y],vel[q][X],vel[q][Y]);
			}
			printf("timestep output: %d\n",t);
		}
		/**Calculo de la fuerza total que actua en
		 * cada una de las particulas que interctuan
		 * entre si ***/
		for(int q=0;q<N;q++){
			for(int k=0;k<N;k++){
				if(k!=q){
					x_diff = pos[q][X]-pos[k][X];
					y_diff = pos[q][Y]-pos[k][Y];
					dist = sqrt(x_diff*x_diff + y_diff*y_diff);
					dist_cubed = dist*dist*dist;
					forces[q][X] -= G*masses[q]*masses[k]/dist_cubed * x_diff;
					forces[q][Y] -= G*masses[q]*masses[k]/dist_cubed * y_diff;
				}
			}
		}
		/**Calculo de la posicion y velocidad de cada
		 * particula ***/
		for(int q=0;q<N;q++){
			pos[q][X] += delta_t*vel[q][X];
			pos[q][Y] += delta_t*vel[q][Y];
			vel[q][X] += delta_t/masses[q]*forces[q][X];
			vel[q][Y] += delta_t/masses[q]*forces[q][Y];
		}
	}
	/**Impresion de las posiciones y velocidades de
	 * las particulas ***/
	for(int q=0;q<N;q++){
		printf("particula nro: %d\tx: %.4f\ty: %.4f\tvx: %.4f\tvy: %.4f\n",q+1,pos[q][X],pos[q][Y],vel[q][X],vel[q][Y]);
	}
	printf("timestep output: %.0f\n",T);

	return 0;
}
