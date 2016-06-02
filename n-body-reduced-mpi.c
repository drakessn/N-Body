#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glut.h>
#include <assert.h>
#include <mpi.h>

#define N 1500 /**Numero de particulas**/
#define T 1500 /**Numero de iteraciones 600**/
#define S 3e11 /**x,y,z 3e11**/
#define DIM 2
#define X 0 
#define Y 1
#define thread_count 4 /**Numero de threads**/
const double G = 6.673e-11f; /**Constante de Gravedad**/
const double delta_t = 3600.0; /**Intervalo de tiempo**/
const double EARTH_MASS	= 5.9742e24;	// kg
const double EARTH_SUN	= 149600000;	// meters
const double SPEED_FACTOR	= 1.45e10;

static double pos[N][DIM];
static double forces[N][DIM];
//static double newPos[N][DIM];
static double vel[N][DIM];
//static double newVel[N][DIM];
static double masses[N];

MPI_Status status;
static int nproc; /* numero de nodos MPI */
static int myid; /* mi propio rank */
static int num_per_thread;

float Ranf( float, float );

void solve(int ite){
	/**iteraciones por cada intervalo de tiempo ***/
	for(int t=0;t<ite;t++){			
		/**Calculo de la fuerza total que actua en cada una de las particulas 
		 * que interctuan entre si ***/
		int source = (myid+1)%nproc;
		int dest = (myid+nproc-1)%nproc;
				
		double loc_pos[num_per_thread][DIM];
		double tmp_pos[num_per_thread][DIM];
		double loc_masses[num_per_thread];
		double tmp_masses[num_per_thread];
		double loc_forces[N][DIM];
		double tmp_forces[N][DIM];
		if(myid==0){
			for(int i=1; i<nproc; i++){
				for(int j=i, k=0 ; k<num_per_thread ;k++,j+=nproc){
					loc_pos[k][X] = pos[j][X];
					loc_pos[k][Y] = pos[j][Y];
					loc_masses[k] = masses[j];
				}
				MPI_Send(loc_pos, num_per_thread*DIM, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
                MPI_Send(loc_masses, num_per_thread*DIM, MPI_DOUBLE, i, 2, MPI_COMM_WORLD);
			}
			for(int j=0, k=0 ; k<num_per_thread ;k++,j+=nproc){
				loc_pos[k][X] = pos[j][X];
				loc_pos[k][Y] = pos[j][Y];
				loc_masses[k] = masses[j];
			}
		}
		else{
			MPI_Recv(loc_pos, num_per_thread*DIM, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(loc_masses, num_per_thread*DIM, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
		}		
		for(int i=0; i<num_per_thread ;i++){
			loc_forces[i][X] = 0;
			loc_forces[i][Y] = 0;
			tmp_forces[i][X] = 0;
			tmp_forces[i][Y] = 0;
			tmp_pos[i][X] = loc_pos[i][X];
			tmp_pos[i][Y] = loc_pos[i][Y];
			tmp_masses[i] = loc_masses[i];
			tmp_masses[i] = loc_masses[i];
		}
		//Compute forces due to interactions among local particles
		for(int q=0; q<num_per_thread-1 ;q++){
			for(int k=q+1; k<num_per_thread ;k++){
				double x_diff = loc_pos[q][X]-tmp_pos[k][X];
				double y_diff = loc_pos[q][Y]-tmp_pos[k][Y];
				double dist = sqrt(x_diff*x_diff + y_diff*y_diff);
				double dist_cubed = dist*dist*dist;
				if(dist_cubed<0)
					dist_cubed = -1*dist_cubed;
				double force_qkx = G*loc_masses[q]*tmp_masses[k]/dist_cubed * x_diff;
				double force_qky = G*loc_masses[q]*tmp_masses[k]/dist_cubed * y_diff;
				loc_forces[q][X] -= force_qkx;
				loc_forces[q][Y] -= force_qky;
				tmp_forces[k][X] += force_qkx;
				tmp_forces[k][Y] += force_qky;
			}
		}
		for(int phase=1; phase<nproc; phase++){
			MPI_Sendrecv_replace(tmp_masses, num_per_thread, MPI_DOUBLE, dest, 1, source, 1, MPI_COMM_WORLD, &status);
			MPI_Sendrecv_replace(tmp_pos, num_per_thread*DIM, MPI_DOUBLE, dest, 2, source, 2, MPI_COMM_WORLD, &status);
			MPI_Sendrecv_replace(tmp_forces, num_per_thread*DIM, MPI_DOUBLE, dest, 3, source, 3, MPI_COMM_WORLD, &status);
			for(int q=0; q<num_per_thread-1 ;q++){
				for(int k=q+1; k<num_per_thread ;k++){
					double x_diff = loc_pos[q][X]-tmp_pos[k][X];
					double y_diff = loc_pos[q][Y]-tmp_pos[k][Y];
					double dist = sqrt(x_diff*x_diff + y_diff*y_diff);
					double dist_cubed = dist*dist*dist;
					if(dist_cubed<0)
						dist_cubed = -1*dist_cubed;
					double force_qkx = G*loc_masses[q]*tmp_masses[k]/dist_cubed * x_diff;
					double force_qky = G*loc_masses[q]*tmp_masses[k]/dist_cubed * y_diff;
					loc_forces[q][X] -= force_qkx;
					loc_forces[q][Y] -= force_qky;
					tmp_forces[k][X] += force_qkx;
					tmp_forces[k][Y] += force_qky;
				}
			}
		}
		MPI_Sendrecv_replace(tmp_masses, num_per_thread, MPI_DOUBLE, dest, 1, source, 1, MPI_COMM_WORLD, &status);
		MPI_Sendrecv_replace(tmp_pos, num_per_thread*DIM, MPI_DOUBLE, dest, 2, source, 2, MPI_COMM_WORLD, &status);
		MPI_Sendrecv_replace(tmp_forces, num_per_thread*DIM, MPI_DOUBLE, dest, 3, source, 3, MPI_COMM_WORLD, &status);
		for(int k=0; k<num_per_thread ;k++){
			loc_forces[k][X] += tmp_forces[k][X];
			loc_forces[k][Y] += tmp_forces[k][Y];
		}
		if(myid==0){		
			for(int k=0 ; k<num_per_thread ;k++){
				forces[k*nproc][X] = loc_forces[k][X];
				forces[k*nproc][Y] = loc_forces[k][Y];
			}	
			for(int i=1; i<nproc; i++){
				MPI_Recv(loc_forces, num_per_thread*DIM, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
				for(int k=0 ; k<num_per_thread ;k++){
					forces[k*nproc+i][X] = loc_forces[k][X];
					forces[k*nproc+i][Y] = loc_forces[k][Y];
				}	
			}
		}
		else{
			MPI_Send(loc_forces, num_per_thread*DIM, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		}	
		double newVel[num_per_thread][DIM];
		double bVel[num_per_thread][DIM];
		double newPos[num_per_thread][DIM];
		
		MPI_Scatter(pos, num_per_thread*DIM, MPI_DOUBLE, newPos, num_per_thread*DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(vel, num_per_thread*DIM, MPI_DOUBLE, newVel, num_per_thread*DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(masses, num_per_thread, MPI_DOUBLE, loc_masses, num_per_thread, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(vel, num_per_thread*DIM, MPI_DOUBLE, bVel, num_per_thread*DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(forces, num_per_thread*DIM, MPI_DOUBLE, loc_forces, num_per_thread*DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
		//printf("antes: %f from %d\n",newPos[1][0],myid);	
		for(int i=0; i<num_per_thread ;i++){
			newPos[i][X] += delta_t*bVel[i][X];
			newPos[i][Y] += delta_t*bVel[i][Y];
			newVel[i][X] += delta_t/loc_masses[i]*loc_forces[i][X];
			newVel[i][Y] += delta_t/loc_masses[i]*loc_forces[i][Y];
		}
		//printf("despi: %f from %d\n",newPos[1][0],myid);
		MPI_Gather(newPos, num_per_thread*DIM, MPI_DOUBLE, pos, num_per_thread*DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(newVel, num_per_thread*DIM, MPI_DOUBLE, vel, num_per_thread*DIM, MPI_DOUBLE, 1, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
			/**Calculo de la posicion y velocidad de cada
			 * particula ***
			for(int q=0;q<N;q++){
				pos[q][X] = newPos[q][X];
				pos[q][Y] = newPos[q][Y];
				vel[q][X] = newVel[q][X];
				vel[q][Y] = newVel[q][Y];
			}*/
	}
}
 
 
void start(){
	if (myid == 0) {
		masses[0] = EARTH_MASS*500000.0;
		pos[0][X]=0*EARTH_SUN;
		pos[0][Y]=0*EARTH_SUN;
		vel[0][X]=0*EARTH_SUN;
		vel[0][Y]=0*EARTH_SUN;
		
		for(int i=1;i<N;i++){
			double distance = 0;
			masses[i] = EARTH_MASS*Ranf( 0.999f, 1.0f );
			do {
				pos[i][X]=EARTH_SUN*Ranf( -500.0f, 500.f );
				pos[i][Y]=EARTH_SUN*Ranf( -500.0f, 500.f );
				distance = sqrt(pos[i][Y]*pos[i][Y]+pos[i][X]*pos[i][X]);
			}while(distance<EARTH_SUN*100 );
					vel[i][X]=pos[i][Y]/(distance*sqrt(distance))*SPEED_FACTOR;
					vel[i][Y]=-pos[i][X]/(distance*sqrt(distance))*SPEED_FACTOR;
		}
		printf("Procesando...\n");
	}
	double time0 = MPI_Wtime();
    solve(T);
	double time1 = MPI_Wtime();;
	/**Impresion del tiempo que tardo el algoritmo para determinar las
	 * posiciones y velocidades de las particulas ***/
	if (myid == 0) {
		printf("duracion: %f\n",time1-time0);	
	}
}
 
void init(){
    glClearColor(0,0,0,0);    
}

void reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0-S, S, 0-S, S, 0-S, S);
    glMatrixMode(GL_MODELVIEW);
}
 
void display(){
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1,1,1);
    glLoadIdentity();
    for(int i=0;i<N;i++){
		glBegin(GL_POINTS);
			glVertex3f(pos[i][X], pos[i][Y], 0.0f);
		glEnd();  
	}
    glFlush();
}

void idle(){
    display();
    solve(1);
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    num_per_thread = N/nproc;
    start();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowPosition(1, 1);
	glutInitWindowSize(700, 700);
		glutCreateWindow("N BODY");
	init();
	glutDisplayFunc(display); 
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutMainLoop();
    MPI_Finalize();
	return 0;
}

float Ranf( float low, float high ) {
	float r = (float) rand();		// 0 - RAND_MAX
	return(low + r*(high-low) / (float)RAND_MAX);
}
