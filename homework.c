#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pthread.h"
#include "math.h"

#define min(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


typedef struct {
	char p[2];
	int x, y;
	int maxval;
	unsigned char *data;
}image;


image input;
image output;
float filterMatrix[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
float smooth[3][3] = {{1/(float)9, 1/(float)9, 1/(float)9}, {1/(float)9, 1/(float)9, 1/(float)9}, {1/(float)9, 1/(float)9, 1/(float)9}};
float blur[3][3] = {{1/(float)16, 2/(float)16, 1/(float)16}, {2/(float)16, 4/(float)16, 2/(float)16}, {1/(float)16, 2/(float)16, 1/(float)16}};
float sharpen[3][3] = {{0, -2/(float)3, 0}, {-2/(float)3, 11/(float)3, -2/(float)3}, {0, -2/(float)3, 0}};
float mean[3][3] = {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}};
float emboss[3][3] = {{0, 1, 0}, {0, 0, 0}, {0, -1, 0}};

void setFilter(float a[3][3]){
	int i, j;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			filterMatrix[i][j] = a[i][j];
}

float matrixMul(unsigned char a[3][3]) {
	float res = 0.0f;
	int i, j;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
				res += (float)a[i][j] * filterMatrix[i][j];
	return res;

}

void createOutput(MPI_Comm channel)
{

	int i, j, m, n, start, end;
	int rank;
	int nProcesses;
	MPI_Status status;
	MPI_Comm_rank(channel, &rank);
	MPI_Comm_size(channel, &nProcesses);

	start = ceil(output.x / (double)nProcesses) * rank;
	end = min(ceil(output.x / (double)nProcesses) * (rank + 1), output.x - 1);


	if (strncmp(input.p, "P6", 2) == 0) {
		if (rank == 0) {
			start ++;
			for (i = 0; i < output.x; i++) {
				for (j = 0; j < 3; j++) {
					output.data[i * output.y * 3 + j] = input.data[i * input.y * 3 + j];	
					output.data[i * output.y * 3 + output.y * 3 - 1 - j] = input.data[i * input.y * 3 + input.y * 3 - 1 - j];
				}
			}
			for (j = 0; j < output.y * 3; j++) {
				output.data[j] = input.data[j];
				output.data[(output.x - 1) * output.y * 3 + j] = input.data[(input.x - 1) * input.y * 3 + j];
			}
		}
		for (i = start; i < end; i++) {
			for (j = 3; j < output.y * 3 - 3; j += 3) {
				unsigned char aR[3][3];
				unsigned char aG[3][3];
				unsigned char aB[3][3];
				for (m = i - 1; m <= i + 1; m++) {
					for (n = j - 3; n <= j + 5; n += 3) {
						aR[m - i + 1][(n - j + 5) / 3] = input.data[m * output.y * 3 + n];
						aG[m - i + 1][(n - j + 5) / 3] = input.data[m * output.y * 3 + n + 1];
						aB[m - i + 1][(n - j + 5) / 3] = input.data[m * output.y * 3 + n + 2];
					}
				}

				output.data[i * output.y * 3 + j] = matrixMul(aR);
				output.data[i * output.y * 3 + j + 1] = matrixMul(aG);
				output.data[i * output.y * 3 + j + 2] = matrixMul(aB);
			}
		}

		if (rank == 0) {
			for (m = 1; m < nProcesses; m++) {
				MPI_Recv(input.data, output.x * output.y * 3, MPI_C_BOOL, MPI_ANY_SOURCE, 0, channel, &status);
				start = ceil(output.x / (double)nProcesses) * status.MPI_SOURCE;
				end = min(ceil(output.x / (double)nProcesses) * (status.MPI_SOURCE + 1), output.x - 1);
				for (i = start; i < end; i++) {
					for (j = 3; j < output.y * 3 - 3; j++) {
						output.data[i * output.y * 3 + j] = input.data[i * output.y * 3 + j];
					}
				}
			}
		} else {
	      MPI_Send(output.data, output.x * output.y * 3, MPI_UNSIGNED_CHAR, 0, 0, channel);
		}

	} else if (strncmp(input.p, "P5", 2) == 0) {
		if (rank == 0) {
			start++;
			for (i = 0; i < output.x; i++) {
				output.data[i * output.y] = input.data[i * input.y];
				output.data[i * output.y + output.y - 1] = input.data[i * input.y + input.y - 1];
			}
			for (j = 0; j < output.y; j++) {
				output.data[j] = input.data[j];
				output.data[(output.x - 1) * output.y + j] = input.data[(input.x - 1) * input.y + j];
			}
		}


		for (i = start; i < end; i++) {
			for (j = 1; j < output.y - 1; j++) {
				unsigned char a[3][3];
				for (m = i - 1; m <= i + 1 ; m++) {
					for (n = j - 1; n <= j + 1; n++) {
						a[m - i + 1][n - j + 1] = input.data[m * input.y + n];
					}
				}
				output.data[i * output.y + j] = matrixMul(a);
			}
		}

		if (rank == 0) {
			for (m = 1; m < nProcesses; m++) {
				MPI_Recv(input.data, output.x * output.y, MPI_C_BOOL, MPI_ANY_SOURCE, 0, channel, &status);
				start = ceil(output.x / (double)nProcesses) * status.MPI_SOURCE;
				end = min(ceil(output.x / (double)nProcesses) * (status.MPI_SOURCE + 1), output.x - 1);
				for (i = start; i < end; i++) {
					for (j = 1; j < output.y - 1; j++) {
						output.data[i * output.y + j] = input.data[i * output.y + j];
					}
				}
			}
		} else {
	      MPI_Send(output.data, output.x * output.y, MPI_UNSIGNED_CHAR, 0, 0, channel);
		}
	}
}


void applyFilter(MPI_Comm channel) {

	int rank;
	int nProcesses;
	MPI_Comm_rank(channel, &rank);
	MPI_Comm_size(channel, &nProcesses);

	if (strncmp(input.p, "P6", 2) == 0) {
		output.p[0] = input.p[0];
		output.p[1] = input.p[1];
		output.x = input.x;
		output.y = input.y;
		output.maxval = input.maxval;
		output.data = malloc(output.x * output.y * 3 * sizeof(unsigned char));
	} else 	if(strncmp(input.p, "P5", 2) == 0) {
		output.p[0] = input.p[0];
		output.p[1] = input.p[1];
		output.x = input.x;
		output.y = input.y;
		output.maxval = input.maxval;
		output.data = malloc(output.x * output.y * sizeof(unsigned char));

	}

	createOutput(channel);
}

void readInput(const char * fileName) {
	FILE *f = fopen(fileName, "rb");

	if (f == NULL) {
		return;
	}

	char format[2];
	int width, height, maxval;

	fscanf(f, "%s\n", format);
	fscanf(f, "%d %d\n", &width, &height);
	fscanf(f, "%d\n", &maxval);

	if (strcmp(format, "P6") == 0) {
		input.p[0] = format[0];
		input.p[1] = format[1];

		input.x = height;
		input.y = width;
		input.maxval = maxval;
		input.data = malloc(input.x * input.y * 3 * sizeof(unsigned char));

		fread(input.data, 3, input.y * input.x, f);

	} else if (strcmp(format, "P5") == 0) {
		input.p[0] = format[0];
		input.p[1] = format[1];

		input.x = height;
		input.y = width;
		input.maxval = maxval;
		input.data = malloc(input.x * input.y * sizeof(unsigned char));

		fread(input.data, 1, input.y * input.x, f);
	}
}


void writeData(const char * fileName) {
	FILE *f = fopen(fileName, "w");

	if (f == NULL) {
		return;
	}
	fwrite(output.p, 2, 1, f);
	fwrite("\n", 1, 1, f);
	fprintf(f, "%d %d\n%d\n", output.y, output.x, output.maxval);

	if (strncmp(output.p, "P6", 2) == 0) {
		fwrite(output.data, 3, output.y * output.x, f);
		
	} else if (strncmp(output.p, "P5", 2) == 0) {
		fwrite(output.data, 1, output.y * output.x, f);
	}

	free(output.data);

	fclose(f);
}


int main(int argc, char * argv[]) {
	int rank, i;
	int nProcesses;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

	if(argc < 4) {
		exit(-1);
	}
	if (rank == 0) {
		readInput(argv[1]);
	}
	MPI_Bcast(&input.p, 2, MPI_SIGNED_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(&input.x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&input.y, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank != 0) {
		if (strcmp(input.p, "P6") == 0) {
			input.data = malloc(input.x * input.y * 3 * sizeof(unsigned char));
		} else if (strcmp(input.p, "P5") == 0) {
			input.data = malloc(input.x * input.y * sizeof(unsigned char));
		}
	}

	if (strcmp(input.p, "P6") == 0) {
		MPI_Bcast(input.data, input.x * input.y * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	} else if (strcmp(input.p, "P5") == 0) {
		MPI_Bcast(input.data, input.x * input.y, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	}


	for (i = 3; i < argc; i++) {
		if(strcmp(argv[i], "smooth") == 0) {
			setFilter(smooth);
		}
		if(strcmp(argv[i], "blur") == 0) {
			setFilter(blur);
		}
		if(strcmp(argv[i], "sharpen") == 0) {
			setFilter(sharpen);
		}
		if(strcmp(argv[i], "mean") == 0) {
			setFilter(mean);
		}
		if(strcmp(argv[i], "emboss") == 0) {
			setFilter(emboss);
		}
		applyFilter(MPI_COMM_WORLD);
		if (i != argc) {
			if (strcmp(input.p, "P6") == 0) {
				if (rank == 0)
					memcpy(input.data, output.data, input.x * input.y * 3);
				MPI_Bcast(input.data, input.x * input.y * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
			} else if (strcmp(input.p, "P5") == 0) {
				if (rank == 0)
					memcpy(input.data, output.data, input.x * input.y);
				MPI_Bcast(input.data, input.x * input.y, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
			}
		}


		MPI_Barrier(MPI_COMM_WORLD);
	}

	free(input.data);

	if (rank == 0) {
		writeData(argv[2]);
	}


	MPI_Finalize();
	return 0;
	
}