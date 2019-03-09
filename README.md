Works on .pnm and .pgm files

Run with: mpiexec -np <number_of_processes> ./homework <input_file> <output_file> <any_number_of_filters> 

Example: mpiexec -np 4 ./homework test.pnm out.pnm blur sharpen smooth mean emboss 
