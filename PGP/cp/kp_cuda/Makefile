all: gpu
gpu: main.o model.o TexturePlane.o gpu_render.o
	nvcc --std=c++11 -Werror cross-execution-space-call -lm main.o model.o TexturePlane.o gpu_render.o -o gpu
main.o: main.cu Matrix.h model.h TexturePlane.h gpu_render.h
	nvcc -c main.cu -dc
model.o: model.cu model.h
	nvcc -c model.cu -dc
TexturePlane.o: TexturePlane.cu TexturePlane.h
	nvcc -c TexturePlane.cu -dc
gpu_render.o: gpu_render.cu gpu_render.h
	nvcc -c gpu_render.cu -dc
clean:
	rm -rf *.o gpu