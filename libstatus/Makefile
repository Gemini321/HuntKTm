all: compile link executable

compile:
	nvcc -Xcompiler -fPIC -x cu -rdc=true -c device.cpp -o device.o
	nvcc -Xcompiler -fPIC -x cu -rdc=true -c system.cpp -o system.o
	nvcc -Xcompiler -fPIC -x cu -rdc=true -c profiler.cpp -o profiler.o

link: compile
	nvcc \
	  -Xcompiler -fPIC --shared \
	  device.o system.o profiler.o \
	  -L/root/spack/opt/spack/linux-ubuntu22.04-zen2/clang-14.0.6/cuda-12.4.0-axu7qpl3bldv54ezkgo7hpw2loqheaot/lib64 \
	  -lcudart -lnvidia-ml -lcublas -lcurand -ldcgm \
	  -o libstatus.so

executable: link
	nvcc \
	  example.cpp \
	  -L. \
	  -L/root/spack/opt/spack/linux-ubuntu22.04-zen2/clang-14.0.6/cuda-12.4.0-axu7qpl3bldv54ezkgo7hpw2loqheaot/lib64 \
	  -lcudart -lnvidia-ml -lcublas -lcurand -lstatus -ldcgm \
	  -o example 

executable_no_so:
	nvcc \
	  example.cpp device.cpp system.cpp profiler.cpp \
	  -L/root/spack/opt/spack/linux-ubuntu22.04-zen2/clang-14.0.6/cuda-12.4.0-axu7qpl3bldv54ezkgo7hpw2loqheaot/lib64 \
	  -lcudart -lnvidia-ml -lcublas -lcurand -ldcgm \
	  -o example 
