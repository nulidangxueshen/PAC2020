all: logVS

.PHONY: logVS

CC	= icpc
CFLAGS	= -std=c++11

logVS :
	$(CC) $(CFLAGS) -O3 -qopenmp -fno-alias -xCORE-AVX512 -qopt-zmm-usage=high -o logVS main.cpp

.PHONY: clean
	rm -f logVS
