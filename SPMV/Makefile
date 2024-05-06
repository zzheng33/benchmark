CC = mpicc
CFLAGS = -O2 -Wall
TARGET = sparse_matvec

all: $(TARGET)

$(TARGET): sparse_matvec.o
	$(CC) $(CFLAGS) -o $(TARGET) sparse_matvec.o

sparse_matvec.o: sparse_matvec.c
	$(CC) $(CFLAGS) -c sparse_matvec.c

clean:
	rm -f *.o $(TARGET)
