SRC := main.cpp

CPPC := mpicxx
CPPFLAGS := -Wall
LDFLAGS := -lgtest -lgtest_main -lpthread

.PHONY: all run-src clean

all: run-src

run-src: main.out
	@./main.out --test
	@mpiexec -np 4 ./main.out

main.out: $(SRC)
	$(CPPC) $(CPPFLAGS) $(SRC) $(LDFLAGS) -o main.out

clean:
	rm -f main.out
