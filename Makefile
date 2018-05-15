CC=g++
FLAGS=-O3 -DNDEBUG -pthread -std=c++11 -Werror -Wextra -Weffc++ -pedantic -pedantic-errors

all:
	$(CC) $(FLAGS) test_optim.cpp matrix.cpp -I./ -o optim

clean:
	rm -rf ./optim
	
.PHONY: all release debug test install

