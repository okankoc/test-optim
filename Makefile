CC=g++
FLAGS=-O3 -DNDEBUG -pthread -std=c++11 -Werror -Wall -Weffc++ -Wextra -pedantic -pedantic-errors 
LIBS=-larmadillo
INCLUDE=-I./ -isystem/usr/local/include/eigen3

all:
	$(CC) $(FLAGS) test_optim.cpp matrix.cpp $(LIBS) $(INCLUDE) -o optim

clean:
	rm -rf ./optim
	
.PHONY: all release debug test install

