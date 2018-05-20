CC=g++
FLAGS=-O3 -DNDEBUG -pthread -std=c++11 -Werror -Wall -Weffc++ -Wextra -pedantic -pedantic-errors 
LIBS=-larmadillo
INCLUDE=-I./ -isystem/usr/local/include/eigen3

all:
	$(CC) $(FLAGS) test_template.cpp ex_template.cpp \
	$(LIBS) $(INCLUDE) -o template

clean:
	rm -rf ./add
	
.PHONY: all release debug test install

