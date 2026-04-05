.PHONY: linux win

CC		= g++
WCC		= x86_64-w64-mingw32-gcc-win32

CFLAGS	= -Wall -Wextra -Iinclude -std=c++20

EXE		= arduino-interface

SRC=$(wildcard src/*.cpp)

all: linux win

linux:
	$(CC) $(CFLAGS) -o $(EXE) $(SRC)

win:
	$(WCC) $(CFLAGS) -o $(EXE).exe $(SRC)