.PHONY: all native linux mac win

CXX		?= g++
WCC		= x86_64-w64-mingw32-gcc-win32

CXXFLAGS	= -Wall -Wextra -Iinclude -std=c++20

EXE		= arduino-interface

SRC=$(wildcard src/*.cpp)

all: native

native:
	$(CXX) $(CXXFLAGS) -o $(EXE) $(SRC)

linux: native

mac: native

win:
	$(WCC) $(CXXFLAGS) -o $(EXE).exe $(SRC)