.PHONY: all native linux mac

CXX			?= g++

CXXFLAGS	= -Wall -Wextra -Iinclude -std=c++20

EXE			= arduino-interface

SRC			=$(wildcard src/*.cpp)

all: native

native:
	$(CXX) $(CXXFLAGS) -o $(EXE) $(SRC)

linux: native

mac: native