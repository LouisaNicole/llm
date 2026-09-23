CXX = g++
CC = gcc

WARNFLAGS = -Wall -Wno-unused-function -Wno-unused-variable
OPT := -O2
CFLAGS += $(WARNFLAGS) $(OPT) -g
CXXFLAGS += -Ithirdparty -Ithirdparty/ggml -Ithirdparty/mastik-0.02/src $(WARNFLAGS) $(OPT) -g -std=c++17
LDFLAGS += $(OPT) -g -lpthread 

.PHONY: all clean

all: collect 

gen-dataset: gen-dataset.o thirdparty/ggml/gguf.c.o thirdparty/ggml/tokenizer.o util.o
	$(CXX) $^ $(LDFLAGS) -o $@

collect: collect.o config-database.o attack.o thirdparty/ggml/gguf.c.o thirdparty/ggml/tokenizer.o util.o
	$(CXX) $^ $(LDFLAGS) -o $@

collect-cache-trace: collect-cache-trace.o config-database.o attack.o thirdparty/ggml/gguf.c.o thirdparty/ggml/tokenizer.o util.o
	$(CXX) $^ $(LDFLAGS) -o $@

exp_mastik: exp_mastik.o thirdparty/mastik-0.02/src/fr.o thirdparty/mastik-0.02/src/util.o thirdparty/mastik-0.02/src/vlist.o thirdparty/mastik-0.02/src/timestats.o util.o
		$(CXX) $^ $(LDFLAGS) -o $@

# tools

tsc-info: tsc-info.o
	$(CXX) $^ $(LDFLAGS) -o $@

-include $(wildcard *.d)

%.c.o: %.c $(wildcard %.d)
	$(CC) -MMD -MP -c $< $(CFLAGS) -o $@

%.o: %.cc $(wildcard %.d)
	$(CXX) -MMD -MP -c $< $(CXXFLAGS) -o $@

clean:
	-rm ./*.o ./*.c.o ./*.d
	-rm ./main ./main.exe
	-rm ./collect ./collect.exe

