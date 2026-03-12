
BINARIES = main-two-cvs-while main-two-cvs-if main-one-cv-while
HEADERS = mythreads.h main-header.h main-common.c pc-header.h

all: $(BINARIES)

clean:
	rm -f $(BINARIES)

main-one-cv-while: main-one-cv-while.c $(HEADERS)
	gcc -o main-one-cv-while main-one-cv-while.c -Wall -Wno-pointer-to-int-cast -pthread

main-two-cvs-if: main-two-cvs-if.c $(HEADERS)
	gcc -o main-two-cvs-if main-two-cvs-if.c -Wall -Wno-pointer-to-int-cast -pthread

main-two-cvs-while: main-two-cvs-while.c $(HEADERS)
	gcc -o main-two-cvs-while main-two-cvs-while.c -Wall -Wno-pointer-to-int-cast -pthread

