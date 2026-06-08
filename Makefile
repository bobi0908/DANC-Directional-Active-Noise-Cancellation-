CC ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra

libfxlms.so: fxlms.c
	$(CC) $(CFLAGS) -shared -o $@ $<

clean:
	rm -f libfxlms.so

.PHONY: clean
