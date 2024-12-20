.PHONY: clean 3d

clean:
	rm -f 3d
	
3d:
	mpicxx -O3 -o 3d ../../src/mpi/3d.cpp -lm


# export TMPDIR=$HOME/tmp  # Create a tmp directory in your home
# mkdir -p $HOME/tmp       # Make sure the directory exists
# make 3d                  # Try compiling again