.PHONY: clean 3d
HIPCC = hipcc
HIP_FLAGS = -x hip -O2 -DUSE_HIP -Wno-unused-result

clean:
	rm -f 3d
	
3d:
	$(HIPCC) $(HIP_FLAGS) `mpicxx --showme:compile` `mpicxx --showme:link` -o 3d ../../src/gpuopt/3d.cu -lm


# export TMPDIR=$HOME/tmp  # Create a tmp directory in your home
# mkdir -p $HOME/tmp       # Make sure the directory exists
# make 3d                  # Try compiling again