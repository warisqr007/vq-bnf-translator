# cuda related
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# path related
eval "$(conda shell.bash hook)"
conda activate transformerv2

# python related
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg