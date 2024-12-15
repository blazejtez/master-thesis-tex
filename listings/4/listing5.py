__global__ void potential(cudaTextureObject_t texture_input,
                                            cudaSurfaceObject_t surface_output,
                                            float *xl, float *yl, float *zl,
                                            int XLEN, int YLEN, int ZLEN,
                                            float Z1, float Z2, float eps) {
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z * blockDim.z;
    float value = 0;
    float potential = 0;
    #define BLOCKSIZE 8 //have be same like the Potential.BLOCKSIZE class atribute
    __shared__ float xls[BLOCKSIZE];
    __shared__ float yls[BLOCKSIZE];
    __shared__ float zls[BLOCKSIZE];
    if((idx_x < XLEN) && (idx_y < YLEN) && (idx_z < ZLEN)){
        xls[threadIdx.x] = xl[idx_x];
        yls[threadIdx.y] = yl[idx_y];
        zls[threadIdx.z] = zl[idx_z];
        value = tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z);
        float r_squared = xls[threadIdx.x]*xls[threadIdx.x] + yls[threadIdx.y]*yls[threadIdx.y] + zls[threadIdx.z]*zls[threadIdx.z];
        float r = __frsqrt_rn(r_squared < eps ? eps : r_squared);
        value *= Z1*Z2/r;
        surf3Dwrite<float>(value, surface_output,idx_x*sizeof(float),idx_y,idx_z);
    }
}