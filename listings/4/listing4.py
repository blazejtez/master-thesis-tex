__global__ void test(cudaTextureObject_t texture_input,
                    cudaSurfaceObject_t surface_output,
                    int XLEN,
                    int YLEN,
                    int ZLEN,
                    float h) {
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z * blockDim.z;
    float value = tex3D<float>(texture_input, idx_x, idx_y, idx_z);

    if((idx_x < XLEN) && (idx_y < YLEN) && (idx_z < ZLEN)){
        value = -tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z)*6;
        value += (idx_x == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y, (float)idx_z));
        value += (idx_x == XLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y, (float)idx_z));
        value += (idx_y == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y - 1, (float)idx_z));
        value += (idx_y == YLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y + 1, (float)idx_z));
        value += (idx_z == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z - 1));
        value += (idx_z == ZLEN -1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z + 1));
        value /= h*h;
        surf3Dwrite<float>(value, surface_output,idx_x*sizeof(float),idx_y,idx_z);
    }
}