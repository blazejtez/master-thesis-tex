texture <float, cudaTextureType3D, cudaReadModeElementType> tx_input
surface <float, cudaSurfaceType3D> surf_output

__global__ void Laplace3D_K5 ()
{
    ...
    if((i<size) && (j<size) && (k<size))
    {
        //Add left cell value
        if (i==0) value += tex2D(tx2_left, j, k);
        else value += tex3D(tx_input, i-1, j, k);
        
        // Similarly add 5 other points here
        
        surf3Dwrite(value/6.0, surf_output, i * sizeof(float), j, k);
    }

}