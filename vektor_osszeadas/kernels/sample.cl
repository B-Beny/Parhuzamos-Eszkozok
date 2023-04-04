__kernel void sample_kernel(__global int* v1, __global int* v2, __global int* output, int n)
{
    size_t i = get_global_id(0);
    if(i < n)
    {
        output[i] = v1[i] + v2[i];
    }
}
