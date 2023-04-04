__kernel void rand_vector(__global int* output, int n)
{
    size_t GID = get_global_id(0);
    
    uint2 seed;
    seed.x = 6;
    seed.y = 9;

    if (GID < n)
    {
        uint Seed = seed.x + GID;
        uint t = Seed ^ (Seed << 11);  
        uint result = seed.y ^ (seed.y >> 19) ^ (t ^ (t >> 8));
        output[GID] = result % 50;
    }
}