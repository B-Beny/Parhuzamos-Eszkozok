int pow_mod(int a, int b, int n);

__kernel void sample_kernel(int n, int d, int s, int minv, __global int* output)
{
    size_t gid = get_global_id(0);
    double x, y = 0;

    //printf("n=%d, d=%d, s=%d, minv=%d, gid=%d\n", n,d,s,minv,gid);

    for(int i = 2; i < minv; i++)
    {
        x = pow_mod(i, d, n);
        //printf("x=%d\n",x);
        for(int j = 0; j <= s; j++)
        {
            y = pow_mod(x, 2, n);
            //printf("y=%d\n",y);
            if( y == 1 && x != 1 && x != n - 1)
            {
                *output = 0;
                //printf("First if.");
                return;
            }
            x = y;
        }
        if( y != 1)
        {
            *output = 0;
            //printf("Second if.");
            return;
        }
    }
    *output = 1;
    return;
}

int pow_mod(int a, int b, int n) {
  int res = 1;
  a = a % n;
  while (b > 0) {
    if (b & 1) {
      res = (res * a) % n;
    }
    b = b >> 1;
    a = (a * a) % n;
  }
  return res;
}