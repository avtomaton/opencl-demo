// faster is using int4 instead of each byte
__kernel void memset_int(__global int* mem, int val, int size) {
    int px = get_global_id(0);
    if (px >= size)
        return;

    mem[px] = val;
}

// possible optimization: use int4 types
// and corresponding arrays dimensions

__kernel void search_simple(__global int *dst,
    __global const char *a, int a_rows, int a_cols,
    __global const char *b, int b_rows, int b_cols)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px > a_cols - b_cols || py > a_rows - b_rows)
        return;

    int have_match = 1;
    for (int r = 0; r < b_rows; ++r)
    {
        for (int c = 0; c < b_cols; ++c)
        {
            have_match &=
                (a[(py + r) * a_cols + px + c] == b[r * b_cols + c]);
        }
    }
    dst[py * a_cols + px] = have_match;
}

__kernel void search_by_row(__global int *dst,
    __global const char *a, int a_rows, int a_cols,
    __global const char *b, int b_rows, int b_cols, int b_row)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px > a_cols - b_cols || py > a_rows - b_rows)
        return;

    // first row was not found
    if (b_row != 0 && dst[py * a_cols + px] < b_row)
        return;

    int have_match = 1;
    for (int c = 0; c < b_cols; ++c)
        have_match &= (a[(py + b_row) * a_cols + px + c] == b[b_row * b_cols + c]);

    dst[py * a_cols + px] += have_match;
}
