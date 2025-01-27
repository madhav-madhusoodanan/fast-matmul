#include <immintrin.h>
#include <stdalign.h>

/* 
    The AVX512-backed library for matrix multiplication

    1. Targeting very specific dimensions (multiples of 32 bits)
    2. Targeting very specific types (binary strings)
    3. The input are many 8x8-dimension vector
 */
const MATRIX_LEN = 16;
typedef __m512i Matrix[MATRIX_LEN];

__m512i matmul(
        __m512i a, 
        Matrix b
    ) {

    /* 
        Logical steps:
        1. load all rows (from top to bottom) from A and fill an AVX512 register (zmmA)
        2. Load a column (from left to right) from B and repeat across another AVX512 register (zmmB)
        3. Set a result register (zmmC) and a scratch register (zmmD) 
        4. in a loop, perform the following:
            a. vpandq zmmD zmmA zmmB                              ;Bitwise AND  (0.5 CPI, 1 latency)
            b. vpopcnt{b/w/d/q} zmmD, zmmD                        ;Count number of 1 bits in each segment  (1 CPI, 3 latency)
            c. vpternlogq zmmC, zmmD, 0x0101010101010101*8, 0xF8  ;Write count%2 to result for each segment  (0.5 CPI, 1 latency)
            d. vprolq zmmC, zmmC, 0x01                            ;shift values left by 1 step  (1 CPI, 1 latency)
        

        Dimension of A: {64*8, 32*16, 16*32, 8*64} 
        Dimension of B: {8, 16, 32, 64} * n
        Output dimension: {64, 32, 16, 8} * n

        Note: Highly recommend "n" to be the number of columns in A for max usage of register
        Note: "n" to be as max as possible to increase valuable computation (and reduce copying)
        Note: for max usage of all 32 AVX512 registers, make n = 16 and 7 matrices of A ( = 7 result registers)
              implies a calculation of [(7 * 16) * 32] x [32 * 16] in one complete register filling

        Note: Additionally, this operation can be parallelized across cores

        Pending: Create a function that performs matrix breaking operations 
                 and delegates multiplications to these low-level functions

        Pending: Create a function that creates worker thread pools
                 and delegates matmul operations to threads 


     */
    for (unsigned short int i = 0; i < MATRIX_LEN; i++)
    {
        __m512i out = _mm512_and_si512(a, b[i]);
        __m512i ones = _mm512_popcnt_epi64(out);
        __m512i rot1 = _mm512_ror_epi64(ones, i+1);
        __m512i rot2 = _mm512_bslli_epi128(rot1, 16);
        __m512i rot3 = _mm512_bslli_epi128(rot2, 16);
        __m512i rot4 = _mm512_bslli_epi128(rot3, 16);
        __m512i rot5 = _mm512_bslli_epi128(rot4, 8);
        __m512i out1 = _mm512_and_epi64(rot1, rot3);
        __m512i out2 = _mm512_and_epi64(out1, rot4);
        __m512i out3 = _mm512_and_epi64(out1, rot5);


    }
    // __m512i diff = _mm512_ternarylogic_epi32(expected_output, actual_output, update_mask, 0x28);


}