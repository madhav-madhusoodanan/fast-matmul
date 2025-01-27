#include <immintrin.h>
#include <stdalign.h>
#include <stdio.h>

// Helper function to print 64-bit integers in binary
void print_binary_64(unsigned long long x) {
    for (int i = 63; i >= 0; i--) {
        printf("%lld", (x >> i) & 1);
        if (i % 8 == 0) printf(" "); // Space every 8 bits for readability
    }
    printf("\n");
}

// Helper function to print __m512i contents
void print_m512i(__m512i v) {
    unsigned long long arr[8];
    _mm512_store_epi64(arr, v);
    
    for (int i = 0; i < 8; i++) {
        printf("Element %d:\n", i);
        printf("  Hex: 0x%016llx\n", arr[i]);
        printf("  Bin: ");
        print_binary_64(arr[i]);
        printf("\n");
    }
}