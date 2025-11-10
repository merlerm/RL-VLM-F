// powf_fix.c
#define _GNU_SOURCE
#include <math.h>

/* Provide the old symbol as a wrapper to powf */
float __powf_finite(float x, float y) {
    return powf(x, y);
}
