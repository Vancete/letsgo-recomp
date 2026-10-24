// FP helpers for one precision; included twice by simd.h (F = s/float, d/double). No include guard.
// Scalar entry points take/return T, vector ones VT (all lanes; callers zero the unused half for Q=0).

#define FN(x) SIMD_CAT(x, F)
#define SGN ((U)1 << (NB - 1))
#define EXPM ((((U)1 << EB) - 1) << MB)
#define QB ((U)1 << (MB - 1))
#define DNAN (EXPM | QB)
#define MINN ((U)1 << MB) // smallest normal
#define VK(v) ((VT){} + (T)(v))
#define K(v) ((T)(v))

SI U FN(fb)(T x) { U u; memcpy(&u, &x, sizeof u); return u; }
SI T FN(fv)(U u) { T x; memcpy(&x, &u, sizeof x); return x; }
SI int FN(isnan)(U u) { return (u & ~SGN) > EXPM; }
SI int FN(issnan)(U u) { return FN(isnan)(u) && !(u & QB); }
SI int FN(isqnan)(U u) { return FN(isnan)(u) && (u & QB); }
SI int FN(isinf)(U u) { return (u & ~SGN) == EXPM; }
SI int FN(iszero)(U u) { return !(u & ~SGN); }
SI int FN(isden)(U u) { return !(u & EXPM) && (u & ~SGN); }
SI T FN(flush)(T x) { U u = FN(fb)(x); return FN(isden)(u) ? FN(fv)(u & SGN) : x; }
SI T FN(fz)(Cpu *c, T x) { return __builtin_expect(c->fpcr & FPCR_FZ, 0) ? FN(flush)(x) : x; }
SI T FN(neg)(T x) { return FN(fv)(FN(fb)(x) ^ SGN); }

// Fused multiply-add x*y + z, one rounding in the host's current mode. With FMA (x86-64-v3, AArch64) the host
// instruction. Without it (x86-64-v2):
// - single precision in double: x*y is exact there, x*y + z is rounded to odd (rounded in the current mode, error by
//   Fast2Sum, which is exact in every rounding mode, then the last bit set from the side the error is on) and that rounds
//   to float exactly as the fused result would (53 >= 24 + 2 bits). Exact sums never underflow in double (operands are
//   multiples of 2^-298).
// - double precision in integers: the 106-bit product and the addend aligned in 128 bits (bits shifted out kept as a
//   sticky bit), rounded in the MXCSR mode. (Not the C library's fma: zig 0.17's returns NaN for some operands.)
#if defined(__FMA__) || defined(__aarch64__)
#define FFMA(x, y, z) __builtin_elementwise_fma(x, y, z)
#elif NB == 64
SI int simd_clz128(u128 v) {
    uint64_t hi = (uint64_t)(v >> 64);
    return hi ? __builtin_clzll(hi) : 64 + __builtin_clzll((uint64_t)v);
}
SI u128 simd_shr_sticky(u128 v, int n) {
    if (n <= 0) return v;
    if (n >= 128) return v != 0;
    return (v >> n) | ((v & (((u128)1 << n) - 1)) != 0);
}
SN T FN(sfma)(T x, T y, T z) {
    if (!__builtin_isfinite(x) || !__builtin_isfinite(y) || x == 0 || y == 0) return x * y + z;  // (the product is exact)
    if (!__builtin_isfinite(z)) return z + z;
    if (z == 0) return x * y;  // (a product rounded to zero keeps its sign, as the fused op's result would)
    U ux = FN(fb)(x), uy = FN(fb)(y), uz = FN(fb)(z);
    int sp = (int)((ux ^ uy) >> 63), sz = (int)(uz >> 63);
    int ex = (int)((ux >> MB) & 0x7ff), ey = (int)((uy >> MB) & 0x7ff), ez = (int)((uz >> MB) & 0x7ff);
    U fx = ux & (((U)1 << MB) - 1), fy = uy & (((U)1 << MB) - 1), fz = uz & (((U)1 << MB) - 1);
    if (ex) fx |= (U)1 << MB; else ex = 1;  // value = significand * 2^(e - 1075)
    if (ey) fy |= (U)1 << MB; else ey = 1;
    if (ez) fz |= (U)1 << MB; else ez = 1;
    u128 p = (u128)fx * fy, q = fz;
    int ep = ex + ey - 2150, eq = ez - 1075;  // exponents of the last bits
    int k = simd_clz128(p) - 2;  // both with their top bit at bit 125
    p <<= k, ep -= k;
    k = simd_clz128(q) - 2;
    q <<= k, eq -= k;
    if (ep >= eq) q = simd_shr_sticky(q, ep - eq);  // (a shift of 2 or more cancels at most a bit below: the
    else p = simd_shr_sticky(p, eq - ep), ep = eq;   // sticky bit stays far below the rounding point)
    u128 r;
    int s;
    if (sp == sz) r = p + q, s = sp;
    else if (p >= q) r = p - q, s = sp;
    else r = q - p, s = sz;
    int mode = (int)((__builtin_ia32_stmxcsr() >> 13) & 3);  // 0 nearest, 1 down, 2 up, 3 toward zero
    if (r == 0) return mode == 1 ? -K(0) : K(0);
    int m = 127 - simd_clz128(r), sh = m - MB;  // keep 53 bits; fewer below the normal range
    if (ep + m < -1022) sh = -1074 - ep;
    U kept;
    if (sh <= 0) {
        kept = (U)(r << -sh);
    } else {
        u128 rem = sh >= 128 ? r : r & (((u128)1 << sh) - 1), half = sh > 128 ? 0 : (u128)1 << (sh - 1);
        kept = sh >= 128 ? 0 : (U)(r >> sh);
        int up = mode == 0 ? (sh <= 128 && (rem > half || (rem == half && (kept & 1)))) : mode == 1 ? s : mode == 2 ? !s : 0;
        kept += rem != 0 && up;
    }
    int el = ep + sh;  // the exponent of kept's last bit: -1074 for a subnormal, whose encoding is kept itself
    U bits = el + 1074 > 2046 ? EXPM : ((U)(el + 1074) << MB) + kept;  // (a carry into the exponent field is right)
    if (bits >= EXPM) bits = mode == 0 || (mode == 1 && s) || (mode == 2 && !s) ? EXPM : EXPM - 1;
    return FN(fv)(bits | (U)s << 63);
}
SI VT FN(vsfma)(VT x, VT y, VT z) { return (VT){FN(sfma)(x[0], y[0], z[0]), FN(sfma)(x[1], y[1], z[1])}; }
#define FFMA(x, y, z) _Generic((x), T: FN(sfma), VT: FN(vsfma))(x, y, z)
#else
typedef double simd_vd4 __attribute__((vector_size(32)));
typedef uint64_t simd_vu4 __attribute__((vector_size(32)));
typedef int64_t simd_vs4 __attribute__((vector_size(32)));
SI T FN(sfma)(T x, T y, T z) {
    double p = (double)x * y, q = z, s = p + q;
    int big = __builtin_fabs(p) >= __builtin_fabs(q);
    double e = (big ? q : p) - (s - (big ? p : q));
    uint64_t u;
    memcpy(&u, &s, sizeof u);
    if (e != 0 && !(u & 1) && __builtin_fabs(s) < __builtin_inf()) {
        uint64_t ue;
        memcpy(&ue, &e, sizeof ue);
        u += ((u ^ ue) >> 63) ? (uint64_t)-1 : 1;  // one ulp toward the error (sign-magnitude bits)
        memcpy(&s, &u, sizeof s);
    }
    return (T)s;
}
SI VT FN(vsfma)(VT x, VT y, VT z) {
    simd_vd4 p = CVT(x, simd_vd4) * CVT(y, simd_vd4), q = CVT(z, simd_vd4), s = p + q;
    simd_vu4 big = (simd_vu4)(__builtin_elementwise_abs(p) >= __builtin_elementwise_abs(q));
    simd_vd4 hi = (simd_vd4)(((simd_vu4)p & big) | ((simd_vu4)q & ~big));
    simd_vd4 lo = (simd_vd4)(((simd_vu4)q & big) | ((simd_vu4)p & ~big));
    simd_vd4 e = lo - (s - hi);
    simd_vu4 u = (simd_vu4)s;
    simd_vu4 odd = (simd_vu4)((e != 0) & ((u & 1) == 0) & (__builtin_elementwise_abs(s) < __builtin_inf()));
    simd_vu4 toward = (simd_vu4)((simd_vs4)(u ^ (simd_vu4)e) >> 63);  // 0: away from zero, all ones: toward it
    u += odd & (toward + toward + 1);
    return CVT((simd_vd4)u, VT);
}
#define FFMA(x, y, z) _Generic((x), T: FN(sfma), VT: FN(vsfma))(x, y, z)
#endif

// FPProcessNaN(s): first signalling NaN (quietened), else first quiet NaN, else (invalid op) default NaN
SN T FN(nan1)(Cpu *c, T a) {
    U x = FN(fb)(a);
    if (!(c->fpcr & FPCR_DN) && FN(isnan)(x)) return FN(fv)(x | QB);
    return FN(fv)(DNAN);
}
SN T FN(nan2)(Cpu *c, T a, T b) {
    U x = FN(fb)(a), y = FN(fb)(b);
    if (!(c->fpcr & FPCR_DN)) {
        if (FN(issnan)(x)) return FN(fv)(x | QB);
        if (FN(issnan)(y)) return FN(fv)(y | QB);
        if (FN(isnan)(x)) return a;
        if (FN(isnan)(y)) return b;
    }
    return FN(fv)(DNAN);
}
SN T FN(nan3)(Cpu *c, T a, T b, T d) {
    U x = FN(fb)(a), y = FN(fb)(b), z = FN(fb)(d);
    if (!(c->fpcr & FPCR_DN)) {
        if (FN(issnan)(x)) return FN(fv)(x | QB);
        if (FN(issnan)(y)) return FN(fv)(y | QB);
        if (FN(issnan)(z)) return FN(fv)(z | QB);
        if (FN(isnan)(x)) return a;
        if (FN(isnan)(y)) return b;
        if (FN(isnan)(z)) return d;
    }
    return FN(fv)(DNAN);
}

// The arithmetic itself, for scalars (KK = K) and vectors (KK = VK). MADD: a + b*d (a = addend).
// RECPS/RSQRTS get their first operand already negated (FPNeg happens before NaN processing).
#define FCALC(op, a, b, d, KK)                                                                       \
    ((op) == FOP_ADD ? (a) + (b) : (op) == FOP_SUB ? (a) - (b)                                     \
   : (op) == FOP_MUL || (op) == FOP_MULX ? (a) * (b) : (op) == FOP_NMUL ? -((a) * (b))              \
   : (op) == FOP_DIV ? (a) / (b) : (op) == FOP_ABD ? __builtin_elementwise_abs((a) - (b))          \
   : (op) == FOP_SQRT ? __builtin_elementwise_sqrt(a)                                              \
   : (op) == FOP_MADD ? FFMA(b, d, a)                                                              \
   : (op) == FOP_RECPS ? FFMA(a, b, KK(2))                                                         \
   : (op) == FOP_RSQRTS ? FFMA(a, b, KK(3)) * KK(0.5) : (a))

// Lanes whose fast result needs fixing: NaN results (NaN operand or invalid op), RSQRTS overflow.
#define FBAD(op, r) ((op) == FOP_RSQRTS ? ((r) != (r)) | (__builtin_elementwise_abs(r) == K(INFINITY)) : (r) != (r))

// (3 + a*b) / 2 with one rounding (a already negated): halve a normal operand first, so a product beyond
// the format's range (which the fast path rounds to inf, or to the largest normal when rounding down) is exact.
SN T FN(rsqrts)(T a, T b) {
    if (__builtin_elementwise_abs(a) >= FN(fv)(2 * MINN)) return FFMA(a * K(0.5), b, K(1.5));
    if (__builtin_elementwise_abs(b) >= FN(fv)(2 * MINN)) return FFMA(a, b * K(0.5), K(1.5));
    return FFMA(a, b, K(3)) * K(0.5);
}

// ARM result for a lane flagged by FBAD (operands already flushed).
SN T FN(fix)(Cpu *c, int op, T a, T b, T d) {
    U x = FN(fb)(a), y = FN(fb)(b), z = FN(fb)(d);
    int nan = FN(isnan)(x) || FN(isnan)(y);
    switch (op) {
    case FOP_SQRT: return FN(nan1)(c, a); // NaN in, or negative -> default NaN
    case FOP_MADD:
        if (FN(isnan)(x) || FN(isnan)(y) || FN(isnan)(z)) {
            if (FN(isqnan)(x) && ((FN(isinf)(y) && FN(iszero)(z)) || (FN(iszero)(y) && FN(isinf)(z))))
                return FN(fv)(DNAN);
            return FN(nan3)(c, a, b, d);
        }
        return FN(fv)(DNAN);
    case FOP_MULX:
        if (!nan) return FN(fv)(((x ^ y) & SGN) | FN(fb)(K(2)));
        break;
    case FOP_RECPS:
        if (!nan) return K(2);
        break;
    case FOP_RSQRTS:
        if (!nan) {
            if ((FN(isinf)(x) && FN(iszero)(y)) || (FN(iszero)(x) && FN(isinf)(y))) return K(1.5);
            return FN(rsqrts)(a, b);
        }
        break;
    case FOP_NMUL: return FN(neg)(FN(nan2)(c, a, b));
    case FOP_ABD: return FN(fv)(FN(fb)(FN(nan2)(c, a, b)) & ~SGN);
    }
    return FN(nan2)(c, a, b);
}

// FMAX/FMIN/FMAXNM/FMINNM (operands already flushed)
SI T FN(minmax)(Cpu *c, int op, T a, T b) {
    U x = FN(fb)(a), y = FN(fb)(b);
    if (__builtin_expect(FN(isnan)(x) || FN(isnan)(y), 0)) {
        if (op == FOP_MAXNM || op == FOP_MINNM) {
            if (FN(isqnan)(x) && !FN(isnan)(y)) return b;
            if (FN(isqnan)(y) && !FN(isnan)(x)) return a;
        }
        return FN(nan2)(c, a, b);
    }
    if (op == FOP_MAX || op == FOP_MAXNM) return a == b ? FN(fv)(x & y) : a > b ? a : b;
    return a == b ? FN(fv)(x | y) : a < b ? a : b;
}
#define FMINMAX(op) ((op) >= FOP_MAX && (op) <= FOP_MINNM)

// FPCR.FZ / RMode path: flush inputs, round in the FPCR mode, flush tiny results (tininess is detected
// before rounding on ARM: a result rounded up to the smallest normal is recomputed toward zero).
SN T FN(slow)(Cpu *c, int op, T a, T b, T d) {
#pragma STDC FENV_ACCESS ON
    uint64_t fpcr = c->fpcr;
    if (fpcr & FPCR_FZ) a = FN(flush)(a), b = FN(flush)(b), d = FN(flush)(d);
    if (FMINMAX(op)) return FN(minmax)(c, op, a, b);
    int old = fegetround();
    fesetround(simd_fe_mode(fpcr));
    volatile T va = a, vb = b, vd = d;
    volatile T vr = op == FOP_RSQRTS ? FN(rsqrts)(va, vb) : FCALC(op, va, vb, vd, K);
    T r = vr;
    if (FBAD(op, r)) vr = FN(fix)(c, op, a, b, d), r = vr;
    if ((fpcr & FPCR_FZ) && (FN(fb)(r) & ~SGN) <= MINN && !FN(iszero)(FN(fb)(r))) {
        if ((FN(fb)(r) & ~SGN) == MINN) {
            fesetround(FE_TOWARDZERO);
            volatile T vz = FCALC(op, va, vb, vd, K);
            if ((FN(fb)(vz) & ~SGN) < MINN) r = FN(fv)(FN(fb)(r) & SGN);
        } else {
            r = FN(fv)(FN(fb)(r) & SGN);
        }
    }
    fesetround(old);
    return r;
}

// Scalar FP op (constant op folds the selection).
SI T FN(op)(Cpu *c, int op, T a, T b, T d) {
    if (__builtin_expect(c->fpcr & FPCR_SLOW, 0)) return FN(slow)(c, op, a, b, d);
    if (FMINMAX(op)) return FN(minmax)(c, op, a, b);
    T r = FCALC(op, a, b, d, K);
    if (__builtin_expect(FBAD(op, r), 0)) r = FN(fix)(c, op, a, b, d);
    return r;
}

SN VT FN(vslow)(Cpu *c, int op, VT a, VT b, VT d) {
    VT r;
    for (int i = 0; i < NL; i++) r[i] = FN(op)(c, op, a[i], b[i], d[i]);
    return r;
}

SI VTU FN(vsel)(VTS m, VTU x, VTU y) { return (x & (VTU)m) | (y & ~(VTU)m); }

// Vector FP op: native lanes, any NaN (or FPCR.FZ/RMode) reruns all lanes through the scalar path.
SI VT FN(vop)(Cpu *c, int op, VT a, VT b, VT d) {
    if (__builtin_expect(c->fpcr & FPCR_SLOW, 0)) return FN(vslow)(c, op, a, b, d);
    VT r;
    if (FMINMAX(op)) {
        if (__builtin_expect(__builtin_reduce_or((a != a) | (b != b)) != 0, 0)) return FN(vslow)(c, op, a, b, d);
        VTU x = (VTU)a, y = (VTU)b;
        if (op == FOP_MAX || op == FOP_MAXNM)
            r = (VT)FN(vsel)(a > b, x, FN(vsel)(a == b, x & y, y));
        else
            r = (VT)FN(vsel)(a < b, x, FN(vsel)(a == b, x | y, y));
        return r;
    }
    r = FCALC(op, a, b, d, VK);
    if (__builtin_expect(__builtin_reduce_or(FBAD(op, r)) != 0, 0)) return FN(vslow)(c, op, a, b, d);
    return r;
}

SI VT FN(vfz)(Cpu *c, VT a) {
    if (__builtin_expect(c->fpcr & FPCR_FZ, 0)) {
        VTU u = (VTU)a, mag = u & ~((VTU){} + SGN);
        VTS den = (mag != 0) & ((u & EXPM) == 0);
        a = (VT)FN(vsel)(den, u & SGN, u);
    }
    return a;
}

// ---- compares
SI uint64_t FN(cmp)(Cpu *c, T a, T b) { // FCMP NZCV
    a = FN(fz)(c, a), b = FN(fz)(c, b);
    return a == b ? 0x60000000ull : a < b ? 0x80000000ull : a > b ? 0x20000000ull : 0x30000000ull;
}
// FCMEQ/FCMGE/FCMGT/FACGE/FACGT (op 0..4)
SI VTU FN(vcmp)(Cpu *c, int op, VT a, VT b) {
    a = FN(vfz)(c, a), b = FN(vfz)(c, b);
    if (op >= 3) a = __builtin_elementwise_abs(a), b = __builtin_elementwise_abs(b);
    switch (op) {
    case 0: return (VTU)(a == b);
    case 1: case 3: return (VTU)(a >= b);
    default: return (VTU)(a > b);
    }
}

// ---- rounding to integral (FRINT*): rm 0 N, 1 P, 2 M, 3 Z, 4 A; 5 X / 6 I = FPCR mode
// (rint = ties-to-even: the host stays in round-to-nearest outside the slow paths; no roundeven in MinGW's CRT)
SI T FN(rnd)(T x, int rm) {
    switch (rm) {
    case 0: return __builtin_elementwise_rint(x);
    case 1: return __builtin_elementwise_ceil(x);
    case 2: return __builtin_elementwise_floor(x);
    case 3: return __builtin_elementwise_trunc(x);
    default: return __builtin_elementwise_round(x);
    }
}
SI T FN(frint)(Cpu *c, T x, int rm) {
    x = FN(fz)(c, x);
    if (rm >= 5) rm = (int)((c->fpcr >> 22) & 3);
    T r = FN(rnd)(x, rm);
    if (__builtin_expect(r != r, 0)) r = FN(nan1)(c, x);
    return r;
}
SI VT FN(vfrint)(Cpu *c, VT x, int rm) {
    x = FN(vfz)(c, x);
    if (rm >= 5) rm = (int)((c->fpcr >> 22) & 3);
    VT r;
    switch (rm) {
    case 0: r = __builtin_elementwise_rint(x); break;
    case 1: r = __builtin_elementwise_ceil(x); break;
    case 2: r = __builtin_elementwise_floor(x); break;
    case 3: r = __builtin_elementwise_trunc(x); break;
    default: r = __builtin_elementwise_round(x); break;
    }
    if (__builtin_expect(__builtin_reduce_or(r != r) != 0, 0))
        for (int i = 0; i < NL; i++)
            if (r[i] != r[i]) r[i] = FN(nan1)(c, x[i]);
    return r;
}

// ---- FP -> integer (FCVT[NPMZA][SU], fixed point): saturating, NaN -> 0. ib = 32/64.
SI uint64_t FN(toint)(Cpu *c, T x, int rm, int fbits, int ib, int uns) {
    x = FN(fz)(c, x);
    if (fbits) x = x * (T)(1ull << (fbits - 1)) * K(2);
    T r = FN(rnd)(x, rm);
    if (r != r) return 0;
    if (uns) {
        if (r <= K(0)) return 0;
        if (ib == 32) return r >= K(4294967296.0) ? 0xffffffffull : (uint32_t)r;
        return r >= K(18446744073709551616.0) ? ~0ull : (uint64_t)r;
    }
    if (ib == 32) return r >= K(2147483648.0) ? 0x7fffffffull : r < K(-2147483648.0) ? 0x80000000ull : (uint32_t)(int32_t)r;
    return r >= K(9223372036854775808.0) ? 0x7fffffffffffffffull : r < K(-9223372036854775808.0) ? 0x8000000000000000ull
                                                                                              : (uint64_t)(int64_t)r;
}

// ---- integer -> FP (SCVTF/UCVTF, fixed point), rounded in the FPCR mode
SN T FN(fromint_slow)(Cpu *c, uint64_t v, int ib, int uns, int fbits) {
#pragma STDC FENV_ACCESS ON
    int old = fegetround();
    fesetround(simd_fe_mode(c->fpcr));
    volatile uint64_t vv = v;
    volatile T r;
    if (uns) r = ib == 32 ? (T)(uint32_t)vv : (T)vv;
    else r = ib == 32 ? (T)(int32_t)vv : (T)(int64_t)vv;
    fesetround(old);
    T x = r;
    if (fbits) x = x / (T)(1ull << (fbits - 1)) / K(2);
    return x;
}
SI T FN(fromint)(Cpu *c, uint64_t v, int ib, int uns, int fbits) {
    if (__builtin_expect(c->fpcr & FPCR_RM, 0)) return FN(fromint_slow)(c, v, ib, uns, fbits);
    T r;
    if (uns) r = ib == 32 ? (T)(uint32_t)v : (T)v;
    else r = ib == 32 ? (T)(int32_t)v : (T)(int64_t)v;
    if (fbits) r = r / (T)(1ull << (fbits - 1)) / K(2);
    return r;
}

// ---- reciprocal estimates (bit-exact ARMv8.0 pseudocode)
SI T FN(recpe)(Cpu *c, T a) {
    U x = FN(fb)(FN(fz)(c, a)), s = x & SGN;
    if (FN(isnan)(x)) return FN(nan1)(c, a);
    if (FN(isinf)(x)) return FN(fv)(s);
    if (FN(iszero)(x)) return FN(fv)(s | EXPM);
    U mag = x & ~SGN;
    if (mag < ((U)1 << (MB - 2))) { // |x| < 2^(emin - 2): result overflows
        int rm = (c->fpcr >> 22) & 3;
        int inf = rm == 0 || (rm == 1 && !s) || (rm == 2 && s);
        return FN(fv)(s | (inf ? EXPM : EXPM - 1));
    }
    int64_t e = (int64_t)(mag >> MB);
    U frac = mag & (((U)1 << MB) - 1);
    if ((c->fpcr & FPCR_FZ) && e >= (int64_t)((1 << EB) - 3)) return FN(fv)(s); // result would be tiny
    if (e == 0) {
        if (!(frac >> (MB - 1))) e = -1, frac = (frac << 2) & (((U)1 << MB) - 1);
        else frac = (frac << 1) & (((U)1 << MB) - 1);
    }
    int scaled = 256 | (int)(frac >> (MB - 8));
    int64_t re = (int64_t)((1 << EB) - 3) - e;
    U est = (U)simd_recip_tab[scaled - 256]; // 8 fraction bits
    U f = est << (MB - 8);
    if (re == 0) f = (f >> 1) | ((U)1 << (MB - 1));
    else if (re == -1) f = (f >> 2) | ((U)1 << (MB - 2)), re = 0;
    return FN(fv)(s | ((U)re << MB) | f);
}
SI T FN(rsqrte)(Cpu *c, T a) {
    U x = FN(fb)(FN(fz)(c, a)), s = x & SGN;
    if (FN(isnan)(x)) return FN(nan1)(c, a);
    if (FN(iszero)(x)) return FN(fv)(s | EXPM);
    if (s) return FN(fv)(DNAN);
    if (FN(isinf)(x)) return K(0);
    int64_t e = (int64_t)(x >> MB);
    U frac = x & (((U)1 << MB) - 1);
    if (e == 0) {
        while (!(frac >> (MB - 1))) frac <<= 1, e--;
        frac = (frac << 1) & (((U)1 << MB) - 1);
    }
    int scaled = (e & 1) ? 128 | (int)(frac >> (MB - 7)) : 256 | (int)(frac >> (MB - 8));
    int64_t re = ((int64_t)(3 * ((1 << (EB - 1)) - 1) - 1) - e) / 2; // (380 - e) / 2, (3068 - e) / 2
    U est = (U)simd_rsqrt_tab[scaled - 128];
    return FN(fv)(((U)re << MB) | (est << (MB - 8)));
}
SI T FN(recpx)(Cpu *c, T a) {
    U x = FN(fb)(FN(fz)(c, a)), s = x & SGN;
    if (FN(isnan)(x)) return FN(nan1)(c, a);
    if (!(x & EXPM)) return FN(fv)(s | (EXPM - MINN));
    return FN(fv)(s | (~x & EXPM));
}

#undef FN
#undef SGN
#undef EXPM
#undef QB
#undef DNAN
#undef MINN
#undef VK
#undef K
#undef FFMA
