#ifndef _CONTRACTIONS_H
#define _CONTRACTIONS_H

#define _PI 3.1415926535897931

#define _CVC_COEFF_IDX(_i,_j,_r,_s,_a,_b) \
    (( ( (_i)*4+(_j) )*4 +2*(_r)+(_s) )*36 + 6*(_a)+(_b) )

#define _SQR(_a) ((_a)*(_a))
#define _NCOLOR 3
#define _NSPIN  4
#define THREADS_PER_BLOCK 256

#define HANDLE_ERROR(_err) {\
    if( (_err) != cudaSuccess ) {\
          fprintf(stderr, "\nError in %s at line %d\n", __FILE__, __LINE__);\
          exit(1);\
        }\
}

// multiply float2 _b_ by float2 _c_ and save in float2 _a_
#define _co_f2_eq_f2_ti_f2(_a_, _b_, _c_) {\
  (_a_).x = (_b_).x * (_c_).x - (_b_).y * (_c_).y; \
  (_a_).y = (_b_).x * (_c_).y + (_b_).y * (_c_).x; \
}

// multiply float2 _b_ by float _c_ and save in float2 _a_
#define _co_f2_eq_f2_ti_f1(_a_,_b_,_c_) {\
  (_a_).x = (_b_).x * (_c_); \
  (_a_).y = (_b_).y * (_c_); \
}

// multiply float2 _a_ by float _c_ and save in float2 _a_
#define _co_f2_ti_eq_f1(_a_,_c_) {\
  (_a_).x *= (_c_); \
  (_a_).y *= (_c_); \
}

// add float2 _b_ to float2 _a_
#define _co_f2_pl_eq_f2(_a_,_b_) {\
  (_a_).x += (_b_).x; \
  (_a_).y += (_b_).y; \
}

// set the phase for on (mu,nu) combination
#define _dev_set_phase(_a_,_p_,_k_,_i_,_j_) {\
  (_a_)[0][0].x = cos( (+(_p_)[(_i_)] + (_p_)[(_j_)] + 0.5*((_k_)[(_i_)] + (_k_)[(_j_)]) ) ); \
  (_a_)[0][0].y = sin( (+(_p_)[(_i_)] + (_p_)[(_j_)] + 0.5*((_k_)[(_i_)] + (_k_)[(_j_)]) ) ); \
  (_a_)[0][1].x = cos( (+(_p_)[(_i_)] - (_p_)[(_j_)] + 0.5*((_k_)[(_i_)] - (_k_)[(_j_)]) ) ); \
  (_a_)[0][1].y = sin( (+(_p_)[(_i_)] - (_p_)[(_j_)] + 0.5*((_k_)[(_i_)] - (_k_)[(_j_)]) ) ); \
  (_a_)[1][0].x = cos( (-(_p_)[(_i_)] + (_p_)[(_j_)] - 0.5*((_k_)[(_i_)] + (_k_)[(_j_)]) ) ); \
  (_a_)[1][0].y = sin( (-(_p_)[(_i_)] + (_p_)[(_j_)] - 0.5*((_k_)[(_i_)] + (_k_)[(_j_)]) ) ); \
  (_a_)[1][1].x = cos( (-(_p_)[(_i_)] - (_p_)[(_j_)] - 0.5*((_k_)[(_i_)] - (_k_)[(_j_)]) ) ); \
  (_a_)[1][1].y = sin( (-(_p_)[(_i_)] - (_p_)[(_j_)] - 0.5*((_k_)[(_i_)] - (_k_)[(_j_)]) ) ); \
}

// accumulate the result for a combination (mu,nu)
#define _cvc_accum(_t_,_m_,_n_,_c_,_p_,_r_,_s_,_a_,_b_) {\
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 0, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 0, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 0, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 0, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 0, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 0, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 1, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 1, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 1, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 1, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 1, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 1, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 2, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 2, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 2, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 2, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 2, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 2, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 3, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 3, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 3, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 3, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 3, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 3, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 4, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 4, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 4, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 4, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 4, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 4, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 5, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 5, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 5, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 5, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 5, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 0, 5, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 0, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 0, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 0, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 0, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 0, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 0, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 1, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 1, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 1, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 1, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 1, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 1, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 2, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 2, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 2, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 2, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 2, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 2, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 3, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 3, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 3, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 3, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 3, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 3, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 4, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 4, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 4, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 4, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 4, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 4, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 5, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 5, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 5, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 5, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 5, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[0][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 0, 1, 5, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 0, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 0, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 0, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 0, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 0, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 0, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 1, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 1, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 1, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 1, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 1, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 1, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 2, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 2, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 2, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 2, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 2, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 2, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 3, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 3, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 3, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 3, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 3, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 3, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 4, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 4, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 4, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 4, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 4, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 4, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 5, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 5, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 5, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 5, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 5, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][0] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 0, 5, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 0, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 0, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 0, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 0, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 0, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[0],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 0, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 1, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 1, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 1, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 1, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 1, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[1],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 1, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 2, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 2, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 2, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 2, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 2, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[2],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 2, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 3, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 3, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 3, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 3, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 3, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[3],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 3, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 4, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 4, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 4, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 4, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 4, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[4],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 4, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[0] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 5, 0)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[1] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 5, 1)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[2] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 5, 2)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[3] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 5, 3)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[4] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 5, 4)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
_co_f2_eq_f2_ti_f2( (_a_),(_r_)[5],(_s_)[5] ); \
_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[1][1] ); \
_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), 1, 1, 5, 5)] ); \
_co_f2_pl_eq_f2( (_t_), (_b_) ); \
}





#endif
