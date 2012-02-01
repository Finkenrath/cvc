#!/bin/bash

#for i in 0 1 2 3; do
#  echo "counter_term[$i].x +=  sinp[$i] * sp[$i].y + sp[4].x * cosp[$i];"
#  echo "counter_term[$i].y += -sinp[$i] * sp[$i].x + sp[4].y * cosp[$i];"
#done
#exit 0

#for r in 0 1; do
#for s in 0 1; do
#for i in 0 1 2 3 4 5; do
#  for j in 0 1 2 3 4 5; do
#
#    echo "_co_f2_eq_f2_ti_f2( (_a_),(_r_)[$i],(_s_)[$j] ); \\"
#    echo "_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[$r][$s] ); \\"
#    echo "_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), $r, $s, $i, $j)] ); \\"
#    echo "_co_f2_pl_eq_f2( (_t_), (_b_) ); \\"
#  done
#done
#done
#done


for r in 0 1; do
for s in 0 1; do
for i in 0 1 2 3 4 5; do
  for j in 0 1 2 3 4 5; do

    echo "_co_f2_eq_f2_ti_f2( (_a_),(_r_)[$i],(_s_)[$j] ); \\"
    echo "_co_f2_eq_f2_ti_f2( (_b_),(_a_), (_p_)[$r][$s] ); \\"
    echo "_co_f2_ti_eq_f1( (_b_), (_c_)[_CVC_COEFF_IDX((_m_), (_n_), $r, $s, $i, $j)] ); \\"
    echo "_co_f2_pl_eq_f2( (_t_), (_b_) ); \\"
  done
done
done
done
