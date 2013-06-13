#ifndef _SPIN_PROJECTION_H
#define _SPIN_PROJECTION_H
void spin_projection_3_2_zero_momentum_tr (spinor_propagator_type *s, spinor_propagator_type *t);
void spin_projection_3_2_zero_momentum (spinor_propagator_type *s, spinor_propagator_type *t);
void spin_projection_1_2_zero_momentum (spinor_propagator_type *s, spinor_propagator_type *t);

void spin_projection_3_2 (spinor_propagator_type *s, spinor_propagator_type *t, double e, double *pvec);
void spin_projection_3_2_field (spinor_propagator_type *sfield, spinor_propagator_type *tfield, double e, double *pvec, unsigned int N);
void spin_projection_3_2_zero_momentum_slice (spinor_propagator_type *s, spinor_propagator_type *t);
void spin_projection_1_2_zero_momentum_slice (spinor_propagator_type *s, spinor_propagator_type *t);
#endif
