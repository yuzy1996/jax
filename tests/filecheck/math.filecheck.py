# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tests for lowerings of elementwise ops to MHLO.

# RUN: %PYTHON %s | FileCheck %s

from absl import app
from functools import partial

import jax
from jax import numpy as jnp
from jax import lax
import numpy as np

from jax.tests.filecheck.jax_filecheck_helpers import print_ir

jax.config.update("jax_enable_x64", True)


def main(_):
  # CHECK-LABEL: TEST: abs int32[]
  # CHECK: mhlo.abs
  # CHECK-SAME: tensor<i32>
  print_ir(np.int32(0))(lax.abs)

  # CHECK-LABEL: TEST: add float32[] float32[]
  # CHECK: mhlo.add
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.add)

  # CHECK-LABEL: TEST: acos float32[]
  # CHECK: mhlo.atan2
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1))(lax.acos)

  # CHECK-LABEL: TEST: acosh float32[]
  # CHECK: chlo.acosh
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.acosh)

  # CHECK-LABEL: TEST: asin float32[]
  # CHECK: mhlo.atan2
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1))(lax.asin)

  # CHECK-LABEL: TEST: asinh float32[]
  # CHECK: chlo.asinh
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.asinh)

  # CHECK-LABEL: TEST: atan float32[]
  # CHECK: mhlo.atan2
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1))(lax.atan)

  # CHECK-LABEL: TEST: atanh float32[]
  # CHECK: chlo.atanh
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.atanh)

  # CHECK-LABEL: TEST: atan2 float64[] float64[]
  # CHECK: mhlo.atan2
  # CHECK-SAME: tensor<f64>
  print_ir(np.float64(1), np.float64(2))(lax.atan2)

  # CHECK-LABEL: TEST: bessel_i0e float32[]
  # CHECK: xla_fallback_bessel_i0e
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.bessel_i0e)

  # CHECK-LABEL: TEST: bessel_i1e float32[]
  # CHECK: xla_fallback_bessel_i1e
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.bessel_i1e)

  # CHECK-LABEL: TEST: betainc float32[] float32[] float32[]
  # CHECK: xla_fallback_regularized_incomplete_beta
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0), np.float32(0), np.float32(0))(lax.betainc)

  # CHECK-LABEL: TEST: bitcast_convert_type uint32[7]
  # CHECK: mhlo.bitcast_convert
  # CHECK-SAME: tensor<7xui32>
  # CHECK-SAME: tensor<7xf32>
  print_ir(np.empty((7,), np.uint32))(
      partial(lax.bitcast_convert_type, new_dtype=np.float32))

  # CHECK-LABEL: TEST: bitwise_and int32[] int32[]
  # CHECK: mhlo.and
  # CHECK-SAME: tensor<i32>
  print_ir(np.int32(1), np.int32(2))(lax.bitwise_and)

  # CHECK-LABEL: TEST: bitwise_and bool[] bool[]
  # CHECK: mhlo.and
  # CHECK-SAME: tensor<i1>
  print_ir(np.bool_(0), np.bool_(0))(lax.bitwise_and)

  # CHECK-LABEL: TEST: bitwise_or int32[] int32[]
  # CHECK: mhlo.or
  # CHECK-SAME: tensor<i32>
  print_ir(np.int32(1), np.int32(2))(lax.bitwise_or)

  # CHECK-LABEL: TEST: bitwise_or bool[] bool[]
  # CHECK: mhlo.or
  # CHECK-SAME: tensor<i1>
  print_ir(np.bool_(0), np.bool_(0))(lax.bitwise_or)

  # CHECK-LABEL: TEST: bitwise_xor int32[] int32[]
  # CHECK: mhlo.xor
  # CHECK-SAME: tensor<i32>
  print_ir(np.int32(1), np.int32(2))(lax.bitwise_xor)

  # CHECK-LABEL: TEST: bitwise_xor bool[] bool[]
  # CHECK: mhlo.xor
  # CHECK-SAME: tensor<i1>
  print_ir(np.bool_(0), np.bool_(0))(lax.bitwise_xor)

  # CHECK-LABEL: TEST: cbrt bfloat16[]
  # CHECK: mhlo.cbrt
  # CHECK-SAME: tensor<bf16>
  print_ir(jnp.bfloat16(0))(lax.cbrt)

  # CHECK-LABEL: TEST: clamp bfloat16[] bfloat16[] bfloat16[]
  # CHECK: mhlo.clamp
  # CHECK-SAME: tensor<bf16>
  print_ir(jnp.bfloat16(0), jnp.bfloat16(0), jnp.bfloat16(0))(lax.clamp)

  # CHECK-LABEL: TEST: ceil float16[7]
  # CHECK: mhlo.ceil
  # CHECK-SAME: tensor<7xf16>
  print_ir(np.empty((7,), np.float16))(lax.ceil)

  # CHECK-LABEL: TEST: convert_element_type float16[7]
  # CHECK: mhlo.convert
  # CHECK-SAME: tensor<7xf16>
  # CHECK-SAME: tensor<7xf32>
  print_ir(np.empty((7,), np.float16))(
      partial(lax.convert_element_type, new_dtype=np.float32))

  # CHECK-LABEL: TEST: convert_element_type complex64[7]
  # CHECK: mhlo.real
  # CHECK-SAME: tensor<7xcomplex<f32>>
  # CHECK-SAME: tensor<7xf32>
  print_ir(np.empty((7,), np.complex64))(
      partial(lax.convert_element_type, new_dtype=np.float32))

  # CHECK-LABEL: TEST: convert_element_type float32[7]
  # CHECK: mhlo.compare
  # CHECK-SAME: tensor<7xf32>
  # CHECK-SAME: tensor<7xi1>
  print_ir(np.empty((7,), np.float32))(
      partial(lax.convert_element_type, new_dtype=np.bool_))

  # CHECK-LABEL: TEST: clz uint32[]
  # CHECK: mhlo.count_leading_zeros
  # CHECK-SAME: tensor<ui32>
  print_ir(np.uint32(0))(lax.clz)

  # CHECK-LABEL: TEST: conj complex64[]
  # CHECK-DAG: mhlo.real
  # CHECK-DAG: mhlo.imag
  # CHECK-DAG: mhlo.neg
  # CHECK-DAG: mhlo.complex
  # CHECK-SAME: tensor<complex<f32>>
  print_ir(np.complex64(0))(lax.conj)

  # CHECK-LABEL: TEST: cos float32[]
  # CHECK: mhlo.cos
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.cos)

  # CHECK-LABEL: TEST: cosh float32[]
  # CHECK: chlo.cosh
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.cosh)

  # CHECK-LABEL: TEST: digamma float32[]
  # CHECK: chlo.digamma
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.digamma)

  # CHECK-LABEL: TEST: div float32[] float32[]
  # CHECK: mhlo.div
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.div)

  # CHECK-LABEL: TEST: eq float32[] float32[]
  # CHECK: mhlo.compare
  # CHECK-SAME: compare_type = #mhlo<"comparison_type FLOAT">
  # CHECK-SAME: comparison_direction = #mhlo<"comparison_direction EQ">
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.eq)

  # CHECK-LABEL: TEST: eq complex128[] complex128[]
  # CHECK: mhlo.compare
  # CHECK-SAME: compare_type = #mhlo<"comparison_type FLOAT">
  # CHECK-SAME: comparison_direction = #mhlo<"comparison_direction EQ">
  # CHECK-SAME: tensor<complex<f64>>
  print_ir(np.complex128(1), np.complex128(2))(lax.eq)

  # CHECK-LABEL: TEST: eq int64[] int64[]
  # CHECK: mhlo.compare
  # CHECK-SAME: compare_type = #mhlo<"comparison_type SIGNED">
  # CHECK-SAME: comparison_direction = #mhlo<"comparison_direction EQ">
  # CHECK-SAME: tensor<i64>
  print_ir(np.int64(1), np.int64(2))(lax.eq)

  # CHECK-LABEL: TEST: eq uint16[] uint16[]
  # CHECK: mhlo.compare
  # CHECK-SAME: compare_type = #mhlo<"comparison_type UNSIGNED">
  # CHECK-SAME: comparison_direction = #mhlo<"comparison_direction EQ">
  # CHECK-SAME: tensor<ui16>
  print_ir(np.uint16(1), np.uint16(2))(lax.eq)

  # CHECK-LABEL: TEST: erf float32[]
  # CHECK: chlo.erf
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.erf)

  # CHECK-LABEL: TEST: erfc float32[]
  # CHECK: chlo.erfc
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.erfc)

  # CHECK-LABEL: TEST: erf_inv float32[]
  # CHECK: xla_fallback_erf_inv
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.erf_inv)

  # CHECK-LABEL: TEST: exp float16[]
  # CHECK: mhlo.exp
  # CHECK-SAME: tensor<f16>
  print_ir(np.float16(0))(lax.exp)

  # CHECK-LABEL: TEST: expm1 bfloat16[]
  # CHECK: mhlo.exponential_minus_one
  # CHECK-SAME: tensor<bf16>
  print_ir(jnp.bfloat16(0))(lax.expm1)

  # CHECK-LABEL: TEST: floor bfloat16[2,3]
  # CHECK: mhlo.floor
  # CHECK-SAME: tensor<2x3xbf16>
  print_ir(np.empty((2, 3), jnp.bfloat16))(lax.floor)

  # CHECK-LABEL: TEST: ge float32[] float32[]
  # CHECK: mhlo.compare
  # CHECK-SAME: compare_type = #mhlo<"comparison_type FLOAT">
  # CHECK-SAME: comparison_direction = #mhlo<"comparison_direction GE">
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.ge)

  # CHECK-LABEL: TEST: gt float32[] float32[]
  # CHECK: mhlo.compare
  # CHECK-SAME: compare_type = #mhlo<"comparison_type FLOAT">
  # CHECK-SAME: comparison_direction = #mhlo<"comparison_direction GT">
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.gt)

  # CHECK-LABEL: TEST: igamma float32[] float32[]
  # CHECK: xla_fallback_igamma
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0), np.float32(0))(lax.igamma)

  # CHECK-LABEL: TEST: igammac float32[] float32[]
  # CHECK: xla_fallback_igammac
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0), np.float32(0))(lax.igammac)

  # CHECK-LABEL: TEST: igamma_grad_a float32[] float32[]
  # CHECK: xla_fallback_igamma_grad_a
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0), np.float32(0))(lax.igamma_grad_a)

  # CHECK-LABEL: TEST: imag complex64[]
  # CHECK: mhlo.imag
  # CHECK-SAME: tensor<complex<f32>>
  print_ir(np.complex64(0))(lax.imag)

  # CHECK-LABEL: TEST: integer_pow float32[]
  # CHECK-DAG: mhlo.mul
  # CHECK-SAME: tensor<f32>
  @print_ir(np.float32(1))
  def integer_pow(x): return lax.integer_pow(x, 3)

  # CHECK-LABEL: TEST: is_finite float64[]
  # CHECK: mhlo.is_finite
  # CHECK-SAME: tensor<f64>
  print_ir(np.float64(0))(lax.is_finite)

  # CHECK-LABEL: TEST: le float32[] float32[]
  # CHECK: mhlo.compare
  # CHECK-SAME: compare_type = #mhlo<"comparison_type FLOAT">
  # CHECK-SAME: comparison_direction = #mhlo<"comparison_direction LE">
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.le)

  # CHECK-LABEL: TEST: lgamma float32[]
  # CHECK: chlo.lgamma
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.lgamma)

  # CHECK-LABEL: TEST: log float32[]
  # CHECK: mhlo.log
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.log)

  # CHECK-LABEL: TEST: log1p float32[]
  # CHECK: mhlo.log_plus_one
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.log1p)

  # CHECK-LABEL: TEST: lt float32[] float32[]
  # CHECK: mhlo.compare
  # CHECK-SAME: compare_type = #mhlo<"comparison_type FLOAT">
  # CHECK-SAME: comparison_direction = #mhlo<"comparison_direction LT">
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.lt)

  # CHECK-LABEL: TEST: max float32[] float32[]
  # CHECK: mhlo.max
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.max)

  # CHECK-LABEL: TEST: min float32[] float32[]
  # CHECK: mhlo.min
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.min)

  # CHECK-LABEL: TEST: mul float32[] float32[]
  # CHECK: mhlo.mul
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.mul)

  # CHECK-LABEL: TEST: ne float32[] float32[]
  # CHECK: mhlo.compare
  # CHECK-SAME: compare_type = #mhlo<"comparison_type FLOAT">
  # CHECK-SAME: comparison_direction = #mhlo<"comparison_direction NE">
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.ne)

  # CHECK-LABEL: TEST: neg int64[]
  # CHECK: mhlo.negate
  # CHECK-SAME: tensor<i64>
  print_ir(np.int64(0))(lax.neg)

  # CHECK-LABEL: TEST: nextafter float32[] float32[]
  # CHECK: chlo.next_after
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0), np.float32(0))(lax.nextafter)

  # CHECK-LABEL: TEST: bitwise_not int64[]
  # CHECK: mhlo.not
  # CHECK-SAME: tensor<i64>
  print_ir(np.int64(0))(lax.bitwise_not)

  # CHECK-LABEL: TEST: bitwise_not bool[]
  # CHECK: mhlo.not
  # CHECK-SAME: tensor<i1>
  print_ir(np.bool_(0))(lax.bitwise_not)

  # CHECK-LABEL: TEST: population_count uint32[]
  # CHECK: mhlo.popcnt
  # CHECK-SAME: tensor<ui32>
  print_ir(np.uint32(0))(lax.population_count)

  # CHECK-LABEL: TEST: pow float32[] float32[]
  # CHECK: mhlo.power
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.pow)

  # CHECK-LABEL: TEST: random_gamma_grad float32[] float32[]
  # CHECK: xla_fallback_random_gamma_grad
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0), np.float32(0))(lax.random_gamma_grad)

  # CHECK-LABEL: TEST: real complex128[]
  # CHECK: mhlo.real
  # CHECK-SAME: tensor<complex<f64>>
  print_ir(np.complex128(0))(lax.real)

  # CHECK-LABEL: TEST: reduce_precision bfloat16[]
  # CHECK: mhlo.reduce_precision
  # CHECK-SAME: tensor<bf16>
  print_ir(jnp.bfloat16(0))(
      partial(lax.reduce_precision, exponent_bits=2, mantissa_bits=2))

  # CHECK-LABEL: TEST: rem float32[] float32[]
  # CHECK: mhlo.rem
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.rem)

  # CHECK-LABEL: TEST: round float64[7,1]
  # CHECK: mhlo.round
  # CHECK-SAME: tensor<7x1xf64>
  print_ir(np.empty((7,1), np.float64))(
      partial(lax.round, rounding_method=lax.RoundingMethod.AWAY_FROM_ZERO))

  # CHECK-LABEL: TEST: rsqrt complex64[]
  # CHECK: mhlo.rsqrt
  # CHECK-SAME: tensor<complex<f32>>
  print_ir(jnp.complex64(0))(lax.rsqrt)

  # CHECK-LABEL: TEST: shift_left uint32[] uint32[]
  # CHECK: mhlo.shift_left
  # CHECK-SAME: tensor<ui32>
  print_ir(np.uint32(0), np.uint32(0))(lax.shift_left)

  # CHECK-LABEL: TEST: shift_right_arithmetic uint8[] uint8[]
  # CHECK: mhlo.shift_right_arithmetic
  # CHECK-SAME: tensor<ui8>
  print_ir(np.uint8(0), np.uint8(0))(lax.shift_right_arithmetic)

  # CHECK-LABEL: TEST: shift_right_logical uint16[] uint16[]
  # CHECK: mhlo.shift_right_logical
  # CHECK-SAME: tensor<ui16>
  print_ir(np.uint16(0), np.uint16(0))(lax.shift_right_logical)

  # CHECK-LABEL: TEST: sign int64[]
  # CHECK: mhlo.sign
  # CHECK-SAME: tensor<i64>
  print_ir(np.int64(0))(lax.sign)

  # CHECK-LABEL: TEST: sign uint32[]
  # CHECK: mhlo.compare
  # CHECK-SAME: tensor<ui32>
  print_ir(np.uint32(0))(lax.sign)

  # CHECK-LABEL: TEST: sin float32[]
  # CHECK: mhlo.sin
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.sin)

  # CHECK-LABEL: TEST: sinh float32[]
  # CHECK: chlo.sinh
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.sinh)

  # CHECK-LABEL: TEST: sub float32[] float32[]
  # CHECK: mhlo.sub
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(1), np.float32(2))(lax.sub)

  # CHECK-LABEL: TEST: sqrt bfloat16[]
  # CHECK: mhlo.sqrt
  # CHECK-SAME: tensor<bf16>
  print_ir(jnp.bfloat16(0))(lax.sqrt)

  # CHECK-LABEL: TEST: tan float16[]
  # CHECK: chlo.tan
  # CHECK-SAME: tensor<f16>
  print_ir(np.float16(0))(lax.tan)

  # CHECK-LABEL: TEST: tanh float32[]
  # CHECK: mhlo.tanh
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.tanh)


if __name__ == "__main__":
  app.run(main)
