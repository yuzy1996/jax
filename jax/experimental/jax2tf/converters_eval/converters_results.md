# JAX Converters Evaluation Results

*Last generated on: 2022-05-30* (YYYY-MM-DD)

This file contains the evaluation results for all converters in table format.
Please see [README.md](README.md) for more details.

## Summary Table

| Example | jax2tf_xla | jax2tf_no_xla | jax2tfjs | jax2tflite | jax2tflite+flex | convert_hlo | convert_hlo+flex |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flax/actor_critic | YES | YES | YES | YES | YES | YES | YES |
| flax/bilstm | YES | YES | YES | [NO](#example-flaxbilstm--converter-jax2tflite) |  [NO](#example-flaxbilstm--converter-jax2tfliteflex) |  [NO](#example-flaxbilstm--converter-convert_hlo) |  [NO](#example-flaxbilstm--converter-convert_hloflex) | 
| flax/cnn | YES | YES | YES | YES | YES | YES | YES |
| flax/resnet50 | YES | YES | YES | YES | YES | YES | YES |
| flax/seq2seq_lstm | YES | YES | [NO](#example-flaxseq2seq_lstm--converter-jax2tfjs) |  [NO](#example-flaxseq2seq_lstm--converter-jax2tflite) |  YES | [NO](#example-flaxseq2seq_lstm--converter-convert_hlo) |  YES |
| flax/transformer_lm1b | [NO](#example-flaxtransformer_lm1b--converter-jax2tf_xla) |  [NO](#example-flaxtransformer_lm1b--converter-jax2tf_no_xla) |  [NO](#example-flaxtransformer_lm1b--converter-jax2tfjs) |  [NO](#example-flaxtransformer_lm1b--converter-jax2tflite) |  [NO](#example-flaxtransformer_lm1b--converter-jax2tfliteflex) |  [NO](#example-flaxtransformer_lm1b--converter-convert_hlo) |  [NO](#example-flaxtransformer_lm1b--converter-convert_hloflex) | 
| flax/transformer_nlp_seq | YES | YES | YES | [NO](#example-flaxtransformer_nlp_seq--converter-jax2tflite) |  YES | YES | YES |
| flax/transformer_wmt | [NO](#example-flaxtransformer_wmt--converter-jax2tf_xla) |  [NO](#example-flaxtransformer_wmt--converter-jax2tf_no_xla) |  [NO](#example-flaxtransformer_wmt--converter-jax2tfjs) |  [NO](#example-flaxtransformer_wmt--converter-jax2tflite) |  [NO](#example-flaxtransformer_wmt--converter-jax2tfliteflex) |  [NO](#example-flaxtransformer_wmt--converter-convert_hlo) |  [NO](#example-flaxtransformer_wmt--converter-convert_hloflex) | 
| flax/vae | YES | YES | YES | YES | YES | YES | YES |

## Errors

### Example: `flax/bilstm` | Converter: `jax2tflite`
```
ConverterError('
... (CROPPED)...
Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select 
TF Select ops: Abs, ScatterNd, Sign
Details:
	tf.Abs(tensor<2xi32>) -> (tensor<2xi32>) : {device = ""}
	tf.ScatterNd(tensor<?x1xi32>, tensor<2xi1>, tensor<1xi32>) -> (tensor<6xi1>) : {device = ""}
	tf.Sign(tensor<2xi32>) -> (tensor<2xi32>) : {device = ""}

')
```
[Back to top](#summary-table)

### Example: `flax/bilstm` | Converter: `jax2tflite+flex`
```
RuntimeError('third_party/tensorflow/lite/kernels/concatenation.cc:158 t->dims->data[d] != t0->dims->data[d] (3 != 1)Node number 11 (CONCATENATION) failed to prepare.Node number 29 (WHILE) failed to invoke.')
```
[Back to top](#summary-table)

### Example: `flax/bilstm` | Converter: `convert_hlo`
```
ConverterError('
... (CROPPED)...
Some ops in the model are custom ops, See instructions to implement custom ops: https://www.tensorflow.org/lite/guide/ops_custom 
Custom ops: Mod
Details:
	tf.Mod(tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>)

')
```
[Back to top](#summary-table)

### Example: `flax/bilstm` | Converter: `convert_hlo+flex`
```
ConverterError('
... (CROPPED)...
Some ops in the model are custom ops, See instructions to implement custom ops: https://www.tensorflow.org/lite/guide/ops_custom 
Custom ops: Mod
Details:
	tf.Mod(tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>)

')
```
[Back to top](#summary-table)

## `flax/seq2seq_lstm`
### Example: `flax/seq2seq_lstm` | Converter: `jax2tfjs`
```
ValueError('Unsupported Ops in the model before optimization
RightShift, BitwiseAnd, LeftShift, BitwiseOr, Bitcast, BitwiseXor')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm` | Converter: `jax2tflite`
```
ConverterError('
... (CROPPED)...
Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select 
TF Select ops: Bitcast, BitwiseOr, BitwiseXor, ConcatV2, LeftShift, Pack, RightShift, ScatterNd, SelectV2, Slice, StridedSlice
Details:
	tf.Bitcast(tensor<1x4xui32>) -> (tensor<1x4xf32>) : {device = ""}
	tf.BitwiseOr(tensor<1x4xui32>, tensor<ui32>) -> (tensor<1x4xui32>) : {device = ""}
	tf.BitwiseOr(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>) : {device = ""}
	tf.BitwiseOr(tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>) : {device = ""}
	tf.BitwiseXor(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>) : {device = ""}
	tf.BitwiseXor(tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>) : {device = ""}
	tf.BitwiseXor(tensor<ui32>, tensor<ui32>) -> (tensor<ui32>) : {device = ""}
	tf.ConcatV2(tensor<1xui32>, tensor<1xui32>, tensor<i32>) -> (tensor<2xui32>)
	tf.ConcatV2(tensor<2xui32>, tensor<2xui32>, tensor<i32>) -> (tensor<4xui32>) : {device = ""}
	tf.LeftShift(tensor<1xui32>, tensor<ui32>) -> (tensor<1xui32>) : {device = ""}
	tf.LeftShift(tensor<2xui32>, tensor<ui32>) -> (tensor<2xui32>) : {device = ""}
	tf.Pack(tensor<ui32>, tensor<ui32>) -> (tensor<2xui32>) : {axis = 0 : i64}
	tf.RightShift(tensor<1x4xui32>, tensor<ui32>) -> (tensor<1x4xui32>) : {device = ""}
	tf.RightShift(tensor<1xui32>, tensor<ui32>) -> (tensor<1xui32>) : {device = ""}
	tf.RightShift(tensor<2xui32>, tensor<ui32>) -> (tensor<2xui32>) : {device = ""}
	tf.ScatterNd(tensor<?x1xi32>, tensor<2xi1>, tensor<1xi32>) -> (tensor<4xi1>) : {device = ""}
	tf.ScatterNd(tensor<?x1xi32>, tensor<4xi1>, tensor<1xi32>) -> (tensor<4xi1>) : {device = ""}
	tf.SelectV2(tensor<i1>, tensor<1x4xui32>, tensor<1x4xui32>) -> (tensor<1x4xui32>) : {device = ""}
	tf.SelectV2(tensor<i1>, tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>) : {device = ""}
	tf.SelectV2(tensor<i1>, tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>) : {device = ""}
	tf.Slice(tensor<1x2xui32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<1x2xui32>) : {device = ""}
	tf.StridedSlice(tensor<2xui32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<1xui32>) : {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}
')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm` | Converter: `convert_hlo`
```
ConverterError("
... (CROPPED)...
Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select 
TF Select ops: Bitcast, BitwiseOr, BitwiseXor, ConcatV2, LeftShift, Mul, RightShift, Slice, StridedSlice, Sub
Details:
	tf.Bitcast(tensor<1x4xui32>) -> (tensor<1x4xf32>)
	tf.BitwiseOr(tensor<1x4xui32>, tensor<1x4xui32>) -> (tensor<1x4xui32>)
	tf.BitwiseOr(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>)
	tf.BitwiseOr(tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>)
	tf.BitwiseXor(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>)
	tf.BitwiseXor(tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>)
	tf.BitwiseXor(tensor<ui32>, tensor<ui32>) -> (tensor<ui32>)
	tf.ConcatV2(tensor<1xui32>, tensor<1xui32>, tensor<i64>) -> (tensor<2xui32>)
	tf.ConcatV2(tensor<2xui32>, tensor<2xui32>, tensor<i64>) -> (tensor<4xui32>)
	tf.LeftShift(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>)
	tf.LeftShift(tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>)
	tf.Mul(tensor<ui32>, tensor<2xui32>) -> (tensor<2xui32>)
	tf.RightShift(tensor<1x4xui32>, tensor<1x4xui32>) -> (tensor<1x4xui32>)
	tf.RightShift(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>)
	tf.RightShift(tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>)
	tf.Slice(tensor<1x2xui32>, tensor<2xi32>, tensor<2xi64>) -> (tensor<1x2xui32>)
	tf.StridedSlice(tensor<2xui32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> (tensor<1xui32>) : {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}
	tf.StridedSlice(tensor<4xui32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> (tensor<1xui32>) : {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}
	tf.Sub(tensor<ui32>, tensor<ui32>) -> (tensor<ui32>)

")
```
[Back to top](#summary-table)

## `flax/transformer_lm1b`
### Example: `flax/transformer_lm1b` | Converter: `jax2tf_xla`
```
InvalidArgumentError()
```
[Back to top](#summary-table)

### Example: `flax/transformer_lm1b` | Converter: `jax2tf_no_xla`
```
TypeError("The DType <class 'numpy._FloatAbstractDType'> could not be promoted by <class 'numpy.dtype[str_]'>. This means that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtype[str_]'>, <class 'numpy._FloatAbstractDType'>)")
```
[Back to top](#summary-table)

### Example: `flax/transformer_lm1b` | Converter: `jax2tfjs`
```
ValueError("in user code:


    ValueError: Got a non-Tensor value FrozenDict({
        cache: {
            decoder: {
                encoderdecoderblock_0: {
                    SelfAttention_0: {
                        cache_index: <tf.Tensor 'StatefulPartitionedCall:1' shape=() dtype=int32>,
                        cached_key: <tf.Tensor 'StatefulPartitionedCall:2' shape=(2, 1, 1, 2) dtype=float32>,
                        cached_value: <tf.Tensor 'StatefulPartitionedCall:3' shape=(2, 1, 1, 2) dtype=float32>,
                    },
                },
                posembed_output: {
                    cache_index: <tf.Tensor 'StatefulPartitionedCall:4' shape=() dtype=uint32>,
                },
            },
        },
    }) for key 'output_1' in the output of the function __inference_tf_graph_302181 used to generate the SavedModel signature 'serving_default'. Outputs for functions used as signatures must be a single Tensor, a sequence of Tensors, or a dictionary from string to Tensor.
")
```
[Back to top](#summary-table)

### Example: `flax/transformer_lm1b` | Converter: `jax2tflite`
```
ConverterError('
... (CROPPED)...
Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select 
TF Select ops: Sign
Details:
	tf.Sign(tensor<2x1xf32>) -> (tensor<2x1xf32>) : {device = ""}

')
```
[Back to top](#summary-table)

### Example: `flax/transformer_lm1b` | Converter: `jax2tflite+flex`
```
ValueError('Returned output tuples lengths do not match: TF length vs JAX length: 5 != 2')
```
[Back to top](#summary-table)

### Example: `flax/transformer_lm1b` | Converter: `convert_hlo`
```
ValueError('Returned output tuples lengths do not match: TF length vs JAX length: 5 != 2')
```
[Back to top](#summary-table)

### Example: `flax/transformer_lm1b` | Converter: `convert_hlo+flex`
```
ValueError('Returned output tuples lengths do not match: TF length vs JAX length: 5 != 2')
```
[Back to top](#summary-table)

## `flax/transformer_nlp_seq`
### Example: `flax/transformer_nlp_seq` | Converter: `jax2tflite`
```
ConverterError('
... (CROPPED)...
Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select 
TF Select ops: Sign
Details:
	tf.Sign(tensor<2x1xf32>) -> (tensor<2x1xf32>) : {device = ""}

')
```
[Back to top](#summary-table)

## `flax/transformer_wmt`
### Example: `flax/transformer_wmt` | Converter: `jax2tf_xla`
```
InvalidArgumentError()
```
[Back to top](#summary-table)

### Example: `flax/transformer_wmt` | Converter: `jax2tf_no_xla`
```
TypeError("The DType <class 'numpy._FloatAbstractDType'> could not be promoted by <class 'numpy.dtype[str_]'>. This means that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtype[str_]'>, <class 'numpy._FloatAbstractDType'>)")
```
[Back to top](#summary-table)

### Example: `flax/transformer_wmt` | Converter: `jax2tfjs`
```
ValueError("in user code:


    ValueError: Got a non-Tensor value FrozenDict({
        cache: {
            decoder: {
                encoderdecoderblock_0: {
                    SelfAttention_0: {
                        cache_index: <tf.Tensor 'StatefulPartitionedCall:1' shape=() dtype=int32>,
                        cached_key: <tf.Tensor 'StatefulPartitionedCall:2' shape=(2, 1, 1, 2) dtype=float32>,
                        cached_value: <tf.Tensor 'StatefulPartitionedCall:3' shape=(2, 1, 1, 2) dtype=float32>,
                    },
                },
                posembed_output: {
                    cache_index: <tf.Tensor 'StatefulPartitionedCall:4' shape=() dtype=uint32>,
                },
            },
        },
    }) for key 'output_1' in the output of the function __inference_tf_graph_329451 used to generate the SavedModel signature 'serving_default'. Outputs for functions used as signatures must be a single Tensor, a sequence of Tensors, or a dictionary from string to Tensor.
")
```
[Back to top](#summary-table)

### Example: `flax/transformer_wmt` | Converter: `jax2tflite`
```
ConverterError('
... (CROPPED)...
Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select 
TF Select ops: Sign
Details:
	tf.Sign(tensor<2x1xf32>) -> (tensor<2x1xf32>) : {device = ""}

')
```
[Back to top](#summary-table)

### Example: `flax/transformer_wmt` | Converter: `jax2tflite+flex`
```
ValueError('Returned output tuples lengths do not match: TF length vs JAX length: 5 != 2')
```
[Back to top](#summary-table)

### Example: `flax/transformer_wmt` | Converter: `convert_hlo`
```
ValueError('Returned output tuples lengths do not match: TF length vs JAX length: 5 != 2')
```
[Back to top](#summary-table)

### Example: `flax/transformer_wmt` | Converter: `convert_hlo+flex`
```
ValueError('Returned output tuples lengths do not match: TF length vs JAX length: 5 != 2')
```
[Back to top](#summary-table)

See `models_test.py` for instructions on how to regenerate this table.
