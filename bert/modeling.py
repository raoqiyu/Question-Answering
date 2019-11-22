# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor

def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def is_special_none_tensor(tensor):
  """Checks if a tensor is a special None Tensor."""
  return tensor.shape.ndims == 0 and tensor.dtype == tf.int32

def pack_inputs(inputs):
  """Pack a list of `inputs` tensors to a tuple.

  Args:
    inputs: a list of tensors.

  Returns:
    a tuple of tensors. if any input is None, replace it with a special constant
    tensor.
  """
  inputs = tf.nest.flatten(inputs)
  outputs = []
  for x in inputs:
    if x is None:
      outputs.append(tf.constant(0, shape=[], dtype=tf.int32))
    else:
      outputs.append(x)
  return tuple(outputs)

def unpack_inputs(inputs):
  """unpack a tuple of `inputs` tensors to a tuple.

  Args:
    inputs: a list of tensors.

  Returns:
    a tuple of tensors. if any input is a special constant tensor, replace it
    with None.
  """
  inputs = tf.nest.flatten(inputs)
  outputs = []
  for x in inputs:
    if is_special_none_tensor(x):
      outputs.append(None)
    else:
      outputs.append(x)
  x = tuple(outputs)

  # To trick the very pointless 'unbalanced-tuple-unpacking' pylint check
  # from triggering.
  if len(x) == 1:
    return x[0]
  return tuple(outputs)

class EmbeddingLookUp(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size=768, initializer_range=0.02, **kwargs):
        super(EmbeddingLookUp, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range

    def build(self, unused_input_shapes):
        self.embeddings = self.add_weight(
            "embeddings",
            shape=[self.vocab_size, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range),
            dtype=self.dtype
        )
        super(EmbeddingLookUp, self).build(unused_input_shapes)

    def call(self, inputs):
        input_shapes = get_shape_list(inputs)
        flat_inputs = tf.reshape(inputs, [-1])
        flat_outputs = tf.gather(self.embeddings, flat_inputs)
        outputs = tf.reshape(flat_outputs, input_shapes+[self.embedding_size])
        return outputs

class EmbeddingPostprocessor(tf.keras.layers.Layer):
    def __init__(self, use_token_type=False,
                    token_type_vocab_size=16,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=0.02,
                    max_position_embeddings=512,
                    dropout_prob=0.1,
                    **kwargs,
                ):
        super(EmbeddingPostprocessor, self).__init__(**kwargs)

        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.token_type_embedding_name = token_type_embedding_name

        self.use_position_embeddings = use_position_embeddings
        self.position_embedding_name = position_embedding_name
        self.max_position_embeddings = max_position_embeddings

        self.initializer_range = initializer_range

        self.dropout_prob = dropout_prob

        if self.use_token_type and not self.token_type_vocab_size:
            raise ValueError("`token_type_vocab_size` must be specified if"
                             "`use_token_type` is True.")
        self.token_type_embeddings = None
        self.position_embeddings = None

    def build(self, input_shapes):
        """
        input_shapes: word_embeddings shape, token_type_ids shape
        :param input_shapes:
        :return:
        """
        word_embeddings_shape, _ = input_shapes
        width = word_embeddings_shape.as_list()[-1]

        if self.use_token_type:
            self.token_type_embeddings = self.add_weight(
                self.token_type_embedding_name,
                shape=[self.token_type_vocab_size, width],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range),
                dtype=self.dtype
            )
        if self.use_position_embeddings:
            self.position_embeddings = self.add_weight(
                self.position_embedding_name,
                shape=[self.max_position_embeddings, width],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range),
                dtype=self.dtype)

        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm",
                                                             axis=-1, epsilon=1e-12, dtype=tf.float32)
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_prob,
                                                      dtype=tf.float32)

        super(EmbeddingPostprocessor, self).build(input_shapes)

    def __call__(self, word_embeddings, token_type_ids=None, **kwargs):
        inputs = pack_inputs([word_embeddings, token_type_ids])
        return super(EmbeddingPostprocessor, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        word_embeddings, token_type_ids = unpack_inputs(inputs)

        input_shape = get_shape_list(word_embeddings, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        output = word_embeddings
        if self.use_token_type:
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            flat_token_type_embeddings = tf.gather(self.token_type_embeddings,
                                              flat_token_type_ids)
            token_type_embeddings = tf.reshape(flat_token_type_embeddings,
                                               [batch_size, seq_length, width])
            output += token_type_embeddings

        if self.use_position_embeddings:
            position_embeddings = tf.expand_dims(
                tf.slice(self.position_embeddings, [0, 0], [seq_length, width]),
                axis=0)

            output += position_embeddings

        output = self.layer_norm(output)
        output = self.dropout(output)

        return output

class Attention(tf.keras.layers.Layer):

    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

      This is an implementation of multi-headed attention based on "Attention
      is all you Need". If `from_tensor` and `to_tensor` are the same, then
      this is self-attention. Each timestep in `from_tensor` attends to the
      corresponding sequence in `to_tensor`, and returns a fixed-with vector.

      This function first projects `from_tensor` into a "query" tensor and
      `to_tensor` into "key" and "value" tensors. These are (effectively) a list
      of tensors of length `num_attention_heads`, where each tensor is of shape
      [batch_size, seq_length, size_per_head].

      Then, the query and key tensors are dot-producted and scaled. These are
      softmaxed to obtain attention probabilities. The value tensors are then
      interpolated by these probabilities, then concatenated back to a single
      tensor and returned.

      In practice, the multi-headed attention are done with transposes and
      reshapes rather than actual separate tensors.

      Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
          from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
          from_seq_length, to_seq_length]. The values should be 1 or 0. The
          attention scores will effectively be set to -infinity for any positions in
          the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
          attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
          * from_seq_length, num_attention_heads * size_per_head]. If False, the
          output will be of shape [batch_size, from_seq_length, num_attention_heads
          * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
          of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `to_tensor`.

      Returns:
        float Tensor of shape [batch_size, from_seq_length,
          num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
          true, this will be of shape [batch_size * from_seq_length,
          num_attention_heads * size_per_head]).

      Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
      """
    def __init__(self, num_attention_heads=1,
                    size_per_head=512,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=True,
                    query_act=None,key_act=None,value_act=None,
                    **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.query_act = query_act
        self.key_act = key_act
        self.value_act = value_act

    def build(self, input_shapes):
        from_tensor_shape, to_tensor_shape, attention_mask_shape = input_shapes

        """Implements build() for the layer."""
        # self.query_dense = self._projection_dense_layer("query")
        # self.key_dense = self._projection_dense_layer("key")
        # self.value_dense = self._projection_dense_layer("value")

        # `query_layer` = [B*F, N*H]
        self.query_dense = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
                                                 activation=self.query_act,
                                                 name="query",
                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                                                    stddev=self.initializer_range))
        # `key_layer` = [B*T, N*H]
        self.key_dense = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
                                                 activation=self.key_act,
                                                 name="key",
                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                     stddev=self.initializer_range))

        # `value_layer` = [B*T, N*H]
        self.value_dense = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
                                                 activation=self.value_act,
                                                 name="value",
                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                     stddev=self.initializer_range))

        self.attention_probs_dropout = tf.keras.layers.Dropout(
                                                 rate=self.attention_probs_dropout_prob)
        super(Attention, self).build(input_shapes)

    def reshape_to_matrix(self, inputs):
        """Reshape N > 2 rank tensor to rank 2 tensor for performance."""
        ndims = inputs.shape.ndims
        if ndims < 2:
            raise ValueError("Input tensor must have at least rank 2."
                             "Shape = %s" % (inputs.shape))
        if ndims == 2:
            return inputs

        width = inputs.shape[-1]
        outputs = tf.reshape(inputs, [-1, width])
        return outputs

    def transpose_for_scores(self, input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def __call__(self,  from_tensor, to_tensor, attention_mask=None, **kwargs):
        inputs = pack_inputs([from_tensor, to_tensor, attention_mask])
        return super(Attention, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        from_tensor, to_tensor, attention_mask = unpack_inputs(inputs)
        from_shape = get_shape_list(from_tensor, expected_rank=[3])
        to_shape = get_shape_list(to_tensor, expected_rank=[3])
        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)

        # [B*F, N*H]
        query_tensor = self.query_dense(from_tensor_2d)

        # [B*T, N*H]
        key_tensor = self.key_dense(to_tensor_2d)

        # [B*T, N*H]
        value_tensor = self.value_dense(to_tensor_2d)

        # `query_tensor` = [B, N, F, H]
        query_tensor = self.transpose_for_scores(query_tensor, batch_size,
                                           self.num_attention_heads,from_seq_length,
                                           self.size_per_head)

        # `key_tensor` = [B, N, T, H]
        key_tensor = self.transpose_for_scores(key_tensor, batch_size,
                                            self.num_attention_heads,to_seq_length,
                                            self.size_per_head)

        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_tensor, key_tensor, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_probs_dropout(attention_probs)

        # `value_layer` = [B, T, N, H]
        value_tensor = tf.reshape(value_tensor,
                                [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head])

        # `value_layer` = [B, N, T, H]
        value_tensor = tf.transpose(value_tensor, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_tensor)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if self.do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, self.num_attention_heads * self.size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head])

        return context_layer

class TransformerBlock(tf.keras.layers.Layer):
    """Single transformer layer.

    It has two sub-layers. The first is a multi-head self-attention mechanism, and
    the second is a positionwise fully connected feed-forward network.
    """

    def __init__(self,hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    intermediate_act_fn='gelu',
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    initializer_range=0.02,
                    **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = get_activation(intermediate_act_fn)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
              "The hidden size (%d) is not a multiple of the number of attention "
              "heads (%d)" % (self.hidden_size, self.num_attention_heads))
        self.size_per_head = int(self.hidden_size / self.num_attention_heads)

    def build(self, unused_input_shapes):
        # input_tensor_shape, attention_mask_shape = input_shapes

        self.attention_layer = Attention(
            num_attention_heads=self.num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            do_return_2d_tensor=True,
            name='self_attention')

        self.linear_projection_layer = tf.keras.layers.Dense(self.hidden_size,
                                                            name="self_attention_output",
                                                            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                                                    stddev=self.initializer_range))
        self.attention_dropout_layer = tf.keras.layers.Dropout(rate=self.attention_probs_dropout_prob)
        self.attention_norm_layer =  tf.keras.layers.LayerNormalization(name="self_attention_layer_norm", axis=-1, epsilon=1e-12,
                                                              dtype=tf.float32)

        self.intermediate_layer = tf.keras.layers.Dense(self.intermediate_size,
                                                            name="intermediate",
                                                            activation=self.intermediate_act_fn,
                                                            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                                                    stddev=self.initializer_range))
        self.output_layer = tf.keras.layers.Dense(self.hidden_size,
                                                        name="output",
                                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                            stddev=self.initializer_range))
        self.output_dropout_layer = tf.keras.layers.Dropout(rate=self.attention_probs_dropout_prob)
        self.output_norm_layer = tf.keras.layers.LayerNormalization(name="output_layer_norm", axis=-1,
                                                                       epsilon=1e-12,
                                                                       dtype=tf.float32)
        super(TransformerBlock, self).build(unused_input_shapes)

    def __call__(self, input_tensor, attention_mask=None, **kwargs):
        inputs = pack_inputs([input_tensor, attention_mask])
        return super(TransformerBlock, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        (input_tensor, attention_mask) = unpack_inputs(inputs)

        attention_outputs = self.attention_layer(from_tensor=input_tensor,
                                                 to_tensor=input_tensor,
                                                 attention_mask=attention_mask)
        attention_outputs = self.linear_projection_layer(attention_outputs)
        attention_outputs = self.attention_dropout_layer(attention_outputs)
        attention_outputs = self.attention_norm_layer(input_tensor + attention_outputs)

        intermediate_output = self.intermediate_layer(attention_outputs)

        outputs = self.output_layer(intermediate_output)
        outputs = self.output_dropout_layer(outputs)
        outputs = self.output_norm_layer(outputs + attention_outputs)

        return  outputs

class Transformer(tf.keras.layers.Layer):
    """
    Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """
    def __init__(self,hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    intermediate_act_fn=gelu,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    initializer_range=0.02,
                    **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

    def build(self, unused_input_shapes):
        self.layers = []
        for i in range(self.num_hidden_layers):
            self.layers.append(TransformerBlock(hidden_size=self.hidden_size,
                        num_hidden_layers=self.num_hidden_layers,
                        num_attention_heads=self.num_attention_heads,
                        intermediate_size=self.intermediate_size,
                        intermediate_act_fn = self.intermediate_act_fn,
                        hidden_dropout_prob = self.hidden_dropout_prob,
                        attention_probs_dropout_prob = self.attention_probs_dropout_prob,
                        initializer_range = self.initializer_range,
                        name='layer_%d'%i))
        super(Transformer, self).build(unused_input_shapes)

    def __call__(self, input_tensor, attention_mask=None, **kwargs):
        inputs = pack_inputs([input_tensor, attention_mask])
        return super(Transformer, self).__call__(inputs=inputs, **kwargs)

    def call(self, inputs, do_return_all_layers=False):
        input_tensor,attention_mask = unpack_inputs(inputs)

        all_layer_outputs,output_tensor = [], input_tensor
        for layer in self.layers:
            output_tensor = layer(output_tensor, attention_mask)
            all_layer_outputs.append(output_tensor)

        if do_return_all_layers:
            return all_layer_outputs

        return all_layer_outputs[-1]

class BertModel(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(BertModel, self).__init__(**kwargs)
        self.config = config

    def build(self, unused_input_shapes):

        self.embedding_lookup = EmbeddingLookUp(
            vocab_size=self.config.vocab_size,
            embedding_size=self.config.hidden_size,
            initializer_range=self.config.initializer_range,
            name='word_embeddings'
        )
        self.embedding_postprocessor = EmbeddingPostprocessor(
            use_token_type=True,
            token_type_vocab_size=self.config.type_vocab_size,
            use_position_embeddings=True,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_prob=self.config.hidden_dropout_prob,
            name='embeddings_postprocessor'
        )
        self.encoder = Transformer(
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_act_fn=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            name='encoder'
        )
        self.pooler = tf.keras.layers.Dense(self.config.hidden_size,
                                            activation=tf.tanh,
                                            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                            stddev=self.config.initializer_range),
                                            name='pooler_transform'
                                            )
        super(BertModel, self).build(unused_input_shapes)

    def __call__(self,
                 input_word_ids,
                 input_mask=None,
                 input_type_ids=None,
                 **kwargs):
        inputs = pack_inputs([input_word_ids, input_mask, input_type_ids])
        return super(BertModel, self).__call__(inputs, **kwargs)

    def call(self, inputs, mode='bert'):
        input_word_ids, input_mask, input_type_ids = unpack_inputs(inputs)

        word_embeddings = self.embedding_lookup(input_word_ids)
        embedding_tensor =  self.embedding_postprocessor(word_embeddings=word_embeddings,
                                                         token_type_ids=input_type_ids)
        attention_mask = None
        if input_mask is not None:
            attention_mask = create_attention_mask_from_input_mask(input_word_ids,
                                                                   input_mask)
        if mode == 'encoder':
            return self.encoder(embedding_tensor, attention_mask, do_return_all_layers=True)

        sequence_output = self.encoder(embedding_tensor, attention_mask)
        first_token_tensor = tf.squeeze(sequence_output[:,0:1,:], axis=1)
        pooled_output = self.pooler(first_token_tensor)

        return (pooled_output, sequence_output)
