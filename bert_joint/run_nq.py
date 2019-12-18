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
"""This code modified  https://github.com/google-research/language.git"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import os
import random
import re
from tqdm import tqdm
import enum
from bert import modeling,tokenization
import numpy as np
import pandas as pd
import tensorflow as tf

from config import FLAGS
from qa_utils.logger import logger



TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
  """Type of NQ answer."""
  UNKNOWN = 0
  YES = 1
  NO = 2
  SHORT = 3
  LONG = 4


class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
  """Answer record.

  An Answer contains the type of the answer and possibly the text (for
  long) as well as the offset (for extractive).
  """

  def __new__(cls, type_, text=None, offset=None):
    return super(Answer, cls).__new__(cls, type_, text, offset)


class NqExample(object):
  """A single training/test example."""

  def __init__(self,
               example_id,
               qas_id,
               questions,
               doc_tokens,
               doc_tokens_map=None,
               answer=None,
               start_position=None,
               end_position=None):
    self.example_id = example_id
    self.qas_id = qas_id
    self.questions = questions
    self.doc_tokens = doc_tokens
    self.doc_tokens_map = doc_tokens_map
    self.answer = answer
    self.start_position = start_position
    self.end_position = end_position
  def __repr__(self):
      example_id = 'example_id:%s'%(str(self.example_id))
      qas_id = 'qas_id:%s'%(self.qas_id)
      questions = 'questions:%s'%(str(self.questions))
      doc_tokens = 'doc_tokens:%s'%(str(self.doc_tokens))
      doc_tokens_map = 'doc_tokens_map:%s'%(str(self.doc_tokens_map))
      answer = 'answer:%s'%(str(self.answer))
      start_position = 'start_position:%s'%(str(self.start_position))
      end_position = 'end_position:%s'%(str(self.end_position))

      return '\n'.join([example_id, qas_id, questions, doc_tokens,
                        doc_tokens_map, answer, start_position, end_position])


def has_long_answer(a):
  return (a["long_answer"]["start_token"] >= 0 and
          a["long_answer"]["end_token"] >= 0)


def should_skip_context(e, idx):
    if FLAGS.skip_nested_contexts and not ( (('top_level' in e["long_answer_candidates"][idx]) and
                                          e["long_answer_candidates"][idx]["top_level"]) or \
                                        (('topOlevel' in e["long_answer_candidates"][idx]) and \
                                          e["long_answer_candidates"][idx]["topOlevel"]) ):
      return True
    elif not get_candidate_text(e, idx).text.strip():
      # Skip empty contexts.
      return True
    else:
      return False


def get_first_annotation(e):
  """Returns the first short or long answer in the example.

  Args:
    e: (dict) annotated example.

  Returns:
    annotation: (dict) selected annotation
    annotated_idx: (int) index of the first annotated candidate.
    annotated_sa: (tuple) char offset of the start and end token
        of the short answer. The end token is exclusive.
  """
  if "annotations" not in e:
      return None, -1, (-1, -1)

  positive_annotations = sorted(
      [a for a in e["annotations"] if has_long_answer(a)],
      key=lambda a: a["long_answer"]["candidate_index"])

  for a in positive_annotations:
    if a["short_answers"]:
      idx = a["long_answer"]["candidate_index"]
      start_token = a["short_answers"][0]["start_token"]
      end_token = a["short_answers"][-1]["end_token"]
      return a, idx, (token_to_char_offset(e, idx, start_token),
                      token_to_char_offset(e, idx, end_token) - 1)

  for a in positive_annotations:
    idx = a["long_answer"]["candidate_index"]
    return a, idx, (-1, -1)

  return None, -1, (-1, -1)


def get_text_span(example, span):
  """Returns the text in the example's document in the given token span."""
  token_positions = []
  tokens = []
  for i in range(span["start_token"], span["end_token"]):
    t = example["document_tokens"][i]
    if not t["html_token"]:
      token_positions.append(i)
      token = t["token"].replace(" ", "")
      tokens.append(token)
  return TextSpan(token_positions, " ".join(tokens))


def token_to_char_offset(e, candidate_idx, token_idx):
  """Converts a token index to the char offset within the candidate."""
  c = e["long_answer_candidates"][candidate_idx]
  char_offset = 0
  for i in range(c["start_token"], token_idx):
    t = e["document_tokens"][i]
    if not t["html_token"]:
      token = t["token"].replace(" ", "")
      char_offset += len(token) + 1
  return char_offset


def get_candidate_type(e, idx):
  """Returns the candidate's type: Table, Paragraph, List or Other."""
  c = e["long_answer_candidates"][idx]
  first_token = e["document_tokens"][c["start_token"]]["token"]
  if first_token == "<Table>":
    return "Table"
  elif first_token == "<P>":
    return "Paragraph"
  elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
    return "List"
  elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
    return "Other"
  else:
    logger.warning("Unknoww candidate type found: %s", first_token)
    return "Other"


def add_candidate_types_and_positions(e):
  """Adds type and position info to each candidate in the document."""
  counts = collections.defaultdict(int)
  for idx, c in candidates_iter(e):
    context_type = get_candidate_type(e, idx)
    if counts[context_type] < FLAGS.max_position:
      counts[context_type] += 1
    c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "[NoLongAnswer]"
  else:
    return e["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(e, idx):
  """Returns a text representation of the candidate at the given index."""
  # No candidate at this index.
  if idx < 0 or idx >= len(e["long_answer_candidates"]):
    return TextSpan([], "")

  # This returns an actual candidate.
  return get_text_span(e, e["long_answer_candidates"][idx])


def candidates_iter(e):
  """Yield's the candidates that should not be skipped in an example."""
  for idx, c in enumerate(e["long_answer_candidates"]):
    if should_skip_context(e, idx):
      continue
    yield idx, c


def create_example_from_jsonl(line):
  """Creates an NQ example from a given line of JSON."""
  try:
    e = json.loads(line, object_pairs_hook=collections.OrderedDict)
  except Exception as ex:
    print(ex)
    return None
  if "document_title" not in e:
    e["document_title"] = e["example_id"]
  if "document_tokens" not in e:
    e["document_tokens"] = []
    document_tokens = e["document_text"].split(" ")
    for token in document_tokens:
      e["document_tokens"].append({"token":token, "start_byte":-1, "end_byte":-1, "html_token":'<' in token})

  add_candidate_types_and_positions(e)
  annotation, annotated_idx, annotated_sa = get_first_annotation(e)

  # annotated_idx: index of the first annotated context, -1 if null.
  # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
  question = {"input_text": e["question_text"]}
  answer = {
      "candidate_id": annotated_idx,
      "span_text": "",
      "span_start": -1,
      "span_end": -1,
      "input_text": "long",
  }

  # Yes/no answers are added in the input text.
  if annotation is not None:
    assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
    if annotation["yes_no_answer"] in ("YES", "NO"):
      answer["input_text"] = annotation["yes_no_answer"].lower()

  # Add a short answer if one was found.
  if annotated_sa != (-1, -1):
    answer["input_text"] = "short"
    span_text = get_candidate_text(e, annotated_idx).text
    answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
    answer["span_start"] = annotated_sa[0]
    answer["span_end"] = annotated_sa[1]
    expected_answer_text = get_text_span(
        e, {
            "start_token": annotation["short_answers"][0]["start_token"],
            "end_token": annotation["short_answers"][-1]["end_token"],
        }).text
    assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])

  # Add a long answer if one was found.
  elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
    answer["span_text"] = get_candidate_text(e, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = len(answer["span_text"])

  context_idxs = [-1]
  context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
  context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
  for idx, _ in candidates_iter(e):
    context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
    context["text_map"], context["text"] = get_candidate_text(e, idx)
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= FLAGS.max_contexts:
      break

  # Assemble example.
  example = {
      "name": e["document_title"],
      "id": str(e["example_id"]),
      "questions": [question],
      "answers": [answer],
      "has_correct_context": annotated_idx in context_idxs
  }

  single_map = []
  single_context = []
  offset = 0
  for context in context_list:
    single_map.extend([-1, -1])
    single_context.append("[ContextId=%d] %s" %
                          (context["id"], context["type"]))
    offset += len(single_context[-1]) + 1
    if context["id"] == annotated_idx:
      answer["span_start"] += offset
      answer["span_end"] += offset

    # Many contexts are empty once the HTML tags have been stripped, so we
    # want to skip those.
    if context["text"]:
      single_map.extend(context["text_map"])
      single_context.append(context["text"])
      offset += len(single_context[-1]) + 1

  example["contexts"] = " ".join(single_context)
  example["contexts_map"] = single_map
  if annotated_idx in context_idxs:
    expected = example["contexts"][answer["span_start"]:answer["span_end"]]

    # This is a sanity check to ensure that the calculated start and end
    # indices match the reported span text. If this assert fails, it is likely
    # a bug in the data preparation code above.
    assert expected == answer["span_text"], (expected, answer["span_text"])

  return example


def make_nq_answer(contexts, answer):
  """Makes an Answer object following NQ conventions.

  Args:
    contexts: string containing the context
    answer: dictionary with `span_start` and `input_text` fields

  Returns:
    an Answer object. If the Answer type is YES or NO or LONG, the text
    of the answer is the long answer. If the answer type is UNKNOWN, the text of
    the answer is empty.
  """
  start = answer["span_start"]
  end = answer["span_end"]
  input_text = answer["input_text"]

  if (answer["candidate_id"] == -1 or start >= len(contexts) or
      end > len(contexts)):
    answer_type = AnswerType.UNKNOWN
    start = 0
    end = 1
  elif input_text.lower() == "yes":
    answer_type = AnswerType.YES
  elif input_text.lower() == "no":
    answer_type = AnswerType.NO
  elif input_text.lower() == "long":
    answer_type = AnswerType.LONG
  else:
    answer_type = AnswerType.SHORT

  return Answer(answer_type, text=contexts[start:end], offset=start)


def read_nq_entry(entry, is_training):
  """Converts a NQ entry into a list of NqExamples."""

  def is_whitespace(c):
    return c in " \t\r\n" or ord(c) == 0x202F

  examples = []
  contexts_id = entry["id"]
  contexts = entry["contexts"]
  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True
  for c in contexts:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append(len(doc_tokens) - 1)

  questions = []
  for i, question in enumerate(entry["questions"]):
    qas_id = "{}".format(contexts_id)
    question_text = question["input_text"]
    start_position = None
    end_position = None
    answer = None
    if is_training:
      answer_dict = entry["answers"][i]
      answer = make_nq_answer(contexts, answer_dict)

      # For now, only handle extractive, yes, and no.
      if answer is None or answer.offset is None:
        continue
      start_position = char_to_word_offset[answer.offset]
      end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

      # Only add answers where the text can be exactly recovered from the
      # document. If this CAN'T happen it's likely due to weird Unicode
      # stuff so we will just skip the example.
      #
      # Note that this means for training mode, every example is NOT
      # guaranteed to be preserved.
      actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
      cleaned_answer_text = " ".join(
          tokenization.whitespace_tokenize(answer.text))
      if actual_text.find(cleaned_answer_text) == -1:
        logger.warning("Could not find answer: '%s' vs. '%s'", actual_text,
                           cleaned_answer_text)
        continue

    questions.append(question_text)
    example = NqExample(
        example_id=int(contexts_id),
        qas_id=qas_id,
        questions=questions[:],
        doc_tokens=doc_tokens,
        doc_tokens_map=entry.get("contexts_map", None),
        answer=answer,
        start_position=start_position,
        end_position=end_position)
    examples.append(example)
  return examples


def convert_examples_to_features(examples, tokenizer, is_training, output_fn):
  """Converts a list of NqExamples into InputFeatures."""
  num_spans_to_ids = collections.defaultdict(list)

  for example in tqdm(examples):
    # print('\n\n\n\n',example)
    example_index = example.example_id
    features = convert_examples_to_features(example, tokenizer, is_training)
    num_spans_to_ids[len(features)].append(example.qas_id)

    for feature in features:
      # print(feature, '\n')
      feature.example_index = example_index
      feature.unique_id = feature.example_index + feature.doc_span_index
      output_fn(feature)

  return num_spans_to_ids


def check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""
  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def convert_single_example(example, tokenizer, is_training):
  """Converts a single NqExample into a list of InputFeatures."""
  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = []
  features = []
  for (i, token) in enumerate(example.doc_tokens):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenize(tokenizer, token)
    tok_to_orig_index.extend([i] * len(sub_tokens))
    all_doc_tokens.extend(sub_tokens)

  # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
  # tokenized word tokens in the contexts. The word tokens might themselves
  # correspond to word tokens in a larger document, with the mapping given
  # by `doc_tokens_map`.
  if example.doc_tokens_map:
    tok_to_orig_index = [
        example.doc_tokens_map[index] for index in tok_to_orig_index
    ]

  # QUERY
  query_tokens = []
  query_tokens.append("[Q]")
  query_tokens.extend(tokenize(tokenizer, example.questions[-1]))
  if len(query_tokens) > FLAGS.max_query_length:
    query_tokens = query_tokens[-FLAGS.max_query_length:]

  # ANSWER
  tok_start_position = 0
  tok_end_position = 0
  if is_training:
    tok_start_position = orig_to_tok_index[example.start_position]
    if example.end_position < len(example.doc_tokens) - 1:
      tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    else:
      tok_end_position = len(all_doc_tokens) - 1

  # The -3 accounts for [CLS], [SEP] and [SEP]
  max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of up to our max length with a stride of `doc_stride`.
  _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length"])
  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    length = min(length, max_tokens_for_doc)
    doc_spans.append(_DocSpan(start=start_offset, length=length))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, FLAGS.doc_stride)

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    tokens.extend(query_tokens)
    segment_ids.extend([0] * len(query_tokens))
    tokens.append("[SEP]")
    segment_ids.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

      is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                            split_token_index)
      token_is_max_context[len(tokens)] = is_max_context
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    assert len(tokens) == len(segment_ids)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (FLAGS.max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length

    start_position = None
    end_position = None
    answer_type = None
    answer_text = ""
    if is_training:
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      contains_an_annotation = (
          tok_start_position >= doc_start and tok_end_position <= doc_end)
      if ((not contains_an_annotation) or
          example.answer.type == AnswerType.UNKNOWN):
        # If an example has unknown answer type or does not contain the answer
        # span, then we only include it with probability --include_unknowns.
        # When we include an example with unknown answer type, we set the first
        # token of the passage to be the annotated short span.
        if (FLAGS.include_unknowns < 0 or
            random.random() > FLAGS.include_unknowns):
          continue
        start_position = 0
        end_position = 0
        answer_type = AnswerType.UNKNOWN
      else:
        doc_offset = len(query_tokens) + 2
        start_position = tok_start_position - doc_start + doc_offset
        end_position = tok_end_position - doc_start + doc_offset
        answer_type = example.answer.type

      answer_text = " ".join(tokens[start_position:(end_position + 1)])

    feature = InputFeatures(
        unique_id=-1,
        example_index=-1,
        doc_span_index=doc_span_index,
        tokens=tokens,
        token_to_orig_map=token_to_orig_map,
        token_is_max_context=token_is_max_context,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        start_position=start_position,
        end_position=end_position,
        answer_text=answer_text,
        answer_type=answer_type)

    features.append(feature)

  return features


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)


def tokenize(tokenizer, text, apply_basic_tokenization=False):
  """Tokenizes text, optionally looking up special tokens separately.

  Args:
    tokenizer: a tokenizer from bert.tokenization.FullTokenizer
    text: text to tokenize
    apply_basic_tokenization: If True, apply the basic tokenization. If False,
      apply the full tokenization (basic + wordpiece).

  Returns:
    tokenized text.

  A special token is any text with no spaces enclosed in square brackets with no
  space, so we separate those out and look them up in the dictionary before
  doing actual tokenization.
  """
  tokenize_fn = tokenizer.tokenize
  if apply_basic_tokenization:
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
  tokens = []
  for token in text.split(" "):
    if _SPECIAL_TOKENS_RE.match(token):
      if token in tokenizer.vocab:
        tokens.append(token)
      else:
        tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
    else:
      tokens.extend(tokenize_fn(token))
  return tokens


class CreateTFExampleFn(object):
  """Functor for creating NQ tf.Examples."""

  def __init__(self, is_training):
    self.is_training = is_training
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  def process(self, example):
    """Coverts an NQ example in a list of serialized tf examples."""
    nq_examples = read_nq_entry(example, self.is_training)
    input_features = []
    for nq_example in nq_examples:
      input_features.extend(
          convert_single_example(nq_example, self.tokenizer, self.is_training))

    for input_feature in input_features:
      input_feature.example_index = int(example["id"])
      input_feature.unique_id = (
          input_feature.example_index + input_feature.doc_span_index)

      def create_int_feature(values):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))

      features = collections.OrderedDict()
      features["unique_id"] = create_int_feature([input_feature.unique_id])
      features["input_ids"] = create_int_feature(input_feature.input_ids)
      features["input_mask"] = create_int_feature(input_feature.input_mask)
      features["segment_ids"] = create_int_feature(input_feature.segment_ids)

      if self.is_training:
        features["start_positions"] = create_int_feature(
            [input_feature.start_position])
        features["end_positions"] = create_int_feature(
            [input_feature.end_position])
        features["answer_types"] = create_int_feature(
            [input_feature.answer_type])
      else:
        token_map = [-1] * len(input_feature.input_ids)
        for k, v in input_feature.token_to_orig_map.iteritems():
          token_map[k] = v
        features["token_map"] = create_int_feature(token_map)

      yield tf.train.Example(features=tf.train.Features(
          feature=features)).SerializeToString()


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               answer_text="",
               answer_type=AnswerType.SHORT):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.answer_text = answer_text
    self.answer_type = answer_type

  def __repr__(self):
      unique_id = 'unique_id: '+str(self.unique_id)
      example_index = 'example_index: '+str(self.example_index)
      doc_span_index = 'doc_span_index: '+str(self.doc_span_index)
      tokens = 'tokens: ' + str(self.tokens)
      token_to_orig_map = 'token_to_orig_map: ' + str(self.token_to_orig_map)
      token_is_max_context = 'token_is_max_context: '+ str(self.token_is_max_context)
      input_ids = 'input_ids: '+ str(self.input_ids)
      input_mask = 'input_mask:' + str(self.input_mask)
      segment_ids = 'segment_ids: '+ str(self.segment_ids)
      start_position = 'start_position:' + str(self.start_position)
      end_position = 'end_position: '+str(self.end_position)
      answer_text = 'answer_text: ' +str(self.answer_text)
      answer_type = 'answer_type: '+ str(self.answer_type)

      return '\n'.join([unique_id, example_index, doc_span_index, tokens, token_to_orig_map,
                        token_is_max_context,input_ids, input_mask, segment_ids, start_position,
                        end_position, answer_text, answer_type])



def read_nq_examples(input_file, is_training, sample_begin_idx=50000,n_samples=50000):
  """Read a NQ json file into a list of NqExample."""
  input_paths = tf.io.gfile.glob(input_file)
  input_data = []
  sample_end_idx = sample_begin_idx + n_samples
  def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.io.gfile.GFile(path, "rb"))
    else:
      return open(path, "r")

  for path in input_paths:
    logger.info("Reading: %s", path)
    with _open(path) as input_file:
      line_cnt = -1
      for line in tqdm(input_file):
        line_cnt += 1
        if line_cnt < sample_begin_idx:
          continue
        if line_cnt > sample_end_idx:
            break
        example = create_example_from_jsonl(line)
        if example is not None:
          input_data.append(example)
  assert line_cnt == sample_end_idx+1
  examples = []
  for entry in input_data:
    examples.extend(read_nq_entry(entry, is_training))
  logger.info('Processing samples from %d-th to %d-th', sample_begin_idx, sample_end_idx)
  return examples


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling_einsum.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Get the logits for the start and end predictions.
  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling_einsum.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/nq/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/nq/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  # Get the logits for the answer type prediction.
  answer_type_output_layer = model.get_pooled_output()
  answer_type_hidden_size = answer_type_output_layer.shape[-1].value

  num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
  answer_type_output_weights = tf.get_variable(
      "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  answer_type_output_bias = tf.get_variable(
      "answer_type_output_bias", [num_answer_types],
      initializer=tf.zeros_initializer())

  answer_type_logits = tf.matmul(
      answer_type_output_layer, answer_type_output_weights, transpose_b=True)
  answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                      answer_type_output_bias)

  return (start_logits, end_logits, answer_type_logits)


def input_fn_builder(input_file, seq_length, is_training, drop_remainder,
                     batch_size,n_repeats=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_id": tf.io.FixedLenFeature([], tf.int64),
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.io.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.io.FixedLenFeature([], tf.int64)
    name_to_features["answer_types"] = tf.io.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features=name_to_features):

      example = tf.io.parse_single_example(serialized=record, features=name_to_features)
      for name in example.keys():
        t = example[name]
        if name != 'unique_id':  # t.dtype == tf.int64:
          t = tf.cast(t, dtype=tf.int32)
        example[name] = t

      if not is_training:
        return example
      else:
        x, y = {}, {}
        y['tf_op_layer_start_positions']=example['start_positions']
        y['tf_op_layer_end_positions']=example['end_positions']
        y['answer_types'] = example['answer_types']
        y['unique_id'] = example['unique_id']

        x['unique_id'] = example['unique_id']
        x['input_ids'] = example['input_ids']
        x['input_mask'] = example['input_mask']
        x['segment_ids'] = example['segment_ids']


        return x,y

  data = tf.data.TFRecordDataset(input_file)

  if is_training:
     if n_repeats:
       data = data.repeat(n_repeats)
     data = data.shuffle(buffer_size=100)

  data_batch = data.map(_decode_record).batch(batch_size=batch_size,
                                              drop_remainder=drop_remainder)

  return data, data_batch


RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_id"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      features["answer_types"] = create_int_feature([feature.answer_type])
    else:
      token_map = [-1] * len(feature.input_ids)
      for k, v in feature.token_to_orig_map.items():
        token_map[k] = v
      features["token_map"] = create_int_feature(token_map)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])


class EvalExample(object):
  """Eval data available for a single example."""

  def __init__(self, example_id, candidates):
    self.example_id = example_id
    self.candidates = candidates
    self.results = {}
    self.features = {}


class ScoreSummary(object):

  def __init__(self):
    self.predicted_label = None
    self.short_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None


def read_candidates_from_one_split(input_path):
  """Read candidates from a single jsonl file."""
  candidates_dict = {}
  logger.info("Reading examples from: %s", input_path)
  if input_path.endswith('gz'):
    with gzip.GzipFile(fileobj=tf.io.gfile.GFile(input_path,'rb')) as input_file:
      for line in input_file:
        e = json.loads(line)
        candidates_dict[e["example_id"]] = e["long_answer_candidates"]
  else:
    with tf.io.gfile.GFile(input_path) as input_file:
      for line in input_file:
        e = json.loads(line)
        candidates_dict[e["example_id"]] = e["long_answer_candidates"]
  return candidates_dict


def read_candidates(input_pattern):
  """Read candidates with real multiple processes."""
  input_paths = tf.io.gfile.glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    final_dict.update(read_candidates_from_one_split(input_path))
  return final_dict


def get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(
      enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def compute_predictions(example):
  """Converts an example into an NQEval object for evaluation."""
  predictions = []
  n_best_size = 10
  max_answer_length = 30

  for unique_id, result in example.results.items():
    if unique_id not in example.features:
      raise ValueError("No feature found with unique_id:", unique_id)
    token_map = np.array(example.features[unique_id]["token_map"])

    start_indexes = get_best_indexes(result["start_logits"], n_best_size)
    end_indexes = get_best_indexes(result["end_logits"], n_best_size)
    if len(start_indexes) == 0 or len(end_indexes) == 0:
      continue

    for start_index in start_indexes:
      for end_index in end_indexes:
        if end_index < start_index:
          continue
        if token_map[start_index] == -1:
          continue
        if token_map[end_index] == -1:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue
        summary = ScoreSummary()
        summary.short_span_score = (
            result["start_logits"][start_index] +
            result["end_logits"][end_index])
        summary.cls_token_score = (
            result["start_logits"][0] + result["end_logits"][0])
        summary.answer_type_logits = result["answer_type_logits"]-np.mean(result["answer_type_logits"])
        start_span = token_map[start_index]
        end_span = token_map[end_index] + 1

        # Span logits minus the cls logits seems to be close to the best.
        score = summary.short_span_score - summary.cls_token_score
        predictions.append((score, summary, start_span, end_span))

  short_span = Span(-1, -1)
  long_span = Span(-1, -1)
  if predictions:
    try:
      score, summary, start_span, end_span = sorted(predictions, reverse=True, key=lambda x:x[0])[0]
    except Exception as e:
      print(predictions)
      raise e
    short_span = Span(start_span, end_span)
    for c in example.candidates:
      start = short_span.start_token_idx
      end = short_span.end_token_idx
      if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
        long_span = Span(c["start_token"], c["end_token"])
        break
  #TypeError: Object of type int64 is not JSON serializable
  summary.predicted_label = {
      "example_id": int(example.example_id),
      "long_answer": {
          "start_token": int(long_span.start_token_idx),
          "end_token": int(long_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      },
      "long_answer_score": float(score),
      "short_answers": [{
          "start_token": int(short_span.start_token_idx),
          "end_token": int(short_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      }],
      "short_answers_score": float(score),
      "yes_no_answer": "NONE",
      "answer_type_logits": summary.answer_type_logits.tolist(),
      "answer_type": int(np.argmax(summary.answer_type_logits))

  }

  return summary


def compute_pred_dict(candidates_dict, dev_features, raw_results):
  """Computes official answer key from raw logits."""
  raw_results_by_id = [(int(res["unique_id"]), 1, res) for res in raw_results]

  examples_by_id = [(int(k), 0, v) for k, v in candidates_dict.items()]

  features_by_id = [(int(d['unique_id']), 2, d) for d in dev_features]  # list(zip(feature_ids, features))

  # Join examplew with features and raw results.
  examples = []
  logger.info('Merging examples')
  merged = sorted(examples_by_id + raw_results_by_id + features_by_id)

  for idx, data_type, datum in merged:
    if data_type == 0:
      examples.append(EvalExample(idx, datum))
    elif data_type == 1:
      examples[-1].results[idx] = datum
    else:
      examples[-1].features[idx] = datum

  # Construct prediction objects.
  logger.info("Computing predictions...")
  nq_pred_dict = {}
  for e in examples:
    summary = compute_predictions(e)
    nq_pred_dict[e.example_id] = summary.predicted_label
    if len(nq_pred_dict) % 100 == 0:
      logger.info("Examples processed: %d", len(nq_pred_dict))
  logger.info("Done computing predictions.")

  return nq_pred_dict


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `{do_train,do_predict}` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_precomputed_file:
      raise ValueError("If `do_train` is True, then `train_precomputed_file` "
                       "must be specified.")
    if not FLAGS.train_num_precomputed:
      raise ValueError("If `do_train` is True, then `train_num_precomputed` "
                       "must be specified.")

  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))

def make_tfrecords(fname, is_training=True, num_shards=10):
  tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  num_spans_to_ids = collections.defaultdict(list)

  eval_writers = []
  for i in range(num_shards):
    tf_fname = FLAGS.train_tfrecord_file.replace('??','%2d'%i)
    logger.info(tf_fname)
    eval_writer = FeatureWriter(filename=tf_fname,is_training=True)
    eval_writers.append(eval_writer)

  with open(fname,'r') as input_file:
    line_cnt = -1
    n_examples,n_features = 0,0
    for line in tqdm(input_file):
      line_cnt += 1
      example_line = create_example_from_jsonl(line)
      if example_line is None:
        continue
      examples = read_nq_entry(example_line, is_training)
      for example in examples:
        n_examples += 1
        example_index = example.example_id
        features = convert_single_example(example, tokenizer, is_training)
        num_spans_to_ids[len(features)].append(example.qas_id)
        for feature in features:
          n_features += 1
          feature.example_index = example_index
          feature.unique_id = feature.example_index + feature.doc_span_index
          shard_idx = n_features % num_shards
          try:
            eval_writers[shard_idx].process_feature(feature)
          except:
            logger.info('Processing line:%d example:%d features:%d wrong',line_cnt, n_examples, n_features)
        del features,  example,
      del example_line, examples

  logger.info("  Num line examples = %d", line_cnt)
  logger.info("  Num orig examples = %d", n_examples)
  logger.info("  Num split examples = %d", n_features)
  for spans, ids in num_spans_to_ids.items():
    logger.info("  Num split into %d = %d", spans, len(ids))


def create_short_answer(entry):
  # if entry["short_answer_score"] < 1.5:
  #     return ""
  answer = []
  for short_answer in entry["short_answers"]:
    if short_answer["start_token"] > -1:
      answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))
  if entry["yes_no_answer"] != "NONE":
    answer.append(entry["yes_no_answer"])
  return " ".join(answer)

def create_long_answer(entry):
  # if entry["long_answer_score"] < 1.5:
  # return ""
  answer = []
  if entry["long_answer"]["start_token"] > -1:
    answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))
  return " ".join(answer)

def create_short_answer_v2(entry):
  answer = ''
  if entry["short_answers_score"] < FLAGS.short_answer_score_threshold:
    if entry['answer_type'] == 0:
        answer = ''
    if entry['answer_type'] == 1:
        answer = 'YES'
    if entry['answer_type'] == 2:
        answer = 'NO'
  else:
    for short_answer in entry["short_answers"]:
      if  short_answer["start_token"] > -1:
        answer = str(short_answer["start_token"]) + ":" + str(short_answer["end_token"])
  return answer

def create_long_answer_v2(entry):
    answer = ''
    if entry["long_answer_score"] < FLAGS.long_answer_score_threshold:
        answer = ''
    elif entry["long_answer"]["start_token"] > -1:
        answer = str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"])
    return answer

def make_submission(prediction_json_fname):
  test_answers_df = pd.read_json(prediction_json_fname)
  for var_name in ['long_answer_score', 'short_answers_score', 'answer_type']:
    test_answers_df[var_name] = test_answers_df['predictions'].apply(lambda q: q[var_name])

  test_answers_df["long_answer"] = test_answers_df["predictions"].apply(create_long_answer_v2)
  test_answers_df["short_answer"] = test_answers_df["predictions"].apply(create_short_answer_v2)
  test_answers_df["example_id"] = test_answers_df["predictions"].apply(lambda q: str(q["example_id"]))


  long_answers = dict(zip(test_answers_df["example_id"], test_answers_df["long_answer"]))
  short_answers = dict(zip(test_answers_df["example_id"], test_answers_df["short_answer"]))

  sample_submission = pd.read_csv(FLAGS.sample_submission)

  long_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_long")].apply(
    lambda q: long_answers[q["example_id"].replace("_long", "")], axis=1)
  short_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_short")].apply(
    lambda q: short_answers[q["example_id"].replace("_short", "")], axis=1)

  sample_submission.loc[
    sample_submission["example_id"].str.contains("_long"), "PredictionString"] = long_prediction_strings
  sample_submission.loc[
    sample_submission["example_id"].str.contains("_short"), "PredictionString"] = short_prediction_strings

  sample_submission.to_csv("submission.csv", index=False)