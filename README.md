# Question-Answering


yes_no_answer long answer 并存

```python
class AnswerType(enum.IntEnum):
  """Type of NQ answer."""
  UNKNOWN = 0
  YES = 1
  NO = 2
  SHORT = 3
  LONG = 4

def create_short_answer_v2(entry):
  answer = ''
  if entry["short_answer_score"] < 0:
    if entry['answer_type'] == 0:
        answer = ''
    if entry['answer_type'] == 1:
        answer = 'YES'
    if entry['answer_type'] == 2:
        answer = 'NO'
  elif entry['short_answer']["start_token"] > -1:
     answer = str(entry['short_answer']["start_token"]) + ":" + str(entry['short_answer']["end_token"])
  return answer

def create_long_answer_v2(entry):
    answer = ''
    if entry["answer_type"] == 0:
        answer = ''     
    elif entry["long_answer"]["start_token"] > -1:
        answer = str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"])
    return answer

    


```


Error
topOlevel


 