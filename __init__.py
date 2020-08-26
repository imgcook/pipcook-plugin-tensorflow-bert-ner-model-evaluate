from .bert_ner_evaluate.index import evaluate

def main(data, model, args):
  testCsvPath = data.testCsvPath
  evaluateParam = model.config
  evaluateParam.update({
    "ner": model.model,
    "data_dir": testCsvPath,
    "eval_batch_size": 8
  })
  evaluateResult = evaluate(evaluateParam)

  return {
    "result": evaluateResult
  }
