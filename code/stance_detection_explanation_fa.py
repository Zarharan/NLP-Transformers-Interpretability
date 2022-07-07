import torch
from transformers import AutoTokenizer
from transformers import XLMRobertaForSequenceClassification
import pandas as pd
from code.shap_explainer import ShapExplainer


device = None
print("-"*20)
if torch.cuda.is_available():
  device = torch.device('cuda')
  print("Device is cuda.\n")
else:
  device = torch.device('cpu')
  print("Device is cpu.\n")


vectorizer_path = "xlm_roberta_longformer_base_4096"
vectorizer = AutoTokenizer.from_pretrained(vectorizer_path)

"""# Define Model"""

# The path of your saved model or the name of Hugging Face transformers.
saveed_model_path= ""
assert len(saveed_model_path)>0, "Please set model save directory!"
num_labels=4
model= XLMRobertaForSequenceClassification.from_pretrained(saveed_model_path, num_labels=num_labels).cuda()
print(saveed_model_path + " load successfully.")
result_explainer = ShapExplainer(model, vectorizer)


"""## Read Test Set"""

# You may need some preprocessing before passing your test instances to the model. I eliminated this section for brevity.
original_df = pd.read_excel("test_set_path")
claims = original_df["claim"]
texts = original_df["text"]
labels = original_df["stance"]


"""# Explain Model """

for claim, text, label in zip(claims, texts, labels):
  try:
    print("claim: ", claim)
    print("True label is: ", result_explainer.orginal_label[label])
    prediction = result_explainer.get_prediction(claim, text)
    print("Predicted label is: ", prediction[0], prediction[1])
    print()
    result_explainer.explain(claim, text, False)
    print("-"*100)
    print()
  except Exception as e:
    print(e)