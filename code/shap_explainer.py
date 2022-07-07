import shap
import torch
import numpy as np


class ShapExplainer():
    def __init__(self, model, vectorizer):
        self.model = model
        self.model.eval()
        self.vectorizer = vectorizer
        self.orginal_label = ["Disagree","Agree","Discuss","Unrelated"]
        self.gpu_explainer = shap.Explainer(self.__model_prediction_gpu, self.vectorizer, output_names=self.orginal_label)


    def __model_prediction_gpu(self, x):
        tv = torch.tensor([self.vectorizer.encode(v, padding='max_length', 
                                            max_length=512, truncation=True) for v in x]).cuda()
        attention_mask = (tv!=0).type(torch.int64).cuda()
        outputs = self.model(tv, attention_mask=attention_mask)[0]
        scores = torch.nn.Softmax(dim=-1)(outputs)
        val = torch.logit(scores).detach().cpu().numpy()

        return val      


    def get_prediction(self, claim, text):
        input_text = claim + self.vectorizer.sep_token + text
        logits = self.__model_prediction_gpu([input_text])
        return self.orginal_label[np.argmax(logits)], logits


    def explain(self, claim, text, show_detailed_labels= False):
        input_text = claim + self.vectorizer.sep_token + text
        shap_values = self.gpu_explainer([input_text], fixed_context=1)
        output = shap.plots.text(shap_values)
        
        if show_detailed_labels:
            for label in self.orginal_label:
                print()
                print()
                print("-"*50, " " + label + " ", "-"*50)
                shap.plots.text(shap_values[0, :, label])