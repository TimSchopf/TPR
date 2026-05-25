# Think-Probe-Respond: Improving Large Language Models as Judges of Research Idea Novelty


> This repository accompanies the paper *"Think-Probe-Respond: Improving Large Language Models as Judges of Research Idea Novelty"*.  
> We demonstrate that LLMs are miscalibrated judges of research idea novelty, which stems from a systematic bias towards judging ideas as "medium novel". To mitigate this, we propose **T**hink-**P**robe-**R**espond (TPR), a lightweight approach that probes latent novelty judgments from hidden states during the reasoning phase and uses the probed judgments to condition the final response. Across strong baselines, TPR improves novelty judgment performance by 22.30% on average and successfully mitigates the prevalent "medium novelty" bias.

---

## **T**hink-**P**robe-**R**espond (TPR)

![TPR.png](figures/TPR.png)

Building on the finding that LLMs internalize beliefs about research idea novelty that closely mirror those of human experts—and therefore generate comparable novelty rationales—yet are biased towards predicting medium novelty categories, we propose the TPR approach for research idea novelty judgment. TPR explicitly exploits the model’s internal beliefs during reasoning about novelty to yield less biased and more accurate quantitative judgments. These judgments are then reused as conditioning signals to generate textual justifications that are coherent and well-aligned with the predicted numerical scores.

As illustrated in the Figure above, TPR consists of three stages.

1. **Think:** We instruct an LLM to evaluate the novelty of a research idea and to think step by step before producing a final response. For reasoning models that generate think tokens by default, we omit any explicit “think step by step” instruction. Importantly, we provide only textual descriptions of the novelty categories—without numerical scores—and instruct the model to evaluate novelty solely based on these descriptions without generating numerical judgments. This design encourages the model to think about the novelty of research ideas qualitatively, avoiding anchoring its internal representations to explicit numeric outcomes that could bias the reasoning process.
2. **Probe:** Given an LLM with $L$ hidden layers, let $H = h^{(1)}, \dots, h^{(L)}$ represent the stack of hidden states. For a generated sequence of reasoning (“think”) tokens $T = t_{1}, \dots, t_{n}$, we terminate generation upon the production of the final think token $t_{n}$ and extract the hidden state $h^{(L)}_{t_{n}}$. This representation $h^{(L)}_{t_{n}}$ is then used as the input feature vector for a logistic regression probing classifier.
3. **Respond:** We append the textual description of the predicted novelty class to the LLM-generated output and resume generation. Conditioned on both its prior reasoning and the predicted novelty judgment, the LLM generates the final response, which is used as the justification of the novelty judgment. This stage is *critical for ensuring alignment between the numerical novelty judgment of the probing classifier and the LLM-generated textual justification*. Without explicitly conditioning the generation process on the predicted numerical judgment, the LLM may generate justifications that are misaligned with the underlying novelty judgment produced by the probing classifier.

TPR is computationally efficient and lightweight. All LLM parameters remain frozen and are used only at inference. Training is limited to a simple logistic regression classifier, which can be learned efficiently on CPU, in contrast to resource-intensive fine-tuning that requires substantial GPU resources.

---

## 📂 Repository Structure

```
TPR/
│
├── figures/                                   # includes figures from the paper
│
├── src/                                       # Scripts and LLM prompts for experiments
│
├── .gitignore
├── README.md 
└── requirements.txt  
```

Clean implementations of TPR and baseline appraoches will be made available for the camera-ready version of the paper.
