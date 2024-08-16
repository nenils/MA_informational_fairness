The repository includes the technical implementation of the conducted study on informational fairness.
The technical implementation includes the training of several fair and unfair classification model on the use-case of the home loan application dataset.
The final two models, one fair and one unfair, are then used to predict three exemplary instances accompanied by different sets of explanations.
The sets of explanations presented in the Study are: 
- baseline explanations: local feature importance & demographics
- advanced explanations: counterfactuals & normative explanations
The explantions are generated on the instances selected using the counterfactual algorithm DiCE and the local feature importance algorithm LIME.
For the adversarial models to be able to run, the conda environment described in the file "environment.yml" needs to be set up.
All other files are running in the environment described in the file "requirements.txt".

Where to find what?

The development of the fair model and the counterfacutals and local feature importance of the selected fair prediction: 
- "Adversarial_classifier_undersampled_females.ipynb"

The development of the unfair model and the counterfacutals and local feature importance of the selected fair prediction: 
- "MLP_undersampling_females.ipynb"

The evaluation of several models is done by each model in the Notebooks that have the prefix "Fairness_audit_" and then the model name that is being evaluated.


The last important file is the "Benchmark_counterfactual", that compares the sets of counterfactuals being generated for either the purpose of contestability or recourse with the fair or unfair models prediction function.


The Master thesis can be read in the file: "MA_informational_fairness"
