EvoPep Operation Instructions
All executable scripts are located in the script folder. Below is the detailed workflow description and visualization for the standard EvoPep pipeline.
Standard Workflow Overview
The standard EvoPep workflow consists of two main modules executed sequentially.
● Module 1: Focuses on generation, screening, and ranking.
● Module 2: Focuses on sequence calculation and machine learning screening.
Module 1: Generation and Screening
1. Run generation_and_experience_screening
2. Run batch_generate_property
3. Run Transformer_screen
○ Note: Alternative model versions are available. To use the CNN or LSTM versions, replace this step with CNN_screen or LSTM_screen respectively.
4. Run rank
○ Configuration: Within the rank script, you can specify the number of peptides to pass to Module 2. The default is set to 1000.
Module 2: Property Calculation and Screening
1. Run cal_pep_from_seq
2. Run ML_screen
