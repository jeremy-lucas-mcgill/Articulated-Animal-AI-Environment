# **Articulated Animal AI Environment**

**Link to research paper**: /LINK/

This paper presents the **Articulated Animal AI Environment for Animal Cognition**, an enhanced version of the previous AnimalAI Environment. Key improvements include the addition of **agent limbs**, enabling more complex behaviors and interactions with the environment that closely resemble real animal movements. The testbench features an **integrated curriculum training sequence** and **evaluation tools**, eliminating the need for users to develop their own training programs. Additionally, the **tests and training procedures are randomized**, enhancing the agent's generalization capabilities. These advancements significantly expand upon the original AnimalAI framework and will be used to evaluate agents on various aspects of animal cognition.

---

## **How to Use the Articulated Animal AI Interface**

### **Requirements:**
- Please use **Python version 3.9.13**.

### **Ease of Use:**
This GitHub repository contains only 5 Python scripts, the Unity build, and a requirements text file. It's simple to get started.

### **File Descriptions:**

- **`gymAPI.py`:**  
  This file contains the necessary implementation to use the Unity build as a Gym environment.

- **`trainSB3Model.py`:**  
  This file is used specifically for training Stable Baseline models. You can edit the model name, type, and any additional configurations here.

- **`playSB3Model.py`:**  
  This file is used to run any SB3 model that is implemented. Simply change the model name to play your SB3 model.
  
- **`trainCustomModel.py`:**  
  This file is used to train any model that is implemented. The while loop in this file can be configured to fit the training logic your model may require.

- **`playCustomModel.py`:**  
  This file is used to run any model that is implemented. The while loop can be configured to fit the evaluation logic required by your model.

- **`requirements.txt`:**  
  This file lists all the necessary packages to run the environment and the above scripts. Ensure these dependencies are installed.

---

### **If Any Questions Arise:**
Please feel free to email **jeremy.lucas@mail.mcgill.ca** for assistance.
